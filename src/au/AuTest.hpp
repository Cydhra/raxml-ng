#ifndef RAXML_AUTEST_HPP_
#define RAXML_AUTEST_HPP_

#include <memory>
#include "../types.hpp"
#include "../PartitionedMSA.hpp"

extern const doubleVector AU_DEFAULT_SCALES;
extern const uintVector AU_DEFAULT_REPS;

/**
 * Calculates RELL bootstrap replicates at different replicate sequence lengths in parallel, calculate empirical CDF
 * functions from the resulting BP counts, and calculate the AU test p-value from that.
 *
 * The class is instanced with a pointer to the MSA and persite-loglikelihood matrix, as well as scaling factors for the
 * replicate sequence lengths and number of replicates per scale, a random seed, and the number of trees.
 *
 * It then allows repeated calculation of the AU test in parallel, as long as the number of trees remains constant.
 * The bootstrapping and p value calculation happens in three steps, since some of them can be run in parallel, and
 * some happen on one thread only, after gathering the data.
 */
class AuTest {
public:
    AuTest(std::shared_ptr<PartitionedMSA> msa,
           const std::vector<std::vector<doubleVector> > &persite_loglh,
           const doubleVector &scales,
           const uintVector &num_replicates,
           long seed) : AuTest(msa, make_persite_vec(persite_loglh), scales, num_replicates, seed) {
    }

    AuTest(std::shared_ptr<PartitionedMSA> msa,
           const std::vector<std::vector<doubleVector>> &reference_logh_matrix,
           const std::vector<std::vector<doubleVector>> &comparison_logh_matrix,
           const doubleVector &scales,
           const uintVector &num_replicates,
           long seed) : AuTest(msa, combine_persite_vec(reference_logh_matrix, comparison_logh_matrix), scales, num_replicates, seed) {
    }

    AuTest(std::shared_ptr<PartitionedMSA> msa,
           std::vector<std::vector<const double *> > loglh_matrices,
           const doubleVector &scales,
           const uintVector &num_replicates,
           long seed) : msa(msa), persite_loglh(std::move(loglh_matrices)),
                        test_statistics(nullptr),
                        normalized_statistics(nullptr),
                        p_values(persite_loglh.size()),
                        finished(false),
                        scales(scales), num_replicates(num_replicates), num_trees(persite_loglh.size()),
                        seed(seed) {
    }

    // custom move constructor
    AuTest(AuTest &&other) noexcept : msa(std::move(other.msa)),
                                      persite_loglh(std::move(other.persite_loglh)),
                                      test_statistics(other.test_statistics),
                                      normalized_statistics(other.normalized_statistics),
                                      p_values(std::move(other.p_values)),
                                      finished(other.finished),
                                      scales(std::move(other.scales)),
                                      num_replicates(std::move(other.num_replicates)),
                                      num_trees(other.num_trees),
                                      seed(other.seed) {
        other.test_statistics = nullptr;
        other.normalized_statistics = nullptr;
    }


    virtual ~AuTest() {
        free_test_statistics();
    }

    // disable copy-construction and assignment, since we are guarding a C-style allocation
    AuTest(const AuTest &) = delete;

    AuTest &operator=(const AuTest &) = delete;

    /**
     * Replace a range of the input likelihood vectors for this AU-Test with new vectors.
     * Input vectors are contiguously replaced starting from the given index.
     * This method will also clear the relevant parts of the test statistics.
     * If the input vectors are replaced incompletely, but the test statistics are finalized inplace, they need to be
     * cleared manually.
     *
     * @param start_index the tree in the input tree list where to start replacing
     * @param new_vectors the per-site log-likelihood matrices of the new trees
     */
    void replace_persite_loglh(unsigned int start_index, const std::vector<std::vector<doubleVector> > &new_vectors);

    /**
     *  Allocate the test statistic arrays for bootstrap replicates for all trees.
     *
     *  @param inplace if true, normalized statistics are stored in-place in the statistics array, overwriting
     *  bootstrap replicates.
     */
    void allocate_test_statistics(bool inplace = true);

    /**
     * Run the RELL bootstrapping on the per-site log-likelihood vectors assigned to this instance.
     * Since the test assumes that no fine-grained parallelization happens across MPI ranks, the vectors are all
     * complete.
     *
     * @param num_rows how many trees to run bootstrap for. This should match `num_trees` given to the constructor,
     *                  except if the method is called with `offset` to run partial bootstraps in parallel
     * @param offset    offset in the tree array. The function will run bootstrap for all trees in the range
     *                  `offset..offset + num_trees` (exclusive).
     */
    void run_bootstrap(size_t num_rows, size_t offset = 0);

    /**
     * Run the normalization step. This can only be done on the entire replicate matrix, it cannot be called on partial
     * matrices, so before this is called (on the master thread), all other threads need to be done.
     */
    void finalize_test_statistics();

    /**
     * Calculate the p-values from the previously generated bootstrap replicates.
     * These are cheap to compute compared to bootstrapping, but require finalized test statistics, so they can either
     * be calculated on the master worker which holds all bootstrap matrices, or the bootstrap matrices need to be
     * broadcasted to other workers.
     */
    void calculate_p_values();

    /**
     * Free the allocated statistics without freeing the AuTest instance.
     * If the AuTest instance is freed, this function is called automatically, so it is not mandatory to call it.
     */
    void free_test_statistics();

    /**
     * Get the finished p-values after calling `calculate_p_values`.
     *
     * @return A vector of p values, in the order of the rows of the persite log-likelihood matrix
     */
    doubleVector &get_p_values();

    /**
     * Reset the test statistics to be able to re-run bootstrapping.
     * This doesn't affect the normalized statistics, because those are overwritten by subsequent calls to
     * finalize_test_statistics. Only the bootstrap replicates are additive between subsequent calls.
     */
    void reset_test_statistics();

    /**
     * @return true, if the p-values have been calculated
     */
    bool is_finished() const;

private:
    const std::shared_ptr<PartitionedMSA> msa;

    std::vector<std::vector<const double *> > persite_loglh;

    /**
     * The bootstrap test statistics as generated by `run_bootstrap` and finalized by `finalize_test_statistics`.
     * When these change, the `p_values` become invalidated until `calculate_p_values` is called.
     */
    double **test_statistics;

    /**
     * The normalized test statistics generated by a call to normalize_test_statistics
     */
    double **normalized_statistics;

    /**
     * The finished AU test p values, which are calculated by `calculate_p_values`. These values are invalid as long
     * as `finished` is false.
     */
    doubleVector p_values;

    /**
     * True if calculate_p_values has been called and the test statistics have not been changed since then
     */
    bool finished;

    const doubleVector &scales;

    const uintVector &num_replicates;

    const unsigned int num_trees;

    const int seed;

    /**
     * Construct arrays of raw pointers from a persite_loglh matrix array, where each matrix gets turned into a linear
     * array of pointers onto the matrix rows.
     * This prepares the per-partition matrices as double** parameters for coraxlib, and allows us to flexibly combine
     * per-site log-likelihood vectors from various sources.
     * @param persite_loglh a list of vectors (one per tree search) of vectors (one per partition)
     *                      of per-site log-likelihoods
     * @return a vector of double** pointers.
     */
    static std::vector<std::vector<const double *> > make_persite_vec(
        const std::vector<std::vector<doubleVector> > &persite_loglh) {
        std::vector<std::vector<const double *>> loglh_matrices;
        loglh_matrices.reserve(persite_loglh.size());

        append_persite_lnl_vectors(loglh_matrices, persite_loglh);

        return loglh_matrices;
    }

    /**
     * Construct arrays of raw pointers from a persite_loglh matrix array, where each matrix gets turned into a linear
     * array of pointers onto the matrix rows.
     * This prepares the per-partition matrices as double** parameters for coraxlib, and allows us to flexibly combine
     * per-site log-likelihood vectors from various sources.
     *
     * @param first the first vector of per-site log-likelihood vectors
     * @param second the second vector of per-site log-likelihood vectors that gets appended to the matrix
     * @return a vector of double** pointers.
     */
    static std::vector<std::vector<const double *> > combine_persite_vec(
    const std::vector<std::vector<doubleVector>> &first,
       const std::vector<std::vector<doubleVector>> &second) {
        std::vector<std::vector<const double *>> loglh_matrices;

        loglh_matrices.reserve(first.size() + second.size());

        append_persite_lnl_vectors(loglh_matrices, first);
        append_persite_lnl_vectors(loglh_matrices, second);

        return loglh_matrices;
    }

    /**
     * Append pointers to the entries in persite_loglh to the back of loglh_matrices.
     */
    static void append_persite_lnl_vectors(
        std::vector<std::vector<const double *>> &loglh_matrices,
        const std::vector<std::vector<doubleVector> > &persite_loglh
        ) {
        for (auto &partitions: persite_loglh) {
            std::vector<const double *> partition_logh;
            partition_logh.reserve(partitions.size());
            for (auto &partition: partitions) {
                partition_logh.push_back(partition.data());
            }
            loglh_matrices.push_back(partition_logh);
        }
    }
};

#endif //RAXML_AUTEST_HPP_
