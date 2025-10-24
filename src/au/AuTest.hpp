#ifndef RAXML_AUTEST_HPP_
#define RAXML_AUTEST_HPP_

#include <memory>
#include "../types.hpp"
#include "../PartitionedMSA.hpp"

class AuTest {
public:
    AuTest(const std::shared_ptr<PartitionedMSA> &msa,
           const std::vector<std::vector<doubleVector> > &persite_loglh,
           const doubleVector &scales,
           const uintVector &num_replicates,
           long seed);

    virtual ~AuTest() {
        if (!test_statistics) return;

        // free individual matrices
        for (unsigned int i = 0; i < scales.size(); i++) {
            if (!test_statistics[i]) continue;
            free(test_statistics[i]);
        }
        free(test_statistics);
    };

    /**
     *  Allocate the test statistic arrays for bootstrap replicates for all trees.
     */
    void allocate_test_statistics();

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
     * Calculate the p-values from the previously generated bootstrap replicates.
     */
    void calculate_p_values();

private:
    const std::shared_ptr<PartitionedMSA> msa;
    const std::vector<std::vector<doubleVector> > &persite_loglh;
    double **test_statistics;
    const doubleVector &scales;
    const uintVector &num_replicates;
    const unsigned int num_trees;
    const int seed;
};

#endif //RAXML_AUTEST_HPP_
