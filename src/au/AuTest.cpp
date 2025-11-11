#include "AuTest.hpp"

#include "../log.hpp"
#include "corax/statistics/au.h"
#include "corax/statistics/bootstrap.h"

const doubleVector AU_DEFAULT_SCALES = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4};
const uintVector AU_DEFAULT_REPS = { 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000 };

AuTest::AuTest(const std::shared_ptr<PartitionedMSA> &msa,
               const std::vector<std::vector<doubleVector> > &persite_loglh,
               const doubleVector &scales,
               const uintVector &num_replicates, const long seed) : msa(msa),
                                                                    persite_loglh(persite_loglh),
                                                                    test_statistics(nullptr),
                                                                    scales(scales),
                                                                    num_replicates(num_replicates),
                                                                    num_trees(persite_loglh.size()), seed(seed) {
}

void AuTest::allocate_test_statistics() {
    if (!corax_RELL_allocate_multiscale_matrices(&test_statistics,
                                                 num_trees,
                                                 num_replicates.data(),
                                                 num_replicates.size())) {
        // TODO handle properly
        exit(-1);
    }
}


void AuTest::run_bootstrap(const size_t num_rows, const size_t offset) {
    // reset random state to ensure reproducibility independent of previous events
    const auto rstate = corax_random_create(seed);

    // some debug information
    LOG_INFO << "Run Bootstrapping..." << std::endl;
    LOG_INFO << "There are " << msa->part_count() << " partitions in the MSA with " << msa->taxon_count() << " taxa." <<
            std::endl;

    // prepare matrix array with offset matrices
    // we collect subarray pointers in `test_statistics_views` and since corax expects double pointers,
    // we create another pointer array onto the subarrays in `test_statistics_pointers`
    std::vector<double*> test_statistics_views(AU_DEFAULT_SCALES.size(), nullptr);
    std::vector<double**> test_statistics_pointers(AU_DEFAULT_SCALES.size(), nullptr);

    for (unsigned int scale_id = 0; scale_id < AU_DEFAULT_SCALES.size(); scale_id++) {
        test_statistics_views[scale_id] = corax_RELL_submatrix(test_statistics[scale_id], offset, num_replicates[scale_id]);
        test_statistics_pointers[scale_id] = &test_statistics_views[scale_id];
    }

    for (unsigned int part_id = 0; part_id < msa->part_count(); part_id++) {
        const MSA &part_msa = msa.get()->part_msa(part_id);

        // create a row of pointers for the per-site likelihoods
        std::vector<const double *> per_site_lnl_matrix;
        for (size_t i = offset; i < offset + num_rows; ++i) {
            per_site_lnl_matrix.push_back(persite_loglh[i][part_id].data());
        }

        // add up the partial replicates of this partition into test_statistics
        // calling the method like this automatically adds partition replicates together, since the test_statistics
        // array is never cleared by the bootstrap method
        corax_RELL_multiscale_bootstrap(rstate,
                                        test_statistics_pointers.data(),
                                        per_site_lnl_matrix.data(),
                                        part_msa.weights().data(),
                                        part_msa.num_sites(),
                                        part_msa.num_patterns(),
                                        num_rows,
                                        num_replicates.data(),
                                        AU_DEFAULT_SCALES.data(),
                                        AU_DEFAULT_SCALES.size());
    }

    // clean up
    corax_random_destroy(rstate);
}

void AuTest::finalize_test_statistics() {
    for (unsigned int id_scale = 0; id_scale < AU_DEFAULT_SCALES.size(); id_scale++) {
        corax_normalize_lnl_bootstrap(test_statistics[id_scale], num_replicates[id_scale], num_trees);
    }
}


void AuTest::calculate_p_values() {
    // TODO handle fine-grained parallelization: after all workers have generated their bootstrap replicates, we need
    //  to collect them on a master-worker of all worker groups and add them there, and the master will do the AU test

    // find the scale closest to 1.0
    int best_index = 0;
    double best_score = fabs(1.0 - AU_DEFAULT_SCALES[best_index]);
    for (size_t scale_index = 0; scale_index < AU_DEFAULT_SCALES.size(); scale_index++) {
        if (best_score > fabs(1.0 - AU_DEFAULT_SCALES[scale_index])) {
            best_score = fabs(1.0 - AU_DEFAULT_SCALES[scale_index]);
            best_index = scale_index;
        }
    }

    for (unsigned int tree = 0; tree < 100; tree++) {
        double d, c;
        double p_value = 0.0;
        corax_au_p_value(test_statistics,
                         tree,
                         AU_DEFAULT_SCALES.data(),
                         num_replicates.data(),
                         AU_DEFAULT_SCALES.size(),
                         corax_bootstrap_expectation(test_statistics[best_index], num_replicates[best_index], tree),
                         &d,
                         &c,
                         &p_value);

        LOG_INFO << "p-value for " << tree << ". tree: " << p_value << std::endl;
    }
}
