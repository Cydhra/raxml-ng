#include "AuTest.hpp"

#include "../log.hpp"
#include "corax/statistics/au.h"
#include "corax/statistics/bootstrap.h"

const doubleVector AU_DEFAULT_SCALES = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4};
const uintVector AU_DEFAULT_REPS = {10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000};

void AuTest::replace_persite_loglh(const unsigned int start_index, const std::vector<std::vector<doubleVector> > &new_vectors) {
    assert(start_index + new_vectors.size() <= this->persite_loglh.size());
    finished = false;

    unsigned int current_index = start_index;
    for (auto &partitions: new_vectors) {
        std::vector<const double *> partition_logh;
        partition_logh.reserve(partitions.size());
        for (auto &partition: partitions) {
            partition_logh.push_back(partition.data());
        }
        this->persite_loglh[current_index++] = partition_logh;
    }

    for (unsigned int scale = 0; scale < scales.size(); scale++) {
        memset(test_statistics[scale], sizeof(double) * start_index * num_replicates[scale], sizeof(double) * new_vectors.size() * num_replicates[scale]);
    }
}

void AuTest::allocate_test_statistics(bool inplace) {
    if (!corax_RELL_allocate_multiscale_matrices(&test_statistics,
                                                 num_trees,
                                                 num_replicates.data(),
                                                 num_replicates.size())) {
        // TODO handle properly
        exit(-1);
    }

    if (inplace) {
        normalized_statistics = test_statistics;
    } else {
        if (!corax_RELL_allocate_multiscale_matrices(&normalized_statistics,
                                                     num_trees,
                                                     num_replicates.data(),
                                                     num_replicates.size())) {
            // TODO handle properly
            exit(-1);
        }
    }
}


void AuTest::run_bootstrap(const size_t num_rows, const size_t offset) {
    if (!test_statistics) {
        throw RaxmlException("AU test statistics were not allocated");
    }

    // reset random state to ensure reproducibility independent of previous events
    const auto rstate = corax_random_create(seed);

    // mark p values as dirty
    finished = false;

    // some debug information
    LOG_DEBUG_TS << "Run Bootstrapping..." << std::endl;

    // prepare matrix array with offset matrices
    // we collect subarray pointers in `test_statistics_views` and since corax expects double pointers,
    // we create another pointer array onto the subarrays in `test_statistics_pointers`
    std::vector<double *> test_statistics_views(AU_DEFAULT_SCALES.size(), nullptr);
    std::vector<double **> test_statistics_pointers(AU_DEFAULT_SCALES.size(), nullptr);

    for (unsigned int scale_id = 0; scale_id < AU_DEFAULT_SCALES.size(); scale_id++) {
        test_statistics_views[scale_id] = corax_RELL_submatrix(test_statistics[scale_id], offset,
                                                               num_replicates[scale_id]);
        test_statistics_pointers[scale_id] = &test_statistics_views[scale_id];
    }

    for (unsigned int part_id = 0; part_id < msa->part_count(); part_id++) {
        const MSA &part_msa = msa.get()->part_msa(part_id);

        // create a row of pointers for the per-site likelihoods
        std::vector<const double *> per_site_lnl_matrix;
        for (size_t i = offset; i < offset + num_rows; ++i) {
            per_site_lnl_matrix.push_back(persite_loglh[i][part_id]);
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
    LOG_DEBUG_TS << "Calculating test statistics" << std::endl;
    for (unsigned int id_scale = 0; id_scale < AU_DEFAULT_SCALES.size(); id_scale++) {
        corax_normalize_lnl_bootstrap(test_statistics[id_scale], normalized_statistics[id_scale],
                                      num_replicates[id_scale], num_trees);
    }

    // mark p values as dirty
    finished = false;
}


void AuTest::calculate_p_values() {
    LOG_DEBUG_TS << "Calculating " << num_trees << " p-values" << std::endl;
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

    for (unsigned int tree = 0; tree < num_trees; tree++) {
        double d, c;
        double p_value = 0.0;
        const int status = corax_au_p_value(normalized_statistics,
                         tree,
                         AU_DEFAULT_SCALES.data(),
                         num_replicates.data(),
                         AU_DEFAULT_SCALES.size(),
                         corax_bootstrap_expectation(normalized_statistics[best_index], num_replicates[best_index], tree),
                         &d,
                         &c,
                         &p_value);

        if (status == AU_MATH_ERROR) {
            LOG_WARN << "p-value for " << tree << ". tree is unstable." << std::endl;
        }

        p_values[tree] = p_value;

        LOG_DEBUG_TS << "p-value for " << tree << ". tree: " << p_value << std::endl;
    }

    // mark p values as finished
    finished = true;
}

void AuTest::free_test_statistics() {
    if (test_statistics) {
        // free individual matrices
        for (unsigned int i = 0; i < scales.size(); i++) {
            if (!test_statistics[i]) continue;
            free(test_statistics[i]);
        }
        free(test_statistics);
    }

    // if we have normalized statistics that are different from test statistics
    if (normalized_statistics && normalized_statistics != test_statistics) {
        for (unsigned int i = 0; i < scales.size(); i++) {
            if (!normalized_statistics[i]) continue;
            free(normalized_statistics[i]);
        }
        free(normalized_statistics);
    }

    test_statistics = nullptr;
    normalized_statistics = nullptr;
}

void AuTest::reset_test_statistics() {
    finished = false;

    for (unsigned int scale = 0; scale < scales.size(); scale++) {
        memset(test_statistics[scale], 0, sizeof(double) * num_trees * num_replicates[scale]);
    }
}


doubleVector &AuTest::get_p_values() {
    if (!finished) {
        LOG_ERROR << "please call calculate_p_values before calling get_p_values" << std::endl;
        exit(-1);
    }

    return p_values;
}

bool AuTest::is_finished() const {
    return finished;
}
