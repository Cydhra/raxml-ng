#include "AuTest.hpp"

#include "../log.hpp"
#include "corax/statistics/bootstrap.h"

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


void AuTest::calculate_p_values() {
}

void AuTest::estimate_parameters() {
    // reset random state to ensure reproducibility independent of previous events
    const auto rstate = corax_random_create(seed);

    // create a row of pointers for the per-site likelihoods
    std::vector<const double *> per_site_lnl_matrix;
    for (auto& tree_parts: persite_loglh) {
        if (tree_parts.size() > 1) {
            // TODO handle multiple partitions: we need to pass all of them into the bootstrapping at once to avoid
            //  having to iterate through all replicates and add them together.
            LOG_ERROR << "AU test cannot handle multiple partitions yet" << std::endl;
            exit(-1);
        }
        per_site_lnl_matrix.push_back(tree_parts[0].data());
    }

    // some debug information
    LOG_INFO << "Run Bootstrapping..." << std::endl;
    LOG_INFO << "There are " << msa->part_count() << " partitions in the MSA with " << msa->taxon_count() << " taxa." << std::endl;



    for (unsigned int part_id = 0; part_id < msa->part_count(); part_id++) {
        const MSA& part_msa = msa.get()->part_msa(part_id);
        corax_RELL_multiscale_bootstrap(rstate,
                                    &test_statistics,
                                    per_site_lnl_matrix.data(),
                                    part_msa.weights().data(),
                                    part_msa.num_sites(),
                                    part_msa.num_patterns(),
                                    num_trees,
                                    num_replicates.data(),
                                    scales.data(),
                                    scales.size());
    }
    // TODO handle fine-grained parallelization: after all workers have generated their bootstrap replicates, we need
    //  to collect them on a master-worker of all worker groups and add them there, and the master will do the AU test
}
