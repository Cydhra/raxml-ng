#include "FixedSpr.hpp"
#include "../batch/SharedBatchResources.hpp"

void FixedSpr::do_optimize(std::optional<TreeInfo> &tree, const unsigned int tree_id, const Options &opts,
                           const TaskGroup &context,
                           SharedBatchResources &, const unsigned int worker_id, const unsigned int thread_id) {
    const auto round_name = thorough ? "SLOW" : (keep_top_k_topol < 20 ? " GREEDY" : " FAST");

    // make sure the spr-params are set correctly for fast/slow rounds
    auto spr_params = this->auto_configure(opts);
    const auto loglh = tree->loglh();
    spr_params.reset_cutoff_info(loglh, true);

    if (context.is_group_leader(worker_id, thread_id) && tree_id == 0) {
        LOG_INFO_TS << this->batch_name << ": Optimizing topology (" << this->num_spr << " " << round_name <<
                " spr rounds, radius: " << spr_params.radius_max << ")" << std::endl;
    }

    // run optimization kernel
    // reset cutoff between tree searches to avoid under-optimizing a tree with cutoffs from previous trees.
    // this also prevents the cutoff info to have invalid data due to uninitialized instantiation
    for (unsigned int spr_round = 0; spr_round < this->num_spr; ++spr_round) {
        tree->spr_round(spr_params);
        tree->optimize_branches(1.0, 1);
    }

    LOG_WORKER_TS(LogLevel::debug) << "performed " << num_spr << " " << round_name << " spr rounds (radius: " <<
            spr_params.radius_min << ") for tree search #" << tree_id << std::endl;
}
