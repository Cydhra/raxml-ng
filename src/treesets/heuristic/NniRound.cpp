#include "NniRound.hpp"
#include "../SharedBatchResources.hpp"

void NniRound::do_optimize(std::optional<TreeInfo> &tree, unsigned int tree_id, const Options &opts, const TaskGroup &context, SharedBatchResources &, const unsigned int worker_id, const unsigned int thread_id) {
    if (context.is_group_leader(worker_id, thread_id) && tree_id == 0) {
        LOG_INFO_TS << this->batch_name << ": Performing NNI round." << std::endl;
    }

    spr_round_params spr_params;

    this->meta_parameters->auto_configure(opts, spr_params, 2);
    spr_params.radius_max = 1;
    spr_params.thorough = false;
    spr_params.ntopol_keep = 1;

    // reset cutoff info for each tree. This has to be done, even if it is just one tree, to avoid
    // uninitialized cutoff problems
    const auto loglh = tree->loglh();
    spr_params.reset_cutoff_info(loglh, true);

    tree->spr_round(spr_params);
    tree->optimize_branches(1.0, 1);
}
