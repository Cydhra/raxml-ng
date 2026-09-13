#include "AdaptiveRaxml.hpp"
#include "../batch/SharedBatchResources.hpp"

void AdaptiveRaxml::do_optimize(std::optional<TreeInfo> &tree, const unsigned int tree_id, const Options &,
                            const TaskGroup &context, SharedBatchResources &resources, const unsigned int worker_id,
                            const unsigned int thread_id) {
    auto &optimizer = resources.get_adaptive_optimizer();

    if (context.is_group_leader(worker_id, thread_id) && tree_id == 0) {
        LOG_INFO_TS << this->batch_name << ": Calling RAxML --search..." << std::endl;
    }

    // reset search state
    auto &cm = resources.get_adaptive_cm();
    cm.reset_search_state();

    // optimize using standard raxml
    optimizer.optimize_topology(tree.value(), cm);
}
