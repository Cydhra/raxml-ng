#include "FastRaxml.hpp"
#include "../SharedBatchResources.hpp"

void FastRaxml::do_optimize(std::optional<TreeInfo> &tree, unsigned int, const Options&, const TaskGroup &context, SharedBatchResources &resources, const unsigned int worker_id, const unsigned int thread_id) {
    auto &optimizer = resources.get_fast_optimizer();
    const auto stop_criterion = resources.get_fast_stop_criterion();

    if (context.is_group_leader(worker_id, thread_id)) {
        LOG_INFO_TS << this->batch_name << ": Calling RAxML --fast..." << std::endl;
    }

    auto &tree_info = tree.value();

    // reset search state
    auto &cm = resources.get_fast_cm();
    cm.reset_search_state();

    // initialize stop criterion
    stop_criterion->initialize_persite_lnl_vectors(&tree_info);
    stop_criterion->set_thread_offset(&tree_info, part_assignments->at(thread_id), ParallelContext::local_proc_id());
    optimizer.set_stopping_criterion(stop_criterion);

    // optimize using standard raxml
    optimizer.optimize_topology(tree_info, cm);
}
