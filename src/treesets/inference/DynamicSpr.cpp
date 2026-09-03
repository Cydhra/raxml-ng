#include "DynamicSpr.hpp"
#include "../batch/SharedBatchResources.hpp"

void DynamicSpr::do_optimize(std::optional<TreeInfo> &tree, const unsigned int tree_id, const Options &opts, const TaskGroup &context, SharedBatchResources &resources, const unsigned int worker_id, const unsigned int thread_id) {
    auto &optimizer = resources.get_fast_optimizer();
    const auto stop_criterion = resources.get_fast_stop_criterion();
    const auto round_name = thorough ? "SLOW" : (keep_top_k_topol < 20 ? " GREEDY" : " FAST");

    if (context.is_group_leader(worker_id, thread_id) && tree_id == 0) {
        LOG_INFO_TS << this->batch_name << ": Performing " << round_name << " SPR rounds dynamically." << std::endl;
    }

    auto spr_params = this->auto_configure(opts);
    auto &tree_info = tree.value();

    // reset search state
    auto &cm = resources.get_fast_cm();
    cm.reset_search_state();

    // initialize stop criterion
    stop_criterion->initialize_persite_lnl_vectors(&tree_info);
    stop_criterion->set_thread_offset(&tree_info, part_assignments->at(thread_id), ParallelContext::local_proc_id());
    optimizer.set_stopping_criterion(stop_criterion);

    // initialize per-site log-lh for pseudo-bootstrap
    vector<double *> persite_lnl = stop_criterion->get_persite_lnl(ParallelContext::group_id(), ParallelContext::local_thread_id());
    vector<double *> persite_lnl_new = stop_criterion->get_persite_lnl_new(ParallelContext::group_id(),  ParallelContext::local_thread_id());

    // run SPR rounds until stop criterion says there is no more improvement
    double loglh = tree->loglh();
    bool improving = true;
    int iter = 0;

    do
    {
        ++iter;

        stop_criterion->compute_loglh(tree_info, persite_lnl, true);

        const double old_loglh = loglh;
        LOG_PROGRESS(old_loglh) << (spr_params.thorough ? "SLOW" : "FAST") <<
            " spr round " << iter << " (radius: " << spr_params.radius_max << ") for tree #" << tree_id << endl;

        loglh = tree_info.spr_round(spr_params);
        /* optimize ALL branches */
        loglh = tree_info.optimize_branches(1.0, 1);

        stop_criterion->compute_loglh(tree_info, persite_lnl_new, false);

        // if(stop_criterion->multi_test_correction())
        //     stop_criterion->set_increasing_moves((*increasing_moves));

        if(ParallelContext::group_master_thread())
            stop_criterion->run_test();

        ParallelContext::barrier();

        double epsilon = stop_criterion->get_epsilon(ParallelContext::group_id());
        LOG_DEBUG << "KH criterion epsilon = " << epsilon << endl;
        improving = (loglh - old_loglh > epsilon);
    }
    while (improving);
}
