#include "FixedSpr.hpp"
#include "../SharedBatchResources.hpp"

void FixedSpr::do_optimize(TreeInfo &tree, const Options &opts, const TaskGroup &context, SharedBatchResources &,
                           const unsigned int worker_id, const unsigned int thread_id) {
    while (this->meta_parameters->num_fast_spr > this->num_fast_spr_performed || this->meta_parameters->num_slow_spr >
           this->num_slow_spr_performed) {
        const auto fast = this->meta_parameters->num_fast_spr > this->num_fast_spr_performed;
        const auto rounds_performed = fast ? this->num_fast_spr_performed : this->num_slow_spr_performed;
        auto total_rounds = fast ? meta_parameters->num_fast_spr : meta_parameters->num_slow_spr;
        auto num_rounds = total_rounds - rounds_performed;

        // make sure the spr-params are set correctly for fast/slow rounds
        spr_round_params spr_params;
        this->meta_parameters->auto_configure(opts, spr_params, num_fast_spr_performed);
        const auto loglh = tree.loglh();
        spr_params.reset_cutoff_info(loglh, true);

        if (context.is_group_leader(worker_id, thread_id)) {
            auto round_name = fast ? "FAST" : "SLOW";
            LOG_INFO_TS << this->batch_name << ": Optimizing topology (" << num_rounds << " of " << total_rounds <<
                    " total "
                    << round_name << " spr rounds, radius: " << spr_params.radius_max << ")" << std::endl;
        }

        auto begin = std::chrono::steady_clock::now();

        // run optimization kernel
        // reset cutoff between tree searches to avoid under-optimizing a tree with cutoffs from previous trees.
        // this also prevents the cutoff info to have invalid data due to uninitialized instantiation
        for (unsigned int spr_round = rounds_performed; spr_round < total_rounds; ++spr_round) {
            if (context.is_group_leader(worker_id, thread_id)) {
                begin = std::chrono::steady_clock::now();
            }

            tree.spr_round(spr_params);
            tree.optimize_branches(1.0, 1);

            if (context.is_group_leader(worker_id, thread_id)) {
                const auto end = std::chrono::steady_clock::now();

                const unsigned int elapsed = static_cast<unsigned int>(std::chrono::duration_cast<
                    std::chrono::milliseconds>(end - begin).count());
                this->wall_time += elapsed;
            }
        }

        LOG_WORKER_TS(LogLevel::debug) << "performed " << (total_rounds - rounds_performed)
                << (spr_params.ntopol_keep < 20 ? " GREEDY" : " FAST") << " spr rounds (radius: " << spr_params.
                radius_min
                << ") for tree search #?" << std::endl;

        // update the TunedBatch status
        if (context.is_group_leader(worker_id, thread_id)) {
            if (fast) {
                this->num_fast_spr_performed = this->meta_parameters->num_fast_spr;
            } else {
                this->num_slow_spr_performed = this->meta_parameters->num_slow_spr;
            }
        }

        // make sure the spr_performed-variables are updated for all threads before they call auto_configure to avoid
        // desynchronization of whether we perform slow or fast spr rounds, or re-evaluate the loop condition
        context.enter_barrier();
    }
}
