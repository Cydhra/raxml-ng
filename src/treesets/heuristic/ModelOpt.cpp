#include "ModelOpt.hpp"
#include "../SharedBatchResources.hpp"

void ModelOpt::do_optimize(std::optional<TreeInfo> &tree, unsigned int, const Options&, const TaskGroup &context, SharedBatchResources &, const unsigned int worker_id, const unsigned int thread_id) {
    const auto opt_model = model && (!this->meta_parameters->skip_model || force);
    const auto opt_branches = branches;

    if (!force && meta_parameters->early_commit) {
        // force hyper-optimization if this batch is early-committing
        epsilon = 0.1;
    }

    const auto begin = std::chrono::steady_clock::now();

    if (opt_model && opt_branches) {
        if (context.is_group_leader(worker_id, thread_id)) {
            LOG_INFO_TS << this->batch_name << ": Optimizing all params (eps: " << epsilon << ")" << std::endl;
        }

        // run all parameters optimization
        tree->optimize_params(CORAX_OPT_PARAM_ALL, epsilon);
    } else if (opt_model) {
        if (context.is_group_leader(worker_id, thread_id)) {
            LOG_INFO_TS << this->batch_name << ": Optimizing model (eps: " << epsilon << ")" << std::endl;
        }

        // run model optimization
        tree->optimize_model(epsilon);
    } else if (branches) {
        if (context.is_group_leader(worker_id, thread_id)) {
            LOG_INFO_TS << this->batch_name << ": Optimizing branches (eps: " << epsilon << ")" << std::endl;
        }

        // run model optimization
        tree->optimize_params(CORAX_OPT_PARAM_BRANCHES_ITERATIVE, epsilon);
    }

    if (context.is_group_leader(worker_id, thread_id)) {
        if (!force) {
            const auto end = std::chrono::steady_clock::now();
            this->wall_time += static_cast<unsigned int>(std::chrono::duration_cast<
                std::chrono::milliseconds>(end - begin).count());
        }

        LOG_INFO_TS << this->batch_name << ": Model Opt complete (eps: " << epsilon << ")" << std::endl;
    }
}
