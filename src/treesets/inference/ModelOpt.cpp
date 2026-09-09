#include "ModelOpt.hpp"
#include "../batch/SharedBatchResources.hpp"

void ModelOpt::do_optimize(std::optional<TreeInfo> &tree, const unsigned int tree_id, const Options &,
                           const TaskGroup &context, SharedBatchResources &, const unsigned int worker_id,
                           const unsigned int thread_id) {
    if (model && branches) {
        if (context.is_group_leader(worker_id, thread_id) && tree_id == 0) {
            LOG_INFO_TS << this->batch_name << ": Optimizing all params (eps: " << epsilon << ")" << std::endl;
        }

        double loglh = tree->optimize_params(CORAX_OPT_PARAM_ALL & ~CORAX_OPT_PARAM_BRANCHES_ITERATIVE, epsilon);
        double new_loglh = tree->optimize_params(CORAX_OPT_PARAM_BRANCHES_ITERATIVE, epsilon);

        while (new_loglh - loglh > epsilon) {
            loglh = new_loglh;
            new_loglh = tree->optimize_params(CORAX_OPT_PARAM_ALL, epsilon);
        }
    } else if (model) {
        if (context.is_group_leader(worker_id, thread_id) && tree_id == 0) {
            LOG_INFO_TS << this->batch_name << ": Optimizing model (eps: " << epsilon << ")" << std::endl;
        }

        double loglh = tree->loglh(true);
        double new_loglh = tree->optimize_model(epsilon);

        while (new_loglh - loglh > epsilon) {
            loglh = new_loglh;
            new_loglh = tree->optimize_model(epsilon);
        }
    } else if (branches) {
        if (context.is_group_leader(worker_id, thread_id) && tree_id == 0) {
            LOG_INFO_TS << this->batch_name << ": Optimizing branches (eps: " << epsilon << ")" << std::endl;
        }

        // run model optimization
        tree->optimize_params(CORAX_OPT_PARAM_BRANCHES_ITERATIVE, epsilon);
    }

    if (context.is_group_leader(worker_id, thread_id) && tree_id == 0) {
        LOG_INFO_TS << this->batch_name << ": Model Opt complete (eps: " << epsilon << ")" << std::endl;
    }
}
