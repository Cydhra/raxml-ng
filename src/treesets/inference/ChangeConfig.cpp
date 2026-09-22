#include "ChangeConfig.hpp"
#include "../batch/SharedBatchResources.hpp"

void ChangeConfig::do_optimize(std::optional<TreeInfo> &tree, const unsigned int tree_id, const Options &opts,
    const TaskGroup &context, SharedBatchResources &resources, const unsigned int worker_id, const unsigned int thread_id) {

    if (context.is_group_leader(worker_id, thread_id) && tree_id == 0) {
        LOG_INFO_TS << this->batch_name << ": Changing inference configuration..." << std::endl;
    }

    if (this->model_override) {
        const auto model = Model(*this->model_override);
        tree.emplace(opts, this->batch_start_trees->at(tree_id), *this->msa,
                                                 *this->tip_msa_idmap, this->part_assignments->at(thread_id), &model);
    } else {
        if (site_masks) {
            tree.emplace(opts, this->batch_start_trees->at(tree_id), *this->msa,
                     *this->tip_msa_idmap, this->part_assignments->at(thread_id), *site_masks);

        } else {
            tree.emplace(opts, this->batch_start_trees->at(tree_id), *this->msa,
                                 *this->tip_msa_idmap, this->part_assignments->at(thread_id));
        }
        assign_models(*tree, *this->initial_model);
    }
}
