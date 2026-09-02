#include "StartTrees.hpp"
#include <optional>
#include "../../TreeInfo.hpp"

void StartTrees::do_optimize(std::optional<TreeInfo> &tree, const unsigned int tree_id, const Options &opts, const TaskGroup &, SharedBatchResources &, const unsigned int, const unsigned int thread_id) {
    assert(!tree.has_value());

    // create context for tree inference and assign the initial model
    if (this->model_override) {
        const auto model = Model(*this->model_override);
        tree.emplace(opts, this->batch_start_trees->at(tree_id), *this->msa,
                                                 *this->tip_msa_idmap, this->part_assignments->at(thread_id), &model);
    } else {
        tree.emplace(opts, this->batch_start_trees->at(tree_id), *this->msa,
                                                 *this->tip_msa_idmap, this->part_assignments->at(thread_id));
        assign_models(*tree, *this->initial_model);
    }
}
