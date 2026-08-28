#include "StartTrees.hpp"
#include <optional>
#include "../../TreeInfo.hpp"

void StartTrees::do_optimize(std::optional<TreeInfo> &tree, const unsigned int tree_id, const Options &opts, const TaskGroup &context, SharedBatchResources &, const unsigned int worker_id, const unsigned int thread_id) {
    assert(!tree.has_value());

    // time measurement
    const auto begin = std::chrono::steady_clock::now();

    // barrier so we dont start building tree-info objects without finished trees (since the thread assignment changes)
    context.enter_barrier();

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

    if (context.is_group_leader(worker_id, thread_id)) {
        const auto end = std::chrono::steady_clock::now();

        const unsigned int elapsed = static_cast<unsigned int>(std::chrono::duration_cast<
            std::chrono::milliseconds>(end - begin).count());
        this->cumulative_wall_time += elapsed;

        LOG_INFO_TS << this->batch_name << ": total batch time after generating starting trees: " << this->cumulative_wall_time << "ms."
                <<
                std::endl;
    }
}
