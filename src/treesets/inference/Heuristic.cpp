#include "Heuristic.hpp"

void InferenceHeuristic::optimize(std::optional<TreeInfo> &tree, const unsigned int tree_id, const Options &opts, // NOLINT(*-no-recursion)
              const TaskGroup &context, SharedBatchResources &resources,
              const unsigned int worker_id, const unsigned int thread_id) {
    if (inner) {
        inner->optimize(tree, tree_id, opts, context, resources, worker_id, thread_id);
    }

    if (passed[thread_id * num_trees + tree_id] == 0) {
        const auto begin = std::chrono::steady_clock::now();
        do_optimize(tree, tree_id, opts, context, resources, worker_id, thread_id);
        const auto end = std::chrono::steady_clock::now();
        const unsigned int elapsed = static_cast<unsigned int>(std::chrono::duration_cast<
            std::chrono::milliseconds>(end - begin).count());

        cumulative_wall_time->fetch_add(elapsed);
        per_tree_wall_time[tree_id]->fetch_add(elapsed);
        passed[thread_id * num_trees + tree_id] = 1;
    }
}

unsigned int InferenceHeuristic::get_total_wall_time() const  { // NOLINT(*-no-recursion)
    unsigned int total = 0;
    if (inner) {
        total += inner->get_total_wall_time();
    }
    return *cumulative_wall_time + total;
}

void InferenceHeuristic::get_total_tree_times(std::vector<unsigned int> &accumulator) const { // NOLINT(*-no-recursion)
    if (inner) {
        inner->get_total_tree_times(accumulator);
    }

    for (size_t i = 0; i < per_tree_wall_time.size(); ++i) {
        accumulator[i] += *per_tree_wall_time[i];
    }
}
