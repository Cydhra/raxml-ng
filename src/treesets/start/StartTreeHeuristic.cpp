#include "StartTreeHeuristic.hpp"

void StartTreeHeuristic::generate(Tree &tree, const unsigned int tree_id, const RaxmlInstance &instance, // NOLINT(*-no-recursion)
                                  const TaskGroup &context, SharedBatchResources &resources,
                                  const unsigned int worker_id, const unsigned int thread_id) {
    if (inner) {
        inner->generate(tree, tree_id, instance, context, resources, worker_id, thread_id);
    }

    const auto begin = std::chrono::steady_clock::now();
    do_generate(tree, tree_id, instance, context, resources, worker_id, thread_id);
    const auto end = std::chrono::steady_clock::now();
    const unsigned int elapsed = static_cast<unsigned int>(std::chrono::duration_cast<
        std::chrono::milliseconds>(end - begin).count());

    cumulative_wall_time->fetch_add(elapsed);
}

unsigned int StartTreeHeuristic::get_total_wall_time() const { // NOLINT(*-no-recursion)
    unsigned int total = 0;
    if (inner) {
        total += inner->get_total_wall_time();
    }
    return *cumulative_wall_time + total;
}
