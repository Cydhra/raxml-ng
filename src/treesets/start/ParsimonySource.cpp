#include "ParsimonySource.hpp"

#include "../../pool/SmartBarrier.hpp"

void ParsimonySource::ensure(const RaxmlInstance &instance, const SmartBarrier &barrier, const unsigned int threads_per_worker, const unsigned int worker_id, const unsigned int thread_id, const unsigned int required_trees) {
    // lock the mutex to ensure all threads see the same list size
    if (worker_id + thread_id == 0) {
        tree_list_mutex->lock();
    }

    barrier.enter();

    const auto old_list_size = tree_list.size();
    const auto coarse_thread_id = threads_per_worker * worker_id + thread_id;

    barrier.enter();

    if (required_trees > old_list_size) {
        // resize the list and wait for it
        if (worker_id + thread_id == 0) {
            this->tree_list.resize(required_trees);
        }
        barrier.enter();

        const auto new_trees = required_trees - old_list_size;

        CoarseAssignment tree_ids(new_trees);
        std::iota(tree_ids.begin(), tree_ids.end(), 0);

        auto seeds = std::vector<int>(new_trees);
        std::iota(seeds.begin(), seeds.end(), starting_seed);

        // generate trees according to a coarse thread assignment
        const auto assignment = load_balancer->get_proc_assignments(tree_ids, barrier.threads_required(), coarse_thread_id);

        for (const auto tree_id : assignment) {
            const auto begin = std::chrono::steady_clock::now();
            tree_list.at(old_list_size + tree_id) = generate_tree(instance, StartingTree::parsimony, seeds[tree_id], false);
            const auto end = std::chrono::steady_clock::now();
            const unsigned int elapsed = static_cast<unsigned int>(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
            cumulative_wall_time->fetch_add(elapsed);
            trees_generated->fetch_add(1);
        }

        barrier.enter();
        if (worker_id + thread_id == 0) {
            this->starting_seed += static_cast<int>(new_trees);
        }
    }

    // update starting tree for next method call and unlock mutex
    if (worker_id + thread_id == 0) {
        tree_list_mutex->unlock();
    }
}

double ParsimonySource::amortized_time(unsigned int batch_size) const {
    // not thread-safe but we stay silly
    return static_cast<double>(*cumulative_wall_time) / static_cast<double>(*trees_generated) * static_cast<double>(batch_size);
}
