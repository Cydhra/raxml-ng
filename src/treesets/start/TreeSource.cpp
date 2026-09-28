#include "TreeSource.hpp"

std::tuple<unsigned int, unsigned int> TreeSource::consume_batch(const RaxmlInstance &instance, const SmartBarrier &barrier, const unsigned int threads_per_worker, unsigned int worker_id, unsigned int thread_id, unsigned int num_trees) {
    const auto start_index = acquire_reservation(barrier, worker_id, thread_id, num_trees);
    this->ensure(instance, barrier, threads_per_worker, worker_id, thread_id, start_index + num_trees);
    return {start_index, start_index + num_trees};
}

void TreeSource::copy_tree(Tree &target, const unsigned int tree_id) const {
    target = tree_list.at(tree_id);
}

unsigned int TreeSource::acquire_reservation(const SmartBarrier &barrier, const unsigned int worker_id, const unsigned int thread_id, const unsigned int num_trees) {
    if (worker_id + thread_id == 0) {
        tree_reserve_mutex->lock();
    }
    barrier.enter();
    const auto start_index = this->tree_cursor;
    barrier.enter();

    this->tree_cursor += num_trees;
    if (worker_id + thread_id == 0) {
        tree_reserve_mutex->unlock();
    }

    return start_index;
}
