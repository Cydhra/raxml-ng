#include "TreeSource.hpp"

#include <optional>

std::optional<std::tuple<unsigned int, unsigned int> > TreeSource::consume_batch(
    const RaxmlInstance &instance, const SmartBarrier &barrier, const unsigned int threads_per_worker,
    const unsigned int worker_id, const unsigned int thread_id, const unsigned int num_trees) {
    if (const auto start_index = acquire_reservation(barrier, worker_id, thread_id, num_trees); this->ensure(
        instance, barrier, threads_per_worker, worker_id, thread_id, start_index + num_trees)) {
        return std::make_optional<std::tuple<unsigned int, unsigned int> >(start_index, start_index + num_trees);
    }

    return std::nullopt;
}

void TreeSource::copy_tree(Tree &target, const unsigned int tree_id) const {
    target = tree_list.at(tree_id);
}

unsigned int TreeSource::acquire_reservation(const SmartBarrier &barrier, const unsigned int worker_id,
                                             const unsigned int thread_id, const unsigned int num_trees) {
    if (worker_id + thread_id == 0) {
        tree_reserve_mutex->lock();
    }
    barrier.enter();
    const auto start_index = this->tree_cursor;
    barrier.enter();

    if (worker_id + thread_id == 0) {
        this->tree_cursor += num_trees;
        tree_reserve_mutex->unlock();
    }

    return start_index;
}

std::deque<Tree>::const_iterator TreeSource::begin() const {
    return this->tree_list.begin();
}

std::deque<Tree>::const_iterator TreeSource::end() const {
    return this->tree_list.end();
}

ConstTreeRange TreeSource::range(const size_t start, const size_t end) const {
    return ConstTreeRange(*this, start, end);
}

size_t TreeSource::total_trees() const {
    return this->tree_list.size();
}

std::deque<Tree>::const_iterator ConstTreeRange::begin() const {
    return this->source.begin() + static_cast<long>(this->start_index);
}

std::deque<Tree>::const_iterator ConstTreeRange::end() const {
    return this->source.begin() + static_cast<long>(this->end_index);
}

size_t ConstTreeRange::size() const {
    return this->end_index - this->start_index;
}

Tree const &ConstTreeRange::operator[](const size_t index) const {
    return *(this->source.begin() + index);
}
