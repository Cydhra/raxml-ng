#include "TreeRepository.hpp"

void TreeRepository::append_candidate(Tree candidate) {
    std::lock_guard<std::mutex> lock(*mutex);
    candidates.push_back(std::move(candidate));
}

TreeList TreeRepository::take_candidate_batch(unsigned int batch_size) {
    std::lock_guard<std::mutex> lock(*mutex);
    TreeList selected;
    if (candidates.size() < batch_size)
        return selected;
    selected.reserve(batch_size);
    while (selected.size() < batch_size) {
        selected.push_back(std::move(candidates.front()));
        candidates.pop_front();
    }
    return selected;
}

std::size_t TreeRepository::candidate_count() const {
    return this->candidates.size();
}
