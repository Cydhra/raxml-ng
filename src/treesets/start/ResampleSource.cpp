#include "ResampleSource.hpp"

constexpr size_t NUM_SAMPLE_ATTEMPTS = 300;

// extract splits from tree, encoded as bit vectors
SplitList ResampleSource::extract_splits(const Tree &tree, const bool normalize) {
    SplitList result;
    if (tree.empty() || tree.num_splits() == 0)
        return result;

    const auto tip_count = static_cast<unsigned int>(tree.num_tips());
    const auto bits_per_word = static_cast<unsigned int>(sizeof(corax_split_base_t) * 8);
    const auto words_per_split = tip_count / bits_per_word +
                                 static_cast<unsigned int>(tip_count % bits_per_word != 0);
    const PllSplitSharedPtr splits(corax_utree_split_create(&tree.pll_utree_root(), tip_count, nullptr),
                                   corax_utree_split_destroy);
    if (!splits) {
        return result;
    }

    if (normalize) {
        corax_utree_split_normalize_and_sort(
            splits.get(), tip_count,
            static_cast<unsigned int>(tree.num_splits()), 1);
    }

    result.reserve(tree.num_splits());
    for (std::size_t split_id = 0; split_id < tree.num_splits(); ++split_id) {
        result.emplace_back(splits.get()[split_id],
                            splits.get()[split_id] + words_per_split);
    }
    return result;
}

std::vector<corax_split_base_t> ResampleSource::topology_key(const Tree &tree) {
    const auto split_words = extract_splits(tree, true);
    Split result;
    for (const auto &split: split_words)
        result.insert(result.end(), split.begin(), split.end());
    return result;
}

bool ResampleSource::has_majority_split(
    const SplitList &donor_topologies,
    const std::vector<std::size_t> &donor_ids,
    const std::size_t words_per_split) {
    std::map<Split, std::size_t> frequencies;
    for (const auto donor_id: donor_ids) {
        const auto &topology = donor_topologies[donor_id];
        for (std::size_t offset = 0; offset < topology.size(); offset += words_per_split) {
            std::vector split(
                topology.begin() + offset,
                topology.begin() + offset + words_per_split);
            ++frequencies[std::move(split)];
        }
    }
    return std::any_of(frequencies.begin(), frequencies.end(),
                       [&donor_ids](const auto &entry) {
                           return entry.second > donor_ids.size() / 2;
                       });
}

bool ResampleSource::ensure(const RaxmlInstance &, const SmartBarrier &,
                            const unsigned int, const unsigned int,
                            const unsigned int,
                            const unsigned int required_trees) {
    // build_parsimony_msa(instance, false); // TODO initialize in main.cpp in case of checkpoint

    const auto old_list_size = tree_list.size();

    if (required_trees > old_list_size) {
        return false;
    }

    return true;
}

void ResampleSource::generate(const RaxmlInstance &instance, const SmartBarrier &barrier,
                              const unsigned int num_workers, const unsigned int worker_id,
                              const unsigned int num_trees) {
    const auto begin = std::chrono::steady_clock::now();

    donor_tree_source->ensure(instance, barrier, 1, worker_id, 0, (sampled_batches + 1) * NUM_DONOR_TREE);
    const auto donors = donor_tree_source->range(sampled_batches * NUM_DONOR_TREE,
                                                 (sampled_batches + 1) * NUM_DONOR_TREE);

    gate->reset_gate(instance, donors, barrier, num_workers, worker_id);

    if (worker_id == 0) {
        local_lists.resize(num_workers);
    }

    // generate splits from donor trees
    SplitList donor_splits;
    donor_splits.reserve(NUM_DONOR_TREE);
    for (const auto &tree: donors)
        donor_splits.push_back(topology_key(tree));

    // generate assignment
    CoarseAssignment bootstrap_tree_ids(num_trees);
    std::iota(bootstrap_tree_ids.begin(), bootstrap_tree_ids.end(), 0);
    ContiguousCoarseLoadBalancer balancer{};
    const auto assignment = balancer.get_proc_assignments(bootstrap_tree_ids, num_workers, worker_id);

    const auto seed_offset = worker_id * (2 * assignment.size() + 1);
    barrier.enter();

    // infer trees in parallel
    local_lists[worker_id] = generate_candidates(instance, donors, donor_splits, assignment.size(), seed + seed_offset);

    const auto end = std::chrono::steady_clock::now();
    const unsigned int elapsed = static_cast<unsigned int>(std::chrono::duration_cast<
        std::chrono::milliseconds>(end - begin).count());
    cumulative_wall_time->fetch_add(elapsed);

    barrier.enter();

    // collect trees and evaluate
    if (worker_id == 0) {
        auto all_trees = TreeList();
        for (size_t wid = 0; wid < num_workers; ++wid) {
            all_trees.insert(all_trees.end(), local_lists[wid].begin(), local_lists[wid].end());
        }

        const auto generated = gate->gate_and_rank(std::move(all_trees));

        for (auto &candidate: generated) {
            if (is_unique(candidate))
                tree_list.push_back(candidate);
        }
        sampled_batches += 1;
        LOG_INFO_TS << "Generated " << tree_list.size() << " trees." << std::endl;
    }

    barrier.enter();
}

double ResampleSource::amortized_time(const unsigned int batch_size) const {
    // we do not count time spent in this bandit toward the amortized time. If we did, the multi-armed bandit might
    // decide that other tree sources are more efficient, but we already spent the entire time budget anyway, and we
    // will never refill this tree source, so we won't need information about its efficiency anyway. We pretend that the
    // cost of this bandit is the same as parsimony, and then the MAB will select whatever source has higher success rate.
    return this->donor_tree_source->amortized_time(batch_size);
}

bool ResampleSource::is_unique(const Tree &candidate) {
    auto topology = topology_key(candidate);

    std::lock_guard lock(*duplicate_filter_mutex);
    return seen_topologies.insert(std::move(topology)).second;
}
