#include "SplitSamplerSource.hpp"

constexpr size_t NUM_DONOR_TREE = 300;

constexpr size_t NUM_SAMPLE_ATTEMPTS = 300;

// extract splits from tree, encoded as bit vectors
SplitList SplitSamplerSource::extract_splits(const Tree &tree, const bool normalize) {
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

Split SplitSamplerSource::topology_key(const Tree &tree) {
    const auto split_words = extract_splits(tree, true);
    Split result;
    for (const auto &split: split_words)
        result.insert(result.end(), split.begin(), split.end());
    return result;
}

bool SplitSamplerSource::has_majority_split(
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

void SplitSamplerSource::ensure(const RaxmlInstance &instance, const SmartBarrier &barrier,
                                const unsigned int threads_per_worker, const unsigned int worker_id,
                                const unsigned int thread_id,
                                const unsigned int required_trees) {
    const auto begin = std::chrono::steady_clock::now();
    // build_parsimony_msa(instance, false); // TODO initialize in main.cpp in case of checkpoint

    const auto old_list_size = tree_list.size();

    if (required_trees > old_list_size) {
        donor_tree_source->ensure(instance, barrier, threads_per_worker, worker_id, thread_id, (sampled_batches + 1) * NUM_DONOR_TREE);
        const auto donors = donor_tree_source->range(sampled_batches * NUM_DONOR_TREE, (sampled_batches + 1) * NUM_DONOR_TREE);

        filter.reset_filter(instance, donors);

        // generate splits from donor trees
        SplitList donor_splits;
        donor_splits.reserve(NUM_DONOR_TREE);
        for (const auto &tree: donors)
            donor_splits.push_back(topology_key(tree));

        const auto generated = filter.filter_and_rank(generate_candidates(instance, donors, donor_splits, NUM_SAMPLE_ATTEMPTS, seed));
        for (auto &candidate: generated) {
            if (remember_topology(candidate))
                tree_list.push_back(candidate);
        }

        sampled_batches += 1;
    }
}

double SplitSamplerSource::amortized_time(unsigned int batch_size) const {
    return 0.0; // TODO
}

bool SplitSamplerSource::remember_topology(const Tree &candidate) {
    auto topology = topology_key(candidate);

    std::lock_guard lock(*mutex);
    return !topology.empty() && seen_topologies.insert(std::move(topology)).second;
}

Tree SplitSamplerSource::materialize_candidate(const std::vector<CandidateSplit> &selected) const {
    Tree result;
    assert(!selected.empty());

    // CORAX clones every input split during this call. Keep only a borrowed
    // pointer view; the CandidateSplit word vectors outlive materialization.
    std::vector<corax_split_t> split_view;
    split_view.reserve(selected.size());
    for (const auto &split: selected) {
        if (split.words.empty()) {
            return result;
        }

        split_view.push_back(const_cast<corax_split_base_t *>(split.words.data()));
    }

    const auto tip_labels = baseline_tree.tip_labels_cstr();

    corax_split_system_t split_system{};
    split_system.split_count = static_cast<unsigned int>(selected.size());
    split_system.max_support = 1.0;
    split_system.support = nullptr;
    split_system.splits = split_view.data();

    std::unique_ptr<corax_consensus_utree_t, decltype(&corax_utree_consensus_destroy)> materialized(
        corax_utree_from_splits(
            &split_system,
            static_cast<unsigned int>(baseline_tree.num_tips()),
            tip_labels.data()),
        corax_utree_consensus_destroy);

    if (!materialized || !materialized->tree) {
        coraxlib_reset_error();
        return result;
    }

    result.pll_utree(static_cast<unsigned int>(baseline_tree.num_tips()), *materialized->tree);

    // The cloned nodes contain non-owning consensus-data pointers. Clear them
    // before the temporary CORAX consensus object is destroyed.
    auto &utree = const_cast<corax_utree_t &>(result.pll_utree());
    for (std::size_t node_id = 0; node_id < utree.tip_count + utree.inner_count; ++node_id) {
        auto *node = utree.nodes[node_id];
        if (!node) {
            continue;
        }

        auto *current = node;
        do {
            current->data = nullptr;
            current = current->next;
        } while (current && current != node);
    }

    result.reset_brlens();
    return result;
}
