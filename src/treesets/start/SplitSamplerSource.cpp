#include "SplitSamplerSource.hpp"

#include "../../bootstrap/ConsensusTree.hpp"

TreeList SplitSamplerSource::generate_candidates(const RaxmlInstance &, const ConstTreeRange &donor_pool,
                                                  const SplitList &donor_splits, unsigned int requested_candidates,
                                                  unsigned long round_seed) {
    TreeList candidates;

    coraxlib_reset_error();

    const auto tip_count = static_cast<unsigned int>(baseline_tree.num_tips());
    const auto max_splits = tip_count > 3 ? tip_count - 3 : 0;
    const auto bits_per_word = static_cast<unsigned int>(sizeof(corax_split_base_t) * 8);
    const auto words_per_split = tip_count / bits_per_word + static_cast<unsigned int>(
                                     tip_count % bits_per_word != 0);

    if (max_splits == 0 || words_per_split == 0) {
        return candidates;
    }

    std::vector<CandidateSplit> split_pool;

    auto collect_tree_splits = [&](const Tree &tree, const bool from_ml) {
        assert(!tree.empty() && tree.num_tips() == tip_count);

        for (auto &words: extract_splits(tree, false)) {
            auto existing = std::find_if(
                split_pool.begin(),
                split_pool.end(),
                [&words](const CandidateSplit &candidate) {
                    return candidate.words == words;
                });

            if (existing == split_pool.end()) {
                split_pool.push_back({std::move(words), 0, 0});
                existing = split_pool.end() - 1;
            }

            if (from_ml) {
                ++existing->ml_frequency;
            } else {
                ++existing->donor_frequency;
            }
        }
    };

    for (const auto &initial_ml_tree: initial_ml_trees) {
        collect_tree_splits(initial_ml_tree, true);
    }
    for (const auto &tree: donor_pool) {
        collect_tree_splits(tree, false);
    }

    candidates.reserve(donor_pool.size());
    unsigned int incomplete_split_systems = 0;
    unsigned int materialization_failures = 0;

    // Eight donors is the one retained reference-algorithm choice. Keep it
    // local to the generator instead of adding a global policy constant.
    const auto seed_window_size = std::min<std::size_t>(8, donor_pool.size());

    const auto attempt_limit = requested_candidates * 2;
    unsigned int deduplicated_this_call = 0;
    std::set<Split> staged_topologies;

    for (unsigned int attempt = 0; attempt < attempt_limit && candidates.size() < requested_candidates; ++attempt) {
        TreeList seed_trees;
        seed_trees.reserve(seed_window_size);
        std::vector<std::size_t> seed_ids;
        seed_ids.reserve(seed_window_size);
        for (std::size_t offset = 0; offset < seed_window_size; ++offset) {
            const auto donor_id = (attempt + offset) % donor_pool.size();
            seed_ids.push_back(donor_id);
            seed_trees.push_back(donor_pool[donor_id]);
        }

        try {
            if (!has_majority_split(donor_splits, seed_ids, words_per_split)) {
                ++incomplete_split_systems;
                continue;
            }

            ConsensusTree seed_tree(seed_trees, ConsenseCutoff::MR);
            seed_tree.compute_support();

            std::vector<CandidateSplit> selected;
            selected.reserve(max_splits);

            auto seed_splits = extract_splits(seed_tree, false);
            if (seed_splits.empty()) {
                ++materialization_failures;
                continue;
            }

            for (auto &words: seed_splits) {
                const auto ranked = std::find_if(
                    split_pool.begin(),
                    split_pool.end(),
                    [&words](const CandidateSplit &candidate) {
                        return candidate.words == words;
                    });

                selected.push_back(
                    ranked == split_pool.end()
                        ? CandidateSplit{std::move(words), 0, 0}
                        : *ranked);
            }

            auto try_add = [&](const CandidateSplit &candidate) {
                if (selected.size() >= max_splits) {
                    return;
                }

                for (const auto &existing: selected) {
                    if (existing.words == candidate.words ||
                        !corax_utree_split_compatible(const_cast<corax_split_base_t *>(existing.words.data()),
                                                      const_cast<corax_split_base_t *>(candidate.words.data()),
                                                      words_per_split, tip_count)) {
                        return;
                    }
                }

                selected.push_back(candidate);
            };

            // Primary pass: greedily prefer splits seen in the fixed ML set.
            for (const auto &candidate: split_pool) {
                if (candidate.ml_frequency > 0) {
                    try_add(candidate);
                }
            }

            // Completion pass: preserve donor-frequency order. The rotating
            // consensus seed already supplies per-attempt diversity.
            for (std::size_t offset = 0; offset < split_pool.size() && selected.size() < max_splits; ++offset) {
                const auto split_id = (static_cast<std::size_t>(attempt) + offset + round_seed) % split_pool.size();
                try_add(split_pool[split_id]);
            }

            if (selected.size() != max_splits) {
                ++incomplete_split_systems;
                continue;
            }

            Tree candidate = materialize_candidate(selected);
            if (candidate.empty() ||
                !candidate.binary() ||
                !seed_tree.compatible(candidate)) {
                coraxlib_reset_error();
                ++materialization_failures;
                continue;
            }

            if (auto topology = topology_key(candidate);
                topology.empty() || seen_topologies.find(topology) != seen_topologies.end() ||
                !staged_topologies.insert(std::move(topology)).second) {
                ++deduplicated_this_call;
                continue;
            }

            candidates.push_back(std::move(candidate));
        } catch (const std::exception &error) {
            coraxlib_reset_error();
            ++materialization_failures;
            LOG_WORKER_TS(LogLevel::info)
                    << "Treeset seed-greedy candidate skipped: attempt="
                    << attempt
                    << ", reason=" << error.what()
                    << std::endl;
        }
    }

    LOG_WORKER_TS(LogLevel::info)
            << "Treeset seed-greedy generation: initial_ml="
            << initial_ml_trees.size()
            << ", donors=" << donor_pool.size()
            << ", split_pool=" << split_pool.size()
            << ", incomplete=" << incomplete_split_systems
            << ", materialization_failures=" << materialization_failures
            << ", generated=" << candidates.size()
            << ", attempts=" << attempt_limit
            << ", deduplicated_this_call=" << deduplicated_this_call
            << ", seen_topologies=" << seen_topologies.size() + staged_topologies.size()
            << std::endl;

    return candidates;
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
