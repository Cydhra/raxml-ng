#include "SplitSamplerSource.hpp"

#include "../../bootstrap/ConsensusTree.hpp"


constexpr size_t NUM_RETRY = 2;

TreeList SplitSamplerSource::generate_candidates(const RaxmlInstance &,
                                                 const ConstTreeRange &donor_pool,
                                                 const SplitList &donor_splits,
                                                 unsigned int requested_candidates,
                                                 const unsigned int donor_offset,
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
    bitv_hashtable_t *split_hash_table = corax_utree_split_hashtable_create(baseline_tree.num_tips(), 0);

    auto collect_tree_splits = [&](const Tree &tree, const bool from_ml) {
        assert(!tree.empty() && tree.num_tips() == tip_count);

        for (auto &words: extract_splits(tree, false)) {
            if (const auto entry = corax_utree_split_hashtable_lookup(split_hash_table, words.data(),
                                                                      baseline_tree.num_tips())) {
                if (from_ml) {
                    split_pool[entry->bip_number].ml_frequency++;
                } else {
                    split_pool[entry->bip_number].donor_frequency++;
                }
            } else {
                const auto new_entry = corax_utree_split_hashtable_insert_single(split_hash_table, words.data(), 1.0);
                split_pool.push_back(CandidateSplit{std::move(words), from_ml ? 1u : 0, from_ml ? 0u : 1});
                assert(split_pool.size() - 1 == new_entry->bip_number);
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
    unsigned int attempt_counter = 0;

    // Eight donors is the one retained reference-algorithm choice. Keep it
    // local to the generator instead of adding a global policy constant.
    const auto seed_window_size = std::min<std::size_t>(8, donor_pool.size());

    unsigned int deduplicated_this_call = 0;
    std::set<Split> staged_topologies;

    for (unsigned int tree_num = 0; tree_num < requested_candidates; ++tree_num) {
        for (unsigned int attempt = 0; attempt < NUM_RETRY; attempt++) {
            TreeList seed_trees;
            seed_trees.reserve(seed_window_size);
            std::vector<std::size_t> seed_ids;
            seed_ids.reserve(seed_window_size);

            for (std::size_t window_index = 0; window_index < seed_window_size; ++window_index) {
                // the donor ids are chosen such that different workers use different donors, and failed attempts do
                // spread out the donor ids to avoid re-using donors of other workers
                const auto donor_id = (donor_offset + tree_num + window_index * (attempt + 1)) % donor_pool.size();

                seed_ids.push_back(donor_id);
                seed_trees.push_back(donor_pool[donor_id]);
            }
            attempt_counter += 1;

            if (!has_majority_split(donor_splits, seed_ids, words_per_split)) {
                ++incomplete_split_systems;
                continue;
            }

            ConsensusTree seed_tree(seed_trees, ConsenseCutoff::MR);
            seed_tree.compute_support();

            std::vector<CandidateSplit> selected;
            selected.reserve(max_splits);

            auto seed_splits = extract_splits(seed_tree, false);

            for (auto &words: seed_splits) {
                const auto ranked = corax_utree_split_hashtable_lookup(split_hash_table, words.data(),
                                                                       baseline_tree.num_tips());

                if (ranked) {
                    selected.push_back(split_pool[ranked->bip_number]);
                } else {
                    selected.push_back(CandidateSplit{std::move(words), 0, 0});
                }
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

            for (std::size_t offset = 0; offset < split_pool.size() && selected.size() < max_splits; ++offset) {
                const auto split_id = (static_cast<std::size_t>(tree_num) + offset + round_seed * 2) % split_pool.
                                      size();
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
            break; // success, no more retry required
        }
    }

    corax_utree_split_hashtable_destroy(split_hash_table);

    LOG_WORKER_TS(LogLevel::info)
            << "Split-Sampler generation: donors="
            << donor_pool.size()
            << ", split_pool=" << split_pool.size()
            << ", incomplete=" << incomplete_split_systems
            << ", materialization_failures=" << materialization_failures
            << ", generated=" << candidates.size()
            << ", attempts=" << attempt_counter
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
