#include "SplitSamplerSource.hpp"

constexpr size_t NUM_DONOR_TREE = 300;

constexpr size_t NUM_SAMPLE_ATTEMPTS = 300;

// TODO this can probably be made lower
static constexpr unsigned int BOOTSTRAP_SUPPORT_COUNT = 200;

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
        donor_tree_source->ensure(instance, barrier, threads_per_worker, worker_id, thread_id, NUM_DONOR_TREE);

        // TODO change start position if the source was exhausted before
        const auto donors = donor_tree_source->range(0, NUM_DONOR_TREE);

        // generate splits from donor trees
        SplitList donor_splits;
        donor_splits.reserve(NUM_DONOR_TREE);
        for (const auto &tree: donors)
            donor_splits.push_back(topology_key(tree));

        if (reference_splits.empty() && bootstrap_support_trees.empty()) {
            bootstrap_support_trees.reserve(BOOTSTRAP_SUPPORT_COUNT);
            while (bootstrap_support_trees.size() < BOOTSTRAP_SUPPORT_COUNT) {
                bootstrap_support_trees.emplace_back(generate_tree(
                    instance, StartingTree::parsimony, seed + 1234567, true));
            }
        }

        const auto generated = gate_and_rank(generate_candidates(instance, donors, donor_splits, NUM_SAMPLE_ATTEMPTS, seed));
        for (auto &candidate: generated) {
            if (remember_topology(candidate))
                tree_list.push_back(candidate);
        }
    }
}

double SplitSamplerSource::amortized_time(unsigned int batch_size) const {
    return 0.0; // TODO
}

TreeList SplitSamplerSource::gate_and_rank(TreeList candidates) {
    TreeList selected;
    if (candidates.empty() || !prepare_gate()) {
        return selected;
    }
    const auto split_count = reference_splits.size();

    std::vector candidate_frequency(split_count, 0.0);
    for (const auto &candidate: candidates) {
        PllSplitSharedPtr candidate_splits(
            corax_utree_split_create(&candidate.pll_utree_root(),
                                     candidate.num_tips(),
                                     nullptr),
            corax_utree_split_destroy);
        if (!candidate_splits) {
            coraxlib_reset_error();
            return TreeList{};
        }

        for (std::size_t split_id = 0; split_id < split_count; ++split_id) {
            if (corax_utree_split_find(candidate_splits.get(),
                                       reference_splits[split_id].data(),
                                       baseline_tree.num_tips()) >= 0) {
                candidate_frequency[split_id] += 1.0;
            }
        }
    }

    for (auto &support: candidate_frequency) {
        support /= static_cast<double>(candidates.size());
    }

    double ml_mae = 0.0;
    double candidate_mae = 0.0;
    for (std::size_t split_id = 0; split_id < split_count; ++split_id) {
        ml_mae += std::abs(ml_frequency[split_id] - ebg_support[split_id]);
        candidate_mae += std::abs(candidate_frequency[split_id] - ebg_support[split_id]);
    }
    ml_mae /= static_cast<double>(split_count);
    candidate_mae /= static_cast<double>(split_count);

    if (!std::isfinite(ml_mae) || !std::isfinite(candidate_mae) || candidate_mae + mae_margin >= ml_mae) {
        LOG_WORKER_TS(LogLevel::info)
                << "Treeset aggressive gate: source="
                << "split-sampler (either)"
                << ", decision=fail"
                << ", initial_ml=" << initial_ml_trees.size()
                << ", support_trees=" << NUM_DONOR_TREE
                << ", bootstrap_support_trees="
                << bootstrap_support_trees.size()
                << ", ml_ebg_mae=" << ml_mae
                << ", candidate_ebg_mae=" << candidate_mae
                << std::endl;
        return selected;
    }

    const auto promise_reference_count = std::min<std::size_t>(3, initial_ml_trees.size());
    std::vector<PllSplitSharedPtr> promise_splits;
    promise_splits.reserve(promise_reference_count);
    for (std::size_t reference_id = 0; reference_id < promise_reference_count; ++reference_id) {
        const auto &reference = initial_ml_trees[reference_id];
        PllSplitSharedPtr splits(
            corax_utree_split_create(&reference.pll_utree_root(),
                                     reference.num_tips(),
                                     nullptr),
            corax_utree_split_destroy);
        if (splits) {
            promise_splits.push_back(std::move(splits));
        }
    }

    std::deque<RankedCandidate> ranked_candidates;
    unsigned int promise_min = std::numeric_limits<unsigned int>::max();
    unsigned int promise_max = 0;
    unsigned int promise_sum = 0;

    for (auto &candidate: candidates) {
        PllSplitSharedPtr candidate_splits(
            corax_utree_split_create(&candidate.pll_utree_root(),
                                     candidate.num_tips(),
                                     nullptr),
            corax_utree_split_destroy);
        if (!candidate_splits) {
            coraxlib_reset_error();
            return TreeList{};
        }

        unsigned int promise_score = 0;
        for (std::size_t split_id = 0; split_id < candidate.num_splits(); ++split_id) {
            const auto found = std::any_of(
                promise_splits.begin(),
                promise_splits.end(),
                [&](const PllSplitSharedPtr &reference) {
                    return corax_utree_split_find(
                               reference.get(),
                               candidate_splits.get()[split_id],
                               candidate.num_tips()) >= 0;
                });
            if (found) {
                ++promise_score;
            }
        }

        promise_min = std::min(promise_min, promise_score);
        promise_max = std::max(promise_max, promise_score);
        promise_sum += promise_score;
        ranked_candidates.push_back({std::move(candidate), promise_score});
    }

    std::stable_sort(
        ranked_candidates.begin(),
        ranked_candidates.end(),
        [](const RankedCandidate &lhs, const RankedCandidate &rhs) {
            return lhs.promise_score > rhs.promise_score;
        });

    selected.reserve(ranked_candidates.size());
    for (auto &candidate: ranked_candidates) {
        selected.push_back(std::move(candidate.tree));
    }

    const double promise_mean = ranked_candidates.empty()
                                    ? 0.0
                                    : static_cast<double>(promise_sum) /
                                      ranked_candidates.size();
    if (ranked_candidates.empty()) {
        promise_min = 0;
    }

    // LOG_WORKER_TS(LogLevel::info)
    //         << "Treeset aggressive gate: source="
    //         << source_name(source)
    //         << ", decision=pass"
    //         << ", initial_ml=" << initial_ml_trees.size()
    //         << ", support_trees=" << support_tree_count
    //         << ", bootstrap_support_trees="
    //         << bootstrap_support_tree_count
    //         << ", ml_ebg_mae=" << ml_mae
    //         << ", candidate_ebg_mae=" << candidate_mae
    //         << ", selected=" << selected.size()
    //         << std::endl;
    // LOG_WORKER_TS(LogLevel::info)
    //         << "Treeset aggressive ranker: source="
    //         << source_name(source)
    //         << ", promise_min=" << promise_min
    //         << ", promise_mean=" << promise_mean
    //         << ", promise_max=" << promise_max
    //         << std::endl;

    return selected;
}

bool SplitSamplerSource::prepare_gate() {
    if (!reference_splits.empty())
        return true;

    // TODO move start index after source exhaustion
    auto support_trees = TreeList();
    support_trees.reserve(NUM_DONOR_TREE);

    // we have to copy because EBG needs a list
    for (auto &tree: donor_tree_source->range(0, NUM_DONOR_TREE)) {
        support_trees.emplace_back(tree);
    }

    CandidateEbgSupportTree support_tree(baseline_tree, support_trees, bootstrap_support_trees);
    if (!support_tree.compute()) {
        coraxlib_reset_error();
        return false;
    }

    const auto split_count = support_tree.num_splits();
    const auto *splits = support_tree.reference_splits();
    if (split_count == 0 || support_tree.support().size() != split_count || !splits)
        return false;

    const auto tip_count = static_cast<unsigned int>(baseline_tree.num_tips());
    const auto bits_per_word = static_cast<unsigned int>(sizeof(corax_split_base_t) * 8);
    const auto words_per_split = tip_count / bits_per_word +
                                 static_cast<unsigned int>(tip_count % bits_per_word != 0);

    ebg_support = support_tree.support();
    reference_splits.reserve(split_count);
    for (std::size_t split_id = 0; split_id < split_count; ++split_id) {
        reference_splits.emplace_back(words_per_split);
        std::memcpy(reference_splits.back().data(), splits[split_id],
                    words_per_split * sizeof(corax_split_base_t));
    }

    ml_frequency.assign(split_count, 0.0);
    for (const auto &initial_ml_tree: initial_ml_trees) {
        PllSplitSharedPtr initial_splits(
            corax_utree_split_create(&initial_ml_tree.pll_utree_root(),
                                     initial_ml_tree.num_tips(), nullptr),
            corax_utree_split_destroy);
        if (!initial_splits) {
            coraxlib_reset_error();
            reference_splits.clear();
            ebg_support.clear();
            ml_frequency.clear();
            return false;
        }
        for (std::size_t split_id = 0; split_id < split_count; ++split_id) {
            if (corax_utree_split_find(initial_splits.get(), reference_splits[split_id].data(), tip_count) >= 0)
                ml_frequency[split_id] += 1.0;
        }
    }
    for (auto &frequency: ml_frequency)
        frequency /= static_cast<double>(initial_ml_trees.size());

    return true;
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
