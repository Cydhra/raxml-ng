#include "EbgGate.hpp"

// TODO this can probably be made lower
static constexpr unsigned int BOOTSTRAP_SUPPORT_COUNT = 200;

bool EbgGate::reset_gate(const RaxmlInstance &instance, const ConstTreeRange &support_trees) {
    reference_splits.clear();

    if (bootstrap_support_trees.empty()) {
        bootstrap_support_trees.reserve(BOOTSTRAP_SUPPORT_COUNT);
        while (bootstrap_support_trees.size() < BOOTSTRAP_SUPPORT_COUNT) {
            bootstrap_support_trees.emplace_back(generate_tree(instance, StartingTree::parsimony, 1234567, true));
        }
    }

    auto parsimony_trees = TreeList();
    parsimony_trees.reserve(support_trees.size());

    // we have to copy because EBG needs a list
    for (auto &tree: support_trees) {
        parsimony_trees.emplace_back(tree);
    }

    CandidateEbgSupportTree support_tree(baseline_tree, parsimony_trees, bootstrap_support_trees);
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

TreeList EbgGate::gate_and_rank(TreeList candidates) {
    TreeList selected;
    if (candidates.empty()) {
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
        // LOG_WORKER_TS(LogLevel::info)
        //         << "Treeset aggressive gate: source="
        //         << "split-sampler (either)"
        //         << ", decision=fail"
        //         << ", initial_ml=" << initial_ml_trees.size()
        //         << ", support_trees=" << NUM_DONOR_TREE
        //         << ", bootstrap_support_trees="
        //         << bootstrap_support_trees.size()
        //         << ", ml_ebg_mae=" << ml_mae
        //         << ", candidate_ebg_mae=" << candidate_mae
        //         << std::endl;
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