#include "AggressiveCandidateManager.hpp"
#include "../bootstrap/ConsensusTree.hpp"
#include "../bootstrap/EbgSupportTree.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <set>
#include <utility>
#include <vector>


using std::to_string;

void build_parsimony_msa(RaxmlInstance &instance, bool force);
Tree generate_tree(const RaxmlInstance &instance,
                   StartingTree type,
                   int random_seed,
                   bool bootstrap);
Tree generate_constrained_parsimony_tree(
    const RaxmlInstance &instance,
    const Tree &constraint_tree,
    int random_seed);


class CandidateEbgSupportTree : public EbgSupportTree {
public:
    using EbgSupportTree::EbgSupportTree;
    bool compute() { return compute_support(); }
    const corax_split_t *reference_splits() const { return _ref_splits.get(); }
};


namespace {

struct CandidateSplit {
    std::vector<corax_split_base_t> words;
    unsigned int ml_frequency = 0;
    unsigned int donor_frequency = 0;
};

Tree materialize_candidate(const std::vector<CandidateSplit> &selected, const Tree &label_source) {
    Tree result;
    if (selected.empty() || label_source.empty()) {
        return result;
    }

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

    const auto tip_labels = label_source.tip_labels_cstr();

    corax_split_system_t split_system{};
    split_system.split_count = static_cast<unsigned int>(selected.size());
    split_system.max_support = 1.0;
    split_system.support = nullptr;
    split_system.splits = split_view.data();

    std::unique_ptr<corax_consensus_utree_t, decltype(&corax_utree_consensus_destroy)> materialized(
        corax_utree_from_splits(
            &split_system,
            static_cast<unsigned int>(label_source.num_tips()),
            tip_labels.data()),
        corax_utree_consensus_destroy);

    if (!materialized || !materialized->tree) {
        coraxlib_reset_error();
        return result;
    }

    result.pll_utree(static_cast<unsigned int>(label_source.num_tips()), *materialized->tree);

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

} // namespace


TreeList AggressiveCandidateManager::gate_and_rank(TreeList candidates) {
    TreeList selected;
    if (candidates.empty() ||
        baseline_tree.empty() ||
        initial_ml_trees.empty() ||
        support_trees.empty() ||
        bootstrap_support_trees.empty()) {
        return selected;
    }

    CandidateEbgSupportTree support_tree(baseline_tree, support_trees, bootstrap_support_trees);
    if (!support_tree.compute()) {
        coraxlib_reset_error();
        return selected;
    }

    const auto &ebg_support = support_tree.support();
    const auto split_count = support_tree.num_splits();
    const auto *reference_splits = support_tree.reference_splits();
    if (split_count == 0 ||
        ebg_support.size() != split_count ||
        reference_splits == nullptr) {
        return selected;
    }

    std::vector<double> ml_frequency(split_count, 0.0);
    for (const auto &initial_ml_tree: initial_ml_trees) {
        PllSplitSharedPtr initial_ml_splits(
            corax_utree_split_create(&initial_ml_tree.pll_utree_root(),
                                     initial_ml_tree.num_tips(),
                                     nullptr),
            corax_utree_split_destroy);
        if (!initial_ml_splits) {
            coraxlib_reset_error();
            return TreeList{};
        }

        for (std::size_t split_id = 0; split_id < split_count; ++split_id) {
            if (corax_utree_split_find(initial_ml_splits.get(),
                                       reference_splits[split_id],
                                       baseline_tree.num_tips()) >= 0) {
                ml_frequency[split_id] += 1.0;
            }
        }
    }

    for (auto &support: ml_frequency) {
        support /= static_cast<double>(initial_ml_trees.size());
    }

    std::vector<double> candidate_frequency(split_count, 0.0);
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
                                       reference_splits[split_id],
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

    if (!std::isfinite(ml_mae) || !std::isfinite(candidate_mae) || candidate_mae + treeset_aggressive_mae_margin >= ml_mae) {
        LOG_WORKER_TS(LogLevel::info)
            << "Treeset aggressive gate: source="
            << (source == AggressiveSourceFamily::seed_greedy
                    ? "seed_greedy"
                    : "constrained_parsimony")
            << ", decision=fail"
            << ", initial_ml=" << initial_ml_trees.size()
            << ", support_trees=" << support_trees.size()
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

    ranked_candidates.clear();
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

    LOG_WORKER_TS(LogLevel::info)
        << "Treeset aggressive gate: source="
        << (source == AggressiveSourceFamily::seed_greedy
                ? "seed_greedy"
                : "constrained_parsimony")
        << ", decision=pass"
        << ", initial_ml=" << initial_ml_trees.size()
        << ", support_trees=" << support_trees.size()
        << ", bootstrap_support_trees="
        << bootstrap_support_trees.size()
        << ", ml_ebg_mae=" << ml_mae
        << ", candidate_ebg_mae=" << candidate_mae
        << ", selected=" << selected.size()
        << std::endl;
    LOG_WORKER_TS(LogLevel::info)
        << "Treeset aggressive ranker: source="
        << (source == AggressiveSourceFamily::seed_greedy
                ? "seed_greedy"
                : "constrained_parsimony")
        << ", promise_min=" << promise_min
        << ", promise_mean=" << promise_mean
        << ", promise_max=" << promise_max
        << std::endl;

    return selected;
}


void AggressiveCandidateManager::generate_candidates(
    const unsigned int minimum_candidates) {
    if (source == AggressiveSourceFamily::none || initial_ml_trees.empty()) {
        return;
    }

    build_parsimony_msa(instance, false);

    const auto requested_candidates = std::max(std::max(target_tree_count, minimum_candidates), treeset_aggressive_support_trees);
    const unsigned long round_seed = opts.random_seed + 104729UL * static_cast<unsigned long>(++generation_round);

    TreeList donor_pool;
    donor_pool.reserve(requested_candidates);

    for (unsigned int donor_id = 0; donor_id < requested_candidates; ++donor_id) {
        donor_pool.push_back(generate_tree(
            instance,
            StartingTree::parsimony,
            static_cast<int>(round_seed + donor_id),
            false));
    }

    if (support_trees.empty()) {
        support_trees.reserve(treeset_aggressive_support_trees);
        for (std::size_t donor_id = 0; donor_id < treeset_aggressive_support_trees; ++donor_id) {
            support_trees.push_back(donor_pool[donor_id]);
        }
    }

    if (bootstrap_support_trees.empty()) {
        bootstrap_support_trees.reserve(treeset_aggressive_support_trees);
        while (bootstrap_support_trees.size() < treeset_aggressive_support_trees) {
            bootstrap_support_trees.push_back(generate_tree(
                instance,
                StartingTree::parsimony,
                static_cast<int>(round_seed + requested_candidates +
                                treeset_aggressive_support_trees +
                                bootstrap_support_trees.size()),
                true));
        }
    }

    TreeList generated;
    if (source == AggressiveSourceFamily::seed_greedy) {
        generated = generate_seed_greedy_candidates(donor_pool, requested_candidates, round_seed);
    }
    else {
        generated = generate_constrained_parsimony_candidates(donor_pool, requested_candidates, round_seed);
    }

    generated = gate_and_rank(std::move(generated));

    if (generated.size() < minimum_candidates) {
        exhausted = true;
        return;
    }

    for (auto &candidate: generated) {
        candidate_pool.push_back(std::move(candidate));
    }
}


TreeList AggressiveCandidateManager::generate_seed_greedy_candidates(const TreeList &donor_pool, const unsigned int requested_candidates, const unsigned long round_seed) {
    TreeList candidates;

    if (donor_pool.empty() || baseline_tree.empty() || initial_ml_trees.empty()) {
        return candidates;
    }

    coraxlib_reset_error();

    const auto tip_count = static_cast<unsigned int>(baseline_tree.num_tips());
    const auto max_splits = tip_count > 3 ? tip_count - 3 : 0;
    const auto bits_per_word = static_cast<unsigned int>(sizeof(corax_split_base_t) * 8);
    const auto words_per_split = tip_count / bits_per_word + static_cast<unsigned int>(tip_count % bits_per_word != 0);

    if (max_splits == 0 || words_per_split == 0) {
        return candidates;
    }

    std::vector<CandidateSplit> split_pool;

    auto collect_tree_splits = [&](const Tree &tree, const bool from_ml) {
        if (tree.empty() || tree.num_tips() != tip_count) {
            return;
        }

        PllSplitSharedPtr splits(
            corax_utree_split_create(&tree.pll_utree_root(),
                                     tip_count,
                                     nullptr),
            corax_utree_split_destroy);
        if (!splits) {
            coraxlib_reset_error();
            return;
        }

        for (std::size_t split_id = 0; split_id < tree.num_splits(); ++split_id) {
            std::vector<corax_split_base_t> words(words_per_split);
            std::memcpy(words.data(),
                        splits.get()[split_id],
                        words.size() * sizeof(corax_split_base_t));

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

    const auto promise_reference_count = std::min<std::size_t>(3, initial_ml_trees.size());
    for (std::size_t tree_id = 0; tree_id < promise_reference_count; ++tree_id) {
        collect_tree_splits(initial_ml_trees[tree_id], true);
    }
    for (const auto &tree: donor_pool) {
        collect_tree_splits(tree, false);
    }

    std::sort(
        split_pool.begin(),
        split_pool.end(),
        [](const CandidateSplit &lhs, const CandidateSplit &rhs) {
            if (lhs.ml_frequency != rhs.ml_frequency) {
                return lhs.ml_frequency > rhs.ml_frequency;
            }
            if (lhs.donor_frequency != rhs.donor_frequency) {
                return lhs.donor_frequency > rhs.donor_frequency;
            }
            return lhs.words < rhs.words;
        });

    if (split_pool.empty()) {
        return candidates;
    }

    candidates.reserve(donor_pool.size());
    unsigned int incomplete_split_systems = 0;
    unsigned int materialization_failures = 0;

    // Eight donors is the one retained reference-algorithm choice. Keep it
    // local to the generator instead of adding a global policy constant.
    const auto seed_window_size = std::min<std::size_t>(8, donor_pool.size());

    const auto attempt_limit = requested_candidates * 2;
    unsigned int deduplicated_this_call = 0;

    for (unsigned int attempt = 0; attempt < attempt_limit && candidates.size() < requested_candidates; ++attempt) {
        TreeList seed_trees;
        seed_trees.reserve(seed_window_size);
        for (std::size_t offset = 0; offset < seed_window_size; ++offset) {
            seed_trees.push_back(donor_pool[(attempt + offset) % donor_pool.size()]);
        }

        try {
            ConsensusTree seed(seed_trees, ConsenseCutoff::MR);
            seed.compute_support();

            std::vector<CandidateSplit> selected;
            selected.reserve(max_splits);

            PllSplitSharedPtr seed_splits(
                corax_utree_split_create(&seed.pll_utree_root(),
                                         tip_count,
                                         nullptr),
                corax_utree_split_destroy);
            if (!seed_splits) {
                coraxlib_reset_error();
                ++materialization_failures;
                continue;
            }

            for (std::size_t split_id = 0; split_id < seed.num_splits(); ++split_id) {
                std::vector<corax_split_base_t> words(words_per_split);
                std::memcpy(words.data(),
                            seed_splits.get()[split_id],
                            words.size() * sizeof(corax_split_base_t));

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
                        !corax_utree_split_compatible(const_cast<corax_split_base_t *>(existing.words.data()), const_cast<corax_split_base_t *>(candidate.words.data()), words_per_split, tip_count)) {
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

            Tree candidate = materialize_candidate(selected, baseline_tree);
            if (candidate.empty() ||
                !candidate.binary() ||
                !seed.compatible(candidate)) {
                coraxlib_reset_error();
                ++materialization_failures;
                continue;
            }

            PllSplitSharedPtr candidate_splits(corax_utree_split_create(&candidate.pll_utree_root(), tip_count, nullptr), corax_utree_split_destroy);
            if (!candidate_splits) {
                coraxlib_reset_error();
                ++materialization_failures;
                continue;
            }

            corax_utree_split_normalize_and_sort(
                candidate_splits.get(),
                tip_count,
                static_cast<unsigned int>(candidate.num_splits()),
                1);

            std::vector<corax_split_base_t> topology;
            topology.reserve(candidate.num_splits() * words_per_split);

            for (std::size_t split_id = 0;
                split_id < candidate.num_splits();
                ++split_id) {
                topology.insert(topology.end(),
                                candidate_splits.get()[split_id],
                                candidate_splits.get()[split_id] + words_per_split);
            }

            if (!seen_topologies.insert(std::move(topology)).second) {
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
        << ", seen_topologies=" << seen_topologies.size()
        << std::endl;

    return candidates;
}


TreeList AggressiveCandidateManager::generate_constrained_parsimony_candidates(const TreeList &donor_pool, const unsigned int requested_candidates, const unsigned long round_seed) {
    TreeList candidates;

    if (donor_pool.empty() ||
        baseline_tree.empty() ||
        initial_ml_trees.empty()) {
        return candidates;
    }

    coraxlib_reset_error();

    const auto tip_count = static_cast<unsigned int>(baseline_tree.num_tips());
    const auto bits_per_word = static_cast<unsigned int>(sizeof(corax_split_base_t) * 8);
    const auto words_per_split = tip_count / bits_per_word + static_cast<unsigned int>(tip_count % bits_per_word != 0);

    if (tip_count <= 3 || words_per_split == 0) {
        return candidates;
    }

    candidates.reserve(requested_candidates);

    unsigned int materialization_failures = 0;
    unsigned int deduplicated_this_call = 0;
    const auto seed_window_size = std::min<std::size_t>(8, donor_pool.size());
    const auto attempt_limit = requested_candidates * 2;

    for (unsigned int attempt = 0; attempt < attempt_limit && candidates.size() < requested_candidates; ++attempt) {
        try {
            TreeList seed_trees;
            seed_trees.reserve(seed_window_size);
            for (std::size_t offset = 0; offset < seed_window_size; ++offset) {
                const auto donor_id = (static_cast<std::size_t>(attempt) * 2 + offset) % donor_pool.size();
                seed_trees.push_back(donor_pool[donor_id]);
            }

            ConsensusTree constraint(seed_trees, ConsenseCutoff::MR);
            constraint.compute_support();
            if (constraint.num_splits() == 0) {
                ++materialization_failures;
                continue;
            }

            Tree candidate = generate_constrained_parsimony_tree(
                instance,
                constraint,
                static_cast<int>(
                    round_seed +
                    requested_candidates +
                    2 * treeset_aggressive_support_trees +
                    attempt));

            if (candidate.empty() ||
                candidate.num_tips() != tip_count ||
                !candidate.binary() ||
                !constraint.compatible(candidate)) {
                coraxlib_reset_error();
                ++materialization_failures;
                continue;
            }

            PllSplitSharedPtr candidate_splits(
                corax_utree_split_create(
                    &candidate.pll_utree_root(),
                    tip_count,
                    nullptr),
                corax_utree_split_destroy);
            if (!candidate_splits) {
                coraxlib_reset_error();
                ++materialization_failures;
                continue;
            }

            corax_utree_split_normalize_and_sort(
                candidate_splits.get(),
                tip_count,
                static_cast<unsigned int>(candidate.num_splits()),
                1);

            std::vector<corax_split_base_t> topology;
            topology.reserve(candidate.num_splits() * words_per_split);
            for (std::size_t split_id = 0; split_id < candidate.num_splits(); ++split_id) {
                topology.insert(
                    topology.end(),
                    candidate_splits.get()[split_id],
                    candidate_splits.get()[split_id] + words_per_split);
            }

            if (!seen_topologies.insert(std::move(topology)).second) {
                ++deduplicated_this_call;
                continue;
            }

            candidates.push_back(std::move(candidate));
        } catch (const std::exception &error) {
            coraxlib_reset_error();
            ++materialization_failures;
            LOG_WORKER_TS(LogLevel::info)
                << "Treeset constrained-parsimony candidate skipped: "
                << "attempt=" << attempt
                << ", reason=" << error.what()
                << std::endl;
        }
    }

    LOG_WORKER_TS(LogLevel::info)
        << "Treeset constrained-parsimony generation: initial_ml="
        << initial_ml_trees.size()
        << ", donors=" << donor_pool.size()
        << ", generated=" << candidates.size()
        << ", attempts=" << attempt_limit
        << ", materialization_failures="
        << materialization_failures
        << ", deduplicated_this_call="
        << deduplicated_this_call
        << ", seen_topologies=" << seen_topologies.size()
        << std::endl;

    return candidates;
}


TreeList AggressiveCandidateManager::take_batch(const unsigned int batch_size) {
    std::lock_guard<std::mutex> lock(candidate_mutex);

    TreeList selected;
    if (!opts.treeset_aggressive || exhausted || batch_size == 0) {
        return selected;
    }

    while (candidate_pool.size() < batch_size && !exhausted) {
        const auto size_before_refill = candidate_pool.size();
        generate_candidates(batch_size);

        if (candidate_pool.size() == size_before_refill) {
            exhausted = true;
        }
    }

    if (candidate_pool.size() < batch_size) {
        LOG_WORKER_TS(LogLevel::info)
            << "Treeset aggressive source exhausted: source="
            << (source == AggressiveSourceFamily::seed_greedy
                    ? "seed_greedy"
                    : "constrained_parsimony")
            << ", available=" << candidate_pool.size()
            << ", required=" << batch_size
            << std::endl;
        return selected;
    }

    selected.reserve(batch_size);
    while (selected.size() < batch_size) {
        selected.push_back(std::move(candidate_pool.front()));
        candidate_pool.pop_front();
    }

    return selected;
}
