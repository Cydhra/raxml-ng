#ifndef RAXML_NG_AGGRESSIVESOURCE_HPP
#define RAXML_NG_AGGRESSIVESOURCE_HPP

#include "ParsimonySource.hpp"
#include "TreeRepository.hpp"
#include "TreeSource.hpp"
#include "../../bootstrap/SplitsTree.hpp"
#include "../../bootstrap/ConsensusTree.hpp"
#include "../../bootstrap/EbgSupportTree.hpp"

using namespace std;

enum StartingTreeSource {
    seed_greedy,
    constrained_parsimony,
};

Tree generate_parsimony_tree(const RaxmlInstance &instance,
                             int random_seed,
                             bool bootstrap,
                             const Tree &constraint_tree);

class CandidateEbgSupportTree : public EbgSupportTree {
public:
    using EbgSupportTree::EbgSupportTree;
    bool compute() { return compute_support(); }
    const corax_split_t *reference_splits() const { return _ref_splits.get(); }
};

struct RaxmlInstance;

class AggressiveSource : public TreeSource {
public:
    explicit AggressiveSource(const std::shared_ptr<ParsimonySource> &donor, int seed, const RaxmlInstance &instance,
                              const Options &opts, TreeList &initial_ml_trees, Tree &baseline_tree) : instance(instance), opts(opts),
        baseline_tree(baseline_tree), initial_ml_trees(initial_ml_trees), donor(donor), seed(seed) {
    }

    void ensure(const RaxmlInstance &instance, const SmartBarrier &barrier, unsigned int threads_per_worker,
                unsigned int worker_id, unsigned int thread_id, unsigned int num_trees) override;

    [[nodiscard]] double amortized_time(unsigned int batch_size) const override;

protected:
    const RaxmlInstance &instance;
    const Options &opts;
    const Tree baseline_tree;
    const TreeList initial_ml_trees;

    struct RankedCandidate {
        Tree tree;
        unsigned int promise_score = 0;
    };

    // build trees from these
    struct CandidateSplit {
        std::vector<corax_split_base_t> words;
        unsigned int ml_frequency = 0;
        unsigned int donor_frequency = 0;
    };

    std::vector<std::vector<corax_split_base_t> > reference_splits;

    std::vector<double> ebg_support;
    std::vector<double> ml_frequency;

    static constexpr double mae_margin = 0.0;

    // insert topology splits into map
    std::vector<corax_split_base_t> topology_key(const Tree &tree) {
        const auto split_words = extract_splits(tree, true);
        std::vector<corax_split_base_t> result;
        for (const auto &split: split_words)
            result.insert(result.end(), split.begin(), split.end());
        return result;
    }

    // todo move
    using SplitWords = std::vector<std::vector<corax_split_base_t> >;

    // extract splits from tree, encoded as bit vectors
    SplitWords extract_splits(const Tree &tree, const bool normalize) {
        SplitWords result;
        if (tree.empty() || tree.num_splits() == 0)
            return result;

        const auto tip_count = static_cast<unsigned int>(tree.num_tips());
        const auto bits_per_word = static_cast<unsigned int>(sizeof(corax_split_base_t) * 8);
        const auto words_per_split = tip_count / bits_per_word +
                                     static_cast<unsigned int>(tip_count % bits_per_word != 0);
        PllSplitSharedPtr splits(
            corax_utree_split_create(&tree.pll_utree_root(), tip_count, nullptr),
            corax_utree_split_destroy);
        if (!splits) {
            coraxlib_reset_error();
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

    TreeList gate_and_rank(TreeList candidates, const StartingTreeSource source) {
        TreeList selected;
        if (candidates.empty() || !prepare_gate()) {
            return selected;
        }
        const auto split_count = reference_splits.size();

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
            //         << source_name(source)
            //         << ", decision=fail"
            //         << ", initial_ml=" << initial_ml_trees.size()
            //         << ", support_trees=" << support_tree_count
            //         << ", bootstrap_support_trees="
            //         << bootstrap_support_tree_count
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

    bool prepare_gate() {
        if (!reference_splits.empty())
            return true;

        // TODO dont copy, reuse donors
        auto support_trees = TreeList(300);
        for (int i = 0; i < 300; ++i) {
            donor->copy_tree(support_trees[i], i);
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

        // Only the derived support values and split words are needed after initialization.
        // TODO this prevents reuse on further ensure(N) calls, so we shouldnt clear them
        // support_trees.clear();
        // support_trees.shrink_to_fit();
        // bootstrap_support_trees.clear();
        // bootstrap_support_trees.shrink_to_fit();
        return true;
    }

    TreeList generate_seed_greedy_candidates(
        const TreeList &donor_pool,
        const std::vector<std::vector<corax_split_base_t> > &donor_topologies,
        const unsigned int requested_candidates,
        const unsigned long round_seed) {
        TreeList candidates;

        if (donor_pool.empty() || baseline_tree.empty() || initial_ml_trees.empty()) {
            return candidates;
        }

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
            if (tree.empty() || tree.num_tips() != tip_count) {
                return;
            }

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
        std::set<std::vector<corax_split_base_t> > staged_topologies;

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
                if (!has_majority_split(donor_topologies, seed_ids, words_per_split)) {
                    ++incomplete_split_systems;
                    continue;
                }
                ConsensusTree seed(seed_trees, ConsenseCutoff::MR);
                seed.compute_support();

                std::vector<CandidateSplit> selected;
                selected.reserve(max_splits);

                auto seed_splits = extract_splits(seed, false);
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

                Tree candidate = materialize_candidate(selected, baseline_tree);
                if (candidate.empty() ||
                    !candidate.binary() ||
                    !seed.compatible(candidate)) {
                    coraxlib_reset_error();
                    ++materialization_failures;
                    continue;
                }

                auto topology = topology_key(candidate);
                if (topology.empty() || seen_topologies.find(topology) != seen_topologies.end() ||
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

    bool remember_topology(const Tree &candidate) {
        auto topology = topology_key(candidate);

        std::lock_guard lock(*mutex);
        return !topology.empty() && seen_topologies.insert(std::move(topology)).second;
    }

    TreeList generate_constrained_parsimony_candidates(
        const TreeList &donor_pool,
        const std::vector<std::vector<corax_split_base_t> > &donor_topologies,
        const unsigned int requested_candidates,
        const unsigned long round_seed) {
        TreeList candidates;

        if (donor_pool.empty() ||
            baseline_tree.empty() ||
            initial_ml_trees.empty()) {
            return candidates;
        }

        coraxlib_reset_error();

        const auto tip_count = static_cast<unsigned int>(baseline_tree.num_tips());
        const auto bits_per_word = static_cast<unsigned int>(sizeof(corax_split_base_t) * 8);
        const auto words_per_split = tip_count / bits_per_word +
                                     static_cast<unsigned int>(tip_count % bits_per_word != 0);
        if (tip_count <= 3 || words_per_split == 0) {
            return candidates;
        }

        candidates.reserve(requested_candidates);

        unsigned int materialization_failures = 0;
        unsigned int deduplicated_this_call = 0;
        const auto seed_window_size = std::min<std::size_t>(8, donor_pool.size());
        const auto attempt_limit = requested_candidates * 2;
        std::set<std::vector<corax_split_base_t> > staged_topologies;

        for (unsigned int attempt = 0; attempt < attempt_limit && candidates.size() < requested_candidates; ++attempt) {
            try {
                TreeList seed_trees;
                seed_trees.reserve(seed_window_size);
                std::vector<std::size_t> seed_ids;
                seed_ids.reserve(seed_window_size);
                for (std::size_t offset = 0; offset < seed_window_size; ++offset) {
                    const auto donor_id = (static_cast<std::size_t>(attempt) * 2 + offset) % donor_pool.size();
                    seed_ids.push_back(donor_id);
                    seed_trees.push_back(donor_pool[donor_id]);
                }

                if (!has_majority_split(donor_topologies, seed_ids, words_per_split)) {
                    ++materialization_failures;
                    continue;
                }
                ConsensusTree constraint(seed_trees, ConsenseCutoff::MR);
                constraint.compute_support();
                if (constraint.num_splits() == 0) {
                    ++materialization_failures;
                    continue;
                }

                Tree candidate = generate_parsimony_tree(
                    instance,
                    // todo magic number support tree count
                    static_cast<int>(round_seed + requested_candidates + 2 * 300 + attempt), false, constraint);

                if (candidate.empty() ||
                    candidate.num_tips() != tip_count ||
                    !candidate.binary() ||
                    !constraint.compatible(candidate)) {
                    coraxlib_reset_error();
                    ++materialization_failures;
                    continue;
                }

                auto topology = topology_key(candidate);
                if (topology.empty() || seen_topologies.find(topology) != seen_topologies.end() ||
                    !staged_topologies.insert(std::move(topology)).second) {
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
                << ", seen_topologies=" << seen_topologies.size() + staged_topologies.size()
                << std::endl;

        return candidates;
    }

    bool has_majority_split(
        const std::vector<std::vector<corax_split_base_t> > &donor_topologies,
        const std::vector<std::size_t> &donor_ids,
        const std::size_t words_per_split) {
        std::map<std::vector<corax_split_base_t>, std::size_t> frequencies;
        for (const auto donor_id: donor_ids) {
            const auto &topology = donor_topologies[donor_id];
            for (std::size_t offset = 0; offset < topology.size(); offset += words_per_split) {
                std::vector<corax_split_base_t> split(
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


    std::shared_ptr<ParsimonySource> donor;

    TreeList bootstrap_support_trees;

    int seed;

    bool seed_greedy_exhausted = false;

    bool constrained_parsimony_exhausted = false;

    std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();

    std::set<std::vector<corax_split_base_t> > seen_topologies;

    TreeRepository seed_greedy_repository = {};

    TreeRepository constrained_parsimony_repository = {};
};


#endif //RAXML_NG_AGGRESSIVESOURCE_HPP
