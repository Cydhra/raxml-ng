#include "ParsimonySamplerSource.hpp"

#include "../../bootstrap/ConsensusTree.hpp"

TreeList ParsimonySamplerSource::generate_candidates(const RaxmlInstance &instance, const ConstTreeRange &donor_pool,
                                                     const SplitList &donor_splits, const unsigned int requested_candidates,
                                                     const unsigned long round_seed) {
    TreeList candidates;

    const auto tip_count = static_cast<unsigned int>(baseline_tree.num_tips());
    constexpr auto bits_per_word = static_cast<unsigned int>(sizeof(corax_split_base_t) * 8);
    const auto words_per_split = tip_count / bits_per_word +
                                 static_cast<unsigned int>(tip_count % bits_per_word != 0);

    candidates.reserve(requested_candidates);

    unsigned int materialization_failures = 0;
    unsigned int deduplicated_this_call = 0;
    const auto seed_window_size = std::min<std::size_t>(8, donor_pool.size());
    const auto attempt_limit = requested_candidates * 2;
    std::set<Split> staged_topologies;

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

            if (!has_majority_split(donor_splits, seed_ids, words_per_split)) {
                ++materialization_failures;
                continue;
            }
            ConsensusTree constraint(seed_trees, ConsenseCutoff::MR);
            constraint.compute_support();
            if (constraint.num_splits() == 0) {
                ++materialization_failures;
                continue;
            }

            Tree candidate = generate_parsimony_tree(instance, static_cast<int>(round_seed + attempt), false,
                                                     constraint);

            if (auto topology = topology_key(candidate); seen_topologies.find(topology) != seen_topologies.end() ||
                                                         !staged_topologies.insert(std::move(topology)).second) {
                ++deduplicated_this_call;
                continue;
            }

            candidates.push_back(std::move(candidate));
        } catch (const std::exception &error) {
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
