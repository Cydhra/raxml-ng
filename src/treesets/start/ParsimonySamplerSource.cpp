#include "ParsimonySamplerSource.hpp"

#include "../../bootstrap/ConsensusTree.hpp"

constexpr size_t NUM_RETRY = 2;

TreeList ParsimonySamplerSource::generate_candidates(const RaxmlInstance &instance, const ConstTreeRange &donor_pool,
                                                     const SplitList &donor_splits,
                                                     const unsigned int requested_candidates,
                                                     const unsigned int donor_offset,
                                                     const unsigned long base_seed) {
    TreeList candidates;

    const auto tip_count = static_cast<unsigned int>(baseline_tree.num_tips());
    constexpr auto bits_per_word = static_cast<unsigned int>(sizeof(corax_split_base_t) * 8);
    const auto words_per_split = tip_count / bits_per_word +
                                 static_cast<unsigned int>(tip_count % bits_per_word != 0);

    candidates.reserve(requested_candidates);

    unsigned int materialization_failures = 0;
    unsigned int deduplicated_this_call = 0;
    const auto seed_window_size = std::min<std::size_t>(8, donor_pool.size());
    std::set<Split> staged_topologies;

    unsigned int attempt_counter = 0;
    for (std::size_t tree_num = 0; tree_num < requested_candidates; ++tree_num) {
        for (size_t attempt = 0; attempt < NUM_RETRY; attempt++) {
            TreeList constraint_seed_trees;
            constraint_seed_trees.reserve(seed_window_size);
            std::vector<std::size_t> seed_ids;
            seed_ids.reserve(seed_window_size);
            for (std::size_t window_index = 0; window_index < seed_window_size; ++window_index) {
                // the donor ids are chosen such that different workers use different donors, and failed attempts do
                // spread out the donor ids to avoid re-using donors of other workers
                const auto donor_id = (donor_offset + tree_num + window_index * (attempt + 1)) % donor_pool.size();
                seed_ids.push_back(donor_id);
                constraint_seed_trees.push_back(donor_pool[donor_id]);
            }
            attempt_counter += 1;

            if (!has_majority_split(donor_splits, seed_ids, words_per_split)) {
                ++materialization_failures;
                continue;
            }

            ConsensusTree constraint(constraint_seed_trees, ConsenseCutoff::MR);
            constraint.compute_support();

            // seed is chosen to be non-overlapping between workers (donor_offset), and add an additional large offset
            // on retries to avoid reusing seeds of other workers
            Tree candidate = generate_parsimony_tree(instance, static_cast<int>(base_seed + donor_offset + tree_num + attempt * 7131337),
                                                 false,
                                                 constraint);

            if (auto topology = topology_key(candidate); seen_topologies.find(topology) != seen_topologies.end() ||
                                                     !staged_topologies.insert(std::move(topology)).second) {
                ++deduplicated_this_call;
            } else {
                candidates.push_back(std::move(candidate));
                break; // break attempt loop, we have succeeded
            }
        }
    }

    LOG_WORKER_TS(LogLevel::info)
            << "Constrained-Parsimony generation:"
            << " donors=" << donor_pool.size()
            << ", generated=" << candidates.size()
            << ", attempts=" << attempt_counter
            << ", materialization_failures="
            << materialization_failures
            << ", deduplicated_this_call="
            << deduplicated_this_call
            << ", seen_topologies=" << seen_topologies.size() + staged_topologies.size()
            << std::endl;

    return candidates;
}
