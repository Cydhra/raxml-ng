#include "AggressiveSource.hpp"

constexpr size_t NUM_DONOR_TREE = 300;

// TODO this can probably be made lower
static constexpr unsigned int BOOTSTRAP_SUPPORT_COUNT = 200;

void AggressiveSource::ensure(const RaxmlInstance &instance, const SmartBarrier &barrier,
                              unsigned int threads_per_worker, unsigned int worker_id, unsigned int thread_id,
                              unsigned int num_trees) {
    const auto begin = std::chrono::steady_clock::now();
    // build_parsimony_msa(instance, false); // TODO initialize in main.cpp in case of checkpoint

    const auto requested_candidates = num_trees;

    donor->ensure(instance, barrier, threads_per_worker, worker_id, thread_id, NUM_DONOR_TREE); // TODO magic value

    // TODO dont copy
    auto donor_list = TreeList(NUM_DONOR_TREE);
    for (int i = 0; i < NUM_DONOR_TREE; ++i) {
        donor->copy_tree(donor_list[i], i);
    }

    // move parsimony trees into duplicate checker
    //
    std::vector<std::vector<corax_split_base_t>> donor_topologies;
    donor_topologies.reserve(NUM_DONOR_TREE);
    for (const auto &tree: donor_list)
        donor_topologies.push_back(topology_key(tree));

    if (reference_splits.empty() && bootstrap_support_trees.empty()) {
        bootstrap_support_trees.reserve(BOOTSTRAP_SUPPORT_COUNT);
        while (bootstrap_support_trees.size() < BOOTSTRAP_SUPPORT_COUNT) {
            bootstrap_support_trees.emplace_back(generate_tree(
                instance, StartingTree::parsimony, seed + 1234567, true));
        }
    }

    const auto seed_size = seed_greedy_repository.candidate_count();
    const auto constrained_size = constrained_parsimony_repository.candidate_count();

    // Fixed ordering makes global cross-source deduplication reproducible.
    if (!seed_greedy_exhausted) {
        auto generated = gate_and_rank(
            generate_seed_greedy_candidates(
                donor_list, donor_topologies, requested_candidates, seed),
            StartingTreeSource::seed_greedy);
        for (auto &candidate: generated) {
            if (remember_topology(candidate))
                seed_greedy_repository.append_candidate(std::move(candidate));
        }
        if (seed_greedy_repository.candidate_count() == seed_size)
            seed_greedy_exhausted = true;
    }

    if (!constrained_parsimony_exhausted) {
        auto generated = gate_and_rank(
            generate_constrained_parsimony_candidates(
                donor_list, donor_topologies, requested_candidates, seed),
            StartingTreeSource::constrained_parsimony);
        for (auto &candidate: generated) {
            if (remember_topology(candidate))
                constrained_parsimony_repository.append_candidate(std::move(candidate));
        }
        if (constrained_parsimony_repository.candidate_count() == constrained_size)
            constrained_parsimony_exhausted = true;
    }

    // TODO this indirection is kinda useless, split both repositories in separate TreeSources, while sharing the common gating logic and resources (bootstrap parsimony trees and donor parsimony trees)
    auto list = constrained_parsimony_repository.take_candidate_batch(constrained_parsimony_repository.candidate_count());
    for (auto &tree : list) {
        tree_list.push_back(tree);
    }

    list = seed_greedy_repository.take_candidate_batch(seed_greedy_repository.candidate_count());
    for (auto &tree : list) {
        tree_list.push_back(tree);
    }
}

double AggressiveSource::amortized_time(unsigned int batch_size) const {
    return 0.0; // TODO
}
