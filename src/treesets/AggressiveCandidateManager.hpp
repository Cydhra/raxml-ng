#ifndef RAXML_AGGRESSIVECANDIDATEMANAGER_HPP_
#define RAXML_AGGRESSIVECANDIDATEMANAGER_HPP_

#include <atomic>
#include <deque>
#include <mutex>
#include <set>
#include <utility>

#include "MetaParameters.hpp"
#include "../Options.hpp"
#include "../Tree.hpp"

struct RaxmlInstance;

/** Thread-safe storage shared by all aggressive proposal sources. */
class AggressiveCandidateRepository {
public:
    bool reserve_topology(std::vector<corax_split_base_t> topology);
    bool contains_topology(const std::vector<corax_split_base_t> &topology) const;
    std::size_t topology_count() const;
    void append_candidate(StartingTreeSource source, Tree candidate);
    TreeList take_candidate_batch(StartingTreeSource source, unsigned int batch_size);
    std::size_t candidate_count(StartingTreeSource source) const;
    void mark_exhausted(StartingTreeSource source);
    bool is_exhausted(StartingTreeSource source) const;

    void add_reusable_donor(const Tree &donor);
    TreeList take_reusable_donors(unsigned int batch_size);
    std::size_t reusable_donor_count() const;

private:
    struct SourceState {
        std::deque<Tree> candidates;
        bool exhausted = false;
    };

    SourceState &state(StartingTreeSource source);
    const SourceState &state(StartingTreeSource source) const;

    mutable std::mutex mutex;
    SourceState seed_greedy;
    SourceState constrained_parsimony;
    std::set<std::vector<corax_split_base_t>> seen_topologies;
    std::deque<Tree> reusable_parsimony;
};

/**
 * Thread-safe repository for aggressive candidates and reusable parsimony donors.
 * Candidate preparation is deliberately outside TunedBatch timing so that bandit
 * measurements remain compatible with the original aggressive implementation.
 */
class AggressiveCandidateManager {
public:
    AggressiveCandidateManager(RaxmlInstance &instance,
                               const Options &opts,
                               const Tree &baseline_tree,
                               TreeList initial_ml_trees,
                               unsigned int target_tree_count)
        : instance(instance),
          opts(opts),
          baseline_tree(baseline_tree),
          initial_ml_trees(std::move(initial_ml_trees)),
          target_tree_count(target_tree_count) {}

    TreeList take_batch(StartingTreeSource source, unsigned int batch_size);

    /** Return up to batch_size cached ordinary parsimony donors in FIFO order. */
    TreeList take_reusable_parsimony(unsigned int batch_size);

private:
    struct RankedCandidate {
        Tree tree;
        unsigned int promise_score = 0;
    };

    RaxmlInstance &instance;
    const Options &opts;
    const Tree baseline_tree;
    const TreeList initial_ml_trees;
    unsigned int target_tree_count;

    AggressiveCandidateRepository repository;
    std::mutex preparation_mutex;
    unsigned int generation_round = 0;
    std::atomic_size_t generated_donor_count{0};
    std::atomic_size_t generated_bootstrap_count{0};
    std::atomic_size_t reused_donor_count{0};

    TreeList support_trees;
    TreeList bootstrap_support_trees;
    std::vector<std::vector<corax_split_base_t>> reference_splits;
    std::vector<double> ebg_support;
    std::vector<double> ml_frequency;
    static constexpr unsigned int support_tree_count = 300;
    static constexpr unsigned int bootstrap_support_tree_count = 100;
    static constexpr double mae_margin = 0.0;

    bool prepare_gate();
    void generate_round(unsigned int minimum_candidates);
    TreeList gate_and_rank(TreeList candidates, StartingTreeSource source);
    TreeList generate_seed_greedy_candidates(const TreeList &donor_pool,
                                             const std::vector<std::vector<corax_split_base_t>> &donor_topologies,
                                             unsigned int requested_candidates,
                                             unsigned long round_seed);
    TreeList generate_constrained_parsimony_candidates(const TreeList &donor_pool,
                                                       const std::vector<std::vector<corax_split_base_t>> &donor_topologies,
                                                       unsigned int requested_candidates,
                                                       unsigned long round_seed);
    bool remember_topology(const Tree &candidate);
};

#endif
