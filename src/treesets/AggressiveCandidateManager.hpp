// tools/raxml-ng/src/treesets/AggressiveCandidateManager.hpp
#ifndef RAXML_AGGRESSIVECANDIDATEMANAGER_HPP_
#define RAXML_AGGRESSIVECANDIDATEMANAGER_HPP_

#include <cmath>
#include <deque>
#include <limits>
#include <mutex>
#include <set>
#include <utility>

#include "MetaParameters.hpp"
#include "../Options.hpp"
#include "../Tree.hpp"

class BatchQueue;
struct RaxmlInstance;

class AggressiveCandidateManager {
public:
    AggressiveCandidateManager(RaxmlInstance &instance,
                               const Options &opts,
                               const Tree &baseline_tree,
                               TreeList initial_ml_trees,
                               unsigned int target_tree_count,
                               AggressiveSourceFamily source)
        : instance(instance),
          opts(opts),
          baseline_tree(baseline_tree),
          initial_ml_trees(std::move(initial_ml_trees)),
          target_tree_count(target_tree_count),
          source(source) {}

    void generate_candidates(const unsigned int minimum_candidates);
    TreeList generate_seed_greedy_candidates(const TreeList &donor_pool, unsigned int requested_candidates, unsigned long round_seed);
    TreeList generate_constrained_parsimony_candidates(const TreeList &donor_pool, unsigned int requested_candidates, unsigned long round_seed);

    bool ensure_candidates(BatchQueue &queue);
    void print_report(const BatchQueue &) const;
    TreeList take_batch(unsigned int batch_size);


private:
    struct RankedCandidate {
        Tree tree;
        unsigned int promise_score = 0;
    };

    RaxmlInstance &instance;
    const Options &opts;
    const Tree &baseline_tree;
    const TreeList initial_ml_trees;
    unsigned int target_tree_count;
    AggressiveSourceFamily source;

    std::set<std::vector<corax_split_base_t>> seen_topologies;

    std::mutex candidate_mutex;
    std::deque<Tree> candidate_pool;
    bool exhausted = false;
    unsigned int generation_round = 0;

    TreeList support_trees;
    TreeList bootstrap_support_trees;
    unsigned int treeset_aggressive_support_trees = 300;
    double treeset_aggressive_mae_margin = 0.0;

    unsigned int pending_preparation_ms = 0;
    unsigned int pending_preparation_batches = 0;

    std::deque<RankedCandidate> ranked_candidates;

    TreeList gate_and_rank(TreeList candidates);
};

#endif
