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
    explicit AggressiveSource(const std::shared_ptr<ParsimonySource> &donor, const int seed,
                              const TreeList &initial_ml_trees, const Tree &baseline_tree) : baseline_tree(baseline_tree),
        initial_ml_trees(initial_ml_trees),
        donor(donor), seed(seed) {
    }

    void ensure(const RaxmlInstance &instance, const SmartBarrier &barrier, unsigned int threads_per_worker,
                unsigned int worker_id, unsigned int thread_id, unsigned int num_trees) override;

    [[nodiscard]] double amortized_time(unsigned int batch_size) const override;

protected:
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

    TreeList gate_and_rank(TreeList candidates);

    bool prepare_gate();

    TreeList generate_seed_greedy_candidates(
        const TreeList &donor_pool,
        const std::vector<std::vector<corax_split_base_t> > &donor_topologies,
        unsigned int requested_candidates,
        unsigned long round_seed);

    bool remember_topology(const Tree &candidate);

    TreeList generate_constrained_parsimony_candidates(
        const RaxmlInstance &instance,
        const TreeList &donor_pool,
        const std::vector<std::vector<corax_split_base_t> > &donor_topologies,
        unsigned int requested_candidates,
        unsigned long round_seed);


    Tree materialize_candidate(const std::vector<CandidateSplit> &selected, const Tree &label_source);

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
