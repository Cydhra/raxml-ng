#ifndef RAXML_NG_AGGRESSIVESOURCE_HPP
#define RAXML_NG_AGGRESSIVESOURCE_HPP

#include <utility>
#include <memory>

#include "ParsimonySource.hpp"
#include "TreeRepository.hpp"
#include "TreeSource.hpp"
#include "../../bootstrap/SplitsTree.hpp"
#include "../../bootstrap/EbgSupportTree.hpp"

using namespace std;

/** A vector of bit-vectors (stored as a vector of corax_split_base_t). Each bit vector is one split from the tree. */
using Split = std::vector<corax_split_base_t>;

using SplitList = std::vector<Split>;

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

class SplitSamplerSource : public TreeSource {
public:
    explicit SplitSamplerSource(const std::shared_ptr<ParsimonySource> &donor, const int seed,
                              TreeList initial_ml_trees, Tree baseline_tree) : baseline_tree(std::move(baseline_tree)),
        initial_ml_trees(std::move(initial_ml_trees)),
        donor_tree_source(donor), seed(seed) {
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

    bool remember_topology(const Tree &candidate);

    static Split topology_key(const Tree &tree);

    static bool has_majority_split(
        const SplitList &donor_topologies,
        const std::vector<std::size_t> &donor_ids,
        std::size_t words_per_split);

    static SplitList extract_splits(const Tree &tree, const bool normalize);

    virtual TreeList generate_candidates(
        const RaxmlInstance &instance,
        const ConstTreeRange &donor_pool,
        const SplitList &donor_splits,
        unsigned int requested_candidates,
        unsigned long round_seed) = 0;

    Tree materialize_candidate(const std::vector<CandidateSplit> &selected) const;

    std::shared_ptr<ParsimonySource> donor_tree_source;

    TreeList bootstrap_support_trees;

    int seed;

    std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();

    std::set<Split> seen_topologies;

    TreeRepository candidate_repository = {};

    bool exhausted = false;
};


#endif //RAXML_NG_AGGRESSIVESOURCE_HPP
