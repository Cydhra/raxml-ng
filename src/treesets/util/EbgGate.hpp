#ifndef RAXML_EBGFILTER_HPP
#define RAXML_EBGFILTER_HPP

#include "../start/TreeSource.hpp"

#include "../../Tree.hpp"
#include "../../bootstrap/SplitsTree.hpp"
#include "../../bootstrap/EbgSupportTree.hpp"

/** A vector of bit-vectors (stored as a vector of corax_split_base_t). Each bit vector is one split from the tree. */
using Split = std::vector<corax_split_base_t>;

using SplitList = std::vector<Split>;

struct RaxmlInstance;

// forward declaration of generate_tree in main.cpp to make it accessible. If the function in main.cpp
// changes signature, just update this declaration as well.
Tree generate_tree(const RaxmlInstance &instance, StartingTree type, int random_seed, bool bootstrap);

class CandidateEbgSupportTree : public EbgSupportTree {
public:
    using EbgSupportTree::EbgSupportTree;
    bool compute() { return compute_support(); }
    const corax_split_t *reference_splits() const { return _ref_splits.get(); }
};

/**
 * Helper struct to rank candidates using a score during filtering
 */
struct RankedCandidate {
    Tree tree;
    unsigned int promise_score = 0;
};

/**
 * Verifies that a list of candidate trees approximates the per-split bootstrap distribution estimated by EBG.
 * This serves to avoid using TreeSources which generate a strongly biased selection of trees which does not reflect
 * the expected distribution of splits.
 */
class EbgGate {
public:
    EbgGate(Tree baseline_tree, TreeList initial_ml_trees)
        : baseline_tree(std::move(baseline_tree)),
          initial_ml_trees(std::move(initial_ml_trees)) {
    }

    bool reset_gate(const RaxmlInstance &instance, const ConstTreeRange & support_trees);

    TreeList gate_and_rank(TreeList candidates);

protected:
    /**
     * Best ML tree from reference set to use as the EBG filter reference
     */
    const Tree baseline_tree;

    /**
     * Set of reference ML trees
     */
    const TreeList initial_ml_trees;

    /**
     * Splits extracted from reference ML trees
     */
    SplitList reference_splits;

    TreeList bootstrap_support_trees;

    /**
     * Frequencies of splits in ML trees (referenced by index of split)
     */
    std::vector<double> ml_frequency;

    /**
     * EBG support values of splits (referenced by index of split)
     */
    std::vector<double> ebg_support;

    /**
     * Margin for acceptable mean absolute error of EBG support from ml_frequency, which are accepted during filtering
     */
    double mae_margin = 0.0;
};


#endif //RAXML_EBGFILTER_HPP
