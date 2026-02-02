#ifndef RAXML_METAPARAMETERS_HPP_
#define RAXML_METAPARAMETERS_HPP_

/**
 * Class that holds the meta-parameters of self-tuning tree inference. The parameters are held by a bandit and can
 * be applied to TunedBatches to configure the tree inference.
 */
class MetaParameters {
public:
    /**
     * Number of top topologies to keep during SPR rounds. If lower, the SPR rounds perform substantially less
     * branch-length optimizations. On datasets where most trees are plausible, this saves time because the K top
     * topologies are probably equally likely anyway.
     */
    const unsigned int keep_top_k_topol;

    /**
     * If true, skip the first model optimization by reusing model parameters from a previous search.
     */
    const bool skip_model;

    /**
     * How many fast SPR rounds to perform for each tree search
     */
    const unsigned int num_fast_spr;

    /**
     * If true, skip all model optimization and simply accept parsimony starting trees as the final resulting topology.
     */
    const bool accept_starting_trees;

    MetaParameters(const unsigned int keep_top_k_topol, const bool skip_model, const unsigned int num_fast_spr,
                   const bool accept_starting_trees) : keep_top_k_topol(keep_top_k_topol), skip_model(skip_model),
                                                 num_fast_spr(num_fast_spr),
                                                 accept_starting_trees(accept_starting_trees) {
    }
};

#endif //RAXML_METAPARAMETERS_HPP_