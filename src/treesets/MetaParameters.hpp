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
     * How many slow SPR rounds to perform for each tree search
     */
    const unsigned int num_slow_spr;

    /**
     * If true, skip all model optimization and simply accept parsimony starting trees as the final resulting topology.
     */
    const bool accept_starting_trees;

    /**
     * If true, the 0.1 model optimization is done instantly to commit the tree and model to the current local minimum.
     * This helps with datasets with discordant signal to commit to a random signal instantly, instead of searching
     * a global minimum.
     */
    const bool early_commit;

    /**
     * Maximum SPR radius
     */
    const unsigned int max_adaptive_radius;

    MetaParameters(const unsigned int keep_top_k_topol, const bool skip_model, const unsigned int num_fast_spr,
                   const unsigned int num_slow_spr,
                   const bool accept_starting_trees = false,
                   const bool early_commit = false,
                   const int max_radius = 20) : keep_top_k_topol(keep_top_k_topol), skip_model(skip_model),
                                                         num_fast_spr(num_fast_spr), num_slow_spr(num_slow_spr),
                                                         accept_starting_trees(accept_starting_trees),
                                                         early_commit(early_commit),
                                                         max_adaptive_radius(max_radius) {
    }

    friend bool operator==(const MetaParameters &lhs, const MetaParameters &rhs) {
        return lhs.keep_top_k_topol == rhs.keep_top_k_topol
               && lhs.skip_model == rhs.skip_model
               && lhs.num_fast_spr == rhs.num_fast_spr
               && lhs.num_slow_spr == rhs.num_slow_spr
               && lhs.accept_starting_trees == rhs.accept_starting_trees
               && lhs.early_commit == rhs.early_commit;
    }

    friend bool operator!=(const MetaParameters &lhs, const MetaParameters &rhs) {
        return !(lhs == rhs);
    }
};

#endif //RAXML_METAPARAMETERS_HPP_
