#ifndef RAXML_METAPARAMETERS_HPP_
#define RAXML_METAPARAMETERS_HPP_

#include <filesystem>
#include <optional>
#include <string>

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

    /**
     * Perform an NNI round before the SPR rounds
     */
    const bool nni_round;

    /**
     * If true, constrain the tree to ignore parts of the tree that are probably unresolvable.
     */
    const bool constrain;

    /**
     * If set, override the model with a model built from this model string.
     */
    const std::optional<std::string> model_override;

    MetaParameters(const unsigned int keep_top_k_topol, const bool skip_model, const unsigned int num_fast_spr,
                   const unsigned int num_slow_spr,
                   const bool accept_starting_trees = false,
                   const bool early_commit = false,
                   const int max_radius = 20,
                   const bool nni_round = false,
                   const bool constrain = false,
                   const std::optional<std::string> &model_override =
                           std::nullopt) : keep_top_k_topol(keep_top_k_topol),
                                           skip_model(skip_model),
                                           num_fast_spr(num_fast_spr),
                                           num_slow_spr(num_slow_spr),
                                           accept_starting_trees(accept_starting_trees),
                                           early_commit(early_commit),
                                           max_adaptive_radius(max_radius),
                                           nni_round(nni_round),
                                           constrain(constrain),
                                           model_override(model_override) {
    }

    friend bool operator==(const MetaParameters &lhs, const MetaParameters &rhs) {
        return lhs.keep_top_k_topol == rhs.keep_top_k_topol
               && lhs.skip_model == rhs.skip_model
               && lhs.num_fast_spr == rhs.num_fast_spr
               && lhs.num_slow_spr == rhs.num_slow_spr
               && lhs.accept_starting_trees == rhs.accept_starting_trees
               && lhs.early_commit == rhs.early_commit
               && lhs.max_adaptive_radius == rhs.max_adaptive_radius
               && lhs.nni_round == rhs.nni_round
               && lhs.constrain == rhs.constrain
               && lhs.model_override == rhs.model_override;
    }

    friend bool operator!=(const MetaParameters &lhs, const MetaParameters &rhs) {
        return !(lhs == rhs);
    }

    friend std::size_t hash_value(const MetaParameters &obj) {
        std::size_t seed = 0x51FEBF61;
        seed ^= (seed << 6) + (seed >> 2) + 0x558F3E9C + static_cast<std::size_t>(obj.keep_top_k_topol);
        seed ^= (seed << 6) + (seed >> 2) + 0x144ECD35 + static_cast<std::size_t>(obj.skip_model);
        seed ^= (seed << 6) + (seed >> 2) + 0x72773779 + static_cast<std::size_t>(obj.num_fast_spr);
        seed ^= (seed << 6) + (seed >> 2) + 0x4799CDEB + static_cast<std::size_t>(obj.num_slow_spr);
        seed ^= (seed << 6) + (seed >> 2) + 0x20437926 + static_cast<std::size_t>(obj.accept_starting_trees);
        seed ^= (seed << 6) + (seed >> 2) + 0x3E03390B + static_cast<std::size_t>(obj.early_commit);
        seed ^= (seed << 6) + (seed >> 2) + 0x4AC1F56C + static_cast<std::size_t>(obj.max_adaptive_radius);
        seed ^= (seed << 6) + (seed >> 2) + 0x7BA5D994 + static_cast<std::size_t>(obj.nni_round);
        seed ^= (seed << 6) + (seed >> 2) + 0x1DBD1C1A + static_cast<std::size_t>(obj.constrain);
        seed ^= (seed << 6) + (seed >> 2) + 0x243E770D + (obj.model_override.has_value()
                                                              ? std::filesystem::hash_value(obj.model_override.value())
                                                              : static_cast<std::size_t>(0));
        return seed;
    }
};

template<>
struct std::hash<MetaParameters> {
    size_t operator()(const MetaParameters &p) const noexcept {
        return hash_value(p);
    }
};

#endif //RAXML_METAPARAMETERS_HPP_
