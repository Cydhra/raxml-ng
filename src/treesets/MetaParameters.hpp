#ifndef RAXML_METAPARAMETERS_HPP_
#define RAXML_METAPARAMETERS_HPP_

#include <filesystem>
#include <optional>
#include <string>
#include "../Options.hpp"
#include "../Optimizer.hpp"


enum class AggressiveSourceFamily {
    none,
    seed_greedy,
    constrained_parsimony
};

/**
 * Holds the meta-parameters of self-tuning tree inference. The parameters are held by a bandit and can
 * be applied to TunedBatches to configure the tree inference.
 */
struct MetaParameters {
    /**
     * Number of top topologies to keep during SPR rounds. If lower, the SPR rounds perform substantially less
     * branch-length optimizations. On datasets where most trees are plausible, this saves time because the K top
     * topologies are probably equally likely anyway.
     */
    unsigned int keep_top_k_topol;

    /**
     * If true, does a model optimization before doing topology moves.
     */
    bool do_first_model;

    /**
     * If true, does a final model and branch length optimization.
     */
    bool do_final_model;

    /**
     * How many fast SPR rounds to perform for each tree search
     */
    unsigned int num_fast_spr;

    /**
     * How many slow SPR rounds to perform for each tree search
     */
    unsigned int num_slow_spr;

    /**
     * If true, skip all model optimization and simply accept parsimony starting trees as the final resulting topology.
     */
    bool accept_starting_trees;

    /**
     * Maximum SPR radius
     */
    unsigned int max_adaptive_radius;

    /**
     * Perform an NNI round before the SPR rounds
     */
    bool nni_round;

    /**
     * If true, constrain the tree to ignore parts of the tree that are probably unresolvable.
     */
    bool constrain;

    /**
     * If set, override the model with a model built from this model string.
     */
    std::optional<std::string> model_override;

    /**
     * If set, fall back to fast-raxml for inference
     */
    bool fallback_fast_raxml;

    /**
     * If set, run SPR rounds with stop criterion.
     */
    bool dynamic_spr;

    /**
     * Metadata used by externally generated aggressive starting-tree batches.
     */
    bool aggressive_source = false;
    AggressiveSourceFamily aggressive_source_family = AggressiveSourceFamily::none;
    bool fixed_model_au_precheck = false;
    bool pin_initial_ml_model = false;

    explicit MetaParameters(const unsigned int keep_top_k_topol = 20,
                            const bool do_first_model = false,
                            const unsigned int num_fast_spr = 0,
                            const unsigned int num_slow_spr = 0,
                            const bool accept_starting_trees = false,
                            const unsigned int max_radius = 20,
                            const bool nni_round = false,
                            const bool constrain = false,
                            const std::optional<std::string> &model_override =
                                    std::nullopt,
                            const bool fallback_fast_raxml = false,
                            const bool dynamic_spr = false,
                            const bool do_final_model = true) : keep_top_k_topol(keep_top_k_topol),
                                                                do_first_model(do_first_model),
                                                                do_final_model(do_final_model),
                                                                num_fast_spr(num_fast_spr),
                                                                num_slow_spr(num_slow_spr),
                                                                accept_starting_trees(accept_starting_trees),
                                                                max_adaptive_radius(max_radius),
                                                                nni_round(nni_round),
                                                                constrain(constrain),
                                                                model_override(model_override),
                                                                fallback_fast_raxml(fallback_fast_raxml),
                                                                dynamic_spr(dynamic_spr) {
    }

    static MetaParameters aggressive_starting_tree_acceptance() {
        MetaParameters p = starting_tree_acceptance();
        p.fixed_model_au_precheck = true;
        p.pin_initial_ml_model = true;
        return p;
    }

    static MetaParameters aggressive_candidate_source(AggressiveSourceFamily source) {
        MetaParameters p = aggressive_starting_tree_acceptance();
        p.aggressive_source = true;
        p.aggressive_source_family = source;
        return p;
    }

    static MetaParameters starting_tree_acceptance() {
        return MetaParameters(1, false, 0, 0, true, false);
    }

    bool is_external_aggressive_source() const {
        return aggressive_source;
    }

    AggressiveSourceFamily external_aggressive_source_family() const {
        return aggressive_source_family;
    }

    friend bool operator==(const MetaParameters &lhs, const MetaParameters &rhs) {
        return lhs.keep_top_k_topol == rhs.keep_top_k_topol
               && lhs.do_first_model == rhs.do_first_model
               && lhs.num_fast_spr == rhs.num_fast_spr
               && lhs.num_slow_spr == rhs.num_slow_spr
               && lhs.accept_starting_trees == rhs.accept_starting_trees
               && lhs.max_adaptive_radius == rhs.max_adaptive_radius
               && lhs.nni_round == rhs.nni_round
               && lhs.constrain == rhs.constrain
               && lhs.model_override == rhs.model_override
               && lhs.fallback_fast_raxml == rhs.fallback_fast_raxml
               && lhs.dynamic_spr == rhs.dynamic_spr
               && lhs.do_final_model == rhs.do_final_model
               && lhs.aggressive_source == rhs.aggressive_source
               && lhs.aggressive_source_family == rhs.aggressive_source_family
               && lhs.fixed_model_au_precheck == rhs.fixed_model_au_precheck
               && lhs.pin_initial_ml_model == rhs.pin_initial_ml_model;
    }

    friend bool operator!=(const MetaParameters &lhs, const MetaParameters &rhs) {
        return !(lhs == rhs);
    }

    friend std::size_t hash_value(const MetaParameters &obj) {
        std::size_t seed = 0x148BCA29;
        seed ^= (seed << 6) + (seed >> 2) + 0x26C25243 + static_cast<std::size_t>(obj.keep_top_k_topol);
        seed ^= (seed << 6) + (seed >> 2) + 0x0341DCC9 + static_cast<std::size_t>(obj.do_first_model);
        seed ^= (seed << 6) + (seed >> 2) + 0x674CC085 + static_cast<std::size_t>(obj.num_fast_spr);
        seed ^= (seed << 6) + (seed >> 2) + 0x61730347 + static_cast<std::size_t>(obj.num_slow_spr);
        seed ^= (seed << 6) + (seed >> 2) + 0x49932C6A + static_cast<std::size_t>(obj.accept_starting_trees);
        seed ^= (seed << 6) + (seed >> 2) + 0x74D4612F + static_cast<std::size_t>(obj.max_adaptive_radius);
        seed ^= (seed << 6) + (seed >> 2) + 0x3C871EF5 + static_cast<std::size_t>(obj.nni_round);
        seed ^= (seed << 6) + (seed >> 2) + 0x667D39FE + static_cast<std::size_t>(obj.constrain);
        seed ^= (seed << 6) + (seed >> 2) + 0x72878931 + (obj.model_override.has_value()
                                                              ? std::filesystem::hash_value(obj.model_override.value())
                                                              : 0);
        seed ^= (seed << 6) + (seed >> 2) + 0x5458316A + static_cast<std::size_t>(obj.fallback_fast_raxml);
        seed ^= (seed << 6) + (seed >> 2) + 0x39D34241 + static_cast<std::size_t>(obj.dynamic_spr);
        seed ^= (seed << 6) + (seed >> 2) + 0x72C16A1E + static_cast<std::size_t>(obj.do_final_model);
        seed ^= (seed << 6) + (seed >> 2) + 0x50E7A8B9 + static_cast<std::size_t>(obj.aggressive_source);
        seed ^= (seed << 6) + (seed >> 2) + 0x38D9B7C1 + static_cast<std::size_t>(obj.aggressive_source_family);
        seed ^= (seed << 6) + (seed >> 2) + 0x2228A3E7 + static_cast<std::size_t>(obj.fixed_model_au_precheck);
        seed ^= (seed << 6) + (seed >> 2) + 0x4D1A2F0B + static_cast<std::size_t>(obj.pin_initial_ml_model);
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
