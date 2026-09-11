#ifndef RAXML_METAPARAMETERS_HPP_
#define RAXML_METAPARAMETERS_HPP_

#include <filesystem>
#include <optional>
#include <string>
#include "../Options.hpp"
#include "../Optimizer.hpp"

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
    unsigned int keep_top_k_topol = 20;

    /**
     * If true, does a model optimization before doing topology moves.
     */
    bool do_first_model = false;

    /**
     * If true, does a final model and branch length optimization.
     */
    bool do_final_model = true;

    /**
     * How many fast SPR rounds to perform for each tree search
     */
    unsigned int num_fast_spr = 0;

    /**
     * How many slow SPR rounds to perform for each tree search
     */
    unsigned int num_slow_spr = 0;

    /**
     * If true, skip all model optimization and simply accept parsimony starting trees as the final resulting topology.
     */
    bool accept_starting_trees = false;

    /**
     * Maximum SPR radius
     */
    unsigned int max_adaptive_radius = 20;

    /**
     * Perform an NNI round before the SPR rounds
     */
    bool nni_round = false;

    /**
     * If true, constrain the tree to ignore parts of the tree that are probably unresolvable.
     */
    bool constrain = false;

    /**
     * If set, override the model with a model built from this model string.
     */
    std::optional<std::string> model_override = std::nullopt;

    /**
     * If set, fall back to fast-raxml for inference
     */
    bool fallback_fast_raxml = false;

    /**
     * If set, run SPR rounds with stop criterion
     */
    bool dynamic_spr = false;

    explicit MetaParameters() = default;

    MetaParameters &with_topk(const unsigned int keep_top_k_topol) {
        this->keep_top_k_topol = keep_top_k_topol;
        return *this;
    }

    MetaParameters &with_first_model(const bool do_first_model) {
        this->do_first_model = do_first_model;
        return *this;
    }

    MetaParameters &with_final_model(const bool do_final_model) {
        this->do_final_model = do_final_model;
        return *this;
    }

    MetaParameters &with_fast_rounds(const unsigned int num_fast_spr) {
        this->num_fast_spr = num_fast_spr;
        return *this;
    }

    MetaParameters &with_slow_rounds(const unsigned int num_slow_spr) {
        this->num_slow_spr = num_slow_spr;
        return *this;
    }

    MetaParameters &with_starting_trees(const bool accept_starting_trees) {
        this->accept_starting_trees = accept_starting_trees;
        return *this;
    }

    MetaParameters &with_radius(const unsigned int max_adaptive_radius) {
        this->max_adaptive_radius = max_adaptive_radius;
        return *this;
    }

    MetaParameters &with_nni_round(const bool nni_round) {
        this->nni_round = nni_round;
        return *this;
    }

    MetaParameters &with_constrain(const bool constrain) {
        this->constrain = constrain;
        return *this;
    }

    MetaParameters &with_model_override(std::optional<std::string> model_override) {
        this->model_override = std::move(model_override);
        return *this;
    }

    MetaParameters &with_fallback_fast_raxml(const bool fallback_fast_raxml) {
        this->fallback_fast_raxml = fallback_fast_raxml;
        return *this;
    }

    MetaParameters &with_dynamic_spr(const bool dynamic_spr) {
        this->dynamic_spr = dynamic_spr;
        return *this;
    }

    friend bool operator==(const MetaParameters &lhs, const MetaParameters &rhs) {
        return lhs.keep_top_k_topol == rhs.keep_top_k_topol
               && lhs.do_first_model == rhs.do_first_model
               && lhs.do_final_model == rhs.do_final_model
               && lhs.num_fast_spr == rhs.num_fast_spr
               && lhs.num_slow_spr == rhs.num_slow_spr
               && lhs.accept_starting_trees == rhs.accept_starting_trees
               && lhs.max_adaptive_radius == rhs.max_adaptive_radius
               && lhs.nni_round == rhs.nni_round
               && lhs.constrain == rhs.constrain
               && lhs.model_override == rhs.model_override
               && lhs.fallback_fast_raxml == rhs.fallback_fast_raxml
               && lhs.dynamic_spr == rhs.dynamic_spr;
    }

    friend bool operator!=(const MetaParameters &lhs, const MetaParameters &rhs) {
        return !(lhs == rhs);
    }

    friend std::size_t hash_value(const MetaParameters &obj) {
        std::size_t seed = 0x148BCA29;
        seed ^= (seed << 6) + (seed >> 2) + 0x26C25243 + static_cast<std::size_t>(obj.keep_top_k_topol);
        seed ^= (seed << 6) + (seed >> 2) + 0x0341DCC9 + static_cast<std::size_t>(obj.do_first_model);
        seed ^= (seed << 6) + (seed >> 2) + 0x674CC085 + static_cast<std::size_t>(obj.do_final_model);
        seed ^= (seed << 6) + (seed >> 2) + 0x61730347 + static_cast<std::size_t>(obj.num_fast_spr);
        seed ^= (seed << 6) + (seed >> 2) + 0x49932C6A + static_cast<std::size_t>(obj.num_slow_spr);
        seed ^= (seed << 6) + (seed >> 2) + 0x74D4612F + static_cast<std::size_t>(obj.accept_starting_trees);
        seed ^= (seed << 6) + (seed >> 2) + 0x3C871EF5 + static_cast<std::size_t>(obj.max_adaptive_radius);
        seed ^= (seed << 6) + (seed >> 2) + 0x667D39FE + static_cast<std::size_t>(obj.nni_round);
        seed ^= (seed << 6) + (seed >> 2) + 0x72878931 + static_cast<std::size_t>(obj.constrain);
        seed ^= (seed << 6) + (seed >> 2) + 0x5458316A + (obj.model_override.has_value()
                                                              ? std::filesystem::hash_value(obj.model_override.value())
                                                              : 0);
        seed ^= (seed << 6) + (seed >> 2) + 0x39D34241 + static_cast<std::size_t>(obj.fallback_fast_raxml);
        seed ^= (seed << 6) + (seed >> 2) + 0x72C16A1E + static_cast<std::size_t>(obj.dynamic_spr);
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
