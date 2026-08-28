#ifndef RAXML_FIXEDSPR_HPP_
#define RAXML_FIXEDSPR_HPP_

#include "Heuristic.hpp"

class FixedSpr : public InferenceHeuristic {
public:
    FixedSpr(const std::string &batch_name, std::unique_ptr<InferenceHeuristic> inner,
             const unsigned int spr_rounds, const unsigned int keep_top_k_topol, const bool thorough,
             const unsigned int max_radius)
        : InferenceHeuristic(batch_name, std::move(inner)),
          num_spr(spr_rounds),
          keep_top_k_topol(keep_top_k_topol),
          thorough(thorough),
          max_radius(max_radius) {
    }

    FixedSpr(FixedSpr &&other) noexcept = default;

    FixedSpr &operator=(FixedSpr &&other) noexcept = default;

    void do_optimize(std::optional<TreeInfo> &tree, unsigned int tree_id, const Options &opts, const TaskGroup &context,
                     SharedBatchResources &resources,
                     unsigned int worker_id, unsigned int thread_id) override;

protected:
    /**
     * Number of fast SPR rounds that have already been performed on the tree.
     */
    unsigned int num_spr_performed{0};

    /**
     * Number of SPR rounds for this step
     */
    unsigned int num_spr;

    /**
     * Number of best topologies to keep during SPR rounds.
     */
    unsigned int keep_top_k_topol;

    /**
     * If true, will perform slow SPR rounds.
     */
    bool thorough;

    /**
     * Maximum SPR radius
     */
    unsigned int max_radius;

    /**
     * Obtain spr_round_params for the SPR rounds.
     *
     * @param opts parsed command line options with defaults and user-mandated search parameters
     */
    spr_round_params auto_configure(const Options &opts) const {
        spr_round_params spr_params{};

        // update options according to MetaParameters:
        spr_params.ntopol_keep = static_cast<int>(this->keep_top_k_topol);
        spr_params.subtree_cutoff = opts.spr_cutoff;
        spr_params.radius_min = 1;

        // if all fast spr rounds have been performed, set thorough to true, so further spr rounds are slow
        spr_params.thorough = this->thorough;
        spr_params.lh_epsilon_brlen_full = opts.lh_epsilon;
        spr_params.lh_epsilon_brlen_triplet = opts.lh_epsilon_brlen_triplet;

        // taken from the fast heuristic
        spr_params.radius_max = static_cast<int>(this->max_radius);

        // we don't need those
        spr_params.increasing_moves = nullptr;
        spr_params.total_moves = nullptr;

        return spr_params;
    }
};

#endif //RAXML_FIXEDSPR_HPP_
