#ifndef RAXML_DYNAMICSPR_HPP_
#define RAXML_DYNAMICSPR_HPP_

#include "Heuristic.hpp"

class DynamicSpr : public InferenceHeuristic {

public:
    DynamicSpr(std::string batch_name, std::unique_ptr<InferenceHeuristic> inner, const unsigned num_trees,
        const unsigned threads_per_worker, std::shared_ptr<PartitionAssignmentList> part_assignments)
        : InferenceHeuristic(std::move(batch_name), std::move(inner), num_trees, threads_per_worker),
          part_assignments(std::move(part_assignments)) {
    }

    DynamicSpr(DynamicSpr &&other) noexcept = default;

    DynamicSpr & operator=(DynamicSpr &&other) noexcept = default;

    void do_optimize(std::optional<TreeInfo> &tree, unsigned int tree_id, const Options &opts, const TaskGroup &context, SharedBatchResources &resources,
                     unsigned int worker_id, unsigned int thread_id) override;

protected:
    std::shared_ptr<PartitionAssignmentList> part_assignments;

    /**
     * Obtain spr_round_params for the SPR rounds.
     *
     * @param opts parsed command line options with defaults and user-mandated search parameters
     */
    spr_round_params auto_configure(const Options &opts) const {
        spr_round_params spr_params{};

        // update options according to MetaParameters:
        spr_params.ntopol_keep = static_cast<int>(20); // TODO obtain from parameters
        spr_params.subtree_cutoff = opts.spr_cutoff;
        spr_params.radius_min = 1;

        // if all fast spr rounds have been performed, set thorough to true, so further spr rounds are slow
        spr_params.thorough = false; // start with fast
        spr_params.lh_epsilon_brlen_full = opts.lh_epsilon;
        spr_params.lh_epsilon_brlen_triplet = opts.lh_epsilon_brlen_triplet;

        // taken from the fast heuristic
        spr_params.radius_max = 10; // TODO obtain from parameters

        // we don't need those
        spr_params.increasing_moves = nullptr;
        spr_params.total_moves = nullptr;

        return spr_params;
    }
};


#endif //RAXML_DYNAMICSPR_HPP_
