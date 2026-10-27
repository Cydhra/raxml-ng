#include "HeuristicFactory.hpp"

#include "AdaptiveRaxml.hpp"
#include "ChangeConfig.hpp"
#include "Constrain.hpp"
#include "DynamicSpr.hpp"
#include "FastRaxml.hpp"
#include "FixedSpr.hpp"
#include "StartTrees.hpp"
#include "ModelOpt.hpp"
#include "NniRound.hpp"

constexpr double FIRST_MODEL_EPS = 3.0;

constexpr double FINAL_MODEL_EPS = 0.1;

constexpr double GREEDY_MODEL_EPS = 1.0;

constexpr unsigned int KEEP_TOPOL_NUM = 20;

/**
 * Decorate an existing partial InferenceHeuristic according to the rules of a meta-parameters instance.
 * This function codifies how meta-parameters are converted into a heuristic, and is used to create new
 * strategies, as well as extend others from a delta of meta parameters.
 *
 * @param meta_parameters The building rules for the strategy
 * @param heuristic the existing heuristic that is to be decorated further
 * @param batch_name name of the batch the strategy is built for
 * @param num_trees number of trees in the batch
 * @param threads_per_worker number of threads that infer each tree
 * @param part_assignments partition assignment across the threads
 * @param partitioned_msa the msa for the inference
 * @return
 */
static std::unique_ptr<InferenceHeuristic> from_meta_parameters(const MetaParameters &meta_parameters,
                                                           std::unique_ptr<InferenceHeuristic> heuristic,
                                                           std::string &batch_name,
                                                           const unsigned int num_trees,
                                                           const unsigned int threads_per_worker,
                                                           std::shared_ptr<TreeList> &start_tree_list,
                                                           std::shared_ptr<PartitionAssignmentList> &part_assignments,
                                                           std::shared_ptr<ModelMap> &initial_model,
                                                           std::shared_ptr<PartitionedMSA> &partitioned_msa,
                                                           std::shared_ptr<IDVector> &tip_msa_idmap) {
    if (meta_parameters.do_first_model) {
        heuristic = std::make_unique<ModelOpt>(batch_name, std::move(heuristic), num_trees, threads_per_worker, true, true,
                                          FIRST_MODEL_EPS);
    }

    if (meta_parameters.nni_round) {
        heuristic = std::make_unique<NniRound>(batch_name, std::move(heuristic), num_trees, threads_per_worker);
    }

    if (meta_parameters.constrain) {
        heuristic = std::make_unique<Constrain>(batch_name, std::move(heuristic), num_trees, threads_per_worker,
                                           partitioned_msa, part_assignments);
    }

    if (meta_parameters.num_fast_spr > 0) {
        heuristic = std::make_unique<FixedSpr>(batch_name, std::move(heuristic), num_trees, threads_per_worker,
                                          meta_parameters.num_fast_spr, meta_parameters.keep_top_k_topol, false,
                                          2 * meta_parameters.max_adaptive_radius);
    }

    if (meta_parameters.num_slow_spr > 0) {
        heuristic = std::make_unique<FixedSpr>(batch_name, std::move(heuristic), num_trees, threads_per_worker,
                                          meta_parameters.num_slow_spr, KEEP_TOPOL_NUM, true,
                                          1 * meta_parameters.max_adaptive_radius);
    }

    if (meta_parameters.dynamic_spr) {
        heuristic = std::make_unique<DynamicSpr>(batch_name, std::move(heuristic), num_trees, threads_per_worker,
                                            part_assignments, meta_parameters.keep_top_k_topol, false,
                                            meta_parameters.max_adaptive_radius);
    }

    if (meta_parameters.fallback_fast_raxml) {
        // reset model
        heuristic = std::make_unique<ChangeConfig>(batch_name, std::move(heuristic), num_trees, threads_per_worker,
                                              start_tree_list, part_assignments, std::make_shared<ModelMap>(), std::nullopt,
                                              partitioned_msa, tip_msa_idmap);

        heuristic = std::make_unique<FastRaxml>(batch_name, std::move(heuristic), num_trees, threads_per_worker,
                                           part_assignments);
    } else if (meta_parameters.fallback_adaptive_raxml) {
        // reset model
        heuristic = std::make_unique<ChangeConfig>(batch_name, std::move(heuristic), num_trees, threads_per_worker,
                                              start_tree_list, part_assignments, std::make_shared<ModelMap>(), std::nullopt,
                                              partitioned_msa, tip_msa_idmap);

        heuristic = std::make_unique<AdaptiveRaxml>(batch_name, std::move(heuristic), num_trees, threads_per_worker);
    }

    if (meta_parameters.model_override) {
        // revert model override
        heuristic = std::make_unique<ChangeConfig>(batch_name, std::move(heuristic), num_trees, threads_per_worker,
                                              start_tree_list, part_assignments, initial_model, std::nullopt,
                                              partitioned_msa, tip_msa_idmap);
    }

    if (meta_parameters.do_final_model) {
        heuristic = std::make_unique<ModelOpt>(batch_name, std::move(heuristic), num_trees, threads_per_worker, true, true,
                                          FINAL_MODEL_EPS);
    } else {
        heuristic = std::make_unique<ModelOpt>(batch_name, std::move(heuristic), num_trees, threads_per_worker, true, true,
                                          GREEDY_MODEL_EPS, false);
    }

    return heuristic;
}

std::unique_ptr<InferenceHeuristic> HeuristicFactory::build_heuristic(const MetaParameters &meta_parameters,
                                                                 std::string &batch_name,
                                                                 const unsigned int num_trees,
                                                                 const unsigned int threads_per_worker,
                                                                 std::shared_ptr<TreeList> &start_tree_list,
                                                                 std::shared_ptr<PartitionAssignmentList> &part_assignments,
                                                                 std::shared_ptr<ModelMap> &initial_model,
                                                                 std::shared_ptr<PartitionedMSA> &partitioned_msa,
                                                                 std::shared_ptr<IDVector> &tip_msa_idmap) {
    std::unique_ptr<InferenceHeuristic> heuristic = std::make_unique<StartTrees>(batch_name, nullptr, num_trees,
                                                                       threads_per_worker, start_tree_list,
                                                                       part_assignments, initial_model,
                                                                       meta_parameters.model_override, partitioned_msa,
                                                                       tip_msa_idmap);
    return from_meta_parameters(meta_parameters, std::move(heuristic), batch_name, num_trees, threads_per_worker,
                                start_tree_list, part_assignments, initial_model, partitioned_msa, tip_msa_idmap);
}

std::unique_ptr<InferenceHeuristic> HeuristicFactory::extend_heuristic(const MetaParameters &new_parameters,
                                                                  const MetaParameters &old_parameters,
                                                                  std::unique_ptr<InferenceHeuristic> old_heuristic,
                                                                  std::string &batch_name,
                                                                  const unsigned int num_trees,
                                                                  const unsigned int threads_per_worker,
                                                                  std::shared_ptr<TreeList> &start_tree_list,
                                                                  std::shared_ptr<PartitionAssignmentList> &part_assignments,
                                                                  std::shared_ptr<ModelMap> &initial_model,
                                                                  std::shared_ptr<PartitionedMSA> &partitioned_msa,
                                                                  std::shared_ptr<IDVector> &tip_msa_idmap) {
    MetaParameters delta = new_parameters; // copy everything into the delta
    const bool accept_anything = !old_parameters.any_rounds() || new_parameters.is_fallback();

    // if the new parameters arent a fallback, add whatever is remaining to the bandit, otherwise just do what the
    // parameters say. If the old parameters accepted starting trees, we also don't need to mind old parameters and simply
    // do what the new_parameters say
    if (!new_parameters.is_fallback()) {
        assert(accept_anything || !old_parameters.do_first_model || new_parameters.do_first_model);
        assert(accept_anything || !old_parameters.do_final_model || new_parameters.do_final_model);

        // we cannot undo a previously inferred model
        // skip model is inverse of do-model, so we need == here instead of !=
        delta.do_first_model = old_parameters.do_first_model != new_parameters.do_first_model;
        delta.do_final_model = old_parameters.do_final_model != new_parameters.do_final_model;

        assert(accept_anything || !old_parameters.nni_round || new_parameters.nni_round);
        // we cannot undo an NNI round
        delta.nni_round = old_parameters.nni_round != new_parameters.nni_round;

        assert(accept_anything || !old_parameters.constrain || new_parameters.constrain);
        // we cannot undo a constraint
        delta.constrain = old_parameters.constrain != new_parameters.constrain;

        delta.dynamic_spr = old_parameters.dynamic_spr != new_parameters.dynamic_spr;

        assert(accept_anything || old_parameters.num_fast_spr <= new_parameters.num_fast_spr);
        assert(
            accept_anything || old_parameters.num_fast_spr == new_parameters.num_fast_spr || old_parameters.num_slow_spr
            == 0);
        delta.num_fast_spr = new_parameters.num_fast_spr - old_parameters.num_fast_spr;

        assert(accept_anything || old_parameters.num_slow_spr <= new_parameters.num_slow_spr);
        delta.num_slow_spr = new_parameters.num_slow_spr - old_parameters.num_slow_spr;

        delta.fallback_fast_raxml = new_parameters.fallback_fast_raxml;
    }

    return from_meta_parameters(delta, std::move(old_heuristic), batch_name, num_trees, threads_per_worker,
                                start_tree_list, part_assignments, initial_model, partitioned_msa, tip_msa_idmap);
}
