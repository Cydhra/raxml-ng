#include "HeuristicFactory.hpp"

#include "heuristic/Constrain.hpp"
#include "heuristic/FastRaxml.hpp"
#include "heuristic/FixedSpr.hpp"
#include "heuristic/StartTrees.hpp"
#include "heuristic/ModelOpt.hpp"
#include "heuristic/NniRound.hpp"

unique_ptr<InferenceHeuristic> HeuristicFactory::build_heuristic(const MetaParameters &meta_parameters,
                                                                 std::string &batch_name,
                                                                 shared_ptr<TreeList> &start_tree_list,
                                                                 shared_ptr<PartitionAssignmentList> &part_assignments,
                                                                 shared_ptr<ModelMap> &initial_model,
                                                                 shared_ptr<PartitionedMSA> &partitioned_msa,
                                                                 shared_ptr<IDVector> &tip_msa_idmap) {
    unique_ptr<InferenceHeuristic> heuristic = make_unique<StartTrees>(batch_name, nullptr, start_tree_list, part_assignments, initial_model,
                                               meta_parameters.model_override, partitioned_msa, tip_msa_idmap);

    if (meta_parameters.accept_starting_trees) {
        return heuristic;
    }

    if (!meta_parameters.skip_model) {
        // TODO get rid of magic numbers
        heuristic = make_unique<ModelOpt>(batch_name, std::move(heuristic), true, true, 3.0);
    }

    if (meta_parameters.nni_round) {
        heuristic = make_unique<NniRound>(batch_name, std::move(heuristic));
    }

    if (meta_parameters.constrain) {
        heuristic = make_unique<Constrain>(batch_name, std::move(heuristic), partitioned_msa);
    }

    if (meta_parameters.num_fast_spr > 0) {
        heuristic = make_unique<FixedSpr>(batch_name, std::move(heuristic), meta_parameters.num_fast_spr, meta_parameters.keep_top_k_topol, false, 2 * meta_parameters.max_adaptive_radius);
    }

    if (meta_parameters.num_slow_spr > 0) {
        heuristic = make_unique<FixedSpr>(batch_name, std::move(heuristic), meta_parameters.num_slow_spr, 20, true, 1 * meta_parameters.max_adaptive_radius);
    }

    if (meta_parameters.fallback_fast_raxml) {
        heuristic = make_unique<FastRaxml>(batch_name, std::move(heuristic), part_assignments);
    }

    return heuristic;
}
