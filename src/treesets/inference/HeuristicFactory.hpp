#ifndef RAXML_HEURISTICFACTORY_HPP_
#define RAXML_HEURISTICFACTORY_HPP_
#include "../MetaParameters.hpp"
#include "Heuristic.hpp"
#include "../../Tree.hpp"
#include "../../loadbalance/PartitionAssignment.hpp"


class HeuristicFactory {
public:
    static std::unique_ptr<InferenceHeuristic> build_heuristic(const MetaParameters &meta_parameters,
                                                               std::string &batch_name,
                                                               unsigned int num_trees,
                                                               unsigned int threads_per_worker,
                                                               std::shared_ptr<TreeList> &start_tree_list,
                                                               std::shared_ptr<PartitionAssignmentList> &
                                                               part_assignments,
                                                               std::shared_ptr<ModelMap> &initial_model,
                                                               std::shared_ptr<PartitionedMSA> &partitioned_msa,
                                                               std::shared_ptr<IDVector> &tip_msa_idmap);

    static std::unique_ptr<InferenceHeuristic> extend_heuristic(const MetaParameters &new_parameters,
                                                                const MetaParameters &old_parameters,
                                                                std::unique_ptr<InferenceHeuristic> old_heuristic,
                                                                std::string &batch_name,
                                                                const unsigned int num_trees,
                                                                const unsigned int threads_per_worker,
                                                                std::shared_ptr<TreeList> &start_tree_list,
                                                                std::shared_ptr<PartitionAssignmentList> &
                                                                part_assignments,
                                                                std::shared_ptr<ModelMap> &initial_model,
                                                                std::shared_ptr<PartitionedMSA> &partitioned_msa,
                                                                std::shared_ptr<IDVector> &tip_msa_idmap);
};


#endif //RAXML_HEURISTICFACTORY_HPP_
