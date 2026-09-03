#ifndef RAXML_HEURISTICFACTORY_HPP_
#define RAXML_HEURISTICFACTORY_HPP_
#include "../MetaParameters.hpp"
#include "Heuristic.hpp"


class HeuristicFactory {
public:
    static unique_ptr<InferenceHeuristic> build_heuristic(const MetaParameters &meta_parameters,
                                                          std::string &batch_name,
                                                          unsigned int num_trees,
                                                          unsigned int threads_per_worker,
                                                          shared_ptr<TreeList> &start_tree_list,
                                                          shared_ptr<PartitionAssignmentList> &part_assignments,
                                                          shared_ptr<ModelMap> &initial_model,
                                                          shared_ptr<PartitionedMSA> &partitioned_msa,
                                                          shared_ptr<IDVector> &tip_msa_idmap);

    static unique_ptr<InferenceHeuristic> extend_heuristic(const MetaParameters &new_parameters,
                                                           const MetaParameters &old_parameters,
                                                           unique_ptr<InferenceHeuristic> old_heuristic,
                                                           string &batch_name,
                                                           unsigned int num_trees,
                                                           unsigned int threads_per_worker,
                                                           shared_ptr<PartitionAssignmentList> &part_assignments,
                                                           shared_ptr<PartitionedMSA> &partitioned_msa);
};


#endif //RAXML_HEURISTICFACTORY_HPP_
