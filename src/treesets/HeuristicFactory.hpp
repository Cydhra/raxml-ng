#ifndef RAXML_HEURISTICFACTORY_HPP_
#define RAXML_HEURISTICFACTORY_HPP_
#include "MetaParameters.hpp"
#include "heuristic/Heuristic.hpp"


class HeuristicFactory {
public:
    static unique_ptr<InferenceHeuristic> build_heuristic(const MetaParameters &meta_parameters, std::string &batch_name,
                                              shared_ptr<TreeList> &start_tree_list,
                                              shared_ptr<PartitionAssignmentList> &part_assignments,
                                              /* TODO: shouldn't this be optional? */ shared_ptr<ModelMap> &initial_model,
                                              shared_ptr<PartitionedMSA> &partitioned_msa,
                                              shared_ptr<IDVector> &tip_msa_idmap);
};


#endif //RAXML_HEURISTICFACTORY_HPP_
