#ifndef RAXML_PARSIMONY_HPP
#define RAXML_PARSIMONY_HPP

#include "Heuristic.hpp"
#include "../../loadbalance/CoarseLoadBalancer.hpp"

// forward declaration
struct RaxmlInstance;

// forward declaration of generate_tree in main.cpp to make it accessible. If the function in main.cpp
// changes signature, just update this declaration as well.
Tree generate_tree(const RaxmlInstance &instance, StartingTree type, int random_seed, bool bootstrap);

class StartTrees : ImprovingHeuristic {
public:
    StartTrees(const std::string &batch_name, std::unique_ptr<ImprovingHeuristic> inner,
              const std::string &model_override)
        : ImprovingHeuristic(batch_name, std::move(inner)),
          model_override(model_override) {
    }

    StartTrees(StartTrees &&other) noexcept = default;

    StartTrees &operator=(StartTrees &&other) noexcept = default;

    void do_optimize(std::optional<TreeInfo> &tree, unsigned int tree_id, const Options &opts, const TaskGroup &context,
                     SharedBatchResources &resources,
                     unsigned int worker_id, unsigned int thread_id) override;

protected:
    shared_ptr<TreeList> batch_start_trees;

    std::shared_ptr<PartitionAssignmentList> part_assignments;

    std::shared_ptr<ModelMap> initial_model;

    std::optional<std::string> model_override;

    std::shared_ptr<PartitionedMSA> msa;

    std::shared_ptr<IDVector> tip_msa_idmap;
};

#endif //RAXML_PARSIMONY_HPP
