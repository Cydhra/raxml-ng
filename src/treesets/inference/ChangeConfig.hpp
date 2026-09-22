#ifndef RAXML_NG_CHANGECONFIG_HPP
#define RAXML_NG_CHANGECONFIG_HPP

#include "Heuristic.hpp"
#include "../../loadbalance/CoarseLoadBalancer.hpp"

class ChangeConfig : public InferenceHeuristic {
public:
    ChangeConfig(std::string batch_name, std::unique_ptr<InferenceHeuristic> inner, const unsigned num_trees,
        const unsigned threads_per_worker, shared_ptr<TreeList> batch_start_trees,
        std::shared_ptr<PartitionAssignmentList> part_assignments, std::shared_ptr<ModelMap> initial_model,
        std::optional<std::string> model_override, std::shared_ptr<PartitionedMSA> msa,
        std::shared_ptr<IDVector> tip_msa_idmap, std::optional<std::vector<uintVector>> site_masks = std::nullopt)
        : InferenceHeuristic(std::move(batch_name), std::move(inner), num_trees, threads_per_worker),
          batch_start_trees(std::move(batch_start_trees)),
          part_assignments(std::move(part_assignments)),
          initial_model(std::move(initial_model)),
          model_override(std::move(model_override)),
          msa(std::move(msa)),
          tip_msa_idmap(std::move(tip_msa_idmap)), site_masks(site_masks) {
    }

    ChangeConfig(ChangeConfig &&other) noexcept = default;

    ChangeConfig &operator=(ChangeConfig &&other) noexcept = default;

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

    std::optional<std::vector<uintVector>> site_masks;
};

#endif //RAXML_NG_CHANGECONFIG_HPP
