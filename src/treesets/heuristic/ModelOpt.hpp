#ifndef RAXML_MODELOPT_HPP_
#define RAXML_MODELOPT_HPP_

#include "Heuristic.hpp"

class ModelOpt : ImprovingHeuristic {
public:
    ModelOpt(const std::string &batch_name, std::unique_ptr<ImprovingHeuristic> inner,
             const std::shared_ptr<MetaParameters> &meta_parameters, const bool model, const bool branches,
             const bool force, const double epsilon)
        : ImprovingHeuristic(batch_name, std::move(inner)),
          meta_parameters(meta_parameters),
          model(model),
          branches(branches),
          force(force),
          epsilon(epsilon) {
    }

    ModelOpt(ModelOpt &&other) noexcept = default;

    ModelOpt &operator=(ModelOpt &&other) noexcept = default;

    void do_optimize(std::optional<TreeInfo> &tree, const Options &opts, const TaskGroup &context, SharedBatchResources &resources,
                     unsigned int worker_id, unsigned int thread_id) override;

    // TODO remove
    std::shared_ptr<MetaParameters> meta_parameters;

protected:
    bool model;

    bool branches;

    bool force;

    double epsilon;
};

#endif //RAXML_MODELOPT_HPP_
