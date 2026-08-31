#ifndef RAXML_MODELOPT_HPP_
#define RAXML_MODELOPT_HPP_

#include "Heuristic.hpp"

class ModelOpt : public InferenceHeuristic {
public:
    ModelOpt(std::string batch_name, std::unique_ptr<InferenceHeuristic> inner, const unsigned num_trees,
        const unsigned threads_per_worker, const bool model, const bool branches, const double epsilon)
        : InferenceHeuristic(std::move(batch_name), std::move(inner), num_trees, threads_per_worker),
          model(model),
          branches(branches),
          epsilon(epsilon) {
    }

    ModelOpt(ModelOpt &&other) noexcept = default;

    ModelOpt &operator=(ModelOpt &&other) noexcept = default;

    void do_optimize(std::optional<TreeInfo> &tree, unsigned int tree_id, const Options &opts, const TaskGroup &context,
                     SharedBatchResources &resources,
                     unsigned int worker_id, unsigned int thread_id) override;

protected:
    bool model;

    bool branches;

    double epsilon;
};

#endif //RAXML_MODELOPT_HPP_
