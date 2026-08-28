#ifndef RAXML_NNIROUND_HPP_
#define RAXML_NNIROUND_HPP_

#include "Heuristic.hpp"

class NniRound : public InferenceHeuristic {

public:
    NniRound(const std::string &batch_name, std::unique_ptr<InferenceHeuristic> inner,
        const std::shared_ptr<MetaParameters> &meta_parameters)
        : InferenceHeuristic(batch_name, std::move(inner)),
          meta_parameters(meta_parameters) {
    }

    NniRound(NniRound &&other) noexcept = default;

    NniRound & operator=(NniRound &&other) noexcept = default;

    void do_optimize(std::optional<TreeInfo> &tree, unsigned int tree_id, const Options &opts, const TaskGroup &context, SharedBatchResources &resources,
                     unsigned int worker_id, unsigned int thread_id) override;

    // TODO remove
    std::shared_ptr<MetaParameters> meta_parameters;
};


#endif //RAXML_NNIROUND_HPP_
