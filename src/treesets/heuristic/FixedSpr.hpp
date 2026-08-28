#ifndef RAXML_FIXEDSPR_HPP_
#define RAXML_FIXEDSPR_HPP_

#include "Heuristic.hpp"

class FixedSpr : public InferenceHeuristic {
public:
    FixedSpr(const std::string &batch_name, std::unique_ptr<InferenceHeuristic> inner,
             const std::shared_ptr<MetaParameters> &meta_parameters)
        : InferenceHeuristic(batch_name, std::move(inner)),
          meta_parameters(meta_parameters) {
    }

    FixedSpr(FixedSpr &&other) noexcept = default;

    FixedSpr &operator=(FixedSpr &&other) noexcept = default;

    void do_optimize(std::optional<TreeInfo> &tree, unsigned int tree_id, const Options &opts, const TaskGroup &context, SharedBatchResources &resources,
                     unsigned int worker_id, unsigned int thread_id) override;

    // TODO remove
    std::shared_ptr<MetaParameters> meta_parameters;

protected:
    /**
     * Number of fast SPR rounds that have already been performed on the tree.
     */
    unsigned int num_fast_spr_performed{0};

    /**
     * Number of slow SPR rounds that have already been performed on the tree.
     */
    unsigned int num_slow_spr_performed{0};
};

#endif //RAXML_FIXEDSPR_HPP_
