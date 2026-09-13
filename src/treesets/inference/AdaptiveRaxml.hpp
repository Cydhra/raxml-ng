#ifndef RAXML_NG_ADAPTIVERAXML_HPP
#define RAXML_NG_ADAPTIVERAXML_HPP

#include "Heuristic.hpp"

class AdaptiveRaxml : public InferenceHeuristic {

public:
    AdaptiveRaxml(std::string batch_name, std::unique_ptr<InferenceHeuristic> inner, const unsigned num_trees,
        const unsigned threads_per_worker)
        : InferenceHeuristic(std::move(batch_name), std::move(inner), num_trees, threads_per_worker) {
    }

    AdaptiveRaxml(AdaptiveRaxml &&other) noexcept = default;

    AdaptiveRaxml & operator=(AdaptiveRaxml &&other) noexcept = default;

    void do_optimize(std::optional<TreeInfo> &tree, unsigned int tree_id, const Options &opts, const TaskGroup &context, SharedBatchResources &resources,
                     unsigned int worker_id, unsigned int thread_id) override;
};


#endif //RAXML_NG_ADAPTIVERAXML_HPP
