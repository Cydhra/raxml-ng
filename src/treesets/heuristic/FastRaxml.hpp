#ifndef RAXML_FASTRAXML_HPP_
#define RAXML_FASTRAXML_HPP_

#include "Heuristic.hpp"

class FastRaxml : public InferenceHeuristic {

public:
    FastRaxml(std::string batch_name, std::unique_ptr<InferenceHeuristic> inner, const unsigned num_trees,
        const unsigned threads_per_worker, std::shared_ptr<PartitionAssignmentList> part_assignments)
        : InferenceHeuristic(std::move(batch_name), std::move(inner), num_trees, threads_per_worker),
          part_assignments(std::move(part_assignments)) {
    }

    FastRaxml(FastRaxml &&other) noexcept = default;

    FastRaxml & operator=(FastRaxml &&other) noexcept = default;

    void do_optimize(std::optional<TreeInfo> &tree, unsigned int tree_id, const Options &opts, const TaskGroup &context, SharedBatchResources &resources,
                     unsigned int worker_id, unsigned int thread_id) override;

protected:
    std::shared_ptr<PartitionAssignmentList> part_assignments;
};



#endif //RAXML_FASTRAXML_HPP_
