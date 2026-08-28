#ifndef RAXML_CONSTRAIN_HPP_
#define RAXML_CONSTRAIN_HPP_

#include "Heuristic.hpp"

class Constrain : public InferenceHeuristic {

public:
    Constrain(const std::string &batch_name, std::unique_ptr<InferenceHeuristic> inner, const std::shared_ptr<PartitionedMSA> &msa)
        : InferenceHeuristic(batch_name, std::move(inner)), msa(msa) {
    }

    Constrain(Constrain &&other) noexcept = default;

    Constrain & operator=(Constrain &&other) noexcept = default;

    void do_optimize(std::optional<TreeInfo> &tree, unsigned int tree_id, const Options &opts, const TaskGroup &context, SharedBatchResources &resources,
                     unsigned int worker_id, unsigned int thread_id) override;

public:
    // TODO remove this
    PartitionAssignmentList partition_assignments;

protected:
    std::shared_ptr<PartitionedMSA> msa;

};


#endif //RAXML_CONSTRAIN_HPP_