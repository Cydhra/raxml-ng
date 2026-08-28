#ifndef RAXML_HEURISTIC_HPP_
#define RAXML_HEURISTIC_HPP_

#include <memory>
#include <utility>
#include <string>
#include "../MetaParameters.hpp"
#include "../Threadpool.hpp"

// forward declaration to avoid cyclic header inclusion.
// only implementations of subclasses may import this type, headers must only refer to the forward declaration.
class SharedBatchResources;

/**
 * Defines any heuristic which can be applied to a batch of trees to infer Maximum Likelihood trees from them, or
 * prepare / advance those strategies in advance.
 * The trees must already exist as this heuristic cannot create starting trees.
 * The heuristics implement both a strategy and decorator pattern to allow customization of the full inference heuristic.
 */
class InferenceHeuristic {
public:
    virtual ~InferenceHeuristic() = default;

    InferenceHeuristic(std::string batch_name, std::unique_ptr<InferenceHeuristic> inner)
        : batch_name(std::move(batch_name)),
          inner(std::move(inner)) {
    }

    InferenceHeuristic(const InferenceHeuristic &other) = delete;

    InferenceHeuristic &operator=(const InferenceHeuristic &other) = delete;

    InferenceHeuristic(InferenceHeuristic &&other) noexcept = default;

    InferenceHeuristic &operator=(InferenceHeuristic &&other) noexcept = default;

    /**
     * Run the entire inference heuristic defined by this (and all contained) instances.
     *
     * @param tree reference to the option which will contain the final tree. It may be empty, if the first strategy
     *        allocates the `TreeInfo` object into it. All tree operations are performed on this instance.
     * @param tree_id batch-internal index of the tree. This can be used to index datastructures which hold instances
     *        per batch tree.
     * @param opts command line options of raxml
     * @param context task group context, used for synchronization of workers working on the same `TunedBatch`
     * @param resources `SharedBatchResources` instance of the treeset optimizer
     * @param worker_id raxml-instance-global id of the current thread's worker. Used for work distribution.
     * @param thread_id thread id within the current worker. Used for work distribution.
     */
    void optimize(std::optional<TreeInfo> &tree, const unsigned int tree_id, const Options &opts, // NOLINT(*-no-recursion)
                  const TaskGroup &context, SharedBatchResources &resources,
                  const unsigned int worker_id, const unsigned int thread_id) {
        if (inner) {
            inner->optimize(tree, tree_id, opts, context, resources, worker_id, thread_id);
        }

        do_optimize(tree, tree_id, opts, context, resources, worker_id, thread_id);
    }

    /**
     * Implementation of the concrete inference strategy. Inference will modify the `tree` argument, which is passed
     * to each implementation in the decorator chain. Subclasses of `ImprovingHeuristic` have to implement this
     * method.
     *
     * @param tree reference to the option which will contain the final tree. It may be empty, if the first strategy
     *        allocates the `TreeInfo` object into it. All tree operations are performed on this instance.
     * @param tree_id batch-internal index of the tree. This can be used to index datastructures which hold instances
     *        per batch tree.
     * @param opts command line options of raxml
     * @param context task group context, used for synchronization of workers working on the same `TunedBatch`
     * @param resources `SharedBatchResources` instance of the treeset optimizer
     * @param worker_id raxml-instance-global id of the current thread's worker. Used for work distribution.
     * @param thread_id thread id within the current worker. Used for work distribution.
     */
    virtual void do_optimize(std::optional<TreeInfo> &tree, unsigned int tree_id, const Options &opts,
                             const TaskGroup &context, SharedBatchResources &resources,
                             unsigned int worker_id, unsigned int thread_id) = 0;

protected:
    /**
     * Wall-time spent on the current heuristic. This is cumulative across all trees, even if they were processed in
     * parallel.
     */
    unsigned int cumulative_wall_time = 0;

    /**
     * Batch name of the batch that owns this strategy, used for log output.
     */
    std::string batch_name;

private:
    /**
     * Previous stage of the decorator chain. If this pointer is not null, the strategy contained in this pointer is
     * called before the current stage is executed.
     */
    std::unique_ptr<InferenceHeuristic> inner = {};
};

#endif //RAXML_HEURISTIC_HPP_
