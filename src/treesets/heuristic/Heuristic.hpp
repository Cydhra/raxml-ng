#ifndef RAXML_HEURISTIC_HPP_
#define RAXML_HEURISTIC_HPP_

#include <memory>
#include <utility>
#include <string>
#include "../MetaParameters.hpp"
#include "../Threadpool.hpp"

// forward declaration to avoid cyclic header inclusion
class SharedBatchResources;

/**
 * Defines any heuristic which can be applied to a batch of trees.
 * The trees must already exist as this heuristic cannot create starting trees.
 * The heuristics implement both a strategy and decorator pattern to allow customization of the full inference heuristic.
 */
class ImprovingHeuristic {
public:
    virtual ~ImprovingHeuristic() = default;

    ImprovingHeuristic(std::string batch_name, std::unique_ptr<ImprovingHeuristic> inner)
        : batch_name(std::move(batch_name)),
          inner(std::move(inner)) {
    }

    ImprovingHeuristic(const ImprovingHeuristic &other) = delete;

    ImprovingHeuristic & operator=(const ImprovingHeuristic &other) = delete;

    ImprovingHeuristic(ImprovingHeuristic &&other) noexcept = default;

    ImprovingHeuristic & operator=(ImprovingHeuristic &&other) noexcept = default;

    void optimize(std::optional<TreeInfo> &tree, const Options &opts,
                  const TaskGroup &context, SharedBatchResources &resources,
                  const unsigned int worker_id, const unsigned int thread_id) {
        if (inner) {
            inner->optimize(tree, opts, context, resources, worker_id, thread_id);
        }

        do_optimize(tree, opts, context, resources, worker_id, thread_id);
    }

    virtual void do_optimize(std::optional<TreeInfo> &tree, const Options &opts, const TaskGroup &context, SharedBatchResources &resources,
                             unsigned int worker_id, unsigned int thread_id) = 0;

protected:
    unsigned int wall_time = 0;

    std::string batch_name;

private:
    std::unique_ptr<ImprovingHeuristic> inner = {};
};

#endif //RAXML_HEURISTIC_HPP_
