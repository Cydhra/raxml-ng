#ifndef RAXML_THREADPOOL_HPP_
#define RAXML_THREADPOOL_HPP_

#include "SmartBarrier.hpp"

/**
 * Callable function for tasks scheduled via the threadpool. The function takes the task-group barrier,
 * the task-specific worker_id, and the task-specific thread_id as input. Note that these are likely different from
 * the ones returned by ParallelContext.
 */
using BatchTask = std::function<void(SmartBarrier, unsigned int, unsigned int)>;

/**
 * Callable function that returns new BatchTask callbacks on demand.
 */
using TaskGenerator = std::function<BatchTask&()>;

/**
 * A group of workers (which are a threadgroup).
 * This adds a layer to the hierarchy of parallelization to allow coarse-grained tree searches to be bundled
 * into tasks.
 */
class TaskGroup {
public:
    TaskGroup(const unsigned int num_threads, const unsigned int num_workers) : num_threads(num_threads),
        num_workers(num_workers), task_barrier(num_threads) {
    }

protected:
    /**
     * Total number of threads in this group. Not threads per worker.
     */
    unsigned int num_threads;

    /**
     * Total number of workers in this group.
     */
    unsigned int num_workers;

    /**
     * Memory barrier for all workers in the task group.
     */
    SmartBarrier task_barrier;
};

/**
 * A pool for handling dynamic workloads on singular ranks.
 * That is, the pool is oblivious to MPI and just handles non-uniform workloads.
 * It is designed with the static load balancing of raxml-ng in mind,
 * meaning it still spawns fine-grained parallel workers that can handle the default tree searches without changes to the code-base.
 * The workers group multiple threads like the default code-base, but multiple workers can be bundled into TaskGroups
 * and those can synchronize without interfering with other workers.
 *
 * The batch-groups are designed to take tasks out of a queue until the queue is empty, requiring no finalization of
 * threads between workloads.
 */
class ThreadPool {
    ThreadPool(TaskGenerator &task_generator, const unsigned int total_threads,
               const unsigned int workers_per_task_group,
               const unsigned int num_task_groups) : task_generator(task_generator) {
        task_groups.reserve(num_task_groups);

        // TODO relax this assumption by assigning threads weirdly, or enforce it by disallowing num_task_groups to not be a divisor
        assert(total_threads % num_task_groups == 0);
        auto threads_per_task_group = total_threads / num_task_groups;

        for (unsigned int i = 0; i < num_task_groups; ++i) {
            task_groups.emplace_back(threads_per_task_group, workers_per_task_group);
        }
    }

protected:
    std::vector<TaskGroup> task_groups;

    /**
     * A callback which generates callbacks for the individual task groups.
     * For each task group, the generator is called and then the generated function is called once for each thread in
     * the task group.
     */
    TaskGenerator &task_generator;
};

#endif //RAXML_THREADPOOL_HPP_
