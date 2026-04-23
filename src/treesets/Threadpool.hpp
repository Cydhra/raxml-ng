#ifndef RAXML_THREADPOOL_HPP_
#define RAXML_THREADPOOL_HPP_

#include <cassert>
#include <utility>
#include <atomic>
#include "../ParallelContext.hpp"
#include "SmartBarrier.hpp"

// forward declaration
class TaskGroup;

/**
 * Callable function for tasks scheduled via the threadpool. The function takes the task-group barrier,
 * the task-specific worker_id, and the task-specific thread_id as input. Note that these are likely different from
 * the ones returned by ParallelContext.
 */
using BatchTask = std::function<void(TaskGroup &, unsigned int, unsigned int)>;

/**
 * Callable function that returns new BatchTask callbacks on demand.
 */
using TaskGenerator = std::function<BatchTask()>;

/**
 * A group of workers (which are a threadgroup).
 * This adds a layer to the hierarchy of parallelization to allow coarse-grained tree searches to be bundled
 * into tasks.
 */
class TaskGroup {
public:
    TaskGroup(const unsigned int task_group_id, const unsigned int num_threads,
              const unsigned int num_workers) : task_group_id(task_group_id),
                                                num_threads(num_threads), num_workers(num_workers),
                                                task_barrier(num_threads) {
    }

    /**
     * Determines the leader thread in a task group.
     *
     * @param worker_id the local (within rank) worker (thread group) id of the current thread
     * @param thread_id the local (within worker) thread id of the current thread
     * @return true, if the current thread is the leader of the task group
     */
    bool is_group_leader(const unsigned int worker_id, const unsigned int thread_id) const {
        return thread_id == 0 && (worker_id % num_workers) == 0;
    }

    /**
     * Get the thread id of the current thread local to the TaskGroup: That is, all threads assigned in a task group are
     * numbered from 0 to n. For assignments that cannot be handled by workers, those ids can be used to share work
     * effectively.
     *
     * @param worker_id global (rank local) worker id
     * @param thread_id local (worker local) thread id
     * @return thread id of contiguous ids in the TaskGroup
     */
    unsigned int get_group_thread_id(const unsigned int worker_id, const unsigned int thread_id) const {
        return (worker_id % num_workers) * (num_threads / num_workers) + thread_id;
    }

    /**
     * Assign a new task function to the group. All threads should wait for a task to become assigned, then call it.
     *
     * @param task A callable that points to the (bound) task function
     */
    template<typename F>
    void assign_task(F &&task) {
        this->current_task = std::make_shared<BatchTask>(std::forward<F>(task));
    }

    /**
     * Get a reference to the task currently assigned to the task group.
     */
    BatchTask &get_task() const {
        return *current_task;
    }

    /**
     * Enter the task barrier and wait until all threads have entered.
     */
    void enter_barrier() const {
        this->task_barrier.enter();
    }

    /**
     * @return the rank-local group id of this TaskGroup
     */
    unsigned int group_id() const {
        return task_group_id;
    }

protected:
    /**
     *  Rank-local id of this task group, among all active groups.
     */
    unsigned int task_group_id;

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

    /**
     * Reference to the current task, which will be made available to all threads in the group.
     */
    std::shared_ptr<BatchTask> current_task;
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
public:
    ThreadPool(const TaskGenerator &task_generator, const unsigned int total_threads,
               const unsigned int workers_per_task_group,
               const unsigned int num_task_groups) : total_threads(total_threads),
                                                     workers_per_task_group(workers_per_task_group),
                                                     task_generator(task_generator) {
        task_groups.reserve(num_task_groups);

        // TODO relax this assumption by assigning threads weirdly, or enforce it by disallowing num_task_groups to not be a divisor
        assert(total_threads % num_task_groups == 0);
        auto threads_per_task_group = total_threads / num_task_groups;

        for (unsigned int i = 0; i < num_task_groups; ++i) {
            task_groups.emplace_back(i, threads_per_task_group, workers_per_task_group);
        }
    }

    /**
     * Spawn all threads, and then distribute work to task groups until no more work is left.
     * Only then will the method return.
     */
    void work(const Options &opts);

    /**
     * Thread groups will no longer ask for work and exit their main method.
     * Threads will not cancel their existing work.
     */
    void shutdown() {
        this->running.exchange(0);
    }

    /**
     * Joins all remaining threads. This can only be called from the main thread.
     */
    static void join() {
        ParallelContext::finalize();
    }

    /**
     * @return the total number of raxml workers assigned to each task group.
     */
    unsigned int workers_per_task() const {
        return this->workers_per_task_group;
    }

    /**
     * @return the total number of threads assigned to each task group. Divisible by the number of workers
     */
    unsigned int threads_per_task() const {
        return this->total_threads / this->task_groups.size();
    }

protected:
    /**
     * Total number of threads assigned to the pool (locally within an MPI rank).
     */
    unsigned int total_threads;

    /**
     * How many workers (tree searches) to assign each task group.
     */
    unsigned int workers_per_task_group;

    /**
     * The task groups contain barriers to synchronize tasks and know which threads are the task group leaders.
     */
    std::vector<TaskGroup> task_groups;

    /**
     * A callback which generates callbacks for the individual task groups.
     * For each task group, the generator is called and then the generated function is called once for each thread in
     * the task group.
     */
    TaskGenerator task_generator;

    /**
     * While true, workers search for more work.
     */
    std::atomic_uint running = 1;

    /**
     * Main function for all pool threads, where they organize themselves and select work until none is left.
     */
    void thread_main();
};

#endif //RAXML_THREADPOOL_HPP_
