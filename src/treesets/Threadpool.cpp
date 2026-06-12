#include "Threadpool.hpp"

#include "../log.hpp"

void ThreadPool::work(const Options &opts) {
    const auto num_workers = this->workers_per_task_group * task_groups.size();

    LOG_INFO << "Thread configuration: " << task_groups.size() << " groups x " << workers_per_task_group << " workers x " <<
        threads_per_task() / workers_per_task_group << " threads" << std::endl;

    const auto main_method = [this] { this->thread_main(); };
    ParallelContext::init_pthreads_custom(opts, main_method, this->total_threads, num_workers);
    main_method();
    ParallelContext::finalize_threads();
}

void ThreadPool::thread_main() {
    // determine local context of this thread
    const auto local_thread_id = ParallelContext::local_thread_id();
    const auto worker_id = ParallelContext::local_group_id();

    const auto task_id = worker_id / this->workers_per_task_group;
    TaskGroup &context = this->task_groups[task_id];

    const auto group_worker_id = context.get_group_worker_id(worker_id);

    while (this->running) {
        // obtain new task for this thread
        if (context.is_group_leader(worker_id, local_thread_id)) {
            const auto task = this->task_generator();
            context.assign_task(std::move(task));
            context.enter_barrier(__LINE__, __FILE__, __func__);
        } else {
            context.enter_barrier(__LINE__, __FILE__, __func__);
        }

        printf("thread %ld:%ld moved safely across barrier %s (%s:%d)\n", ParallelContext::group_id(), ParallelContext::local_thread_id(), __func__, __FILE__, __LINE__);

        // solve task
        const auto &task = context.get_task();

        printf("thread %ld:%ld retrieved task %s (%s:%d)\n", ParallelContext::group_id(), ParallelContext::local_thread_id(), __func__, __FILE__, __LINE__);
        task(context, group_worker_id, local_thread_id);

        printf("thread %ld:%ld exited task at %s (%s:%d)\n", ParallelContext::group_id(), ParallelContext::local_thread_id(), __func__, __FILE__, __LINE__);

        // wait at a barrier to make sure the task isn't shutting down the threadpool while some workers are already
        // in the next loop iteration.
        // TODO this still breaks if another thread cancels the pool while some workers are still in this barrier and others already left
        context.enter_barrier(__LINE__, __FILE__, __func__);
    }
}
