#include "Threadpool.hpp"

void ThreadPool::work(const Options &opts) {
    const auto num_workers = this->workers_per_task_group * task_groups.size();

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

    // obtain new task for this thread
    if (context.is_group_leader(local_thread_id, worker_id)) {
        const auto task = this->task_generator();
        context.assign_task(std::move(task));
        context.enter_barrier();
    } else {
        context.enter_barrier();
    }

    // solve task
    const auto &task = context.get_task();
    task(context, worker_id, local_thread_id);
}
