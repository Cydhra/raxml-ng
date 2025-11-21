#include "TreesetHeuristic.hpp"

/**
 * Worker thread during the auto-tuning phase
 */
void run_worker_tuning() {

}

void TreesetHeuristic::infer_treeset(Options &opts, CheckpointManager &cm) {
    // step 1: run tree search for singular batches for auto-tuning
    ParallelContext::init_pthreads_custom(opts, run_worker_tuning,
                                          recommended_thread_count(), recommended_worker_count());
    run_worker_tuning();
}

int TreesetHeuristic::recommended_thread_count() {
    // TODO add parameters to options containing the max thread count, which we just assign to the single worker per rank
    return 4;
}

int TreesetHeuristic::recommended_worker_count() {
    // TODO this is only true for the tuning phase
    return 1;
}


