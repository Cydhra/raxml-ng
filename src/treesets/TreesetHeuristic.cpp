#include "TreesetHeuristic.hpp"


void TreesetHeuristic::infer_treeset(RaxmlInstance &instance, const Options &opts, CheckpointManager &cm) {
    auto batch1 = TunedBatch(true, false, 4, 8, 8);
    batch1.infer_batch(instance, opts);

    LOG_INFO_TS << "treeset inference complete." << std::endl;
}

int TreesetHeuristic::recommended_thread_count() {
    // TODO add parameters to options containing the max thread count, which we just assign to the single worker per rank
    return 4;
}

int TreesetHeuristic::recommended_worker_count() {
    // TODO this is only true for the tuning phase
    return 1;
}

unsigned int TunedBatch::get_batch_size() const {
    return this->batch_start_trees->size();
}


void TunedBatch::infer_batch(RaxmlInstance &instance, const Options &opts) {
    LOG_INFO_TS << "Running inference batch [BLO: " << !this->light_spr << ", MO: " << !this->skip_model << ", SPR: " <<
            this->num_spr << "] with " << this->num_threads << " threads." << std::endl;

    const intVector seeds(this->get_batch_size());

    auto tree_builder = std::bind(thread_start_trees,
                               std::ref(instance),
                               std::ref(*this->batch_start_trees),
                               StartingTree::parsimony,
                               std::cref(seeds),
                               0,
                               false);

    // step 1: infer starting trees
    ParallelContext::init_pthreads_custom(opts, tree_builder, this->num_threads, this->num_threads);
    tree_builder();
    ParallelContext::finalize_threads();

    // TODO store the trees somewhere


}
