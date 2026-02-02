#include "TreesetOptimizer.hpp"

#include "TunedBatch.hpp"

int recommended_thread_count() {
    // TODO add parameters to options containing the max thread count, which we just assign to the single worker per rank
    return 8;
}

int recommended_worker_count() {
    // TODO this is only true for the tuning phase
    return 1;
}


void TreesetOptimizer::prepare_initial_batches(RaxmlInstance &instance, const Options &opts, LoadBalancer &load_balancer, const IDVector &tip_msa_idmap) {
    const auto minimum_batches = (this->target_tree_count + this->batch_size - 1) / this->batch_size;

    const std::string name_prefix = "Batch";

    for (size_t i = 0; i < minimum_batches; i++) {
        std::string batch_name = name_prefix + std::to_string(i);
        this->batches.emplace_back(batch_name,
                                   generate_seed_for_trees(this->batch_size),
                                   this->batch_size,
                                   recommended_thread_count(),
                                   recommended_worker_count(),
                                   msa,
                                   persite_loglh);
        auto &current_batch = this->batches[this->batches.size() - 1];

        // TODO schedule the batches in parallel if enough threads are available
        starting_tree_bandit().apply_parameters(current_batch);
        current_batch.generate_starting_trees(instance, opts, load_balancer, tip_msa_idmap);
        current_batch.perform_au_test(opts);
        starting_tree_bandit().take_measurement(current_batch);
    }
}

void TreesetOptimizer::run(RaxmlInstance &instance, Options &opts, LoadBalancer &load_balancer, const IDVector &tip_msa_idmap) {
    this->prepare_initial_batches(instance, opts, load_balancer, tip_msa_idmap);
    LOG_INFO << "Expected throughput of starting trees: " << (this->starting_tree_bandit().get_mean_throughput() * 1000.0) << " trees per second" << std::endl;
}
