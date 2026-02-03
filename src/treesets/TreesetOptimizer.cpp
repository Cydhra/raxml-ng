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

void TreesetOptimizer::prepare_initial_batches() {
    const auto minimum_batches = (this->target_tree_count + this->batch_size - 1) / this->batch_size;
    this->generate_batches(minimum_batches);
}

/**
 *
 */
void TreesetOptimizer::generate_batches(const unsigned int n) {
    const std::string name_prefix = "Batch";

    for (size_t i = 0; i < n; i++) {
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
        starting_tree_bandit().apply_parameters(opts, current_batch);
        current_batch.generate_starting_trees(this->instance, opts, load_balancer, tip_msa_idmap);
        // TODO: we probably don't need the AU test for all batches, we can save time if we infer it only for enough to
        //  get acceptable estimates of success and throughput
        current_batch.perform_au_test(opts);
        starting_tree_bandit().take_measurement(current_batch);
    }
}

void TreesetOptimizer::initialize_bandits() {
    if (this->starting_tree_bandit().get_mean_success() >= 0.9) {
        LOG_INFO << "Starting trees are so successful, no ML optimization is necessary." << std::endl;
        this->batch_cursor = this->batches.size();
    } else {
        // init default bandits
        this->bandits.emplace_back("Greedy,DoModel,4spr", MetaParameters(1, false, 4, false));
        this->bandits.emplace_back("Greedy,DoModel,2spr", MetaParameters(1, false, 2, false));
        this->bandits.emplace_back("Greedy,NoModel,2spr", MetaParameters(1, true, 2, false));

        this->bandits.emplace_back("Fast,DoModel,4spr", MetaParameters(20, false, 4, false));
        this->bandits.emplace_back("Fast,DoModel,2spr", MetaParameters(20, false, 2, false));
        this->bandits.emplace_back("Fast,NoModel,2spr", MetaParameters(20, true, 2, false));
    }
}

Bandit &TreesetOptimizer::select_next_bandit() {
    const auto current = this->bandit_cursor;

    this->bandit_cursor += 1;
    this->bandit_cursor %= this->bandits.size();

    // TODO select best bandit if the current bandit is definitely worse.
    return this->bandits[current];
}

TunedBatch &TreesetOptimizer::select_next_batch() {
    const auto current = this->batch_cursor;
    // TODO: figure out when to reuse a batch using dependencies between bandits
    this->batch_cursor += 1;

    if (this->batches.size() == current) {
        this->generate_batches(1);
    }

    return this->batches[current];
}

void TreesetOptimizer::run() {
    this->prepare_initial_batches();
    LOG_INFO << std::endl;
    LOG_INFO << "Expected throughput of starting trees: " << (this->starting_tree_bandit().get_mean_throughput() * 1000.0) << " trees per second at a mean success rate of " << (this->starting_tree_bandit().get_mean_success() * 100.0) << "%." << std::endl;

    this->initialize_bandits();

    while (true) {
        auto &current_bandit = this->select_next_bandit();
        auto &current_batch = this->select_next_batch();

        // Todo figure out if current or best batch are supposed to be tested
        current_bandit.apply_parameters(opts, current_batch);
        current_batch.optimize(opts);
        current_batch.perform_au_test(opts);
        current_bandit.take_measurement(current_batch);

        // TODO currently we break after the first round. Instead we should obviously break only when we have enough
        //  plausible trees. Keep in mind, that reusing a batch means we arent getting all the plausible trees from the
        //  batch's previous run.
        if (this->bandit_cursor == 0) {
            break;
        }
    }
}
