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
    this->generate_batches(INITIAL_VARIANCE_WEIGHT);
}

void TreesetOptimizer::generate_batches(const unsigned int n) {
    const std::string name_prefix = "Batch";

    for (size_t i = 0; i < n; i++) {
        std::string batch_name = name_prefix + std::to_string(this->batches.size());
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

        // if this isn't the very first tree, inherit model parameters from previous trees. Since we run the AU test
        // on the first tree, we have parameters optimized once
        if (this->batches.size() > 1) {
            current_batch.assign_batch_models(*this->backup_model);
        }

        // take a few measurements, but no more than necessary to have reasonable values for mean and variance
        if (batches.size() <= INITIAL_VARIANCE_WEIGHT) {
            current_batch.perform_plausibility_check(opts);

            if (batches.size() == 1) {
                current_batch.backup_models(*this->backup_model);
            }

            mab.take_measurement(starting_tree_bandit(), current_batch);
        }
    }
}

void TreesetOptimizer::initialize_bandits() {
    if (this->starting_tree_bandit().get_expected_tree_rate() >= 0.9) {
        LOG_INFO << "Starting trees are so successful, no ML optimization is necessary." << std::endl;
        advance_batch_cursor(this->batches.size() - this->batch_cursor);
    } else {
        // init default bandits
        this->mab.emplace_back("Greedy,DoModel,2spr", MetaParameters(1, false, 2, 0, false));
        this->mab.emplace_back("Greedy,DoModel,4spr", MetaParameters(1, false, 4, 0, false));
        this->mab.emplace_back("Greedy,NoModel,2spr", MetaParameters(1, true, 2, 0, false));

        this->mab.emplace_back("Fast,DoModel,2spr", MetaParameters(20, false, 2, 0, false));
        this->mab.emplace_back("Fast,DoModel,4spr", MetaParameters(20, false, 4, 0, false));
        this->mab.emplace_back("Fast,NoModel,2spr", MetaParameters(20, true, 2, 0, false));

        // experimental thorough bandits
        this->mab.emplace_back("Slow,2spr", MetaParameters(20, false, 0, 2, false));
    }
}

Bandit &TreesetOptimizer::select_next_bandit() {
    return this->mab.select_next_bandit();
}

TunedBatch &TreesetOptimizer::select_next_batch(const Bandit &current_bandit) {
    if (this->batch_cursor == this->batches.size()) {
        // generate a new batch because we need it right now.
        generate_batches(1);
        return this->batches[this->batch_cursor];
    }

    // if we still have batches in the queue, check if we should advance the cursor or reuse the current one:

    // to speed up the initial round of computation where all bandits are executed once,
    // we want to reuse batches. If the total rounds is already higher than the bandit count, we don't do that,
    // so we actually make progress.
    if (this->mab.num_iterations_completed() > this->mab.num_bandits() || !this->batches[this->batch_cursor].is_compatible(
            current_bandit.get_parameters())) {
        advance_batch_cursor(1);

        // if the cursor now surpasses the queue, fill it up
        if (this->batch_cursor == this->batches.size()) {
            generate_batches(1);
        }
    }

    LOG_DEBUG << "Batch cursor: " << this->batch_cursor << std::endl;
    return this->batches[this->batch_cursor];
}

void TreesetOptimizer::advance_batch_cursor(const unsigned int n) {
    const auto target_index = this->batch_cursor + n;

    // if we have not enough batches to move the cursor to that index, generate the missing ones
    if (this->batches.size() < target_index) {
        LOG_DEBUG << "Warning: cursor moved " << (target_index - this->batches.size()) <<
                " batches past the end of the queue. Why are we generating batches that we will finalize instantly?" <<
                std::endl;
        generate_batches(target_index - this->batches.size());
    }

    for (unsigned int batch = this->batch_cursor; batch < target_index; ++batch) {
        // perform AU test to get plausible tree count. If the AU test was already executed, it will transparently
        // return the result of the previous run and not do any work
        const auto batch_plausible_trees = this->batches[batch].perform_plausibility_check(this->opts);
        this->total_plausible_trees += batch_plausible_trees;

        // finalize batch
        this->batches[batch].finalize();
    }

    LOG_INFO_TS << "Progress: " << this->total_plausible_trees << " / " << this->target_tree_count <<
            " plausible trees." << std::endl;
    this->batch_cursor += n;
    assert(this->batch_cursor <= this->batches.size());
}

void TreesetOptimizer::run() {
    LOG_INFO << std::endl;
    LOG_INFO_TS << "Treeset: Inferring at least " << this->target_tree_count <<
            " plausible trees while optimizing throughput." << std::endl;

    this->prepare_initial_batches();
    LOG_INFO << std::endl;
    LOG_INFO << "Expected throughput of starting trees: " << (
                this->starting_tree_bandit().get_mean_throughput() * 1000.0) <<
            " trees per second at a mean success rate of "
            << (this->starting_tree_bandit().get_expected_tree_rate() * 100.0) << "%." << std::endl;

    this->initialize_bandits();

    while (this->total_plausible_trees < this->target_tree_count) {
        auto &current_bandit = this->select_next_bandit();
        auto &current_batch = this->select_next_batch(current_bandit);

        current_bandit.apply_parameters(opts, current_batch);
        current_batch.optimize(opts);
        current_batch.perform_plausibility_check(opts);
        this->mab.take_measurement(current_bandit, current_batch);

        // check if we would exceed the final tree count if we finalized the current batch immediately
        if (this->total_plausible_trees + current_batch.get_plausible_tree_count() > this->target_tree_count) {
            // finalize last batch
            advance_batch_cursor(1);
            break;
        }
    }

    LOG_INFO_TS << "Inferred " << this->total_plausible_trees << " plausible trees in " << this->batch_cursor <<
            " batches." << std::endl;
}

std::vector<Tree> TreesetOptimizer::get_all_trees() const {
    auto full_set = std::vector<Tree>();

    for (unsigned int batch_id = 0; batch_id < this->batch_cursor; ++batch_id) {
        auto &batch = this->batches[batch_id];
        for (unsigned int tree_id = 0; tree_id < batch.get_batch_size(); ++tree_id) {
            full_set.push_back(batch.get_tree(tree_id));
        }
    }

    return full_set;
}


std::vector<Tree> TreesetOptimizer::get_plausible_trees() const {
    auto plausible_set = std::vector<Tree>();

    for (unsigned int batch_id = 0; batch_id < this->batch_cursor; ++batch_id) {
        auto &batch = this->batches[batch_id];
        auto &p_values = batch.get_p_values();
        for (unsigned int tree_id = 0; tree_id < batch.get_batch_size(); ++tree_id) {
            if (p_values[tree_id + 16] > SIGNIFICANCE_LEVEL) {
                plausible_set.push_back(batch.get_tree(tree_id));
            }
        }
    }

    return plausible_set;
}
