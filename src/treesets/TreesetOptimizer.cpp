#include "TreesetOptimizer.hpp"

#include "TunedBatch.hpp"

/**
 * How many samples the initial variance measurement replaces in the bandits.
 */
constexpr unsigned int INITIAL_VARIANCE_WEIGHT = 6;

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

        // take a few measurements, but no more than necessary to have reasonable values for mean and variance
        if (batches.size() <= INITIAL_VARIANCE_WEIGHT) {
            current_batch.perform_plausibility_check(opts);
            starting_tree_bandit().take_measurement(current_batch);
        }
    }
}

void TreesetOptimizer::initialize_bandits() {
    if (this->starting_tree_bandit().get_expected_tree_rate() >= 0.9) {
        LOG_INFO << "Starting trees are so successful, no ML optimization is necessary." << std::endl;
        this->batch_cursor = this->batches.size() - 1; // select_next_batch will advance by one
        this->total_batches_completed = this->batches.size();
    } else {
        // init default bandits
        this->bandits.emplace_back("Greedy,DoModel,2spr", MetaParameters(1, false, 2, 0, false));
        this->bandits.emplace_back("Greedy,DoModel,4spr", MetaParameters(1, false, 4, 0, false));
        this->bandits.emplace_back("Greedy,NoModel,2spr", MetaParameters(1, true, 2, 0, false));

        this->bandits.emplace_back("Fast,DoModel,2spr", MetaParameters(20, false, 2, 0, false));
        this->bandits.emplace_back("Fast,DoModel,4spr", MetaParameters(20, false, 4, 0, false));
        this->bandits.emplace_back("Fast,NoModel,2spr", MetaParameters(20, true, 2, 0, false));
    }
}

Bandit &TreesetOptimizer::select_next_bandit() {
    this->bandit_cursor += 1;
    this->bandit_cursor %= this->bandits.size();

    auto &selected_bandit = this->bandits[this->bandit_cursor];
    auto &best_bandit = this->bandits[this->best_known_bandit];

    if (this->bandit_cursor != this->best_known_bandit && selected_bandit.is_worse_than(
            best_bandit, this->total_batches_completed)) {
        LOG_INFO << std::endl << "Switching to best bandit " << best_bandit.get_name() << " because its mean expected success ("
                << (best_bandit.get_mean_throughput() * 1000.0) <<
                " t/s) exceeds the largest reasonable success of "
                << selected_bandit.get_name() << " (" << (selected_bandit.get_upper_confidence(this->total_batches_completed) * 1000.0) << " t/s)." << std::endl;

        return best_bandit;
    }

    if (!std::isnan(selected_bandit.get_upper_confidence(this->total_batches_completed))) {
        LOG_INFO << std::endl << "Selecting bandit " << selected_bandit.get_name() << " because its largest reasonable success ("
            << (selected_bandit.get_upper_confidence(this->total_batches_completed) * 1000.0) <<
            " t/s) exceeds the mean expected success of current best bandit "
            << best_bandit.get_name() << " (" << (best_bandit.get_mean_throughput() * 1000.0) << " t/s)." << std::endl;
    } else {
        LOG_INFO << std::endl << "Initial estimation of " << selected_bandit.get_name() << "." << std::endl;
    }

    return selected_bandit;
}

TunedBatch &TreesetOptimizer::select_next_batch(const Bandit &current_bandit) {
    auto current = this->batch_cursor;
    LOG_DEBUG << "Batch cursor: " << this->batch_cursor << std::endl;

    // to speed up the initial round of computation where all bandits are executed once,
    // we want to reuse batches. If the total rounds is already higher than the bandit count, we don't do that,
    // so we actually make progress.
    if (this->total_batches_completed > this->bandits.size() || !this->batches[current].is_compatible(current_bandit.get_parameters())) {
        // finalize the plausible trees of the current batch as we won't touch it again
        this->total_plausible_trees += this->batches[current].get_plausible_tree_count();

        current = ++this->batch_cursor;
        if (this->batches.size() == current) {
            generate_batches(1);
        }
    }

    return this->batches[current];
}

void TreesetOptimizer::run() {
    const auto NUM = 500;
    LOG_INFO << std::endl;
    LOG_INFO_TS << "Treeset: Inferring " << NUM << " starting trees and benchmark their potential." << std::endl;

    // generate 100 parsimony trees
    auto batch = TunedBatch("TestTrees",
                               1000,
                               NUM,
                               recommended_thread_count(),
                               recommended_worker_count(),
                               msa,
                               persite_loglh);

    const auto pseudo_bandit = Bandit("pseudo", MetaParameters(opts.ktop, false, opts.thorough ? 0 : opts.num_spr, opts.thorough ? opts.num_spr : 0, false));
    pseudo_bandit.apply_parameters(opts, batch);

    batch.generate_starting_trees(this->instance, opts, load_balancer, tip_msa_idmap);
    batch.optimize_parameters(opts, 3.0);

    batch.perform_plausibility_check(opts);

    const auto initial_loglh = batch.get_tree_likelihoods();
    const auto initial_p_values = batch.get_p_values();

    batch.optimize_topology(opts);
    batch.perform_plausibility_check(opts);

    auto optimized_loglh = batch.get_tree_likelihoods();
    auto optimized_p_values = batch.get_p_values();

    for (unsigned int i = 0; i < initial_loglh.size(); i++) {
        LOG_INFO << i << "\t" << initial_loglh[i] << "\t"  << initial_p_values[i] << "\t" << optimized_loglh[i] << "\t"  << optimized_p_values[i] << std::endl;
    }

    LOG_INFO_TS << "Analyzed " << NUM << " starting trees." << std::endl;
}
