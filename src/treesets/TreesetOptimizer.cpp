#include "TreesetOptimizer.hpp"

#include "TunedBatch.hpp"

/**
 * How many samples the initial variance measurement replaces in the bandits.
 */
constexpr unsigned int INITIAL_VARIANCE_WEIGHT = 6;

/**
 * Multiply the initial variance measured between bandits with this to ensure we over-estimate the initial variance
 * of all distributions.
 */
constexpr double VARIANCE_OVERESTIMATION = 2.0;

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
        this->batch_cursor = this->batches.size();
    } else {
        // init default bandits
        this->bandits.emplace_back("Greedy,DoModel,2spr", MetaParameters(1, false, 2, false));
        this->bandits.emplace_back("Greedy,DoModel,4spr", MetaParameters(1, false, 4, false));
        this->bandits.emplace_back("Greedy,NoModel,2spr", MetaParameters(1, true, 2, false));

        this->bandits.emplace_back("Fast,DoModel,2spr", MetaParameters(20, false, 2, false));
        this->bandits.emplace_back("Fast,DoModel,4spr", MetaParameters(20, false, 4, false));
        this->bandits.emplace_back("Fast,NoModel,2spr", MetaParameters(20, true, 2, false));
    }
}

Bandit &TreesetOptimizer::select_next_bandit() {
    this->bandit_cursor += 1;
    this->bandit_cursor %= this->bandits.size();

    auto &selected_bandit = this->bandits[this->bandit_cursor];
    auto &best_bandit = this->bandits[this->best_known_bandit];

    if (this->bandit_cursor != this->best_known_bandit && selected_bandit.is_worse_than(
            best_bandit, this->total_batches_completed)) {
        LOG_INFO << "Selecting bandit " << best_bandit.get_name() << " because its mean expected success ("
                << (best_bandit.get_mean_throughput() * 1000.0) <<
                " t/s) is far better than the mean expected success of "
                << selected_bandit.get_name() << " (" << (selected_bandit.get_mean_throughput() * 1000.0) << " t/s)." << std::endl;

        return best_bandit;
    }

    LOG_INFO << "Selecting bandit " << selected_bandit.get_name() << " because its largest reasonable success ("
                << (selected_bandit.get_upper_confidence(this->total_plausible_trees) * 1000.0) <<
                " t/s) exceeds the mean expected success of "
                << best_bandit.get_name() << " (" << (best_bandit.get_mean_throughput() * 1000.0) << " t/s)." << std::endl;

    return selected_bandit;
}

TunedBatch &TreesetOptimizer::select_next_batch(const Bandit &current_bandit) {
    auto current = this->batch_cursor;

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

    // run through the bandits once (i.e. until the bandit cursor is 0 again) to collect initial measurements
    do {
        auto &current_bandit = this->select_next_bandit();
        auto &current_batch = this->select_next_batch(current_bandit);

        current_bandit.apply_parameters(opts, current_batch);
        current_batch.optimize(opts);
        current_batch.perform_plausibility_check(opts);
        current_bandit.take_measurement(current_batch);
        this->total_batches_completed += 1;

        // if the current bandit is not the best one, check if the best one has to be updated
        if (current_bandit.get_parameters() != this->bandits[best_known_bandit].get_parameters()) {
            if (current_bandit.get_mean_throughput() > this->bandits[best_known_bandit].get_mean_throughput()) {
                best_known_bandit = bandit_cursor;
            }
        }

        // once all bandits have been selected once, assign variances to the bandits
        if (total_batches_completed == this->bandits.size() - 1) {
            // collect variances
            double mean = 0.0;
            for (unsigned int i = 1; i < bandits.size(); i++) {
                mean += bandits[i].get_mean_throughput();
            }
            mean /= static_cast<double>(bandits.size() - 1);

            double variance = 0.0;
            for (unsigned int i = 1; i < bandits.size(); i++) {
                variance += (mean - bandits[i].get_mean_throughput()) * (mean - bandits[i].get_mean_throughput());
            }
            variance /= static_cast<double>(bandits.size() - 1);

            const auto standard_deviation = sqrt(variance);

            LOG_INFO << std::endl;
            LOG_INFO << "Mean throughput is " << (mean * 1000.0) <<
                    " trees per second with the standard deviation over all bandits being " << (
                        standard_deviation * 1000.0) <<
                    std::endl << std::endl;

            for (auto &bandit: this->bandits) {
                bandit.initialize_variance(variance * VARIANCE_OVERESTIMATION, INITIAL_VARIANCE_WEIGHT);
            }
        }

        if (this->total_plausible_trees + current_batch.get_plausible_tree_count() > this->target_tree_count) {
            this->total_plausible_trees += current_batch.get_plausible_tree_count();
            this->batch_cursor += 1;
            break;
        } else {
            LOG_INFO_TS << "Progress: " << this->total_plausible_trees << " / " << this->target_tree_count << " plausible trees." << std::endl;
        }
    } while (true);

    LOG_INFO_TS << "Inferred " << this->total_plausible_trees << " plausible trees in " << this->batch_cursor <<
            " batches." << std::endl;
}
