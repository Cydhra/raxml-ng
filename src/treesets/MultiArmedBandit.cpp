#include "MultiArmedBandit.hpp"

Bandit &MultiArmedBandit::select_next_bandit() {
    // shortcut if we forced the selection of only one bandit, to keep the logs clean
    if (this->bandits.size() == 1) {
        return this->bandits[0];
    }

    // select next participating bandit
    // TODO guard against all bandits no longer participating
    do {
        this->bandit_cursor += 1;
        this->bandit_cursor %= this->bandits.size();
    } while (!bandits[this->bandit_cursor].participating);

    auto &selected_bandit = this->bandits[this->bandit_cursor];
    auto &best_bandit = this->bandits[this->best_known_bandit];

    if (this->bandit_cursor != this->best_known_bandit && selected_bandit.is_worse_than(
            best_bandit, iterations_completed)) {
        LOG_INFO << std::endl << "Switching to best bandit " << best_bandit.get_name() <<
                " because its mean expected success ("
                << (best_bandit.get_mean_throughput() * 1000.0) <<
                " t/s) exceeds the largest reasonable success of "
                << selected_bandit.get_name() << " (" << (
                    selected_bandit.get_upper_confidence(iterations_completed) * 1000.0) << " t/s)." <<
                std::endl;

        // check if the selected bandit is so bad that we can just delete it from the round-robin
        // because this requires both trees to have been selected thrice, this likely only ever excludes parsimony
        // TODO the exclusion mechanism should be encapsulated a bit better
        if (selected_bandit.is_hopeless(best_bandit, iterations_completed)) {
            LOG_INFO << "Excluding bandit " << selected_bandit.get_name() <<
                    " from algorithm because it is much worse than the others." << std::endl;
            selected_bandit.participating = false;
        }

        return best_bandit;
    }

    if (!std::isnan(selected_bandit.get_upper_confidence(iterations_completed))) {
        if (this->bandit_cursor != this->best_known_bandit) {
            LOG_INFO << std::endl << "Selecting bandit " << selected_bandit.get_name() <<
                    " because its largest reasonable success ("
                    << (selected_bandit.get_upper_confidence(iterations_completed) * 1000.0) <<
                    " t/s) exceeds the mean expected success of current best bandit "
                    << best_bandit.get_name() << " (" << (best_bandit.get_mean_throughput() * 1000.0) << " t/s)." <<
                    std::endl;
        } else {
            LOG_INFO << std::endl << "Selecting bandit " << selected_bandit.get_name() << " (mean: " << (
                best_bandit.get_mean_throughput() * 1000.0) << " t/s)." << std::endl;
        }
    } else {
        LOG_INFO << std::endl << "Initial estimation of " << selected_bandit.get_name() << "." << std::endl;
    }

    return selected_bandit;
}

void MultiArmedBandit::take_measurement(Bandit &current_bandit, TunedBatch &batch) {
    current_bandit.take_measurement(batch);

    this->iterations_completed += 1;

    // if the current bandit is not the best one, check if the best one has to be updated
    if (current_bandit.get_parameters() != this->bandits[best_known_bandit].get_parameters()) {
        if (current_bandit.get_mean_throughput() > this->bandits[best_known_bandit].get_mean_throughput()) {
            best_known_bandit = bandit_cursor;
        }
    }

    // once all bandits have been selected once, assign variances to the bandits
    if (iterations_completed == this->bandits.size()) {
        // collect variances
        double mean = 0.0;
        for (const auto &bandit: bandits) {
            mean += bandit.get_mean_throughput();
        }
        mean /= static_cast<double>(bandits.size());

        double variance = 0.0;
        for (const auto &bandit: bandits) {
            variance += (mean - bandit.get_mean_throughput()) * (mean - bandit.get_mean_throughput());
        }
        variance /= static_cast<double>(bandits.size());

        const auto standard_deviation = sqrt(variance);

        LOG_INFO << std::endl;
        LOG_INFO << "Mean throughput is " << (mean * 1000.0) <<
                " trees per second with the standard deviation over all bandits being " << (
                    standard_deviation * 1000.0) <<
                std::endl << std::endl;

        for (auto &bandit: this->bandits) {
            bandit.initialize_variance(variance, INITIAL_VARIANCE_WEIGHT);
        }
    }
}
