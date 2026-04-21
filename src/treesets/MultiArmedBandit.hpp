#ifndef RAXML_MULTIARMEDBANDIT_HPP_
#define RAXML_MULTIARMEDBANDIT_HPP_
#include "Bandit.hpp"

/**
 * How many samples the initial variance measurement replaces in the bandits.
 */
constexpr unsigned int INITIAL_VARIANCE_WEIGHT = 6;

/**
 * A multi-armed bandit is an algorithm that selects between different heuristics and balances exploration (testing new
 * heuristics and measuring their success) and exploitation (getting as much success as possible).
 *
 * After each call to `select_next_bandit`, the method `take_measurement` should be called to report the reward the
 * bandit generated. Measurements can also be taken without advancing the MAB.
 */
template<class Heuristic>
class MultiArmedBandit {
public:
    /**
     * Add a single-arm bandit to this MAB instance.
     * It will be returned by the round-robin style selection as long it isn't assumed to be worse.
     *
     * @param bandit a bandit instance to add to this MAB
     */
    void register_bandit(Bandit<Heuristic> &&bandit) {
        this->bandits.emplace_back(bandit);
    }

    /**
     * Create a new single-arm bandit at the end of this MAB instance's bandit list.
     * It will be returned by the round-robin style selection as long it isn't assumed to be worse.
     *
     * @param name bandit's name
     * @param parameters the bandit's meta-parameters
     */
    void emplace_back(std::string name, Heuristic parameters) {
        this->bandits.emplace_back(name, parameters);
    }

    /**
     * Get access to a previously registered bandit.
     *
     * @param index The index of the bandit in the MAB. Bandits are added in the order of their registration
     * @return A reference to the index-th bandit in the MAB
     */
    Bandit<Heuristic> &get_bandit(const unsigned int index) {
        return this->bandits.at(index);
    }

    /**
     * @return number of registered bandits (both, participating and non-participating)
     */
    unsigned int num_bandits() const {
        return this->bandits.size();
    }

    /*+
     * @return number of iterations that were completed, i.e., a bandit was selected and a measurement returned.
     */
    unsigned int num_iterations_completed() const {
        return this->iterations_completed;
    }

    /**
     * Select the next bandit according to the knowledge learned so far.
     * Bandits that are expected to be worse than the best known bandit are replaced with the best known bandit.
     *
     * @return A reference to the bandit that should be used according to the selection rule.
     */
    Bandit<Heuristic> &select_next_bandit() {
        // shortcut if we forced the selection of only one bandit, to keep the logs clean
        if (this->bandits.size() == 1) {
            return this->bandits[0];
        }

        selection_mutex.lock();
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

            selection_mutex.unlock();
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

        selection_mutex.unlock();
        return selected_bandit;
    }

    /**
     * Add a measurement to the given bandit and check if that bandit is now better than the current best bandit.
     *
     * @param current_bandit bandit which the measurement is for
     * @param batch the batch that created the measurement
     * @param iteration_completed whether the measurement was taken from a completed iteration. If false, the batch
     * will be modified with the measurement, but the MAB rule will not consider this as generating a reward for the
     * purposes of the selection rule. Calling take_measurement with this flag unset can be used to generated
     * measurements from prior knowledge or heuristics that are not part of the selection rule.
     */
    void take_measurement(Bandit<Heuristic> &current_bandit, const TunedBatch &batch, const bool iteration_completed) {
        measurement_mutex.lock();
        current_bandit.take_measurement(batch);

        // if the current bandit is not the best one, check if the best one has to be updated
        if (current_bandit.get_parameters() != this->bandits[best_known_bandit].get_parameters()) {
            if (current_bandit.get_mean_throughput() > this->bandits[best_known_bandit].get_mean_throughput()) {
                best_known_bandit = bandit_cursor;
            }
        }

        // if this completes an iteration, update the counter for calculating confidence thresholds
        if (iteration_completed) {
            this->iterations_completed += 1;
        }

        measurement_mutex.unlock();
    }

    bool has_bandit(std::string name) const {
        for (auto &bandit : this->bandits) {
            if (bandit.name == name) {
                return true;
            }
        }

        return false;
    }

protected:
    /**
     * Registered bandits in this MAB.
     * Bandits must not be removed from this list.
     * This list is implemented with a deque to avoid reallocations while batches hold references to the bandits
     * in the list.
     */
    std::deque<Bandit<Heuristic> > bandits = std::deque<Bandit<Heuristic> >();

    /**
     * Points to the bandit that was selected last.
     */
    unsigned int bandit_cursor = 0;

    /**
     * Points to the bandit that is assumed to be the best one (it has the highest mean, independent of the confidence
     * interval size)
     */
    unsigned int best_known_bandit = 0;

    /**
     * How many measurements were taken by this MAB.
     */
    unsigned int iterations_completed = 0;

private:
    std::mutex measurement_mutex;

    std::mutex selection_mutex;
};


#endif //RAXML_MULTIARMEDBANDIT_HPP_
