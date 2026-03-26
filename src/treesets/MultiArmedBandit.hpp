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
 */
class MultiArmedBandit {
public:
    /**
     * Add a single-arm bandit to this MAB instance.
     * It will be returned by the round-robin style selection as long it isn't assumed to be worse.
     *
     * @param bandit a bandit instance to add to this MAB
     */
    void register_bandit(Bandit &&bandit) {
        this->bandits.emplace_back(bandit);
    }

    /**
     * Create a new single-arm bandit at the end of this MAB instance's bandit list.
     * It will be returned by the round-robin style selection as long it isn't assumed to be worse.
     *
     * @param name bandit's name
     * @param parameters the bandit's meta-parameters
     */
    void emplace_back(std::string name, MetaParameters parameters) {
        this->bandits.emplace_back(name, parameters);
    }

    /**
     * Get access to a previously registered bandit.
     *
     * @param index The index of the bandit in the MAB. Bandits are added in the order of their registration
     * @return A reference to the index-th bandit in the MAB
     */
    Bandit &get_bandit(const unsigned int index) {
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
    Bandit &select_next_bandit();

    /**
     * Add a measurement to the given bandit and check if that bandit is now better than the current best bandit.
     *
     * @param current_bandit bandit which the measurement is for
     * @param batch the batch that created the measurement
     */
    void take_measurement(Bandit &current_bandit, TunedBatch &batch);

protected:
    std::vector<Bandit> bandits = std::vector<Bandit>();

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
};


#endif //RAXML_MULTIARMEDBANDIT_HPP_
