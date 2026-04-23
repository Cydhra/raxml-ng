#ifndef RAXML_BANDIT_HPP_
#define RAXML_BANDIT_HPP_

#include <memory>
#include <vector>
#include <string>

#include "TunedBatch.hpp"

/**
 * A measurement sample obtained from inferring a TunedBatch with a given set of parameters. The bandits are keeping
 * track of the measurements to estimate the mean and standard deviation of the underlying probability distribution of
 * the reward.
 */
class Measurement {
public:
    Measurement(const unsigned int time_spent, const unsigned int plausible_trees,
                const unsigned int batch_size) : time_spent(time_spent),
                                                 plausible_trees(plausible_trees), batch_size(batch_size) {
    }

    const unsigned int time_spent;

    const unsigned int plausible_trees;

    const unsigned int batch_size;
};

/**
 * A one-armed bandit which represents a certain set of MetaParameters that generate reward (i.e., throughput of plausible
 * trees) with an unknown probability distribution.
 */
template<class Heuristic>
class Bandit {
public:
    explicit Bandit(const std::string &name, Heuristic parameters) : name(name),
                                                                     parameters(std::make_shared<Heuristic>(
                                                                         parameters)) {
    }

    // grant MAB access to protected members, specifically "take_measurement"
    template<class H>
    friend class MultiArmedBandit;

    /**
     * Whether this bandit is participating in the multiarmed bandit algorithm.
     */
    bool participating{true};

    /**
     * Initialize the bandit distribution estimation with a constant variance. This allows comparing bandits with some
     * level of confidence regarding the unknown variance. The set variance will account for n samples in the calculation
     * of the bandit's actual variance, where n is the given weight argument.
     *
     * The set variance will be replaced with the variance of actual measurements successively
     * by replacing the weight of it with actual samples.
     * That is, when the Bandit has taken n measurements, the set variance will have a weight of 0 in the calculation.
     *
     * @param variance the initial estimate of the underlying distribution's variance
     * @param weight the number of samples this variance accounts for. Each sample measurement the bandit takes reduces
     * this weight by one in the calculation of the bandit's variance.
     */
    void initialize_variance(const double variance, const unsigned int weight) {
        this->estimated_variance = variance;
        this->estimated_variance_weight = weight;
    }

    /**
     * Compare two bandit distributions and determine if this one is worse than the `other` bandit.
     * A bandit is worse if the upper bound of its expected success is below the mean of the expected success
     * of the other bandit.
     *
     * @param other Bandit to compare this one to
     * @param total_samples total number of batches inferred so far
     *
     * @return true if this bandit has worse success rate with high probability.
     */
    bool is_worse_than(const Bandit &other, const unsigned int total_samples) const {
        // each bandit needs to be sampled at least once
        if (this->samples.empty() || other.samples.empty()) {
            return false;
        }

        return other.get_mean_throughput() > this->get_upper_confidence(total_samples);
    }

    /**
    * Compare two bandit distributions and determine if this one is hopeless when compared to the other.
    * This way we can exclude it from the algorithm, speeding up the decision on other bandits.
    *
    * @param other Bandit to compare this one to
    * @param total_samples total number of batches inferred so far
    *
    * @return true if this bandit has no conceivable chance of becoming relevant in the algorithm again.
    */
    bool is_hopeless(const Bandit &other, const unsigned int total_samples) const {
        // if either bandit is not sampled enough to allow a good estimate of the mean, return false
        if (this->samples.size() < 3 || other.samples.size() < 3) {
            return false;
        }

        return other.get_mean_throughput() > this->get_upmost_confidence(total_samples);
    }

    /**
     * @return the mean expected reward (throughput) of the underlying distribution.
     */
    double get_mean_throughput() const {
        double expectation = 0.0;

        for (auto &sample: this->samples) {
            // the cost function is TIME / #TREES (time spent per plausible tree), but the throughput (which is our reward
            // function) is the inverse of that: 1 / (TIME / #TREES)) = #TREES / TIME, with a reward of 0 if no plausible
            // trees are found
            expectation += static_cast<double>(sample.plausible_trees) / static_cast<double>(sample.time_spent);
        }

        return expectation / this->samples.size();
    }

    /**
     * @return the mean success rate (between 0 and 1) of yielding a plausible tree under this bandit's parameters.
     */
    double get_expected_tree_rate() const {
        double success = 0.0;
        for (auto &sample: this->samples) {
            success += static_cast<double>(sample.plausible_trees) / static_cast<double>(sample.batch_size);
        }

        return success / this->samples.size();
    }

    /**
     * @return the variance of the reward distribution. Because this is required to be known a priori, the value is
     * being interpolated between an initial value and the measured value depending on how many measurements are
     * available.
     */
    double get_variance() const {
        const double mean = get_mean_throughput();
        double variance_sum = this->estimated_variance;

        // calculate participation of the estimator, gradually replacing it with actual measurements
        const auto weight = this->estimated_variance_weight - min(this->estimated_variance_weight,
                                                                  static_cast<unsigned int>(this->samples.size()));

        variance_sum *= weight;

        for (auto &sample: this->samples) {
            const double sample_throughput = static_cast<double>(sample.plausible_trees) / static_cast<double>(sample.
                                                 time_spent);
            variance_sum += (mean - sample_throughput) * (mean - sample_throughput);
        }

        // bessel correction because the population variance is much more important than the sample variance
        // this likely overestimates the variance because of low sample sizes, but relying less on the estimated variance
        // and thus do a little bit more exploration rarely hurts.
        return variance_sum / (this->samples.size() + weight - 1);
    }

    /**
     * Calculate the upper confident limit of the mean expected success of this bandit. This depends on the total
     * number of samples drawn so far, as well as the number of samples drawn for this bandit.
     *
     * @param total_samples the number of total samples from all bandits that have been drawn so far
     *
     * @return the upper bound on the mean expected success that can be determined with high confidence.
     */
    double get_upper_confidence(const unsigned int total_samples) const {
        const auto mean_throughput = this->get_mean_throughput();
        const auto variance = this->get_variance();

        // as defined by 10.1016/0196-8858(85)90002-8, formula 4.13 with the choice of `a_(n,i)` = `(log n) / i`,
        // where `n` is the total number of samples, and `i` is the number of samples drawn for this bandit.
        // do note that the formula contains the standard deviation, not the variance, so we move the variance into the root.
        return mean_throughput + sqrt(
                   variance * 2.0 * log(static_cast<double>(total_samples)) / static_cast<double>(this->samples.
                       size()));
    }

    /**
     * The upper confidence limit of the mean expected success is one standard-deviation above the sample mean,
     * corrected by the number of samples available (more samples means we increase the confidence interval because we
     * do not have enough data).
     * This method, however, returns the upper limit at two standard deviations away from the sample mean.
     * This can be used to reasonably exclude bandits that are so bad that they will not become relevant even with a
     * high number of total samples.
     *
     * This still takes into account the sample ratio (total samples versus samples for this bandit) to estimate how
     * uncertain the confidence is.
     *
     * @param total_samples the number of total samples from all bandits that have been drawn so far
     *
     * @return the expected success two standard deviations higher than the mean.
     */
    double get_upmost_confidence(const unsigned int total_samples) const {
        const auto mean_throughput = this->get_mean_throughput();
        const auto variance = this->get_variance();

        // see get_upper_confidence
        return mean_throughput + sqrt(
                   4.0 * variance * 2.0 * log(static_cast<double>(total_samples)) / static_cast<double>(this->samples.
                       size()));
    }

    /**
     * @return This bandit's meta parameters
     */
    std::shared_ptr<Heuristic> get_parameters() const {
        return this->parameters;
    }

    /**
     * @return Bandit name for debug output
     */
    std::string get_name() const {
        return this->name;
    }

    unsigned int num_samples() const {
        return this->samples.size();
    }

protected:
    /**
     * Display name of the bandit for debugging
     */
    std::string name;

    /**
     * Heuristics parameters of this bandit
     */
    std::shared_ptr<Heuristic> parameters;

    /**
     * Samples drawn from the reward distribution.
     */
    std::vector<Measurement> samples = {};

    /**
     * The set variance will account for n samples in the calculation of the bandit's actual variance (`get_variance()`).
     * It gets replaced by actual `samples`, where each existing sample reduces the `estimated_variance_weight` by one.
     */
    double estimated_variance{0.0};

    /**
     * Initial weight of the `estimated_variance` in the calculation of the actual variance. Reduced by one for each
     * existing measurement in `samples`.
     */
    unsigned int estimated_variance_weight{0};

    /**
     * Take the benchmark data of a tuned batch which has previously run its inference with the parameters of this
     * bandit.
     * The measurement is stored and updates the expected mean reward of this bandit.
     *
     * @param batch A tuned batch which has been run on the parameter set of this Bandit instance.
     */
    void take_measurement(const TunedBatch &batch) {
        // if we have no other samples yet, we initialize the variance estimate by overestimating it intentionally
        if (samples.size() == 0) {
            const auto mean = static_cast<double>(batch.get_plausible_tree_count()) / static_cast<double>(batch.
                                  elapsed_wall_time());
            constexpr auto worst_case = 0.0;
            const auto best_case = static_cast<double>(batch.get_batch_size()) / static_cast<double>(batch.
                                       elapsed_wall_time());

            auto over_estimated_variance = 0.0;
            // sample mean is equal to sample, so the measured sample has contribution 0
            over_estimated_variance += (mean - worst_case) * (mean - worst_case);
            over_estimated_variance += (mean - best_case) * (mean - best_case);

            // initialize variance with an estimate that overestimates it by the maximum amount obtained from one sample
            // and two extreme value.
            this->initialize_variance(over_estimated_variance / 3, 3);
        }

        LOG_WORKER_TS(LogLevel::info) << "[" << this->name << "]: Takes measurement: " << batch.get_plausible_tree_count() << " plausible trees at " << (
                    static_cast<double>(batch.get_plausible_tree_count()) / static_cast<double>(batch.
                        elapsed_wall_time()) *
                    1000.0)
                << " trees per second." << std::endl;
        this->samples.emplace_back(batch.elapsed_wall_time(), batch.get_plausible_tree_count(), batch.get_batch_size());
    }
};


#endif //RAXML_BANDIT_HPP_
