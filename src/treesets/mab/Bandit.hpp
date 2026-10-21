#ifndef RAXML_BANDIT_HPP_
#define RAXML_BANDIT_HPP_

#include <memory>
#include <utility>
#include <vector>
#include <string>

#include "../batch/TunedBatch.hpp"

/**
 * Consistency factor to convert Mean Absolute Deviation into an estimator for the standard deviation.
 */
constexpr double CONSISTENCY = 1.48;

/**
 * A measurement sample obtained from inferring a TunedBatch with a given set of parameters. The bandits are keeping
 * track of the measurements to estimate the mean and standard deviation of the underlying probability distribution of
 * the reward.
 */
class Measurement {
public:
    Measurement(const unsigned int time_spent, const unsigned int plausible_trees,
                const unsigned int batch_size,
                const std::vector<double> &individual_inverse_times) : time_spent(time_spent),
                                                                       plausible_trees(plausible_trees),
                                                                       batch_size(batch_size),
                                                                       individual_inverse_times(
                                                                           individual_inverse_times) {
    }

    const unsigned int time_spent;

    const unsigned int plausible_trees;

    const unsigned int batch_size;

    std::vector<double> individual_inverse_times;
};

/**
 * A one-armed bandit which represents a certain set of MetaParameters that generate reward (i.e., throughput of plausible
 * trees) with an unknown probability distribution.
 * The class is generic over the algorithm that it is modelling.
 * In the simplest case, a Bandit holds an instance of MetaParameters which
 *
 * There are several models for the underlying distribution.
 *
 * The simplest one measures throughput of each inferred batch modelling it with a normal distribution.
 * This makes estimating the sample variance difficult until we have an adequate number of measurements, so we
 * initially over-estimate variance by simply adding two samples with success rate of 0 and success rate of 1.
 * The weight of those two samples is reduced as more samples are added.
 *
 * The second model only works for lowest level of bandits (i.e., Bandits which explicitly hold MetaParameters).
 * We measure the inference time for each individual tree and thus obtain BATCH_SIZE measurements per batch for inference
 * time.
 * We store 1/time for each measurement and model those measurements with a normal distribution.
 * This allows faster estimation of the variance but is extremely sensitive to changes in the measuring setup:
 * If the system load changes between batches, or slightly different starting conditions are used (e.g., for higher
 * level bandits in the multi-level MAB which model multiple similar but different MetaParameter instances),
 * the initial variance is grossly underestimated.
 * To offset this, we introduce a multiplicative epsilon which overestimates the initial variance.
 * Success rate is modelled as a binomial distribution, so even a single sample is sufficient to get a variance estimate.
 * The final distribution of throughput is obtained through a ratio distribution of success rate samples divided by
 * inference time samples.
 * By multiplying the means of success rate and inverse wall-time, we obtain a mean of the final ratio distribution
 * that models bandit throughput.
 * The variance is obtained analogously.
 */
template<class Heuristic>
class Bandit {
public:
    explicit Bandit(std::string name, Heuristic parameters, const double maximum_success) : name(std::move(name)),
        parameters(std::make_shared<Heuristic>(
            parameters)), maximum_success(maximum_success) {
    }

    // grant MAB access to protected members, specifically "take_measurement" and "disable"
    template<class H>
    friend class MultiArmedBandit;

    /**
     * Configures the bandit to use a normal distribution as the model for the unknown probability distribution its
     * heuristic is distributed with.
     */
    Bandit *with_normal_distribution() {
        this->mean = &Bandit::get_normal_mean;
        this->deviation = &Bandit::get_normal_smart_deviation;
        return this;
    }

    /**
     * Configures the bandit to use a ratio distribution of bandit success and bandit walltime distributions for
     * the unknown probabilitiy distribution its heuristic is distributed with.
     */
    Bandit *with_ratio_distribution() {
        this->mean = &Bandit::get_ratio_mean;
        this->deviation = &Bandit::get_ratio_deviation;
        return this;
    }

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
        this->initial_variance = variance;
        this->initial_variance_weight = weight;
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
    [[nodiscard]] bool is_worse_than(const Bandit &other, const unsigned int total_samples) const {
        // each bandit needs to be sampled at least once
        if (this->samples.empty() || other.samples.empty()) {
            return false;
        }

        return other.get_upper_confidence(total_samples) > this->get_upper_confidence(total_samples);
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
    [[nodiscard]] bool is_hopeless(const Bandit &other, const unsigned int total_samples) const {
        // if either bandit is not sampled enough to allow a good estimate of the mean, return false
        if (this->samples.size() < 3 || other.samples.size() < 3) {
            return false;
        }

        return other.get_throughput_mean() > this->get_upmost_confidence(total_samples);
    }

    /**
     * Get a estimator for the mean using this bandit's selected model for the underlying distribution.
     */
    [[nodiscard]] double get_throughput_mean() const {
        return (this->*mean)();
    }

    /**
     * Get a standard deviation estimator using this bandit's selected model for the underlying distribution.
     */
    [[nodiscard]] double get_throughput_deviation() const {
        return (this->*deviation)();
    }

    /**
     * @return the mean success rate (between 0 and 1) of yielding a plausible tree under this bandit's parameters.
     */
    [[nodiscard]] double get_success_mean() const {
        double success = 0.0;
        for (auto &sample: this->samples) {
            success += static_cast<double>(sample.plausible_trees) / static_cast<double>(sample.batch_size);
        }

        return success / this->samples.size();
    }

    /**
     * @return the variance of the bandit's success rate modelled with a bernoulli distribution.
     */
    [[nodiscard]] double get_success_variance() const {
        return get_success_mean() * (1.0 - get_success_mean());
    }

    /**
     * Calculate the upper confident limit of the mean expected success of this bandit. This depends on the total
     * number of samples drawn so far, as well as the number of samples drawn for this bandit.
     *
     * @param total_samples the number of total samples from all bandits that have been drawn so far
     *
     * @return the upper bound on the mean expected success that can be determined with high confidence.
     */
    [[nodiscard]] double get_upper_confidence(const unsigned int total_samples) const {
        const auto mean_throughput = this->get_throughput_mean();
        // const auto std_variation = this->get_throughput_deviation();

        // as defined by 10.1016/0196-8858(85)90002-8, formula 4.13 with the choice of `a_(n,i)` = `(log n) / i`,
        // where `n` is the total number of samples, and `i` is the number of samples drawn for this bandit.
        // do note that the formula contains the standard deviation, not the variance, so we move the variance into the root.
        return mean_throughput / maximum_success * sqrt(
                   2.0 * log(static_cast<double>(total_samples)) / static_cast<double>(this->samples.size()));
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
    [[nodiscard]] double get_upmost_confidence(const unsigned int total_samples) const {
        const auto mean_throughput = this->get_throughput_mean();
        const auto std_variation = this->get_throughput_deviation();

        // see get_upper_confidence
        return mean_throughput + 2 * std_variation * sqrt(
                   2.0 * log(static_cast<double>(total_samples)) / static_cast<double>(this->samples.size()));
    }

    /**
     * @return This bandit's meta parameters
     */
    [[nodiscard]] Heuristic &get_parameters() const {
        return *this->parameters;
    }

    /**
     * @return Bandit name for debug output
     */
    [[nodiscard]] std::string get_name() const {
        return this->name;
    }

    /**
     * @return Number of samples taken (i.e. number of batches inferred with this bandit).
     */
    [[nodiscard]] unsigned int num_samples() const {
        return this->samples.size();
    }

private:
    /**
     * Get the mean expected reward (throughput) modelled with a single normal distribution.
     */
    [[nodiscard]] double get_normal_mean() const {
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
     * @return the median expected reward (throughput) modelled with a single normal distribution.
     */
    [[nodiscard]] double get_normal_median() const {
        auto throughput_samples = std::vector<double>();


        for (auto &sample: this->samples) {
            throughput_samples.push_back(
                static_cast<double>(sample.plausible_trees) / static_cast<double>(sample.time_spent));
        }

        std::nth_element(throughput_samples.begin(),
                         throughput_samples.begin() + static_cast<long>(throughput_samples.size() / 2),
                         throughput_samples.end());
        return throughput_samples[throughput_samples.size() / 2];
    }

    /**
     * Calculate the median absolute deviation as an outlier-resistant replacement of the sample variance.
     * This value has to be multiplied with a correction factor to obtain a consistent estimator of the standard deviation
     * and cannot be used on its own in the bandit calculations.
     */
    [[nodiscard]] double get_median_abs_dev() const {
        const double median = this->get_normal_median();
        auto abs_devs = std::vector<double>();

        for (auto &sample: this->samples) {
            abs_devs.push_back(
                abs(median - static_cast<double>(sample.plausible_trees) / static_cast<double>(sample.time_spent)));
        }

        std::nth_element(abs_devs.begin(), abs_devs.begin() + static_cast<long>(abs_devs.size() / 2), abs_devs.end());
        return abs_devs[abs_devs.size() / 2];
    }

    /**
     * @return the variance of the reward distribution when modelled with a single normal distribution.
     * Because this is required to be known a priori, the value is being interpolated between an initial value and the
     * sample variance depending on how many measurements are available.
     */
    [[nodiscard]] double get_normal_variance() const {
        const double mean = get_normal_mean();
        double variance_sum = this->initial_variance;

        // calculate participation of the estimator, gradually replacing it with actual measurements
        const auto weight = this->initial_variance_weight - min(this->initial_variance_weight,
                                                                static_cast<unsigned int>(this->samples.size()));

        variance_sum *= weight;

        for (auto &sample: this->samples) {
            const double sample_throughput = static_cast<double>(sample.plausible_trees) / static_cast<double>(sample.
                                                 time_spent);
            variance_sum += (mean - sample_throughput) * (mean - sample_throughput);
        }

        return variance_sum / (this->samples.size() + weight);
    }

    /**
     * Get a best-effort estimator for the standard deviation of the normally-distributed model.
     * When we have under 5 measurements, we use the sample variance (including the initial variance overestimation),
     * otherwise we switch to the MAD to avoid outliers influencing the deviation.
     */
    [[nodiscard]] double get_normal_smart_deviation() const {
        return this->samples.size() < 5 ? sqrt(this->get_normal_variance()) : this->get_median_abs_dev() * CONSISTENCY;
    }

    /**
     * Get the mean individual inverse inference time.
     * That is, the mean of all multiplicative-inverses of time measurements of individual tree inferences, i.e.
     * BATCH_SIZE measurements per batch.
     * Taking individual measurements increases the amount of independent measurements we get per batch, reducing
     * variance.
     *
     * Individual measurements can only be taken for inference time, but not success rate.
     * Therefore, this has to be combined with the ratio distribution calculation.
     */
    [[nodiscard]] double get_iiit_mean() const {
        auto iiit_mean = 0.0;
        auto sample_count = 0;
        for (auto &sample: this->samples) {
            for (auto time: sample.individual_inverse_times) {
                sample_count++;
                iiit_mean += time;
            }
        }
        iiit_mean /= static_cast<double>(sample_count);
        return iiit_mean;
    }

    /**
     * Get the variance of all individual inverse inference time measurements.
     * That is, the mean of all multiplicative-inverses of time measurements of individual tree inferences, i.e.
     * BATCH_SIZE measurements per batch.
     * Taking individual measurements increases the amount of independent measurements we get per batch, reducing
     * variance.
     * It can however introduce systematic bias within the first few batches if the surrounding system is performing
     * other tasks, runs into heat dissipation issues, or batch reusing effects become measurable.
     * Therefore, individual time measurements should be combined with a decreasing multiplicative epsilon factor.
     *
     *
     * Individual measurements can only be taken for inference time, but not success rate.
     * Therefore, this has to be combined with the ratio distribution calculation.
     */
    [[nodiscard]] double get_iiit_variance() const {
        auto iiit_mean = get_iiit_mean();
        auto iiit_variance = 0.0;
        auto sample_count = 0;

        for (auto &sample: this->samples) {
            for (auto time: sample.individual_inverse_times) {
                sample_count++;
                iiit_variance += (iiit_mean - time) * (iiit_mean - time);
            }
        }
        iiit_variance /= static_cast<double>(sample_count);
        return iiit_variance;
    }

    /**
     * Get the mean of the ratio distribution of success rate and wall time measurements.
     * The success rate is modeled as a binomial distribution (which eliminates estimating the variance),
     * and the wall time distribution is modelled by a normal distribution of the multiplicative inverse measurements.
     */
    [[nodiscard]] double get_ratio_mean() const {
        const auto iiit_mean = get_iiit_mean();
        auto success_mean = this->get_success_mean();

        return success_mean * iiit_mean;
    }

    /**
     * Get the standard deviation of the ratio distribution of success rate and wall time measurements.
     * The success rate is modeled as a binomial distribution (which eliminates estimating the variance),
     * and the wall time distribution is modelled by a normal distribution of the multiplicative inverse measurements.
     */
    [[nodiscard]] double get_ratio_deviation() const {
        const auto iiit_mean = get_iiit_mean();
        const auto iiit_variance = get_iiit_variance();
        const auto success_mean = get_success_mean();
        const auto success_variance = get_success_variance();

        return sqrt(
            (success_mean * success_mean + success_variance) * (iiit_mean * iiit_mean + iiit_variance) - success_mean
            * success_mean * iiit_mean * iiit_mean);
    }

    /**
     * Take the benchmark data of a tuned batch which has previously run its inference with the parameters of this
     * bandit.
     * The measurement is stored and updates the expected mean reward of this bandit.
     *
     * @param batch A tuned batch which has been run on the parameter set of this Bandit instance.
     */
    void take_measurement(const TunedBatch &batch) {
        // if we have no other samples yet, we initialize the variance estimate by overestimating it intentionally
        if (samples.empty()) {
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

        // extract individual inverse inference times
        const auto tree_times = batch.per_tree_time();
        auto inverse_times = std::vector<double>(tree_times.size());
        for (unsigned int i = 0; i < tree_times.size(); ++i) {
            inverse_times[i] = (1.0 / static_cast<double>(tree_times[i]));
        }

        this->samples.emplace_back(batch.elapsed_wall_time(), batch.get_plausible_tree_count(), batch.get_batch_size(),
                                   inverse_times);

        const auto deviation = get_normal_smart_deviation();

        LOG_WORKER_TS(LogLevel::info) << "[" << this->name << "]: Takes measurement: " << batch.
                get_plausible_tree_count() << " plausible trees at " << (
                    static_cast<double>(batch.get_plausible_tree_count()) / static_cast<double>(batch.
                        elapsed_wall_time()) * 1000.0)
                << " (+- " << deviation * 1000.0 << ") trees per second." << std::endl;


        // calculate mean of the individual inverse inference times
        const auto ratio_mean = get_ratio_mean();
        const auto ratio_std_deviation = get_ratio_deviation();

        LOG_WORKER_TS(LogLevel::info) << "[" << this->name << "]: Adjusted Measurement: " <<
                (1000.0 * ratio_mean) << " (+- " << ratio_std_deviation * 1000.0 <<
                ") trees per second per worker. (" << (
                    1000.0 * batch.get_num_workers() * ratio_mean) << " (+- " << ratio_std_deviation * 1000.0 * batch.
                get_num_workers()
                << ") trees per second) " << std::endl;
    }

    void disable() {
        this->participating = false;
        LOG_INFO << "Disabled bandit " << this->name << std::endl;
    }

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
    double initial_variance{0.0};

    /**
     * Initial weight of the `estimated_variance` in the calculation of the actual variance. Reduced by one for each
     * existing measurement in `samples`.
     */
    unsigned int initial_variance_weight{0};

    /**
     * Whether this bandit is participating in the multiarmed bandit algorithm.
     */
    bool participating{true};

    double (Bandit::*mean)() const = &Bandit::get_ratio_mean;

    /**
     * The selected model for standard deviation. Can be either `get_ratio_deviation` or `get_smart_deviation` which
     * is a proxy for `get_normal_variance` that switches to Median Absolute Deviation transparently.
     */
    double (Bandit::*deviation)() const = &Bandit::get_ratio_deviation;

    /**
     * Maximum success attainable by any bandit
     */
    double maximum_success;
};
#endif //RAXML_BANDIT_HPP_
