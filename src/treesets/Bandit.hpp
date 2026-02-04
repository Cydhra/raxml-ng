#ifndef RAXML_BANDIT_HPP_
#define RAXML_BANDIT_HPP_

#include "MetaParameters.hpp"
#include "TunedBatch.hpp"
#include <memory>
#include <vector>

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
class Bandit {
public:
    explicit Bandit(const std::string &name, MetaParameters parameters) : name(name),
                                                                          parameters(std::make_shared<MetaParameters>(
                                                                              parameters)) {
    }

    /**
     * Apply this bandit's parameters to the batch.
     */
    void apply_parameters(const Options &opts, TunedBatch &batch) const;

    /**
     * Take the benchmark data of a tuned batch which has previously run its inference with the parameters of this
     * bandit.
     * The measurement is stored and updates the expected mean reward of this bandit.
     *
     * @param batch A tuned batch which has been run on the parameter set of this Bandit instance.
     */
    void take_measurement(const TunedBatch &batch);

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
    void initialize_variance(const double variance, const unsigned int weight);

    /**
     * Compare two bandit distribution and determine if this one is worse than the `other` bandit.
     * A bandit is worse if the upper bound of its expected success is below the mean of the expected success
     * of the other bandit.
     *
     * @param other Bandit to compare this one to
     * @param total_samples total number of batches inferred so far
     *
     * @return true if this bandit has worse success rate with high probability.
     */
    bool is_worse_than(const Bandit &other, unsigned int total_samples) const;

    /**
     * @return the mean expected reward (throughput) of the underlying distribution.
     */
    double get_mean_throughput() const;

    /**
     * @return the mean success rate (between 0 and 1) of yielding a plausible tree under this bandit's parameters.
     */
    double get_mean_success() const;

    /**
     * @return the variance of the reward distribution. Because this is required to be known a priori, the value is
     * being interpolated between an initial value and the measured value depending on how many measurements are
     * available.
     */
    double get_variance() const;

    /**
     * Calculate the upper confident limit on the mean expected success of this bandit. This depends on the total
     * number of samples drawn so far, as well as the number of samples drawn for this bandit.
     *
     * @param total_samples the number of total samples from all bandits that have been drawn so far
     *
     * @return the upper bound on the mean expected success that can be determined with high confidence.
     */
    double get_upper_confidence(unsigned int total_samples) const;

    /**
     * @return This bandit's meta parameters
     */
    MetaParameters &get_parameters() const;

    /**
     * @return Bandit name for debug output
     */
    string get_name() const;

protected:
    /**
     * Display name of the bandit for debugging
     */
    std::string name;

    /**
     * Heuristics parameters of this bandit
     */
    std::shared_ptr<MetaParameters> parameters;

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
};


#endif //RAXML_BANDIT_HPP_
