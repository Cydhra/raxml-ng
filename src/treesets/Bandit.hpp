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
    Measurement(const unsigned int time_spent, const unsigned int plausible_trees, const unsigned int batch_size) : time_spent(time_spent),
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
    explicit Bandit(MetaParameters parameters) : parameters(std::make_shared<MetaParameters>(parameters)) {
    }

    /**
     * Apply this bandit's parameters to the batch.
     */
    void apply_parameters(TunedBatch &batch) const;

    /**
     * Take the benchmark data of a tuned batch which has previously run its inference with the parameters of this
     * bandit.
     * The measurement is stored and updates the expected mean reward of this bandit.
     *
     * @param batch A tuned batch which has been run on the parameter set of this Bandit instance.
     */
    void take_measurement(const TunedBatch &batch);

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

protected:
    std::shared_ptr<MetaParameters> parameters;

    /**
     * Samples drawn from the reward distribution.
     */
    std::vector<Measurement> samples = {};
};


#endif //RAXML_BANDIT_HPP_
