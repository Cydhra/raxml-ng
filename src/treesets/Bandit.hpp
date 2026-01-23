#ifndef RAXML_BANDIT_HPP_
#define RAXML_BANDIT_HPP_
#include <memory>
#include <vector>

// forward declaration of tuned batch
class TunedBatch;

/**
 * Class that holds the meta-parameters of self-tuning tree inference. The parameters are held by a bandit and can
 * be applied to TunedBatches to configure the tree inference.
 */
class MetaParameters {
public:
    /**
     * Number of top topologies to keep during SPR rounds. If lower, the SPR rounds perform substantially less
     * branch-length optimizations. On datasets where most trees are plausible, this saves time because the K top
     * topologies are probably equally likely anyway.
     */
    const unsigned int keep_top_k_topol;

    /**
     * If true, skip the first model optimization by reusing model parameters from a previous search.
     */
    const bool skip_model;

    /**
     * How many fast SPR rounds to perform for each tree search
     */
    const unsigned int num_fast_spr;

    /**
     * If true, skip all model optimization and simply accept parsimony starting trees as the final resulting topology.
     */
    const bool accept_starting_trees;

    MetaParameters(const unsigned int keep_top_k_topol, const bool skip_model, const unsigned int num_fast_spr,
                   const bool accept_starting_trees) : keep_top_k_topol(keep_top_k_topol), skip_model(skip_model),
                                                 num_fast_spr(num_fast_spr),
                                                 accept_starting_trees(accept_starting_trees) {
    }
};

/**
 * A measurement sample obtained from inferring a TunedBatch with a given set of parameters. The bandits are keeping
 * track of the measurements to estimate the mean and standard deviation of the underlying probability distribution of
 * the reward.
 */
class Measurement {
public:
    Measurement(const unsigned int time_spent, const unsigned int plausible_trees) : time_spent(time_spent),
                                                                         plausible_trees(plausible_trees) {
    }

    const unsigned int time_spent;

    const unsigned int plausible_trees;
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
     * @return the mean expected reward of the underlying distribution.
     */
    double get_mean() const;

protected:
    std::shared_ptr<MetaParameters> parameters;

    /**
     * Samples drawn from the reward distribution.
     */
    std::vector<Measurement> samples = {};
};


#endif //RAXML_BANDIT_HPP_
