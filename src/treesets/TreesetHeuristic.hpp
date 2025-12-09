#ifndef RAXML_TREESETHEURISTIC_HPP_
#define RAXML_TREESETHEURISTIC_HPP_

#include "../loadbalance/LoadBalancer.hpp"
#include "../Checkpoint.hpp"

// forward declaration of RaxmlInstance
struct RaxmlInstance;

enum TuningPhase {
    /** First phase: Check how many SPR rounds we need */
    TUNE_BASELINE,

    /** Check if greedy hillclimbing without keeping topologies suffices */
    TUNE_GREEDY,

    /** Check if inheriting models from other tree searches suffices */
    TUNE_MODEL_OPT,

    /** Tuning is done, infer trees with current parameters */
    FINALIZED,
};

/**
 * An aggressive stateful local search heuristic for treeset-search. This differs substantially from other topological
 * heuristics because those are stateless (i.e., they do not depend on previous results of the same raxml search).
 * This heuristic auto-tunes its search parameters in between batches of tree searches to reduce the amount of effort
 * spent on individual searches as much as possible.
 * This does not lead to a deterioration of the final tree accuracy under the assumption that the treeset command is
 * used for degenerate datasets with extremely low signal.
 * If the likelihood surface has pronounced peaks, this aggressive heuristic will likely not yield optimal results (but
 * this will be detectable by the AU test).
 */
class TreesetHeuristic {
public:
    explicit TreesetHeuristic(const std::shared_ptr<PartitionedMSA> &msa,
                              const std::vector<std::vector<doubleVector> > &persite_loglh,
                              const IDVector &tip_msa_idmap)
        : num_spr(0), msa(msa), persite_loglh(persite_loglh), tip_msa_idmap(tip_msa_idmap) {
    }

    /**
     * Main function for the treeset command. it is different from the thread_main function because it has to
     * dynamically adjust load balancing and thread auto-tuning. It is called by master_main instead of starting
     * pthreads in `thread_main` if the treeset command is called.
     */
    void infer_treeset(RaxmlInstance &instance, const Options &opts, CheckpointManager &cm,
                       LoadBalancer &load_balancer);

private:
    /**
     * If true, the heuristic is still tuning parameters,
     */
    TuningPhase tuning_phase{TUNE_BASELINE};

    /**
     * If true, replace fast SPR rounds with light SPR rounds that do even less BLOs.
     */
    bool greedy_spr{false};

    /**
     * If true, skip the first model optimization by reusing model parameters from the previous search.
     */
    bool skip_model{false};

    /**
     * How many SPR rounds to perform for each tree search
     */
    unsigned int num_spr;

    /**
     * If true, skip all model optimization and simply accept parsimony starting trees as the final resulting topology.
     */
    bool accept_starting_trees{false};

    /**
     * A reference to the MSA used in inference. We need it for the AU test.
     */
    const std::shared_ptr<PartitionedMSA> &msa;

    /**
     * Per-site log-likelihoods of the reference trees already inferred before the treeset heuristic kicked in.
     * These cannot change, and constitute the first part of the AU test input for each batch.
     */
    const std::vector<std::vector<doubleVector> > &persite_loglh;

    /**
     * Reference to the MSA tip index map in RaxmlInstance
     */
    const IDVector &tip_msa_idmap;

    /**
     * @return the recommended number of threads for workers
     */
    int recommended_thread_count();

    /**
     * @return the recommended number of workers per rank
     */
    int recommended_worker_count();
};

#endif //RAXML_TREESETHEURISTIC_HPP_
