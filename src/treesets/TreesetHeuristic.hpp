#ifndef RAXML_TREESETHEURISTIC_HPP_
#define RAXML_TREESETHEURISTIC_HPP_

#include "../au/AuTest.hpp"
#include "../loadbalance/LoadBalancer.hpp"
#include "../Checkpoint.hpp"

// forward declaration of RaxmlInstance
struct RaxmlInstance;

// forward declaration of thread_start_trees in main.cpp to make it accessible. If the function in main.cpp
// changes signature, just update this declaration as well.
void thread_start_trees(RaxmlInstance &instance, TreeList &tree_list, StartingTree st_tree_type,
                        const intVector &seeds, size_t offset, bool bootstrap);

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
    bool tuning_phase{true};

    /**
     * If true, replace fast SPR rounds with light SPR rounds that do even less BLOs.
     */
    bool light_spr{false};

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

// TODO we should reuse bootstrap resamplings of the reference trees for the AU test, since they stay the same.
/**
 * A batch of local searches tuned with a specific set of parameters chosen by a {@link TreesetHeuristic} instance.
 */
class TunedBatch final {
public:
    TunedBatch(const bool light_spr, const bool skip_model, const unsigned int num_spr,
               const unsigned int starting_seed, const unsigned int batch_size, const unsigned int num_threads,
               const unsigned int num_workers,
               const std::shared_ptr<PartitionedMSA> &msa,
               const std::vector<std::vector<doubleVector> > &reference_persite_loglh)
        : light_spr(light_spr),
          skip_model(skip_model),
          num_spr(num_spr),
          starting_seed(starting_seed),
          num_threads(num_threads), num_workers(num_workers),
          batch_start_trees(new TreeList(batch_size)),
          msa(msa),
          reference_persite_loglh(reference_persite_loglh),
          batch_persite_logh(std::vector<std::vector<doubleVector> >(batch_size)) {
        for (auto &tree_slh: batch_persite_logh) {
            for (const auto &pinfo: msa->part_list())
                tree_slh.emplace_back(pinfo.msa().length());
        }

        // we can initialize au_test only after initializing the per-site lnl partition vectors
        this->au_test.reset(new AuTest(msa, reference_persite_loglh, batch_persite_logh, AU_DEFAULT_SCALES, AU_DEFAULT_REPS, starting_seed));
        this->au_test->allocate_test_statistics();
    }

    /**
     * If true, replace fast SPR rounds with light SPR rounds that do even less BLOs.
     */
    const bool light_spr;

    /**
     * If true, skip the first model optimization by reusing model parameters from the previous search.
     */
    const bool skip_model;

    /**
     * How many SPR rounds to perform for each tree search
     */
    const unsigned int num_spr;

    /**
     * @return the number of trees that are inferred in this batch.
     */
    unsigned int get_batch_size() const;

    /**
     * Generate parsimony starting trees for this batch, and initialize the tree inference.
     */
    void generate_starting_trees(RaxmlInstance &instance, const Options &opts, LoadBalancer &load_balancer,
                                 const IDVector &tip_msa_idmap);

    /**
     * Using the batch configuration, infer K trees in parallel.
     */
    void infer_batch(RaxmlInstance &instance, const Options &opts, LoadBalancer &load_balancer,
                     const IDVector &tip_msa_idmap);

    /**
     * Perform low-epsilon parameter optimization followed by the AU test against the reference topologies,
     * and calculate the ratio of batch trees which are considered plausible. Returns true, if the ratio reaches
     * the threshold.
     */
    bool is_plausible(const Options &opts);

protected:
    /**
     * The starting seed (starting from 0) for this batch. Batches infer starting trees with ascending seeds, so this
     * number is the number of starting trees in previous batches.
     */
    const unsigned int starting_seed;


    /**
     * How many threads are used in this batch. Divisible by the number of workers.
     */
    const unsigned int num_threads;

    /**
     * Number of workers assigned to this batch.
     */
    const unsigned int num_workers;

    /**
     * Starting trees for this inference batch
     */
    const shared_ptr<TreeList> batch_start_trees;

    /**
     * A reference to the MSA used in inference. We need it for the AU test.
     */
    const shared_ptr<PartitionedMSA> &msa;

    /**
     * Per-site log-likelihoods of the reference trees already inferred before the treeset heuristic kicked in.
     * These cannot change, and constitute the first part of the AU test input.
     */
    const std::vector<std::vector<doubleVector> > &reference_persite_loglh;

    /**
     * Assignment of partitions within the thread assignment of the batch. This differs from the part assignment of the
     * main algorithm, if the batch got assigned different numbers of threads and workers.
     */
    unique_ptr<PartitionAssignmentList> part_assignment;

    /**
     * Treeinfo objects for the trees inferred in this batch. These objects are updated by the inference algorithm.
     */
    std::vector<TreeInfo> batch_trees{std::vector<TreeInfo>()};

    /**
     * Backup for the model parameters, such that the model can be restored after changing it in the TreeInfo instances.
     */
    std::vector<Model> batch_model_backup{std::vector<Model>()};

    /**
     * Per-site log-likelihoods of the trees inferred in this batch. We recalculate these if the tree has changed,
     * and we perform the AU test by combining it with the reference tree loglikelihood vectors.
     */
    std::vector<std::vector<doubleVector> > batch_persite_logh;

    /**
    * AU test instance
    */
    shared_ptr<AuTest> au_test;

    /**
     * Perform the AU test on the trees in the batch, as well as the supplied reference trees.
     */
    unsigned int perform_au_test(const Options &opts);

    /**
     * Perform model and branch length optimization according to the current tuning parameters and the given epsilon.
     * If model optimization is currently disabled, load models from a backup.
     *
     * @param epsilon the likelihood threshold when to stop optimizing
     * @param force if true, model optimization is forced, disregarding batch tuning parameters
     */
    void optimize_all_parameters(double epsilon, bool force = false);

    /**
     * Store the current model parameters in a backup, such that we can restore them if optimization needs to continue.
     * The backup can also be obtained from the outside, to allow sharing models across batches.
     */
    void save_model_backup();

    /**
     * Restore model parameters from the internal backup.
     */
    void restore_model_backup();
};

#endif //RAXML_TREESETHEURISTIC_HPP_
