#ifndef RAXML_TUNEDBATCH_HPP_
#define RAXML_TUNEDBATCH_HPP_

#include "../loadbalance/LoadBalancer.hpp"
#include "../loadbalance/CoarseLoadBalancer.hpp"
#include "../au/AuTest.hpp"
#include "../Checkpoint.hpp"

// forward declaration of RaxmlInstance
struct RaxmlInstance;

// forward declaration of thread_start_trees in main.cpp to make it accessible. If the function in main.cpp
// changes signature, just update this declaration as well.
void thread_start_trees(RaxmlInstance &instance, TreeList &tree_list, StartingTree st_tree_type,
                        const intVector &seeds, size_t offset, bool bootstrap);

// TODO we should reuse bootstrap resamplings of the reference trees for the AU test, since they stay the same.
/**
 * A batch of local searches tuned with a specific set of parameters chosen by a {@link TreesetHeuristic} instance.
 */
class TunedBatch final {
public:
    TunedBatch(const bool light_spr,
               const bool skip_model,
               unsigned int num_spr,
               const unsigned int starting_seed,
               const unsigned int batch_size,
               spr_round_params spr_params,
               const unsigned int num_threads,
               const unsigned int num_workers,
               const std::shared_ptr<PartitionedMSA> &msa,
               const std::vector<std::vector<doubleVector> > &reference_persite_loglh)
        : light_spr(light_spr),
          skip_model(skip_model),
          target_num_spr(num_spr),
          starting_seed(starting_seed),
          num_threads(num_threads),
          num_workers(num_workers), batch_start_trees(new TreeList(batch_size)),
          msa(msa),
          spr_params(spr_params),
          reference_persite_loglh(reference_persite_loglh),
          batch_persite_logh(std::vector<std::vector<doubleVector> >(batch_size)) {
        for (auto &tree_slh: batch_persite_logh) {
            for (const auto &pinfo: msa->part_list())
                tree_slh.emplace_back(pinfo.msa().length());
        }

        // we can initialize au_test only after initializing the per-site lnl partition vectors
        this->au_test.reset(new AuTest(msa, reference_persite_loglh, batch_persite_logh, AU_DEFAULT_SCALES,
                                       AU_DEFAULT_REPS, starting_seed));
        this->au_test->allocate_test_statistics();

        // initialize model maps for storing backups
        this->batch_model_backup.resize(this->get_batch_size());

        // initialize coarse load balancing (i.e. split trees among workers for inference)
        assert(this->num_workers <= this->get_batch_size());
        ContiguousCoarseLoadBalancer load_balancer;
        CoarseAssignment tree_ids(batch_size);
        std::iota(tree_ids.begin(), tree_ids.end(), 0);
        this->coarse_assignments = load_balancer.get_all_assignments(tree_ids, this->num_workers);

        this->per_thread_timing = std::vector<unsigned int>(num_threads);
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
     * How many SPR rounds to perform for each tree search. This parameter can be updated.
     */
    unsigned int target_num_spr;

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
    void infer_batch(const Options &opts);

    /**
     * Perform low-epsilon parameter optimization followed by the AU test against the reference topologies,
     * and calculate the ratio of batch trees which are considered plausible. Returns true, if the ratio reaches
     * the threshold.
     */
    bool is_plausible(const Options &opts);

    /**
     * Copy the model from a previous instance of TunedBatch, which allows restoring the model instead of
     * optimizing it from scratch. The model will be written into this batch's model backup and then directly
     * applied.
     */
    void inherit_model(const TunedBatch &other);

    /**
     * Compute the total CPU (wall) time spent on inferring the batch. This includes the sum of all wall times spent
     * by all threads.
     * @return
     */
    unsigned int elapsed_cpu_time() const;

    /**
     * If an AU test has already been performed, count how many of the batch's trees are plausible.
     * @return Number of trees with a p-value above 0.05. If no AU test has been performed
     */
    unsigned int plausible_tree_count() const;

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
     * SPR round parameters inherited from the default checkpoint manager. They will be updated by the batch according
     * to the batch settings
     */
    spr_round_params spr_params;

    /**
     * Number of SPR rounds that have already been performed on the tree. This is increased by the `infer_batch`
     * method.
     */
    unsigned int num_spr_performed{0};

    /**
     * Per-site log-likelihoods of the reference trees already inferred before the treeset heuristic kicked in.
     * These cannot change, and constitute the first part of the AU test input.
     */
    const std::vector<std::vector<doubleVector> > &reference_persite_loglh;

    /**
     * Assignment of trees to workers for tree inference. The AU test diverges from this assignment because the AU
     * test cannot split partitions between threads.
     */
    CoarseAssignmentList coarse_assignments;

    /**
     * Assignment of partitions within the thread assignment of the batch. This differs from the part assignment of the
     * main algorithm, if the batch got assigned different numbers of threads and workers.
     */
    PartitionAssignmentList part_assignments;

    /**
     * Treeinfo objects for the trees inferred in this batch. These objects are updated by the inference algorithm.
     * The outer vector is indexed by tree, the inner by in-worker thread id (i.e. if each worker has 4 threads,
     * the inner vectors contain 4 TreeInfo instances).
     */
    std::vector<std::vector<TreeInfo>> batch_trees{std::vector<std::vector<TreeInfo>>()};

    /**
     * Backup for the model parameters, such that the model can be restored after changing it in the TreeInfo instances.
     */
    std::vector<ModelMap> batch_model_backup{std::vector<ModelMap>()};

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
     * A vector of millisecond-timing of SPR rounds. The vector contains one entry per thread, which includes
     * the time for all SPR rounds performed by the batch.
     */
    std::vector<unsigned int> per_thread_timing;

    /**
     * Get the number of threads per worker
     */
    unsigned int num_threads_per_worker() const {
        return num_threads / num_workers;
    }

    /**
     * Perform the AU test on the trees in the batch, as well as the supplied reference trees.
     */
    unsigned int perform_au_test(const Options &opts);

    /**
     * Perform model and branch length optimization according to the current tuning parameters and the given epsilon.
     * If model optimization is currently disabled, load models from a backup.
     *
     * @param opts parsed command line options and forced RAxML parameters
     * @param epsilon the likelihood threshold when to stop optimizing
     * @param force if true, model optimization is forced, disregarding batch tuning parameters
     */
    void optimize_all_parameters(const Options &opts, double epsilon, bool force = false);

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

#endif //RAXML_TUNEDBATCH_HPP_
