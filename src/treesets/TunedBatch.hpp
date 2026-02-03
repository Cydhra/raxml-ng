#ifndef RAXML_TUNEDBATCH_HPP_
#define RAXML_TUNEDBATCH_HPP_

#include "MetaParameters.hpp"
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
    TunedBatch(const string name,
               const unsigned int starting_seed,
               const unsigned int batch_size,
               const unsigned int num_threads,
               const unsigned int num_workers,
               const std::shared_ptr<PartitionedMSA> &msa,
               const std::vector<std::vector<doubleVector> > &reference_persite_loglh)
        : name(name),
          starting_seed(starting_seed),
          num_threads(num_threads),
          num_workers(num_workers),
          batch_start_trees(new TreeList(batch_size)),
          msa(msa),
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
        this->batch_models.resize(this->get_batch_size());

        // initialize coarse load balancing (i.e. split trees among workers for inference)
        assert(this->num_workers <= this->get_batch_size());
        ContiguousCoarseLoadBalancer load_balancer;
        CoarseAssignment tree_ids(batch_size);
        std::iota(tree_ids.begin(), tree_ids.end(), 0);
        this->coarse_assignments = load_balancer.get_all_assignments(tree_ids, this->num_workers);
    }

    /**
     * Generate parsimony starting trees for this batch, and initialize the tree inference.
     */
    void generate_starting_trees(RaxmlInstance &instance, const Options &opts, LoadBalancer &load_balancer,
                                 const IDVector &tip_msa_idmap);

    /**
     * Perform model and branch length optimization according to the current tuning parameters and the given epsilon.
     * If model optimization is currently disabled, load models from a backup.
     *
     * @param opts parsed command line options and forced RAxML settings
     * @param epsilon the likelihood threshold when to stop optimizing
     * @param model if true, optimize model parameters
     * @param branches if true, optimize branch lengths
     * @param force if true, model optimization is forced, even if batch tuning parameters turn it off
     */
    void optimize_parameters(const Options &opts, double epsilon, bool model = true, bool branches = true,
                             bool force = false);

    /**
     * Perform SPR rounds up to the target count, with meta-parameters according to the batch settings.
     *
     * @param opts parsed command line options and forced RAxML settings
     */
    void optimize_topology(const Options &opts);

    /**
     * Using the batch configuration, infer K trees in parallel.
     */
    void optimize(const Options &opts);

    /**
     * Perform the AU test on the trees in the batch, as well as the supplied reference trees.
     */
    unsigned int perform_au_test(const Options &opts);

    /**
     * Perform the AU test on the trees in the batch, as well as the supplied reference trees,
     * but backup the model before, optimize the model fully, and then restore the original model.
     *
     * @return The number of plausible trees.
     */
    unsigned int perform_plausibility_check(const Options &opts);

    /**
     * Update the meta heuristical parameters of the batch, reconfiguring the search parameters from them.
     *
     * @param opts Command line options
     * @param new_parameters batch treeset inference meta parameters
     */
    void update_meta_parameters(const Options &opts, const shared_ptr<MetaParameters> &new_parameters);

    /**
     * Compare the batch's current configuration with a set of new parameters, and check whether the inference can
     * continue with the new parameters as if the batch had been inferred under the new parameters from the start.
     *
     * @param new_parameters a set of meta parameters that is compared to this batch's parameters
     * @return true, if the batch can continue inference
     */
    bool is_compatible(const shared_ptr<MetaParameters> &new_parameters) const;

    /**
     * Replace the model parameters with the model parameters of a different batch, which allows restoring the model instead of
     * optimizing it from scratch. The model will be written into this batch's model backup and then directly
     * applied.
     */
    void assign_batch_models(const TunedBatch &other);

    /**
     * @return the number of trees that are inferred in this batch.
     */
    unsigned int get_batch_size() const;

    /**
     * If an AU test has already been performed, count how many of the batch's trees are plausible.
     * If no AU test has been performed, or the last AU test is no longer valid, the method throws a
     * RaxmlException.
     *
     * @return Number of trees with a p-value above 0.05.
     */
    unsigned int get_plausible_tree_count() const;

    /**
     * Compute the total (wall) time spent on inferring the batch, ignoring the parallelization.
     * @return Time spent working on inference in milliseconds.
     */
    unsigned int elapsed_wall_time() const;

    /**
     * @return Whether the starting trees have been generated for this batch.
     */
    bool start_trees_generated() const;

protected:
    /**
     * Name of the batch for outputting debug information.
     */
    string name;

    /**
     * The meta-heuristic parameters for inferring trees. These are not the model parameters, but settings of the
     * inference heuristics which are being optimized for plausible tree throughput during tree set inference.
     */
    shared_ptr<MetaParameters> meta_parameters;

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
    std::vector<std::vector<TreeInfo> > batch_trees{std::vector<std::vector<TreeInfo> >()};

    /**
     * Backup for the model parameters, such that the model can be restored from fixed parameters at any time.
     * The model backup is used during AU-Tests to store unoptimized model parameters while the model is optimized for
     * the current tree.
     * It is also used to store models that are assigned to the batch from the outside
     * instead of model parameter optimization.
     */
    std::vector<ModelMap> batch_models{std::vector<ModelMap>()};

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
     * Flag indicating whether the batch has been configured with meta-parameters.
     */
    bool meta_parameters_set{false};

    /**
     * Flag indicating whether the model has been optimized once (or alternatively, if a pre-optimized model
     * has been loaded).
     * If this is false, the optimize() function needs to perform one model optimization before doing SPR rounds.
     */
    bool initial_model_optimized{false};

    /**
     * Flag indicating whether the au_test instance is outdated.
     * The class must set the flag to true whenever the per-site log-likelihoods for the batch trees change.
     */
    bool au_test_dirty{true};

    /**
     * Number of plausible trees as determined by the last AU test.
     */
    unsigned int plausible_tree_count = 0;

    /**
     * Time spent on this batch. Does not include overhead that could be largely avoided on batches outside the tuning
     * phase.
     * This mostly excludes time spent on model optimization (MO) because we only do a MO after all SPR rounds are finished.
     * During tuning we do BLO between all SPR rounds though, which would throw off the walltime measurement.
     */
    unsigned int wall_time{0};

    /**
     * Update spr_params instance according to the meta_parameters
     */
    void auto_configure(const Options &opts) {
        // update options according to MetaParameters:
        spr_params.ntopol_keep = this->meta_parameters->keep_top_k_topol;
        spr_params.subtree_cutoff = opts.spr_cutoff;
        spr_params.radius_min = 0;
        spr_params.radius_max = 20;
        spr_params.thorough = false;
        spr_params.lh_epsilon_brlen_full = opts.lh_epsilon;
        spr_params.lh_epsilon_brlen_triplet = opts.lh_epsilon_brlen_triplet;
    }

    /**
     * Get the number of threads per worker
     */
    unsigned int num_threads_per_worker() const {
        return num_threads / num_workers;
    }

    /**
     * Called when the per-site log-likelihoods change, overriding the results of the AU-test
     */
    void mark_p_values_dirty();

    /**
     * Store the current model parameters in the batch model store,
     * such that we can restore them if optimization needs to continue.
     */
    void backup_models();

    /**
     * Restore model parameters from the batch model store.
     */
    void load_batch_models();
};

#endif //RAXML_TUNEDBATCH_HPP_
