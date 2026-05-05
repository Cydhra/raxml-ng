#ifndef RAXML_TUNEDBATCH_HPP_
#define RAXML_TUNEDBATCH_HPP_

#include <memory>
#include <optional>
#include <utility>
#include "MetaParameters.hpp"
#include "Threadpool.hpp"
#include "../loadbalance/LoadBalancer.hpp"
#include "../loadbalance/CoarseLoadBalancer.hpp"
#include "../au/AuTest.hpp"
#include "../Checkpoint.hpp"

// forward declaration to avoid cyclic header inclusion
class SharedBatchResources;

/**
 * Probability to reject a tree even if it is plausible.
 * For reference: we reject the null-hypothesis (trees are equally plausible) if the p-value is equal or smaller than
 * the significance level, so a tree is plausible if its p-value is strictly larger than the significance level).
 */
constexpr double SIGNIFICANCE_LEVEL = 0.05;

// forward declaration of RaxmlInstance
struct RaxmlInstance;

// forward declaration of generate_tree in main.cpp to make it accessible. If the function in main.cpp
// changes signature, just update this declaration as well.
Tree generate_tree(const RaxmlInstance &instance, StartingTree type, int random_seed, bool bootstrap);

class TunedBatch final {
public:
    TunedBatch(string name,
               const unsigned int starting_seed,
               const unsigned int batch_size,
               const unsigned int num_threads,
               const unsigned int num_workers,
               const std::shared_ptr<PartitionedMSA> &msa,
               LoadBalancer &thread_load_balancer,
               IDVector &tip_msa_idmap,
               std::vector<std::vector<doubleVector> > &reference_persite_loglh)
        : name(std::move(name)),
          reference_persite_loglh(reference_persite_loglh),
          msa(msa),
          starting_seed(starting_seed),
          batch_start_trees(new TreeList(batch_size)),
          tip_msa_idmap(tip_msa_idmap),
          batch_persite_logh(std::vector<std::vector<doubleVector> >(batch_size)) {
        for (auto &tree_slh: batch_persite_logh) {
            for (const auto &pinfo: msa->part_list())
                tree_slh.emplace_back(pinfo.msa().length());
        }

        const auto threads_per_worker = num_threads / num_workers;

        // prepare space for the tree-info objects
        this->batch_trees = std::vector<std::vector<std::optional<TreeInfo> > >(batch_size);
        for (auto &vector: batch_trees) {
            vector.resize(threads_per_worker);
        }

        // load balance tasks where one tree may be split between multiple threads
        assert(num_workers <= this->get_batch_size());
        ContiguousCoarseLoadBalancer load_balancer;
        CoarseAssignment tree_ids(batch_size);
        std::iota(tree_ids.begin(), tree_ids.end(), 0);
        this->coarse_assignments = load_balancer.get_all_assignments(tree_ids, num_workers);

        // load balance partitions for such tasks between threads
        PartitionAssignment part_sizes;
        for (unsigned int i = 0; i < this->msa->part_list().size(); ++i) {
            /* init the list of partition sizes */
            auto pinfo = &this->msa->part_list()[i];
            part_sizes.assign_sites(i, 0, pinfo->length(), pinfo->model().clv_entry_size());
        }

        this->part_assignments = thread_load_balancer.get_all_assignments(part_sizes, threads_per_worker);

        // load-balance work for AU test, where we have reference trees and trees assigned to one thread need to be
        // contiguous. This is only needed for the first instance of the AU test, afterward we can reuse the bootstrap
        // values for the reference trees and use the exclusive_tree_access assignment
        const unsigned int total_trees_au = reference_persite_loglh.size() + batch_size;
        CoarseAssignment au_tree_ids(total_trees_au);
        std::iota(au_tree_ids.begin(), au_tree_ids.end(), 0);
        this->au_assignment = load_balancer.get_all_assignments(au_tree_ids, num_threads);

        // load-balance work where one tree can be manaaged by one thread only
        CoarseAssignment exclusive_tree_access(batch_size);
        std::iota(exclusive_tree_access.begin(), exclusive_tree_access.end(), 0);
        this->exclusive_assignment = load_balancer.get_all_assignments(exclusive_tree_access, num_threads);
    }

    // delete copy constructor because of corax partition
    TunedBatch(const TunedBatch &other) = delete;

    // TODO check which property violates the copy/move contract so we dont have to implement this manually
    TunedBatch(TunedBatch &&other) noexcept
        : reuse_attempts(other.reuse_attempts),
          name(std::move(other.name)),
          reference_persite_loglh(other.reference_persite_loglh),
          msa(std::move(other.msa)),
          starting_seed(other.starting_seed),
          meta_parameters(std::move(other.meta_parameters)),
          batch_start_trees(std::move(other.batch_start_trees)),
          spr_params(other.spr_params),
          num_fast_spr_performed(other.num_fast_spr_performed),
          num_slow_spr_performed(other.num_slow_spr_performed),
          au_assignment(std::move(other.au_assignment)),
          exclusive_assignment(std::move(other.exclusive_assignment)),
          coarse_assignments(std::move(other.coarse_assignments)),
          part_assignments(std::move(other.part_assignments)),
          tip_msa_idmap(other.tip_msa_idmap),
          batch_trees(std::move(other.batch_trees)),
          batch_persite_logh(std::move(other.batch_persite_logh)),
          meta_parameters_set(other.meta_parameters_set),
          initial_model_optimized(other.initial_model_optimized),
          plausible_tree_count(other.plausible_tree_count),
          wall_time(other.wall_time),
          tree_topologies(std::move(other.tree_topologies)) {
    }

    // explicitly implement move-assign to avoid implicit deletion
    // TODO check which property violates the copy/move contract so we dont have to implement this manually
    TunedBatch &operator=(TunedBatch &&other) noexcept {
        if (this == &other)
            return *this;
        reuse_attempts = other.reuse_attempts;
        name = std::move(other.name);
        meta_parameters = std::move(other.meta_parameters);
        starting_seed = other.starting_seed;
        batch_start_trees = std::move(other.batch_start_trees);
        msa = std::move(other.msa);
        spr_params = other.spr_params;
        num_fast_spr_performed = other.num_fast_spr_performed;
        num_slow_spr_performed = other.num_slow_spr_performed;
        reference_persite_loglh = other.reference_persite_loglh;
        au_assignment = std::move(other.au_assignment);
        exclusive_assignment = std::move(other.exclusive_assignment);
        coarse_assignments = std::move(other.coarse_assignments);
        part_assignments = std::move(other.part_assignments);
        tip_msa_idmap = std::move(other.tip_msa_idmap);
        tree_topologies = std::move(other.tree_topologies);
        batch_trees = std::move(other.batch_trees);
        batch_persite_logh = std::move(other.batch_persite_logh);
        meta_parameters_set = other.meta_parameters_set;
        initial_model_optimized = other.initial_model_optimized;
        plausible_tree_count = other.plausible_tree_count;
        wall_time = other.wall_time;
        return *this;
    }

    /**
     * How often the batch queue has attempted to reuse this batch.
     */
    unsigned int reuse_attempts = 0;

    /**
     * Using the batch configuration, infer K trees in parallel.
     *
     * @param instance Raxml instance, required for tree generation
     * @param opts command line options, required for parameter optimization
     * @param resources
     * @param resources
     */
    void optimize(RaxmlInstance &instance, const Options &opts, SharedBatchResources &resources,
                  const TaskGroup &context,
                  unsigned int worker_id, unsigned int thread_id);

    /**
     * Perform the AU test on the trees in the batch, as well as the supplied reference trees,
     * but backup the model before, optimize the model fully, and then restore the original model.
     *
     * @param resources au test instance and profiling, shared between batches
     * @param initialized if false, the au_test instance is not initialized and memory will be allocated, and the
     * reference trees included in the bootstrap. Otherwise, only the batch trees are included.
     *
     * @return The number of plausible trees.
     */
    void perform_plausibility_check(SharedBatchResources &resources, bool initialized, const TaskGroup &context,
                                    unsigned int worker_id,
                                    unsigned int thread_id);

    /**
     * Update the meta heuristical parameters of the batch, reconfiguring the search parameters from them.
     *
     * @param opts Command line options
     * @param new_parameters batch treeset inference meta parameters
     */
    void update_meta_parameters(const Options &opts, const shared_ptr<MetaParameters> new_parameters);

    /**
     * Compare the batch's current configuration with a set of new parameters, and check whether the inference can
     * continue with the new parameters as if the batch had been inferred under the new parameters from the start.
     *
     * @param new_parameters a set of meta parameters that is compared to this batch's parameters
     * @return true, if the batch can continue inference
     */
    bool is_compatible(const MetaParameters &new_parameters) const;

    /**
     * Replace the model parameters with the model parameters of a different batch, which allows restoring the model instead of
     * optimizing it from scratch. The model will be written into this batch's model backup and then directly
     * applied.
     */
    void assign_batch_models(const ModelMap &other);

    /**
     * Store the model of the first tree in the batch into a given model map.
     *
     * @param target Reference to a ModelMap where to store the model parameters
     */
    void backup_models(ModelMap &target) const;

    /**
     * Free resources used during inference that will not be used again. The batch data stays valid but further
     * inference will not be possible.
     */
    void finalize();

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
     * Compute the output tree of the `index`-th tree of this batch.
     *
     * @param index the index of the tree within the batch's trees
     *
     * @return a tree object generated from the information in all partitions of the TreeInfo object
     */
    Tree get_tree(unsigned int index) const;

    /**
     * @return the log-likelihoods of all trees in order of the trees
     */
    std::vector<double> get_tree_likelihoods();

    /**
     * Append the plausible trees of this batch to the end of a vector.
     * The topologies are copied at the end, possible reallocating the given vector object.
     *
     * @param buffer a vector which can be extended with tree topologies.
     */
    void get_plausible_trees(std::vector<Tree> &buffer) const;

    std::string const &get_name() const {
        return this->name;
    }

protected:
    /**
     * Name of the batch for outputting debug information.
     */
    string name;

    /**
     * Per-site log-likelihoods of the reference trees already inferred before the treeset heuristic kicked in.
     * These cannot change, and constitute the first part of the AU test input.
     */
    std::vector<std::vector<doubleVector> > &reference_persite_loglh;

    /**
     * A reference to the MSA used in inference. We need it for the AU test.
     */
    shared_ptr<PartitionedMSA> msa;

    /**
     * The starting seed (starting from 0) for this batch. Batches infer starting trees with ascending seeds, so this
     * number is the number of starting trees in previous batches.
     */
    unsigned int starting_seed;

    /**
     * The meta-heuristic parameters for inferring trees. These are not the model parameters, but settings of the
     * inference heuristics which are being optimized for plausible tree throughput during tree set inference.
     */
    shared_ptr<MetaParameters> meta_parameters;

    /**
     * Starting trees for this inference batch
     */
    shared_ptr<TreeList> batch_start_trees;

    /**
     * SPR round parameters inherited from the default checkpoint manager. They will be updated by the batch according
     * to the batch settings
     */
    spr_round_params spr_params;

    /**
     * Number of fast SPR rounds that have already been performed on the tree. This is increased by the `infer_batch`
     * method.
     */
    unsigned int num_fast_spr_performed{0};

    /**
     * Number of slow SPR rounds that have already been performed on the tree. This is increased by the `infer_batch`
     * method.
     */
    unsigned int num_slow_spr_performed{0};

    /**
     * Assignment of trees to threads for the AU test. The AU test cannot split between partitions, and so no tree
     * can have more than one thread assigned.
     * This assignment includes the reference trees of the AU test.
     */
    CoarseAssignmentList au_assignment;

    /**
     * Assignment of trees to threads where only one thread can work on a tree. This is relevant for starting trees,
     * but can replace the au_assignment once the AU test stops recalculating the reference bootstraps.
     */
    CoarseAssignmentList exclusive_assignment;

    /**
     * Assignment of trees to workers for tree inference. The AU test diverges from this assignment because the AU
     * test cannot split partitions between threads.
     */
    CoarseAssignmentList coarse_assignments;

    /**
     * Fine-grained assignment of partitions to threads within workers (thread groups).
     */
    PartitionAssignmentList part_assignments;

    /**
     * Vector mapping sequence IDs to the MSA.
     */
    IDVector &tip_msa_idmap;

    /**
     * Treeinfo objects for the trees inferred in this batch. These objects are updated by the inference algorithm.
     * The outer vector is indexed by tree, the inner by in-worker thread id (i.e. if each worker has 4 threads,
     * the inner vectors contain 4 TreeInfo instances).
     */
    std::vector<std::vector<std::optional<TreeInfo> > > batch_trees;

    /**
     * Per-site log-likelihoods of the trees inferred in this batch. We recalculate these if the tree has changed,
     * and we perform the AU test by combining it with the reference tree loglikelihood vectors.
     */
    std::vector<std::vector<doubleVector> > batch_persite_logh;

    /**
     * Initial model parameters that get loaded into the tree info objects upon creation.
     * This is initialized after creating the TunedBatch with a call to assign_batch_models.
     */
    ModelMap initial_model;

    /**
     * Flag indicating whether the batch has been configured with meta-parameters.
     */
    bool meta_parameters_set{false};

    /**
     * How many starting trees have been generated.
     */
    atomic_uint num_trees_generated{0};

    /**
     * Flag indicating whether the model has been optimized once (or alternatively, if a pre-optimized model
     * has been loaded).
     * If this is false, the optimize() function needs to perform one model optimization before doing SPR rounds.
     */
    bool initial_model_optimized{false};

    /**
     * Number of plausible trees as determined by the last AU test.
     */
    unsigned int plausible_tree_count{0};

    /**
     * Time spent on this batch. Does not include overhead that could be largely avoided on batches outside the tuning
     * phase.
     * This mostly excludes time spent on model optimization (MO) because we only do a MO after all SPR rounds are finished.
     * During tuning we do BLO between all SPR rounds though, which would throw off the walltime measurement.
     */
    unsigned int wall_time{0};

    /**
     * Time spent in AU test. This has to be considered for the total wall time, because the amortized cost of AU test
     * increases if less trees become plausible. But it cannot be added directly on top of the wall time, because
     * batch reusing should not double-count the AU test time.
     */
    unsigned int au_wall_time{0};

    /**
     * Mutex guard for the topologies vector, where the topologies of previous optimize() calls are backed up, in case
     * we end the algorithm before the current call to optimize() is finished.
     * Unique pointer to provide inner mutability. Const so the move constructors don't touch it.
     */
    const std::unique_ptr<std::mutex> topology_access = make_unique<std::mutex>();

    /**
     * The final tree topologies. This vector is populated by a call to `finalize()` and is otherwise empty.
     */
    std::vector<Tree> tree_topologies{};

    /**
     * The finished AU test p values, which are updated whenever the AU test is run. These values refer to the backup
     * topologies at all times, since the batch might have progressed since the last AU test.
     * Access to this member has to be guarded with the topology_access mutex to avoid concurrent reading and
     * modification.
     * The vector only contains p-value for the batch trees, the p-values of the reference topologies are not included.
     */
    doubleVector p_values;

    /**
     * Update spr_params instance according to the meta_parameters
     */
    void auto_configure(const Options &opts) {
        // update options according to MetaParameters:
        spr_params.ntopol_keep = this->meta_parameters->keep_top_k_topol;
        spr_params.subtree_cutoff = opts.spr_cutoff;
        spr_params.radius_min = 1;

        // if all fast spr rounds have been performed, set thorough to true, so further spr rounds are slow
        spr_params.thorough = this->num_fast_spr_performed >= this->meta_parameters->num_fast_spr;
        spr_params.lh_epsilon_brlen_full = opts.lh_epsilon;
        spr_params.lh_epsilon_brlen_triplet = opts.lh_epsilon_brlen_triplet;

        // taken from the fast heuristic
        spr_params.radius_max = (spr_params.thorough ? 1 : 2) * meta_parameters->max_adaptive_radius;

        // we don't need those
        spr_params.increasing_moves = nullptr;
        spr_params.total_moves = nullptr;
    }

    /**
     * @return Whether all starting trees have been generated for this batch.
     */
    bool start_trees_generated() const {
        return this->num_trees_generated == get_batch_size();
    }

    /**
     * Generate parsimony starting trees for this batch, and initialize the tree inference.
     */
    void generate_starting_trees(RaxmlInstance &instance, const Options &opts, const TaskGroup &context,
                                 unsigned int worker_id, unsigned int thread_id);

    /**
     * Perform model and branch length optimization according to the current tuning parameters and the given epsilon.
     * If model optimization is currently disabled, load models from a backup.
     *
     * @param resources
     * @param epsilon the likelihood threshold when to stop optimizing
     * @param model if true, optimize model parameters
     * @param branches if true, optimize branch lengths
     * @param force if true, model optimization is forced, even if batch tuning parameters turn it off
     */
    void optimize_parameters(SharedBatchResources &resources, const TaskGroup &context, unsigned int worker_id,
                             unsigned int thread_id,
                             double epsilon, bool model = true, bool branches = true, bool force = false);

    /**
     * Perform SPR rounds up to the target count, with meta-parameters according to the batch settings.
     *
     * @param opts parsed command line options and forced RAxML settings
     */
    void optimize_topology(const Options &opts, const TaskGroup &context, SharedBatchResources &resources,
                           unsigned int worker_id, unsigned int thread_id);

    /**
     * Perform the AU test on the trees in the batch, as well as the supplied reference trees.
     *
     * @param au_test test instance
     * @param initialized if false, the au_test instance is not initialized and memory will be allocated, and the
     * reference trees included in the bootstrap. Otherwise, only the batch trees are included.
     */
    void perform_au_test(AuTest &au_test, bool initialized, const TaskGroup &context, unsigned int worker_id,
                         unsigned int thread_id);
};

#endif //RAXML_TUNEDBATCH_HPP_
