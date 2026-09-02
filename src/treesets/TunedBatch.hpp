#ifndef RAXML_TUNEDBATCH_HPP_
#define RAXML_TUNEDBATCH_HPP_

#include <memory>
#include <optional>
#include <utility>
#include "MetaParameters.hpp"
#include "Threadpool.hpp"
#include "inference/Heuristic.hpp"
#include "../loadbalance/LoadBalancer.hpp"
#include "../loadbalance/CoarseLoadBalancer.hpp"
#include "../au/AuTest.hpp"
#include "../Checkpoint.hpp"
#include "../Optimizer.hpp"
#include "start/StartTreeHeuristic.hpp"

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

class TunedBatch final {
public:
    TunedBatch(std::string name,
               const std::shared_ptr<PartitionedMSA> &msa,
               const std::shared_ptr<IDVector> &tip_msa_idmap,
               const std::shared_ptr<std::vector<std::vector<doubleVector> > > &reference_persite_loglh,
               const unsigned int starting_seed,
               const unsigned int batch_size,
               const unsigned int num_threads,
               const unsigned int num_workers,
               LoadBalancer &thread_load_balancer)
        : name(std::move(name)),
          reference_persite_loglh(reference_persite_loglh),
          msa(msa),
          starting_seed(starting_seed),
          batch_start_trees(new TreeList(batch_size)),
          tip_msa_idmap(tip_msa_idmap),
          batch_persite_logh(std::vector<std::vector<doubleVector> >(batch_size)),
          threads_per_worker(num_threads / num_workers) {
        for (auto &tree_slh: batch_persite_logh) {
            for (const auto &pinfo: msa->part_list())
                tree_slh.emplace_back(pinfo.msa().length());
        }

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
        this->coarse_assignments = make_shared<CoarseAssignmentList>(
            load_balancer.get_all_assignments(tree_ids, num_workers));

        // load balance partitions for such tasks between threads
        PartitionAssignment part_sizes;
        for (unsigned int i = 0; i < this->msa->part_list().size(); ++i) {
            /* init the list of partition sizes */
            auto pinfo = &this->msa->part_list()[i];
            part_sizes.assign_sites(i, 0, pinfo->length(), static_cast<double>(pinfo->model().clv_entry_size()));
        }

        this->part_assignments = make_shared<PartitionAssignmentList>(
            thread_load_balancer.get_all_assignments(part_sizes, threads_per_worker));

        // load-balance work for AU test, where we have reference trees and trees assigned to one thread need to be
        // contiguous. This is only needed for the first instance of the AU test, afterward we can reuse the bootstrap
        // values for the reference trees and use the exclusive_tree_access assignment
        const unsigned int total_trees_au = reference_persite_loglh->size() + batch_size;
        CoarseAssignment au_tree_ids(total_trees_au);
        std::iota(au_tree_ids.begin(), au_tree_ids.end(), 0);
        this->au_assignment = make_shared<CoarseAssignmentList>(
            load_balancer.get_all_assignments(au_tree_ids, num_threads));

        // load-balance work where one tree can be managed by one thread only
        CoarseAssignment exclusive_tree_access(batch_size);
        std::iota(exclusive_tree_access.begin(), exclusive_tree_access.end(), 0);
        this->exclusive_assignment = make_shared<CoarseAssignmentList>(
            load_balancer.get_all_assignments(exclusive_tree_access, num_threads));
    }

    // delete copy constructor because of corax partitions in tree vector
    TunedBatch(const TunedBatch &other) = delete;

    TunedBatch(TunedBatch &&other) noexcept = default;

    // explicitly implement move-assign to avoid implicit deletion
    TunedBatch &operator=(TunedBatch &&other) noexcept = default;

    /**
     * How often the batch queue has attempted to reuse this batch.
     */
    unsigned int reuse_attempts = 0;

    /**
     * Using the batch configuration, infer K trees in parallel.
     *
     * @param instance Raxml instance, required for tree generation
     * @param opts command line options, required for parameter optimization
     * @param resources resources shared between multiple batches, like the AU test instance.
     * @param context task group context
     * @param worker_id raxml-instance-local id of the worker. Within task groups, they need not start at 0.
     * @param thread_id worker-local id of the thread, thread numbering start at 0 for each worker.
     */
    void optimize(const RaxmlInstance &instance, const Options &opts, SharedBatchResources &resources,
                  const TaskGroup &context,
                  unsigned int worker_id, unsigned int thread_id);

    /**
     * Perform the AU test on the trees in the batch, as well as the supplied reference trees,
     * but backup the model before, optimize the model fully, and then restore the original model.
     *
     * @param resources au test instance and profiling, shared between batches
     * @param initialized if false, the au_test instance is not initialized and memory will be allocated, and the
     *                    reference trees included in the bootstrap. Otherwise, only the batch trees are included.
     * @param context task group context
     * @param worker_id raxml-instance-local id of the worker. Within task groups, they need not start at 0.
     * @param thread_id worker-local id of the thread, thread numbering start at 0 for each worker.
     *
     * @return The number of plausible trees.
     */
    void perform_plausibility_check(const Options &opts, SharedBatchResources &resources, bool initialized,
                                    const TaskGroup &context,
                                    unsigned int worker_id,
                                    unsigned int thread_id);

    /**
     * Update the meta heuristical parameters of the batch. This will forcibly update them, even if `is_compatible`
     * returns false.
     *
     * @param new_parameters batch treeset inference meta parameters
     */
    void update_meta_parameters(const MetaParameters &new_parameters);

    /**
     * Compare the batch's current configuration with a set of new parameters, and check whether the inference can
     * continue with the new parameters as if the batch had been inferred under the new parameters from the start.
     *
     * @param new_parameters a set of meta parameters that is compared to this batch's parameters
     * @return true, if the batch can continue inference
     */
    [[nodiscard]] bool is_compatible(const MetaParameters &new_parameters) const;

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
    [[nodiscard]] unsigned int get_batch_size() const;

    /**
     * If an AU test has already been performed, count how many of the batch's trees are plausible.
     * If no AU test has been performed, or the last AU test is no longer valid, the method throws a
     * RaxmlException.
     *
     * @return Number of trees with a p-value above 0.05.
     */
    [[nodiscard]] unsigned int get_plausible_tree_count() const;

    /**
     * Compute the total (wall) time spent on inferring the batch, ignoring the parallelization.
     * @return Time spent working on inference in milliseconds.
     */
    [[nodiscard]] unsigned int elapsed_wall_time() const;

    /**
     * Compute the output tree of the `index`-th tree of this batch.
     *
     * @param index the index of the tree within the batch's trees
     *
     * @return a tree object generated from the information in all partitions of the TreeInfo object
     */
    [[nodiscard]] Tree get_tree(unsigned int index) const;

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

    [[nodiscard]] std::string const &get_name() const {
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
    shared_ptr<std::vector<std::vector<doubleVector> > > reference_persite_loglh;

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
    MetaParameters meta_parameters;

    /**
     * Starting trees for this inference batch
     */
    std::shared_ptr<TreeList> batch_start_trees;

    /**
     * Assignment of trees to threads for the AU test. The AU test cannot split between partitions, and so no tree
     * can have more than one thread assigned.
     * This assignment includes the reference trees of the AU test.
     */
    std::shared_ptr<CoarseAssignmentList> au_assignment;

    /**
     * Assignment of trees to threads where only one thread can work on a tree. This is relevant for starting trees,
     * but can replace the au_assignment once the AU test stops recalculating the reference bootstraps.
     */
    std::shared_ptr<CoarseAssignmentList> exclusive_assignment;

    /**
     * Assignment of trees to workers for tree inference. The AU test diverges from this assignment because the AU
     * test cannot split partitions between threads.
     */
    std::shared_ptr<CoarseAssignmentList> coarse_assignments;

    /**
     * Fine-grained assignment of partitions to threads within workers (thread groups).
     */
    std::shared_ptr<PartitionAssignmentList> part_assignments;

    /**
     * Vector mapping sequence IDs to the MSA.
     */
    std::shared_ptr<IDVector> tip_msa_idmap;

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
    std::shared_ptr<ModelMap> initial_model;

    /**
     * How many starting trees have been generated.
     */
    std::unique_ptr<atomic_uint> num_trees_generated = make_unique<atomic_uint>(0);

    /**
     * Number of plausible trees as determined by the last AU test.
     */
    unsigned int plausible_tree_count{0};

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
    std::unique_ptr<std::mutex> topology_access = make_unique<std::mutex>();

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
     * Decorated generation strategy for starting trees, set when `update_meta_parameters` is called.
     */
    std::unique_ptr<StartTreeHeuristic> start_tree_heuristic = {};

    /**
     * Decorated inference strategy, set when `update_meta_parameters` is called.
     */
    std::unique_ptr<InferenceHeuristic> heuristic = {};

    /**
     * How many threads will infer this batch at once (i.e. threads per task-group)
     */
    unsigned int threads_per_worker;

    /**
     * @return Whether all starting trees have been generated for this batch.
     */
    [[nodiscard]] bool start_trees_generated() const {
        return *this->num_trees_generated == get_batch_size();
    }

    /**
     * Generate parsimony starting trees for this batch, and initialize the tree inference.
     *
     * @param instance RaxmlInstance which is required for tree generation
     * @param context task group context
     * @param worker_id raxml-instance-local id of the worker. Within task groups, they need not start at 0.
     * @param thread_id worker-local id of the thread, thread numbering start at 0 for each worker.
     */
    void generate_starting_trees(const RaxmlInstance &instance, SharedBatchResources &resources, const TaskGroup &context,
                                 unsigned int worker_id, unsigned int thread_id);

    /**
     * Perform the AU test on the trees in the batch, as well as the supplied reference trees.
     *
     * @param au_test test instance
     * @param initialized if false, the au_test instance is not initialized and memory will be allocated, and the
     * reference trees included in the bootstrap. Otherwise, only the batch trees are included.
     * @param context task group context
     * @param worker_id raxml-instance-local id of the worker. Within task groups, they need not start at 0.
     * @param thread_id worker-local id of the thread, thread numbering start at 0 for each worker.
     */
    void perform_au_test(AuTest &au_test, bool initialized, const TaskGroup &context, unsigned int worker_id,
                         unsigned int thread_id);
};

#endif //RAXML_TUNEDBATCH_HPP_
