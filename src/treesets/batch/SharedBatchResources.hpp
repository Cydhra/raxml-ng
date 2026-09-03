#ifndef RAXML_SHAREDBATCHRESOURCES_HPP_
#define RAXML_SHAREDBATCHRESOURCES_HPP_

#include "../../pool/Threadpool.hpp"
#include "../../au/AuTest.hpp"
#include "../../Optimizer.hpp"

/**
 * Shallow replication counts for a faster AU test.
 */
const uintVector SHALLOW_REPS = {500, 500, 500, 500, 500, 500, 500, 500, 500, 500};

/**
 * We share AUTest instances between batches, one per TaskGroup of the threadpool, to save on resource and reuse
 * the computed likelihood values for the reference trees.
 */
class SharedBatchResources {
public:
    SharedBatchResources(const unsigned int num_task_groups,
                         const unsigned int workers_per_group,
                         const unsigned int total_threads,
                         const Options &opts,
                         std::shared_ptr<PartitionedMSA> msa,
                         const Tree &tree,
                         const std::vector<std::vector<doubleVector> > &reference_logh_matrix,
                         const unsigned int batch_size,
                         long seed) {
        // prepare a dummy matrix with empty vectors to correctly initialize the AU-Test. These dummy vectors
        // will be replaced by the TunedBatch instance before the AU test is called.
        std::vector<std::vector<doubleVector> > batch_loglh_dummy(batch_size);
        for (unsigned int i = 0; i < reference_logh_matrix.size(); ++i) {
            batch_loglh_dummy[i] = std::vector<doubleVector>(reference_logh_matrix[i].size());
        }

        for (unsigned int group = 0; group < num_task_groups; ++group) {
            screening_au_tests.emplace_back(msa, reference_logh_matrix, batch_loglh_dummy, AU_DEFAULT_SCALES,
                                            SHALLOW_REPS, seed);
            initialized.emplace_back(false);
        }

        // configure options for the raxml-fast optimizer
        fast_options = make_shared<Options>(opts);
        fast_options->topology_opt_method = TopologyOptMethod::simplified;
        fast_options->stopping_rule = StoppingRule::kh_mult;
        fast_options->nofiles_mode = true;
        fast_options->num_searches = 16; // TODO sync with batch size

        // initialize an optimizer and checkpoint manager for raxml-fast.
        fast_optimizer = make_shared<Optimizer>(*fast_options);
        fast_checkpoint_manager = make_shared<CheckpointManager>(*fast_options);

        fast_stop = make_shared<KHStoppingTest>(msa,
                                                workers_per_group * num_task_groups,
                                                total_threads,
                                                true,
                                                opts.random_seed,
                                                opts.lh_epsilon);

        // unfortunately this method wants a tree. Please do not ask why it wants that, it doesn't deserve the tree.
        // But we have to comply, so we give it one of the reference trees since
        // any tree that conforms to the MSA will do.
        fast_checkpoint_manager->init_checkpoints(tree, msa->models(), num_task_groups * workers_per_group);
    }

    /**
     * @return An Optimizer instance pre-configured to run `RAxML-ng --fast` inference
     */
    Optimizer &get_fast_optimizer() const {
        return *fast_optimizer;
    }

    /**
     * @return The KH stopping criterion instance pre-configured to run `RAxML-ng --fast` inference
     */
    std::shared_ptr<KHStoppingTest> get_fast_stop_criterion() {
        return fast_stop;
    }

    /**
     * @return an instance of the CheckpointManager that is configured to work for raxml fast inference
     */
    CheckpointManager &get_fast_cm() const {
        return *fast_checkpoint_manager;
    }

    /**
     * Get an AuTest instance exclusively for the caller batch.
     *
     * @param context the parallel context for the threads that require an au test instance
     * @return An AuTest instance that is reserved for the caller TaskGroup. Call is_initialized() to find out whether
     * the reference tree bootstrap values have been generated already.
     */
    AuTest &get_screening_test(const TaskGroup &context) {
        return screening_au_tests[context.group_id()];
    }

    /**
     * The batch leader of any batch may ask if the AU Test instance obtained by get_au_test
     * @param context the parallel context for the calling batch leader
     * @return true, if the AuTest instance returned for the calling thread has been initialized before.
     */
    bool is_initialized(const TaskGroup &context) {
        return initialized[context.group_id()];
    }

    /**
     * Set the AU Test instance of the calling batch leader to initialized. Later batches of the same task group no longer
     * have to compute bootstrap values for the reference trees.
     *
     * @param context the parallel context for the calling batch leader
     */
    void set_initialized(const TaskGroup &context) {
        this->initialized[context.group_id()] = true;
    }

protected:
    /**
     * Shared AU test instances, one for each thread group. These are initialized with drastically reduced replication
     * counts to be able to be used for pre-screening.
     */
    std::vector<AuTest> screening_au_tests;

    /**
     * Whether the corresponding Au Test has been used before, initializing the bootstrap values of the reference trees.
     * The flags are not atomic, because only the group leader of each batch is supposed to ask and only for its own
     * context.
     */
    std::vector<bool> initialized;

    /**
     * A copy of the CLI Options instance, with every option required to run raxml-fast forcibly enabled.
     * This instance is shared with fast_optimizer and local_fast_cm, but they do not own it, so we store it here.
     */
    shared_ptr<Options> fast_options;

    /**
     * An instance of the kh stopping criterion used for raxml-fast. It is used in the optimizer but not owned by it,
     * so we store it here.
     */
    std::shared_ptr<KHStoppingTest> fast_stop;

    /**
     * Batch-local instance of the standard RAxML optimizer. This is instanced for the raxml-fast fallback bandit.
     * The options passed to this optimizer are forced to perform the --fast heuristic.
     */
    std::shared_ptr<Optimizer> fast_optimizer;

    /**
     * Batch-local instance of the standard RAxML checkpoint manager configured for raxml-fast.
     * This is instanced for the raxml-fast fallback bandit.
     */
    shared_ptr<CheckpointManager> fast_checkpoint_manager;
};

#endif //RAXML_SHAREDBATCHRESOURCES_HPP_
