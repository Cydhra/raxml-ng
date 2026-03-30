#ifndef RAXML_BATCHQUEUE_HPP_
#define RAXML_BATCHQUEUE_HPP_
#include "TunedBatch.hpp"

// forward declaration of RaxmlInstance
struct RaxmlInstance;

/**
 * Central resource management for batches:
 * Each batch requires starting trees, and batches should be reused if possible
 */
class BatchQueue {
public:
    BatchQueue(RaxmlInstance &instance,
               const Options &opts,
               const IDVector &tip_msa_idmap,
               LoadBalancer &load_balancer,
               const std::shared_ptr<PartitionedMSA> &msa,
               const std::vector<std::vector<doubleVector> > &persite_loglh,
               const unsigned long long seed,
               const unsigned int batch_size) : instance(instance),
                                                opts(opts),
                                                tip_msa_idmap(tip_msa_idmap),
                                                msa(msa),
                                                load_balancer(load_balancer),
                                                persite_loglh(persite_loglh),
                                                batch_size(batch_size),
                                                current_seed(seed) {
    }

    /**
     * How often the queue should try to reuse a batch before finalizing it.
     */
    unsigned int max_reuse_attempts = 10;

    /**
     * Select a TunedBatch instance for inference with the given parameters.
     * If a previously inferred batch has potential for more plausible trees, and the parameters
     * are compatible with the parameters used for that batch before, that batch is returned.
     * Otherwise, a new batch is generated and returned.
     */
    TunedBatch &select_next_batch(const MetaParameters &current_parameters);

    /**
     * Return a batch to the queue after inference is completed.
     */
    void finish_batch(TunedBatch &batch);

    /**
     * Backup the model of a batch
     * @param batch any batch
     */
    void backup_batch_model(const TunedBatch &batch);

    /**
     * @return the number of plausible trees found so far
     */
    unsigned int num_plausible_trees() const {
        return this->total_plausible_trees;
    }

    /**
     * @returnt the number of inferred batches, finalized or not.
     */
    unsigned int num_batches() const {
        return this->batches.size();
    }

    /**
     * @return access to the batch vector
     */
    std::vector<TunedBatch>& get_batches() {
        return this->batches;
    }

    /**
     * @return read-only access to the batch vector
     */
    std::vector<TunedBatch> const& view_batches() const {
        return this->batches;
    }

protected:
    /**
     * Reference to the RAxML-ng instance which is required for all optimization steps
     */
    RaxmlInstance &instance;

    /**
     * Reference to the user options which are required for all optimization steps
     */
    const Options &opts;

    /**
     * Reference to the RAxML-ng instance's tip id-map which is used for starting tree generation.
     */
    const IDVector &tip_msa_idmap;

    /**
     * Pointer to the partitioned msa instance of the current RAxML-ng run
     */
    const std::shared_ptr<PartitionedMSA> msa;

    /**
     * Reference to the user-configured fine-grained load balancer used by the main RAxML instance.
     */
    LoadBalancer &load_balancer;

    /**
     * Pointer to the persite_loglh vector of the reference tree run
     */
    const std::vector<std::vector<doubleVector> > persite_loglh;

    /**
     * List of all batches that are being inferred or were inferred by the optimizer. It is initialized with a
     * minimum number of batches, and is extended whenever more batches are required.
     *
     * Some of the batches might already be finalized.
     * There is, however, no guarantee that the finalized batches are contiguous.
     */
    std::vector<TunedBatch> batches;

    /**
     * Because some batches can be reused after inference if they did not achieve
     */
    std::unordered_set<std::string> unfinished = {};

    /**
     * Because the inference of batches is parallel, and batches could be reused multiple times, the queue tracks
     * which batches are currently in-flight (meaning, there is currently a worker inferring trees for them).
     *
     * This maps the (unique) batch name to the index in the batch vector.
     * When a worker is done, the batch has to be removed from the in-flight map.
     */
    std::unordered_set<std::string> in_flight = {};

    /**
     * Model parameter backup to initialize batch trees with.
     */
    std::unique_ptr<ModelMap> backup_model = unique_ptr<ModelMap>(new ModelMap());

    /**
     * Number of plausible trees that all batches could provide right now.
     */
    unsigned int total_plausible_trees = 0;

    /**
     * The number of trees to generate per batch.
     */
    const unsigned int batch_size;

    /**
     * The current seed for starting tree generation. Offset that by the number of generated trees whenever it is used
     * to generate a batch of trees.
     */
    unsigned long long current_seed;

    /**
     * Finalize a batch and add its plausible tree count to the total count
     *
     * @param batch tuned batch which is in the unfinished queue
     */
    void finalize_batch(TunedBatch &batch);

    /**
     * Return the starting seed for generating `num_trees` trees.
     * The queue keeps track of how many trees have been generated and advances the seed accordingly.
     */
    unsigned long generate_seed_for_trees(const unsigned int num_trees) {
        const unsigned long long current = this->current_seed;
        this->current_seed += num_trees;
        return current;
    }

// TODO: these methods should not remain public
public:
    /**
     * Push-back `n` batches to the end of the batch vector, and infer starting trees for them.
     */
    void generate_batches(unsigned int n);
};


#endif //RAXML_BATCHQUEUE_HPP_
