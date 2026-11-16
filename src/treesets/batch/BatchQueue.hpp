#ifndef RAXML_BATCHQUEUE_HPP_
#define RAXML_BATCHQUEUE_HPP_

#include "TunedBatch.hpp"
#include <atomic>

// forward declaration of RaxmlInstance
struct RaxmlInstance;

/**
 * Central resource management for batches:
 * Each batch requires starting trees, and batches should be reused if possible
 */
class BatchQueue {
public:
    BatchQueue(const std::shared_ptr<IDVector> &tip_msa_idmap,
               LoadBalancer &load_balancer,
               const std::shared_ptr<PartitionedMSA> &msa,
               const std::shared_ptr<std::vector<std::vector<doubleVector> > > &persite_loglh,
               const unsigned int batch_size) : tip_msa_idmap(tip_msa_idmap),
                                                msa(msa),
                                                load_balancer(load_balancer),
                                                persite_loglh(persite_loglh),
                                                batch_size(batch_size) {
    }

    /**
     * How often the queue should try to reuse a batch before finalizing it.
     */
    unsigned int max_reuse_attempts = 10;

    /**
     * Push-back `n` batches to the end of the batch vector, and infer starting trees for them.
     *
     * @return a pointer to the array of newly generated batches. The range [return_val, return_val + n] is a valid
     * range of TunedBatch instances.
     *
     * @param num_workers number of workers assigned to the batch
     * @param num_threads total number of threads (not per worker) assigned to the batch
     * @param lock proof that the calling thread is allowed to modify the batch list. Because this method does not
     * trigger any inference, it does not stall the queue.
     */
    TunedBatch &generate_batch(unsigned int num_workers, unsigned int num_threads, const std::lock_guard<std::mutex> &lock);

    /**
     * Select a TunedBatch instance for inference with the given parameters.
     * If a previously inferred batch has potential for more plausible trees, and the parameters
     * are compatible with the parameters used for that batch before, that batch is returned.
     * Otherwise, a new batch is generated and returned.
     */
    TunedBatch &select_next_batch(const MetaParameters &current_parameters, unsigned int num_workers,
                                  unsigned int num_threads);

    /**
     * Return a batch to the queue after inference is completed.
     */
    void finish_batch(TunedBatch &batch);

    /**
     * @return the number of plausible trees found so far
     */
    unsigned int num_plausible_trees() const {
        return this->finalized_plausible_trees + this->unfinished_plausible_trees;
    }

    /**
     * @returnt the number of inferred batches, finalized or not.
     */
    unsigned int num_batches() const {
        return this->batches.size();
    }

    /**
     * @return read-only access to the batch vector
     */
    std::deque<TunedBatch> const &view_batches() const {
        return this->batches;
    }

protected:
    /**
     * Reference to the RAxML-ng instance's tip id-map which is used for starting tree generation.
     */
    std::shared_ptr<IDVector> tip_msa_idmap;

    /**
     * Pointer to the partitioned msa instance of the current RAxML-ng run
     */
    std::shared_ptr<PartitionedMSA> msa;

    /**
     * Reference to the user-configured fine-grained load balancer used by the main RAxML instance.
     */
    LoadBalancer &load_balancer;

    /**
     * Pointer to the persite_loglh vector of the reference tree run
     */
    std::shared_ptr<std::vector<std::vector<doubleVector> > > persite_loglh;

    /**
     * List of all batches that are being inferred or were inferred by the optimizer. It is initialized with a
     * minimum number of batches, and is extended whenever more batches are required.
     *
     * Some of the batches might already be finalized.
     * There is, however, no guarantee that the finalized batches are contiguous.
     *
     * This needs to be a deque rather than a vector because threads will hold references while elements are inserted,
     * so moving of existing elements must not happen.
     */
    std::deque<TunedBatch> batches;

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
     * Number of plausible trees that finalized batches provide right now.
     */
    unsigned int finalized_plausible_trees = 0;

    /**
     * Number of plausible trees in batches that are unfinished and thus could change again.
     */
    unsigned int unfinished_plausible_trees = 0;

    /**
     * The number of trees to generate per batch.
     */
    unsigned int batch_size;

    std::mutex batch_mutex;

    /**
     * Finalize a batch and add its plausible tree count to the total count
     *
     * @param batch tuned batch which is in the unfinished queue
     */
    void finalize_batch(TunedBatch &batch);
};


#endif //RAXML_BATCHQUEUE_HPP_
