#ifndef RAXML_SHAREDBATCHRESOURCES_HPP_
#define RAXML_SHAREDBATCHRESOURCES_HPP_

#include "Threadpool.hpp"
#include "TreesetProfiling.hpp"
#include "../au/AuTest.hpp"

/**
 * We share AUTest instances between batches, one per TaskGroup of the threadpool, to save on resource and reuse
 * the computed likelihood values for the reference trees.
 */
class SharedBatchResources {
public:
    SharedBatchResources(const unsigned int num_task_groups,
                                  std::shared_ptr<PartitionedMSA> msa,
                                  const std::vector<std::vector<doubleVector> > &reference_logh_matrix,
                                  const unsigned int batch_size,
                                  const doubleVector &scales,
                                  const uintVector &num_replicates,
                                  long seed) {

        // prepare a dummy matrix with empty vectors to correctly initialize the AU-Test. These dummy vectors
        // will be replaced by the TunedBatch instance before the AU test is called.
        std::vector<std::vector<doubleVector>> batch_loglh_dummy(batch_size);
        for (unsigned int i = 0; i < batch_size; ++i) {
            batch_loglh_dummy[i] = std::vector<doubleVector>(reference_logh_matrix[i].size());
        }

        for (unsigned int group = 0; group < num_task_groups; ++group) {
            au_tests.emplace_back(msa, reference_logh_matrix, batch_loglh_dummy, scales, num_replicates, seed);
            initialized.emplace_back(false);
        }
    }

    /**
     * Get the profiling instance. The instance is thread-safe, so there is only one global profiler.
     */
    TreesetProfiling &get_profiling() {
        return profiling;
    }

    /**
     * Get an AuTest instance exclusively for the caller batch.
     *
     * @param context the parallel context for the threads that require an au test instance
     * @return An AuTest instance that is reserved for the caller TaskGroup. Call is_initialized() to find out whether
     * the reference tree bootstrap values have been generated already.
     */
    AuTest &get_au_test(const TaskGroup &context) {
        return au_tests[context.group_id()];
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
     * Handles fine-grained profiling of batch optimization
     */
    TreesetProfiling profiling;

    /**
     * Shared AU test instances, one for each thread group.
     */
    std::vector<AuTest> au_tests;

    /**
     * Whether the corresponding Au Test has been used before, initializing the bootstrap values of the reference trees.
     * The flags are not atomic, because only the group leader of each batch is supposed to ask and only for its own
     * context.
     */
    std::vector<bool> initialized;
};

#endif //RAXML_SHAREDBATCHRESOURCES_HPP_
