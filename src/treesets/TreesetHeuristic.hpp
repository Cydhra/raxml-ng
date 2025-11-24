#ifndef RAXML_TREESETHEURISTIC_HPP_
#define RAXML_TREESETHEURISTIC_HPP_

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
    explicit TreesetHeuristic() : num_spr(0) {
    }

    /**
     * Main function for the treeset command. it is different from the thread_main function because it has to
     * dynamically adjust load balancing and thread auto-tuning. It is called by master_main instead of starting
     * pthreads in `thread_main` if the treeset command is called.
     */
    void infer_treeset(RaxmlInstance &instance, const Options &opts, CheckpointManager &cm);

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
     * @return the recommended number of threads for workers
     */
    int recommended_thread_count();

    /**
     * @return the recommended number of workers per rank
     */
    int recommended_worker_count();
};

/**
 * A batch of local searches tuned with a specific set of parameters chosen by a {@link TreesetHeuristic} instance.
 */
class TunedBatch final {
public:
    TunedBatch(const bool light_spr, const bool skip_model, const unsigned int num_spr, const unsigned int batch_size, const unsigned int num_threads)
        : light_spr(light_spr),
          skip_model(skip_model),
          num_spr(num_spr),
          num_threads(num_threads),
          batch_start_trees(new TreeList(batch_size)) {
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
     * Using the batch configuration, infer K trees in parallel.
     */
    void infer_batch(RaxmlInstance &instance, const Options &opts);

protected:
    /**
     * How many threads are used in this batch.
     */
    const unsigned int num_threads;

    /**
     * Starting trees for this inference batch
     */
    const shared_ptr<TreeList> batch_start_trees;
};

#endif //RAXML_TREESETHEURISTIC_HPP_
