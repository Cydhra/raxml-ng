#ifndef RAXML_NG_STARTTREESOURCE_HPP
#define RAXML_NG_STARTTREESOURCE_HPP
#include <atomic>
#include <memory>

#include "TreeSource.hpp"
#include "../../loadbalance/CoarseLoadBalancer.hpp"
#include "../../Tree.hpp"



// forward declaration of generate_tree in main.cpp to make it accessible. If the function in main.cpp
// changes signature, just update this declaration as well.
Tree generate_tree(const RaxmlInstance &instance, StartingTree type, int random_seed, bool bootstrap);

class ParsimonySource : public TreeSource {

public:
    explicit ParsimonySource(const int starting_seed) : starting_seed(starting_seed) {}

    void ensure(const RaxmlInstance &instance, const SmartBarrier &barrier, unsigned int threads_per_worker, unsigned int worker_id, unsigned int thread_id, unsigned int num_trees) override;

    std::tuple<unsigned int, unsigned int> consume_batch(const RaxmlInstance &instance, const SmartBarrier &barrier, unsigned int threads_per_worker, unsigned int worker_id, unsigned int thread_id, unsigned int num_trees) override;

    void copy_tree(Tree &target, unsigned int tree_id) const override;

    /**
     * @return mean estimate of time spent per tree on parsimony
     */
    [[nodiscard]] double amortized_time(unsigned int batch_size) const override;

protected:
    /**
     * The master thread of the current cohort (i.e. thread where worker_id and thread_id is 0) locks the mutex for
     * the reserve counter, and every other thread waits for them.
     *
     * @param barrier a barrier for all threads involved in the modification of the reserve counter
     * @param worker_id the worker-id of the local thread, exactly one worker (potentially multiple threads) must have worker id 0
     * @param thread_id the worker-local thread id of the local thread, exactly one thread in each worker must have id 0
     */
    unsigned int acquire_reservation(const SmartBarrier &barrier, const unsigned int worker_id, const unsigned int thread_id, const unsigned int num_trees) {
        if (worker_id + thread_id == 0) {
            tree_reserve_mutex->lock();
        }
        barrier.enter();
        const auto start_index = this->tree_cursor;
        barrier.enter();

        this->tree_cursor += num_trees;
        if (worker_id + thread_id == 0) {
            tree_reserve_mutex->unlock();
        }

        return start_index;
    }

private:
    /**
     * A mutex that the master-thread must hold during modification of the tree-list or the starting seed.
     */
    std::unique_ptr<std::mutex> tree_list_mutex = std::make_unique<std::mutex>();

    /**
     * A mutex that the master-thread must hold during modification of the tree list cursor
     */
    std::unique_ptr<std::mutex> tree_reserve_mutex = std::make_unique<std::mutex>();

    /**
     * A list of (pre-generated) tree topologies. The topologies can be used as starting trees (consuming them) or in
     * heuristics that do not use them as a starting tree for inference, in which case consumption is not necessary,
     * and ML heuristics can reuse those trees.
     */
    std::deque<Tree> tree_list = {};

    /**
     * Cursor within the treelist to demarc the boundary of yet-unconsumed tree topologies.
     */
    unsigned int tree_cursor = 0;

    /**
     * Seed where the next round of tree generation attempts begins
     */
    int starting_seed;

    /**
     * Load balancer for tree generation.
     */
    std::unique_ptr<ContiguousCoarseLoadBalancer> load_balancer = std::make_unique<ContiguousCoarseLoadBalancer>();

    /**
     * Walltime measurements of parsimony for later bandit time estimation.
     */
    std::unique_ptr<std::atomic_uint> cumulative_wall_time = std::make_unique<std::atomic_uint>(0);

    std::unique_ptr<std::atomic_uint> trees_generated = std::make_unique<std::atomic_uint>(0);
};


#endif //RAXML_NG_STARTTREESOURCE_HPP
