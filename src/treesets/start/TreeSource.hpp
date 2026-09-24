#ifndef RAXML_NG_TREESOURCE_HPP
#define RAXML_NG_TREESOURCE_HPP

#include "../../pool/SmartBarrier.hpp"
#include "../../Tree.hpp"

// forward declaration of RaxmlInstance
struct RaxmlInstance;

class TreeSource {
public:
    virtual ~TreeSource() = default;

    /**
     * Ensure that at least `num_trees` trees are available in the tree source
     *
     * @param instance the static raxml instance required for tree generation
     * @param barrier a barrier for all threads involved in the tree generation
     * @param threads_per_worker how many threads are assigned to each worker
     * @param worker_id the barrier-local worker id. One worker must have id 0
     * @param thread_id the worker-local thread id. One thread per worker must have id 0.
     */
    virtual void ensure(const RaxmlInstance &instance, const SmartBarrier &barrier, unsigned int threads_per_worker, unsigned int worker_id, unsigned int thread_id, unsigned int num_trees) = 0;

    /**
     * Obtain `num_trees` tree topology from the source with multiple threads at once.
     * If not enough trees are present, some threads may generate new ones.
     */
    virtual std::tuple<unsigned int, unsigned int> consume_batch(const RaxmlInstance &instance, const SmartBarrier &barrier, unsigned int threads_per_worker, unsigned int worker_id, unsigned int thread_id, unsigned int num_trees) = 0;

    /**
     * Copy the tree at the given tree id to the target reference
     */
    virtual void copy_tree(Tree &target, unsigned int tree_id) const = 0;

    /**
     * @return mean estimate of time spent per tree on parsimony
     */
    [[nodiscard]] virtual double amortized_time(unsigned int batch_size) const = 0;
};


#endif //RAXML_NG_TREESOURCE_HPP
