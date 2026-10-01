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

    bool ensure(const RaxmlInstance &instance, const SmartBarrier &barrier, unsigned int threads_per_worker, unsigned int worker_id, unsigned int thread_id, unsigned int num_trees) override;

    /**
     * @return mean estimate of time spent per tree on parsimony
     */
    [[nodiscard]] double amortized_time(unsigned int batch_size) const override;

private:
    /**
     * A mutex that the master-thread must hold during modification of the tree-list or the starting seed.
     */
    std::unique_ptr<std::mutex> tree_list_mutex = std::make_unique<std::mutex>();

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
