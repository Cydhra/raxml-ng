#ifndef RAXML_TREESETOPTIMIZER_HPP_
#define RAXML_TREESETOPTIMIZER_HPP_

#include <vector>
#include "Bandit.hpp"
#include "../loadbalance/LoadBalancer.hpp"
#include "../Checkpoint.hpp"

// forward declaration of RaxmlInstance
struct RaxmlInstance;

constexpr unsigned int DEFAULT_BATCH_SIZE = 16;

class TreesetOptimizer {
protected:
    /**
     * Reference to the RAxML-ng instance which is required for all optimization steps
     */
    const RaxmlInstance &instance;

    /**
     * Reference to the user options which are required for all optimization steps
     */
    const Options &opts;

    /**
     * Reference to the RAxML-ng instance's tip id-map which is used for starting tree generation.
     */
    const IDVector &tip_msa_idmap;

    /**
     * Reference to the user-configured fine-grained load balancer used by the main RAxML instance.
     */
    const LoadBalancer &load_balancer;

    /**
     * The number of plausible trees to infer in total.
     */
    const unsigned int target_tree_count;

    /**
     * The batch size of trees to infer at once.
     */
    const unsigned int batch_size;

    /**
     * The current seed for starting tree generation. Offset that by the number of generated trees whenever it is used
     * to generate a batch of trees.
     */
    unsigned long long current_seed;

    /**
     * Pointer to the partitioned msa instance of the current RAxML-ng run
     */
    const std::shared_ptr<PartitionedMSA> msa;

    /**
     * Pointer to the persite_loglh vector of the reference tree run
     */
    const std::vector<std::vector<doubleVector> > persite_loglh;

    /**
     * List of all batches that are being inferred or were inferred by the optimizer. It is initialized with the
     * minimum required number of batches, and is extended whenever more batches are required.
     */
    std::vector<TunedBatch> batches;

    /**
     * Index of the next batch that is supposed to be optimized.
     * Every batch before the cursor is already finished.
     */
    unsigned int batch_cursor = 0;

    /**
     * List of all bandits registered in the current run. Each bandit is considered during inference.
     */
    std::vector<Bandit> bandits;

    /**
     * Return the starting seed for generating `num_trees` trees.
     */
    unsigned long generate_seed_for_trees(const unsigned int num_trees) {
        const unsigned long long current = this->current_seed;
        this->current_seed += num_trees;
        return current;
    }

    /**
     * Get the bandit that represents the distribution of plausible trees obtained from accepting starting trees.
     */
    Bandit &starting_tree_bandit() {
        return this->bandits[0];
    }

    /**
     * Prepare the initial TunedBatch instances we use for inference.
     * Because we have a number of trees we have to infer, we have a minimum number of batches required even if every
     * tree becomes plausible.
     * Those are inferred here, and they are used to infer initial guesses over meta-parameters, like the variance of
     * the underlying distribution of plausible trees.
     */
    void prepare_initial_batches(RaxmlInstance &instance, const Options &opts, LoadBalancer &load_balancer, const IDVector &tip_msa_idmap);

    /**
     * Initialize the bandit algorithms we use during the inference. These depend on the parameters derivded from initial
     * starting tree inference in `prepare_initial_batches()`
     */
    void initialize_bandits();

public:
    /**
     *
     * @param instance Reference to the RAxML-ng instance
     * @param opts Reference to the user options structure
     * @param tip_msa_idmap Reference to the tip ID mapping of the MSA.
     * @param load_balancer Reference to the user-configured fine-grained load balancer.
     * @param target_tree_count The number of plausible trees to infer
     * @param starting_seed the tree generating seed for the first tree. Subsequent seeds are incremented by one.
     */
    explicit TreesetOptimizer(const RaxmlInstance &instance,
                              const Options &opts,
                              const IDVector &tip_msa_idmap,
                              const LoadBalancer &load_balancer,
                              const unsigned int target_tree_count,
                              const unsigned long long starting_seed,
                              const std::shared_ptr<PartitionedMSA> msa,
                              const std::vector<std::vector<doubleVector> > persite_loglh) : instance(instance),
        opts(opts),
        tip_msa_idmap(tip_msa_idmap),
        load_balancer(load_balancer),
        target_tree_count(target_tree_count),
        batch_size(DEFAULT_BATCH_SIZE),
        current_seed(starting_seed),
        msa(msa),
        persite_loglh(persite_loglh) {
        // place the first bandit that represents the distribution of plausible starting trees
        this->bandits.emplace_back(MetaParameters(1, false, 0, true));
    }

    /**
     * Run the treeset optimizer to infer K plausible trees.
     */
    void run(RaxmlInstance &instance, Options &opts, LoadBalancer &load_balancer, const IDVector &tip_msa_idmap);
};


#endif //RAXML_TREESETOPTIMIZER_HPP_
