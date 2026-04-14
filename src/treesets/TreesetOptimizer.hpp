#ifndef RAXML_TREESETOPTIMIZER_HPP_
#define RAXML_TREESETOPTIMIZER_HPP_

#include <vector>
#include "Bandit.hpp"
#include "BatchQueue.hpp"
#include "MultiArmedBandit.hpp"
#include "../loadbalance/LoadBalancer.hpp"
#include "../Checkpoint.hpp"

constexpr unsigned int DEFAULT_BATCH_SIZE = 16;

class TreesetOptimizer {
protected:
    RaxmlInstance &instance;

    /**
     * Reference to the user options which are required for all optimization steps
     */
    const Options &opts;

    /**
     * Manages the available batches that can be used by the bandits.
     */
    BatchQueue batch_queue;

    /**
     * The number of plausible trees to infer in total.
     */
    const unsigned int target_tree_count;

    MultiArmedBandit<std::shared_ptr<MultiArmedBandit<MetaParameters> > > hierarchical_mab;

    std::shared_ptr<MultiArmedBandit<MetaParameters> > parsimony = std::make_shared<MultiArmedBandit<
        MetaParameters> >();

    /**
     * The multi-armed bandit instance that contains aggressive heuristics
     */
    std::shared_ptr<MultiArmedBandit<MetaParameters> > light_mab = std::make_shared<MultiArmedBandit<
        MetaParameters> >();

    /**
    * The multi-armed bandit instance that contains slower heuristics
    */
    std::shared_ptr<MultiArmedBandit<MetaParameters> > heavy_mab = std::make_shared<MultiArmedBandit<
        MetaParameters> >();

    /**
     * The multi-armed bandit instance that contains light heuristics with fast commitment: we do model optimization to 0.1 EPS before anything
     * to commit to the local minimum.
     */
    std::shared_ptr<MultiArmedBandit<MetaParameters> > commitment_mab = std::make_shared<MultiArmedBandit<
        MetaParameters> >();

    /**
     * A mapping of bandit arms that are successors to previous arms in case they are not yet optimal.
     * For example, the successors to the parsimony arm are the light and commitment arms, so if the parsimony arm
     * does not find enough plausible trees, the successor arms are added to the algorithm.
     */
    unordered_map<MultiArmedBandit<MetaParameters> *, std::vector<std::tuple<std::string, std::shared_ptr<MultiArmedBandit<MetaParameters> >>>> successors;

    /**
     * Get the bandit that represents the distribution of plausible trees obtained from accepting starting trees.
     */
    Bandit<MetaParameters> &starting_tree_bandit() {
        return this->light_mab->get_bandit(0);
    }

    /**
     * Prepare the initial TunedBatch instances we use for inference.
     * Because we have a number of trees we have to infer, we have a minimum number of batches required even if every
     * tree becomes plausible.
     * Those are inferred here, and they are used to infer initial guesses over meta-parameters, like the variance of
     * the underlying distribution of plausible trees.
     */
    void prepare_initial_batches();

    /**
     * Initialize the bandit algorithms we use during the inference. These depend on the parameters derivded from initial
     * starting tree inference in `prepare_initial_batches()`
     */
    void initialize_bandits();

    /**
     * TODO temporary method during refactoring to start threads
     */
    void run_batch(TunedBatch *batch);

public:
    /**
     *
     * @param instance Reference to the RAxML-ng instance
     * @param opts Reference to the user options structure
     * @param msa multiple sequence alignment
     * @param tip_msa_idmap Reference to the tip ID mapping of the MSA.
     * @param persite_loglh reference to the per-site log likelihoods of the originally computed trees
     * @param load_balancer Reference to the user-configured fine-grained load balancer.
     * @param target_tree_count The number of plausible trees to infer
     * @param starting_seed the tree generating seed for the first tree. Subsequent seeds are incremented by one.
     */
    TreesetOptimizer(RaxmlInstance &instance,
                     Options &opts,
                     const std::shared_ptr<PartitionedMSA> &msa,
                     IDVector &tip_msa_idmap,
                     const std::vector<std::vector<doubleVector> > &persite_loglh,
                     LoadBalancer &load_balancer,
                     const unsigned int target_tree_count,
                     const unsigned long long starting_seed) : instance(instance),
                                                               opts(opts),
                                                               batch_queue(
                                                                   instance, opts, tip_msa_idmap, load_balancer, msa,
                                                                   persite_loglh, starting_seed, DEFAULT_BATCH_SIZE),
                                                               target_tree_count(target_tree_count) {
        // place the first bandit that represents the distribution of plausible starting trees
        this->light_mab->emplace_back("Parsimony", MetaParameters(1, true, 0, 0, true, false));
    }

    /**
     * Run the treeset optimizer to infer K plausible trees.
     */
    void run();

    /**
     * Check whether we should insert new arms into the MAB depending on the performance of the current bandit arm.
     * This implements a heuristic that enables exploration for new arms if they have potential to be useful within
     * the algorithm.
     * This enables us to skip exploring arms that have no potential gain over currently explored arms.
     *
     * @param current_arm the arm of the MAB that was last modified
     */
    void check_mab_modification(const Bandit<shared_ptr<MultiArmedBandit<MetaParameters>>> &current_arm);

    /**
     * Obtain all trees (plausible and rejected) inferred during the treeset optimization into a common vector and
     * return the vector.
     * This does not include the reference trees.
     *
     * @return A new vector instance containing all tree topologies.
     */
    std::vector<Tree> get_all_trees() const;

    /**
     * Obtain all plausible trees into a common vector and return the vector.
     * This does not include the reference trees.
     *
     * @return A new vector instance containing all plausible tree topologies.
     */
    std::vector<Tree> get_plausible_trees() const;
};


#endif //RAXML_TREESETOPTIMIZER_HPP_
