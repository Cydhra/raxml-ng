#ifndef RAXML_NG_DYNAMICMAB_HPP
#define RAXML_NG_DYNAMICMAB_HPP
#include "MultiArmedBandit.hpp"

typedef std::shared_ptr<MultiArmedBandit<MetaParameters> > OuterArm;

constexpr unsigned int MIN_PAUSE_BETWEEN_MODIFICATIONS = 4;

/**
 * An opaque wrapper around a multi-level bandit that modifies the bandit during operation to restrict the search
 * space as much as possible.
 */
class DynamicMAB {
public:
    DynamicMAB() = default;

    /**
     * Add an arm to the dynamic bandit, which itself is a multi-armed bandit.
     *
     * @param name name of the MAB for display purposes
     * @param new_arm a MultiArmedBandit which itself acts as a bandit in this multi-level bandit
     */
    void register_new_arm(const std::string &name, const OuterArm &new_arm);

    void register_new_successor(unsigned int level, const std::string &&name, const OuterArm &&new_arm);

    /**
     * Select a set of parameters to infer a batch of trees with using the accumulated knowledge of its performance,
     * balancing it with exploring new bandits. Whenever the batch is inferred, `take_measurement` may
     * be called to record the performance of the selected parameters and modify the expected outcome of the bandit
     * which holds these parameters.
     *
     * @return A set of parameters which can be applied to a tuned batch to infer a batch of trees with.
     */
    MetaParameters &select_next_bandit();

    /**
     * Measure the performance of a set of `MetaParameters` previously chosen by this multi-armed bandit with
     * `select_next_bandit`.
     *
     * @param batch a batch of trees inferred with parameters taken from this MAB
     */
    void take_measurement(TunedBatch &batch);

protected:
    /**
     * The multi-level multi-armed bandit that is being managed and modified by this class.
     */
    MultiArmedBandit<OuterArm> hierarchical_mab;

    /**
     * A list of bandit arms that are successors to previous arms in case they are not yet optimal.
     * They are each given a rank, and are added if the best performing bandit does not have high success rate.
     * Each time, all bandits with a rank as high or lower than the number of currently active bandits are added.
     */
    std::vector<std::tuple<unsigned int, std::string, OuterArm> > successors;

    /**
     * Mapping of meta parameters to the bandit id in the outer MAB. This is used to take measurements of
     * the parameters' performance in the corresponding second-level bandit.
     */
    std::unordered_map<MetaParameters, size_t> outer_mapping = {};

    /**
     * Mapping of meta parameters to the bandit id in their respective inner MAB. This is used to take measurements of
     * the parameters' performance in the corresponding bandit.
     */
    std::unordered_map<MetaParameters, size_t> inner_mapping = {};

    // TODO get rid of this heuristic
    unsigned int last_mab_modification = 0;

    /**
     * Check whether we should insert new arms into the MAB depending on the performance of the current bandit arm.
     * This implements a heuristic that enables exploration for new arms if they have potential to be useful within
     * the algorithm.
     * This enables us to skip exploring arms that have no potential gain over currently explored arms.
     */
    void check_update();
};


#endif //RAXML_NG_DYNAMICMAB_HPP
