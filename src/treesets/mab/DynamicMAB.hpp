#ifndef RAXML_NG_DYNAMICMAB_HPP
#define RAXML_NG_DYNAMICMAB_HPP
#include "MultiArmedBandit.hpp"

typedef std::shared_ptr<MultiArmedBandit<MetaParameters> > OuterArm;

constexpr unsigned int MIN_PAUSE_BETWEEN_MODIFICATIONS = 4;

/**
 * An opaque wrapper around a multi-level bandit that modifies the bandit during operation to restrict the search
 * space as much as possible.
 * It adds new arms to the multi-level bandit on demand, and attempts to modify existing arms by mutation.
 */
class DynamicMAB {
public:
    explicit DynamicMAB(const BanditPolicy policy) : policy(policy) {}

    /**
     * Add an arm to the dynamic bandit, which itself is a multi-armed bandit.
     *
     * @param name name of the MAB for display purposes
     * @param new_arm a MultiArmedBandit which itself acts as a bandit in this multi-level bandit
     */
    void register_new_arm(const std::string &name, const OuterArm &new_arm);

    /**
     * Add a new first level bandit configuration (MetaParameters) to an existing second-level bandit.
     *
     * @param arm existing second-level bandit
     * @param outer_id id of the existing second-level bandit
     * @param new_parameters new parameter object for a new first-level bandit
     * @param name name of the new first-level bandit
     */
    void add_bandit_to_arm(const Bandit<OuterArm> &arm, size_t outer_id, MetaParameters &new_parameters,
                           const std::string &name);

    /**
     * Add a bandit to the progressive successor heuristic, adding bandits unless no more speedup is being generated.
     *
     * @param level The amount of bandits that have to be present in the algorithm to consider adding this new bandit
     * @param name the name of this new bandit
     * @param new_arm the new bandit arm
     */
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
    void take_measurement(const TunedBatch &batch);

    /**
     * Disable the bandit that handles these parameters.
     *
     * @param parameters bandit parameters
     */
    void disable_bandit(const MetaParameters &parameters);

    /**
     * Set the upper bound to the largest achievable throughput.
     * If this is underestimated, bandits with higher throughputs cannot be distinguished as their success will be capped.
     *
     * @param throughput_bound Upper bound to throughput
     */
    void set_upper_bound(double throughput_bound);

    /**
     * @return the upper bound for attainable tree throughput of bandits
     */
    double get_upper_bound() const;

protected:
    /**
     * Which policy the bandits should have when added to the mab
     */
    BanditPolicy policy;

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

    /**
     * A cooldown between attempts to modify the MAB to ensure enough batches have been inferred to collect some data.
     */
    unsigned int last_mab_modification = 0;

    /**
     * Upper bound of bandit throughput
     */
    double upper_bound = 0;

    /**
     * How often the dynamic MAB has tried to increase effort from a bandit.
     * Used to prevent trying out more and more expensive bandits if the current bandit is better than the expensive ones.
     */
    std::unordered_map<std::string, unsigned int> past_increases = {};

    /**
     * How often the dynamic MAB has tried to mutate a bandit.
     * Used to select which mutation to try next.
     */
    std::unordered_map<MetaParameters, unsigned int> past_mutations = {};

    /**
     * Guards against concurrent modification while a bandit is being selected, or another modification of mappings
     * or mutations takes place.
     */
    std::unique_ptr<std::shared_mutex> modification_lock = std::make_unique<std::shared_mutex>();

    /**
     * Check whether we should insert new arms into the MAB depending on the performance of the current bandit arm.
     * This implements a heuristic that enables exploration for new arms if they have potential to be useful within
     * the algorithm.
     * This enables us to skip exploring arms that have no potential gain over currently explored arms.
     */
    void check_update();

    /**
     * Add a bandit to the MAB that tries to infer trees with more effort spent than the current best bandit
     *
     * @param exclusive exclusive lock of the internal read-write mutex
     */
    void propose_more_effort(std::lock_guard<std::shared_mutex> &exclusive);

    /**
     * Add a bandit to the MAB that tries to infer trees with less effort spent than the current best bandit
     *
     * @param exclusive exclusive lock of the internal read-write mutex
     */
    void propose_less_effort(std::lock_guard<std::shared_mutex> &exclusive);

    /**
     * Add a bandit to the MAB that is the next in the line of successor bandits. Used when proposing more effort,
     * or if all current bandits are deactivated.
     *
     * @param exclusive exclusive lock of the internal read-write mutex
     */
    void add_next_successor(std::lock_guard<std::shared_mutex> &exclusive);

    /**
     * Attempt to change the input parameters to reduce effort spent.
     * This method changes parameters which are not already being tested by the multi-level bandit scheme.
     *
     * @param parameters current bandit parameters that we attempt to mutate
     * @return New bandit parameters which are a modification of the input parameters
     */
    MetaParameters mutate(const MetaParameters &parameters, std::lock_guard<std::shared_mutex>&);

    /**
     * Add a new bandit arm to the outer MAB and register its bandits in the mappings.
     * This method requires proof of ownership of a lock guard of the internal mutex.
     * The equivalent public method calls this method after acquiring a lock.
     *
     * @param name bandit name
     * @param new_arm bandit arm
     * @param exclusive exclusive lock of the internal read-write mutex
     */
    void guarded_register_new_arm(const std::string &name, const OuterArm &new_arm, std::lock_guard<std::shared_mutex> &exclusive);

    /**
     * Add a new first level bandit configuration (MetaParameters) to an existing second-level bandit.
     * This method requires proof of ownership of a lock guard of the internal mutex.
     * The equivalent public method calls this method after acquiring a lock.
     *
     * @param arm existing second-level bandit
     * @param outer_id id of the existing second-level bandit
     * @param new_parameters new parameter object for a new first-level bandit
     * @param name name of the new first-level bandit
     */
    void guarded_add_bandit_to_arm(const Bandit<OuterArm> &arm, size_t outer_id, MetaParameters &new_parameters,
                           const std::string &name, std::lock_guard<std::shared_mutex> &exclusive);
};


#endif //RAXML_NG_DYNAMICMAB_HPP
