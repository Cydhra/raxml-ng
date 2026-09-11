#include "DynamicMAB.hpp"

void DynamicMAB::register_new_arm(const std::string &name, const OuterArm &new_arm) {
    auto id = hierarchical_mab.emplace_back(name, new_arm);

    for (size_t inner_id = 0; inner_id < new_arm->num_bandits(); ++inner_id) {
        auto &parameters = new_arm->get_bandit(inner_id).get_parameters();
        assert(outer_mapping.find(parameters) == outer_mapping.end());

        outer_mapping.insert({parameters, id});
        inner_mapping.insert({parameters, inner_id});
    }
}

void DynamicMAB::register_new_successor(const unsigned int level, const std::string &&name, const OuterArm &&new_arm) {
    successors.emplace_back(level, name, new_arm);
}

MetaParameters & DynamicMAB::select_next_bandit() {
    const auto &mab = this->hierarchical_mab.select_next_bandit();
    const auto &current_bandit = mab.get_parameters().get()->select_next_bandit();
    return current_bandit.get_parameters();
}

void DynamicMAB::take_measurement(TunedBatch &batch) {
    auto &parameters = batch.get_parameters();

    // let outer bandit take the measurement
    auto &bandit = this->hierarchical_mab.get_bandit(outer_mapping[parameters]);
    this->hierarchical_mab.take_measurement(bandit, batch, true);

    // let the inner bandit take the measurement
    auto &inner_bandit = bandit.get_parameters()->get_bandit(inner_mapping[parameters]);
    bandit.get_parameters()->take_measurement(inner_bandit, batch, true);

    check_update();
}

void DynamicMAB::check_update() {
    // TODO implement a proper heuristic here. For now, we check if the current arm exceeds 75% success per batch,
    //  and if not, we add arms according to a pre-defined mapping.

    // check if we have more than a trivial amount of data, if the success rate is low, and we had at least 4 batches since the last modification
    if (last_mab_modification + MIN_PAUSE_BETWEEN_MODIFICATIONS < hierarchical_mab.get_iterations_completed() &&
        hierarchical_mab.get_best_bandit().num_samples() >= 4 && hierarchical_mab.get_best_bandit().
        get_expected_tree_rate() < 0.75) {
        const auto current_level = hierarchical_mab.num_bandits();

        // add all bandits of the current level
        for (auto it = this->successors.begin(); it != this->successors.end(); it += 1) {
            if (std::get<0>(*it) <= current_level) {
                if (!hierarchical_mab.has_bandit(std::get<1>(*it))) {
                    LOG_WORKER_TS(LogLevel::info) << std::endl << "Adding bandit " << std::get<1>(*it) <<
                            " to algorithm." << std::endl;
                    register_new_arm(std::get<1>(*it), std::get<2>(*it));
                    last_mab_modification = hierarchical_mab.get_iterations_completed();
                }
            }
        }
    }
}
