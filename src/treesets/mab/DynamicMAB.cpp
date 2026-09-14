#include "DynamicMAB.hpp"

void DynamicMAB::register_new_arm(const std::string &name, const OuterArm &new_arm) {
    const std::lock_guard<std::mutex> lock(state_mutex);
    register_new_arm_unlocked(name, new_arm);
}

void DynamicMAB::register_new_arm_unlocked(const std::string &name, const OuterArm &new_arm) {
    auto id = hierarchical_mab.emplace_back(name, new_arm);

    for (size_t inner_id = 0; inner_id < new_arm->num_bandits(); ++inner_id) {
        auto &parameters = new_arm->get_bandit(inner_id).get_parameters();
        assert(outer_mapping.find(parameters) == outer_mapping.end());

        outer_mapping.insert({parameters, id});
        inner_mapping.insert({parameters, inner_id});
    }
}

void DynamicMAB::register_new_successor(const unsigned int level, const std::string &&name, const OuterArm &&new_arm) {
    const std::lock_guard<std::mutex> lock(state_mutex);
    successors.emplace_back(level, name, new_arm);
}

MetaParameters &DynamicMAB::select_next_bandit() {
    const std::lock_guard<std::mutex> lock(state_mutex);
    const auto &mab = this->hierarchical_mab.select_next_bandit();
    const auto &current_bandit = mab.get_parameters().get()->select_next_bandit();
    return current_bandit.get_parameters();
}

void DynamicMAB::take_measurement(TunedBatch &batch) {
    const std::lock_guard<std::mutex> lock(state_mutex);
    auto &parameters = batch.get_parameters();

    // let outer bandit take the measurement
    auto &bandit = this->hierarchical_mab.get_bandit(outer_mapping[parameters]);
    this->hierarchical_mab.take_measurement(bandit, batch, true);

    // let the inner bandit take the measurement
    auto &inner_bandit = bandit.get_parameters()->get_bandit(inner_mapping[parameters]);
    bandit.get_parameters()->take_measurement(inner_bandit, batch, true);

    check_update();
}

bool DynamicMAB::disable(const MetaParameters &parameters) {
    const std::lock_guard<std::mutex> lock(state_mutex);
    const auto found = outer_mapping.find(parameters);
    if (found == outer_mapping.end())
        throw std::logic_error("cannot disable unregistered meta parameters");
    return hierarchical_mab.disable_bandit(found->second);
}

void DynamicMAB::check_update() {
    // TODO implement a proper heuristic here. For now, we check if the current arm exceeds 75% success per batch,
    //  and if not, we add arms according to a pre-defined mapping.

    // check if we have more than a trivial amount of data, if the success rate is low, and we had at least 4 batches since the last modification
    if (last_mab_modification + MIN_PAUSE_BETWEEN_MODIFICATIONS < hierarchical_mab.get_iterations_completed() &&
        hierarchical_mab.get_best_bandit().num_samples() >= 4) {
        if (hierarchical_mab.get_best_bandit().get_expected_tree_rate() < 0.75) {
            propose_more_effort();
        } else {
            propose_less_effort();
        }
    }
}

void DynamicMAB::propose_more_effort() {
    // Successor ranks describe Cydhra's staged inference progression. Initial
    // alternatives (including aggressive starting-tree sources) do not advance
    // that progression.
    const auto current_level = 1 + registered_successor_arms;

    // add all bandits of the current level
    for (auto it = this->successors.begin(); it != this->successors.end(); it += 1) {
        if (std::get<0>(*it) <= current_level) {
            if (!hierarchical_mab.has_bandit(std::get<1>(*it))) {
                LOG_WORKER_TS(LogLevel::info) << std::endl << "Adding bandit " << std::get<1>(*it) <<
                        " to algorithm." << std::endl;
                register_new_arm_unlocked(std::get<1>(*it), std::get<2>(*it));
                ++registered_successor_arms;
                last_mab_modification = hierarchical_mab.get_iterations_completed();
            }
        }
    }
}

void DynamicMAB::propose_less_effort() {
    const auto &bandit = hierarchical_mab.get_best_bandit();
    const auto &inner_bandit = bandit.get_parameters()->get_best_bandit();
    const auto &parameters = inner_bandit.get_parameters();

    if (auto new_parameters = this->mutate(parameters); outer_mapping.find(new_parameters) == outer_mapping.end()) {
        LOG_WORKER_TS(LogLevel::info) << std::endl << "Mutating bandit " << inner_bandit.get_name() <<
                "." << std::endl;

        auto outer_id = outer_mapping[parameters];
        auto inner_id = bandit.get_parameters()->emplace_back(inner_bandit.get_name() + ",mut", new_parameters);

        // TOdo move this behavior into dedicated method
        outer_mapping.insert({new_parameters, outer_id});
        inner_mapping.insert({new_parameters, inner_id});

        last_mab_modification = hierarchical_mab.get_iterations_completed();
    }
}

MetaParameters DynamicMAB::mutate(const MetaParameters &parameters) {
    if (parameters.accept_starting_trees)
        return parameters;

    const auto elem = past_mutations.find(parameters);
    unsigned int mutation = 0;
    MetaParameters new_parameters = parameters;

    if (elem != past_mutations.end()) {
        mutation += elem->second;
    }

    const bool does_any_rounds = parameters.num_fast_spr + parameters.num_slow_spr > 0 || parameters.dynamic_spr;

    switch (mutation) {
        case 0:
            mutation += 1;
            if (parameters.do_final_model) {
                new_parameters.do_final_model = false;
                break;
            }
            [[fallthrough]];
        case 1:
            mutation += 1;
            if (does_any_rounds && parameters.keep_top_k_topol > 1) {
                new_parameters.keep_top_k_topol /= 2;
                break;
            }
            [[fallthrough]];
        case 2:
            mutation += 1;
            if (parameters.nni_round && !parameters.constrain) {
                new_parameters.nni_round = false;
                break;
            }
            [[fallthrough]];
        case 3:
            mutation += 1;
            if (does_any_rounds && parameters.max_adaptive_radius >= 6) {
                new_parameters.max_adaptive_radius -= 3;
                break;
            }
            [[fallthrough]];
        default: break;
    }

    past_mutations.insert({parameters, mutation});
    return new_parameters;
}
