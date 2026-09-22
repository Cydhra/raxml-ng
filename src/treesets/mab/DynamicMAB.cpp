#include "DynamicMAB.hpp"

void register_mapping(std::unordered_map<MetaParameters, size_t> &outer_mapping,
                      std::unordered_map<MetaParameters, size_t> &inner_mapping,
                      MetaParameters &parameters,
                      size_t outer_id,
                      size_t inner_id) {
    assert(outer_mapping.find(parameters) == outer_mapping.end());

    outer_mapping.insert({parameters, outer_id});
    inner_mapping.insert({parameters, inner_id});
}

void DynamicMAB::register_new_arm(const std::string &name, const OuterArm &new_arm) {
    auto id = hierarchical_mab.emplace_back(name, new_arm);

    for (size_t inner_id = 0; inner_id < new_arm->num_bandits(); ++inner_id) {
        auto &parameters = new_arm->get_bandit(inner_id).get_parameters();
        register_mapping(outer_mapping, inner_mapping, parameters, id, inner_id);
    }
}

void DynamicMAB::add_bandit_to_arm(const Bandit<OuterArm> &arm, const size_t outer_id, MetaParameters &new_parameters,
                                   const std::string &name) {
    const auto inner_id = arm.get_parameters()->emplace_back(name, new_parameters);
    register_mapping(outer_mapping, inner_mapping, new_parameters, outer_id, inner_id);
}

void DynamicMAB::register_new_successor(const unsigned int level, const std::string &&name, const OuterArm &&new_arm) {
    successors.emplace_back(level, name, new_arm);
}

MetaParameters &DynamicMAB::select_next_bandit() {
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
    const auto current_level = hierarchical_mab.num_bandits();

    const auto &bandit = hierarchical_mab.get_best_bandit();
    const auto &name = bandit.get_name();

    const auto previous_successors_spawned = past_increases.find(name);
    unsigned int previous_modifications = 0;
    if (previous_successors_spawned != past_increases.end()) {
        previous_modifications += previous_successors_spawned->second;
    }

    // dont add expensive bandits if that didnt work before and we are already getting trees
    if (previous_modifications < 2 || bandit.get_expected_tree_rate() < 0.1) {
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

        previous_modifications++;
        past_increases[name] = previous_modifications;
    }
}

void DynamicMAB::propose_less_effort() {
    const auto &bandit = hierarchical_mab.get_best_bandit();
    const auto &inner_bandit = bandit.get_parameters()->get_best_bandit();
    const auto &parameters = inner_bandit.get_parameters();

    if (auto new_parameters = this->mutate(parameters); outer_mapping.find(new_parameters) == outer_mapping.end()) {
        LOG_WORKER_TS(LogLevel::info) << std::endl << "Mutating bandit " << inner_bandit.get_name() <<
                "." << std::endl;

        this->add_bandit_to_arm(bandit, outer_mapping[parameters], new_parameters, inner_bandit.get_name() + ",mut");

        last_mab_modification = hierarchical_mab.get_iterations_completed();
    }
}

MetaParameters DynamicMAB::mutate(const MetaParameters &parameters) {
    const auto elem = past_mutations.find(parameters);
    unsigned int mutation = 0;
    MetaParameters new_parameters = parameters;

    if (elem != past_mutations.end()) {
        mutation += elem->second;
    }

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
            if (parameters.any_rounds() && parameters.keep_top_k_topol > 1) {
                new_parameters.keep_top_k_topol /= 2;
                break;
            }
            [[fallthrough]];
        case 2:
            mutation += 1;
            if (parameters.any_spr_rounds() && parameters.max_adaptive_radius >= 6) {
                new_parameters.max_adaptive_radius -= 3;
                break;
            }
            [[fallthrough]];
        default: break;
    }

    past_mutations.insert({parameters, mutation});
    return new_parameters;
}
