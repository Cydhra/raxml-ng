#include "DynamicMAB.hpp"

static void register_mapping(std::unordered_map<MetaParameters, size_t> &outer_mapping,
                             std::unordered_map<MetaParameters, size_t> &inner_mapping,
                             MetaParameters &parameters,
                             size_t outer_id,
                             size_t inner_id,
                             std::lock_guard<std::shared_mutex>&) {
    assert(outer_mapping.find(parameters) == outer_mapping.end());

    outer_mapping.insert({parameters, outer_id});
    inner_mapping.insert({parameters, inner_id});
}

void DynamicMAB::register_new_arm(const std::string &name, const OuterArm &new_arm) {
    std::lock_guard<std::shared_mutex> exclusive(*this->modification_lock);
    this->guarded_register_new_arm(name, new_arm, exclusive);
}

void DynamicMAB::guarded_register_new_arm(const std::string &name, const OuterArm &new_arm,
                                          std::lock_guard<std::shared_mutex> &exclusive) {
    const auto id = hierarchical_mab.emplace_back(name, new_arm, upper_bound);

    switch (this->policy) {
        case BanditPolicy::normal:
            hierarchical_mab.get_bandit(id).with_normal_distribution();
            break;
        case BanditPolicy::ratio:
            hierarchical_mab.get_bandit(id).with_ratio_distribution();
            break;
        case BanditPolicy::greedy:
            hierarchical_mab.get_bandit(id).with_greedy_confidence();
            break;
    }

    for (size_t inner_id = 0; inner_id < new_arm->num_bandits(); ++inner_id) {
        auto &parameters = new_arm->get_bandit(inner_id).get_parameters();
        register_mapping(outer_mapping, inner_mapping, parameters, id, inner_id, exclusive);
    }
}

void DynamicMAB::add_bandit_to_arm(const Bandit<OuterArm> &arm, const size_t outer_id, MetaParameters &new_parameters,
                                   const std::string &name) {
    std::lock_guard<std::shared_mutex> exclusive(*this->modification_lock);
    this->guarded_add_bandit_to_arm(arm, outer_id, new_parameters, name, exclusive);
}

void DynamicMAB::guarded_add_bandit_to_arm(const Bandit<OuterArm> &arm, size_t outer_id, MetaParameters &new_parameters, const std::string &name, std::lock_guard<std::shared_mutex> &exclusive) {
    const auto inner_id = arm.get_parameters()->emplace_back(name, new_parameters, upper_bound);
    register_mapping(outer_mapping, inner_mapping, new_parameters, outer_id, inner_id, exclusive);
}

void DynamicMAB::register_new_successor(const unsigned int level, const std::string &&name, const OuterArm &&new_arm) {
    std::lock_guard<std::shared_mutex> exclusive(*this->modification_lock);
    successors.emplace_back(level, name, new_arm);
}

MetaParameters &DynamicMAB::select_next_bandit() {
    this->modification_lock->lock_shared();
    const auto &mab = this->hierarchical_mab.select_next_bandit();
    const auto &current_bandit = mab.get_parameters().get()->select_next_bandit();
    this->modification_lock->unlock_shared();
    return current_bandit.get_parameters();
}

void DynamicMAB::take_measurement(const TunedBatch &batch) {
    modification_lock->lock_shared(); // read access to mappings, no write access to dynamic bandit required

    auto &parameters = batch.get_parameters();

    // let outer bandit take the measurement
    auto &bandit = this->hierarchical_mab.get_bandit(outer_mapping.at(parameters));
    this->hierarchical_mab.take_measurement(bandit, batch, true);

    // let the inner bandit take the measurement
    auto &inner_bandit = bandit.get_parameters()->get_bandit(inner_mapping.at(parameters));
    bandit.get_parameters()->take_measurement(inner_bandit, batch, true);

    // by unlocking the shared lock, and then in check_update() relocking an exclusive lock, we could potentially
    // introduce thread reordering, where multiple bandits take measurements before the update is performed.
    // However, since check_update acquires the lock before checking the new measurements, all staged calls to
    // check_update() see a valid state of the bandits, the first call would perform all updates, and later calls would
    // determine no more updates are required.
    modification_lock->unlock_shared();

    check_update();
}

void DynamicMAB::disable_bandit(const MetaParameters &parameters) {
    std::lock_guard<std::shared_mutex> exclusive(*this->modification_lock);
    const auto outer_bandit_id = this->outer_mapping.at(parameters);
    const auto &outer_bandit = this->hierarchical_mab.get_bandit(outer_bandit_id);

    const auto inner_bandit_id = this->inner_mapping.at(parameters);

    if (!outer_bandit.get_parameters()->disable_bandit(inner_bandit_id)) {
        if (!this->hierarchical_mab.disable_bandit(outer_bandit_id)) {
            add_next_successor(exclusive);
        }
    }
}

void DynamicMAB::set_upper_bound(const double throughput_bound) {
    this->upper_bound = throughput_bound;
}

double DynamicMAB::get_upper_bound() const {
    return this->upper_bound;
}

void DynamicMAB::check_update() {
    std::lock_guard<std::shared_mutex> exclusive(*this->modification_lock);
    // check if we have more than a trivial amount of data, if the success rate is low, and we had at least 4 batches since the last modification
    if (last_mab_modification + MIN_PAUSE_BETWEEN_MODIFICATIONS < hierarchical_mab.get_iterations_completed() &&
        hierarchical_mab.get_best_bandit().num_samples() >= 4) {
        if (hierarchical_mab.get_best_bandit().get_success_mean() < 0.75) {
            propose_more_effort(exclusive);
        } else {
            propose_less_effort(exclusive);
        }
    }
}

void DynamicMAB::propose_more_effort(std::lock_guard<std::shared_mutex> &exclusive) {
    const auto &bandit = hierarchical_mab.get_best_bandit();
    const auto &name = bandit.get_name();

    const auto previous_successors_spawned = past_increases.find(name);
    unsigned int previous_modifications = 0;
    if (previous_successors_spawned != past_increases.end()) {
        previous_modifications += previous_successors_spawned->second;
    }

    // dont add expensive bandits if that didnt work before and we are already getting trees
    if (previous_modifications < 2 || bandit.get_success_mean() < 0.1) {
        add_next_successor(exclusive);
        previous_modifications++;
        past_increases[name] = previous_modifications;
    }
}

void DynamicMAB::propose_less_effort(std::lock_guard<std::shared_mutex> &exclusive) {
    const auto &bandit = hierarchical_mab.get_best_bandit();
    const auto &inner_bandit = bandit.get_parameters()->get_best_bandit();
    const auto &parameters = inner_bandit.get_parameters();

    if (auto new_parameters = this->mutate(parameters, exclusive); outer_mapping.find(new_parameters) == outer_mapping.end()) {
        LOG_WORKER_TS(LogLevel::info) << std::endl << "Mutating bandit " << inner_bandit.get_name() <<
                "." << std::endl;

        this->guarded_add_bandit_to_arm(bandit, outer_mapping.at(parameters), new_parameters, inner_bandit.get_name() + ",mut", exclusive);

        last_mab_modification = hierarchical_mab.get_iterations_completed();
    }
}

void DynamicMAB::add_next_successor(std::lock_guard<std::shared_mutex> &exclusive) {
    const auto current_level = hierarchical_mab.num_bandits();

    for (auto it = this->successors.begin(); it != this->successors.end(); it += 1) {
        if (std::get<0>(*it) <= current_level) {
            if (!hierarchical_mab.has_bandit(std::get<1>(*it))) {
                LOG_WORKER_TS(LogLevel::info) << std::endl << "Adding bandit " << std::get<1>(*it) <<
                        " to algorithm." << std::endl;
                guarded_register_new_arm(std::get<1>(*it), std::get<2>(*it), exclusive);
                last_mab_modification = hierarchical_mab.get_iterations_completed();
            }
        }
    }
}

MetaParameters DynamicMAB::mutate(const MetaParameters &parameters, std::lock_guard<std::shared_mutex>&) {
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

    past_mutations[parameters] = mutation;
    return new_parameters;
}