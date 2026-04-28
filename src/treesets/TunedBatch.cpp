#include "TunedBatch.hpp"

#include "../coraxlib/src/corax/optimize/opt_generic.h"
#include "../loadbalance/CoarseLoadBalancer.hpp"
#include <chrono>
#include "Bandit.hpp"
#include "Threadpool.hpp"

using namespace std::placeholders;

unsigned int TunedBatch::get_batch_size() const {
    return this->batch_start_trees->size();
}

void TunedBatch::generate_starting_trees(RaxmlInstance &instance, const Options &opts, const TaskGroup &context,
                                         const unsigned int worker_id, const unsigned int thread_id) {
    // time measurement
    const auto begin = std::chrono::steady_clock::now();

    // generate ascending seeds from a starting point to allow coordinating batch seeds reproducibly.
    intVector seeds(this->get_batch_size());
    std::iota(seeds.begin(), seeds.end(), this->starting_seed);

    // generate trees from seeds
    for (const auto id: this->exclusive_assignment.at(context.get_group_thread_id(worker_id, thread_id))) {
        (*this->batch_start_trees)[id] = generate_tree(instance, StartingTree::parsimony, seeds[id], false);
        this->num_trees_generated.fetch_add(1);
    }

    // barrier so we dont start building tree-info objects without finished trees (since the thread assignment changes)
    context.enter_barrier();

    // create context for tree inference and assign the initial model
    for (const auto id: this->coarse_assignments.at(worker_id)) {
        this->batch_trees[id][thread_id].emplace(opts, this->batch_start_trees->at(id), *this->msa,
                                                 this->tip_msa_idmap, this->part_assignments[thread_id]);
        assign_models(batch_trees[id][thread_id].value(), this->initial_model);
    }

    if (context.is_group_leader(worker_id, thread_id)) {
        const auto end = std::chrono::steady_clock::now();

        const unsigned int elapsed = static_cast<unsigned int>(std::chrono::duration_cast<
            std::chrono::milliseconds>(end - begin).count());
        this->wall_time += elapsed;

        LOG_INFO_TS << this->name << ": total batch time after generating starting trees: " << this->wall_time << "ms."
                <<
                std::endl;
    }
}

void TunedBatch::optimize_topology(const Options &opts, const TaskGroup &context, const unsigned int worker_id,
                                   const unsigned int thread_id) {
    const auto &tree_ids = this->coarse_assignments.at(worker_id);

    // copy current status into local variables. This is simpler than putting those states into atomic counters and add
    // barriers to their access
    auto current_spr_fast = this->num_fast_spr_performed;
    auto current_spr_slow = this->num_slow_spr_performed;

    while (this->meta_parameters->num_fast_spr > current_spr_fast || this->meta_parameters->num_slow_spr >
           current_spr_slow) {
        const auto fast = this->meta_parameters->num_fast_spr > current_spr_fast;
        const auto rounds_performed = fast ? current_spr_fast : current_spr_slow;
        auto total_rounds = fast ? meta_parameters->num_fast_spr : meta_parameters->num_slow_spr;
        auto num_rounds = total_rounds - rounds_performed;

        if (context.is_group_leader(worker_id, thread_id)) {
            auto round_name = fast ? "FAST" : "SLOW";

            LOG_INFO_TS << this->name << ": Optimizing topology (" << num_rounds << " of " << total_rounds << " total "
                    <<
                    round_name << " spr rounds)" << std::endl;

            // make sure the spr-params are set correctly for fast/slow rounds
            this->auto_configure(opts);
        }

        context.enter_barrier(); // required to propagate auto-configuration
        const auto begin = std::chrono::steady_clock::now();

        // run optimization kernel
        for (const auto tree_id: tree_ids) {
            for (unsigned int spr_round = rounds_performed; spr_round < total_rounds; ++spr_round) {
                // important: we create a copy of the parameters here and give spr_round a copy, not the shared reference.
                // This doesn't fix any issues or has any effect on the code as written (that I know of) because the
                // first thing the spr round does is copying the values into a per-thread local struct.
                // But if we don't do it, the spr_rounds desynchronize reliably if two threads work on the same tree.
                // I have no idea why, but creating a copy of the parameter here fixes it and is cheaper
                // than ritual sacrifice of a goat each syzygy.
                auto local_copy = spr_params;
                batch_trees[tree_id][thread_id].value().spr_round(local_copy);
                batch_trees[tree_id][thread_id].value().optimize_branches(1.0, 1);
            }

            LOG_WORKER_TS(LogLevel::debug) << "performed " << (total_rounds - rounds_performed)
                    << (spr_params.ntopol_keep < 20 ? " GREEDY" : " FAST") << " spr rounds (radius: " << spr_params.
                    radius_min
                    << ") for tree search #" << (tree_id + 1) << std::endl;
        }

        if (fast) {
            current_spr_fast = this->meta_parameters->num_fast_spr;
        } else {
            current_spr_slow = this->meta_parameters->num_slow_spr;
        }

        // update the TunedBatch status
        if (context.is_group_leader(worker_id, thread_id)) {
            const auto end = std::chrono::steady_clock::now();

            const unsigned int elapsed = static_cast<unsigned int>(std::chrono::duration_cast<
                std::chrono::milliseconds>(end - begin).count());
            this->wall_time += elapsed;

            if (fast) {
                this->num_fast_spr_performed = current_spr_fast;
            } else {
                this->num_slow_spr_performed = current_spr_slow;
            }
        }
    }
}

void TunedBatch::optimize_parameters(const TaskGroup &context, const unsigned int worker_id,
                                     const unsigned int thread_id, double epsilon, const bool model,
                                     const bool branches, const bool force) {
    const auto &tree_ids = this->coarse_assignments.at(worker_id);

    const auto opt_model = model && (!this->meta_parameters->skip_model || force);
    const auto opt_branches = branches;

    if (!force && meta_parameters->early_commit) {
        // force hyper-optimization if this batch is early-committing
        epsilon = 0.1;
    }

    const auto begin = std::chrono::steady_clock::now();

    if (opt_model && opt_branches) {
        if (context.is_group_leader(worker_id, thread_id)) {
            LOG_INFO_TS << this->name << ": Optimizing all params (eps: " << epsilon << ")" << std::endl;
        }

        // run all parameters optimization
        for (const auto tree_id: tree_ids) {
            batch_trees[tree_id][thread_id].value().optimize_params(CORAX_OPT_PARAM_ALL, epsilon);
        }
    } else if (opt_model) {
        if (context.is_group_leader(worker_id, thread_id)) {
            LOG_INFO_TS << this->name << ": Optimizing model (eps: " << epsilon << ")" << std::endl;
        }

        // run model optimization
        for (const auto tree_id: tree_ids) {
            batch_trees[tree_id][thread_id].value().optimize_model(epsilon);
        }
    } else if (branches) {
        if (context.is_group_leader(worker_id, thread_id)) {
            LOG_INFO_TS << this->name << ": Optimizing branches (eps: " << epsilon << ")" << std::endl;
        }

        // run model optimization
        for (const auto tree_id: tree_ids) {
            batch_trees[tree_id][thread_id].value().optimize_params(CORAX_OPT_PARAM_BRANCHES_ITERATIVE, epsilon);
        }
    }

    if (context.is_group_leader(worker_id, thread_id)) {
        if (!force) {
            const auto end = std::chrono::steady_clock::now();
            this->wall_time += static_cast<unsigned int>(std::chrono::duration_cast<
                std::chrono::milliseconds>(end - begin).count());
        }

        LOG_INFO_TS << this->name << ": Model Opt complete (eps: " << epsilon << ")" << std::endl;
    }
}

void TunedBatch::optimize(RaxmlInstance &instance, const Options &opts, SharedBatchResources &resources,
                          const TaskGroup &context, const unsigned int worker_id, const unsigned int thread_id) {
    if (!meta_parameters_set) {
        throw RaxmlException("TunedBatch has not been configured with meta heuristics");
    }

    if (!this->start_trees_generated()) {
        this->generate_starting_trees(instance, opts, context, worker_id, thread_id);
    }

    if (!meta_parameters->accept_starting_trees) {
        // do initial model and branch length optimization
        if (!this->initial_model_optimized) {
            this->optimize_parameters(context, worker_id, thread_id, 3.0);

            if (context.is_group_leader(worker_id, thread_id)) {
                this->initial_model_optimized = true;
            }
        }

        // compute all required SPR rounds
        this->optimize_topology(opts, context, worker_id, thread_id);
    }

    if (context.is_group_leader(worker_id, thread_id)) {
        LOG_INFO_TS << this->name << ": total batch time after heuristics: " << this->wall_time << "ms." << std::endl;
    }

    auto &au_test = resources.get_au_test(context);
    perform_plausibility_check(au_test, resources.is_initialized(context), context, worker_id, thread_id);
    resources.set_initialized(context);

    if (context.is_group_leader(worker_id, thread_id)) {
        auto guard = std::lock_guard(*this->topology_access.get());

        // clear previous backups
        this->tree_topologies.clear();

        // backup tree topologies so we can get the plausible trees on demand
        for (auto &batch_tree: this->batch_trees) {
            this->tree_topologies.push_back(batch_tree.at(0).value().tree());
        }
    }
}

void TunedBatch::perform_au_test(AuTest &au_test, const bool initialized, const TaskGroup &context,
                                 const unsigned int worker_id, const unsigned int thread_id) {
    const auto trees = this->coarse_assignments.at(worker_id);
    for (const auto tree_id: trees) {
        // collect the sub-partitions for the local worker
        auto &tree_likelihood_vec = batch_persite_logh[tree_id];
        std::vector<double *> thread_partition_view(msa.get()->part_count(), nullptr);

        for (const auto &pa: part_assignments.at(thread_id)) {
            thread_partition_view[pa.part_id] = tree_likelihood_vec[pa.part_id].data() + pa.start;
        }

        // calculate site likelihoods for the assigned sub-partitions
        batch_trees[tree_id][thread_id].value().persite_loglh(thread_partition_view);
    }

    // replace with group barrier
    context.enter_barrier();

    // next, change the parallelization scheme to avoid splitting trees between workers. If we have more workers than
    // trees, this sucks, but currently AU doesn't support per-partition parallelization because that would require
    // synchronizing accesses to the bootstrap replicate likelihood sums.
    // we therefore use as many workers as possible with one thread each now.
    const std::vector<size_t> &tree_ids =
            initialized ? exclusive_assignment.at(context.get_group_thread_id(worker_id, thread_id)) : au_assignment.at(context.get_group_thread_id(worker_id, thread_id));
    const unsigned int slice_start = initialized
                                         ? *tree_ids.begin() + reference_persite_loglh.size()
                                         : *tree_ids.begin();

    au_test.run_bootstrap(tree_ids.size(), slice_start);
    context.enter_barrier();

    if (context.is_group_leader(worker_id, thread_id)) {
        // guard the calculation of AU test p-values with a guard so we don't use partially updated p-values to obtain
        // tree topologies.
        auto guard = std::lock_guard(*this->topology_access.get());
        au_test.finalize_test_statistics();
        au_test.calculate_p_values();
    }
}

void TunedBatch::perform_plausibility_check(AuTest &au_test, const bool initialized,
                                            const TaskGroup &context, const unsigned int worker_id,
                                            const unsigned int thread_id) {
    if (context.is_group_leader(worker_id, thread_id)) {
        if (!initialized) {
            au_test.allocate_test_statistics(false);
        }
        au_test.replace_persite_loglh(reference_persite_loglh.size(), batch_persite_logh);
    }

    // TODO should we backup the less optimized model or just accept that we overspecify the model
    const auto begin = std::chrono::steady_clock::now();
    this->optimize_parameters(context, worker_id, thread_id, 0.1, true, true, true);

    this->perform_au_test(au_test, initialized, context, worker_id, thread_id);

    // no barrier required, since batch leader is the one who finishes the AU test
    if (context.is_group_leader(worker_id, thread_id)) {
        this->plausible_tree_count = 0;
        auto first = au_test.get_p_values().begin() + reference_persite_loglh.size();
        for (const auto last = au_test.get_p_values().end(); first != last; ++first) {
            if (*first > SIGNIFICANCE_LEVEL) {
                this->plausible_tree_count += 1;
            }
        }

        LOG_INFO_TS << "AU test found " << plausible_tree_count << " plausible trees for " << this->name << "." <<
                std::endl;

        const auto end = std::chrono::steady_clock::now();
        this->au_wall_time = static_cast<unsigned int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                end - begin)
            .count());
    }
}

void TunedBatch::update_meta_parameters(const Options &opts, const shared_ptr<MetaParameters> new_parameters) {
    this->meta_parameters = new_parameters;
    this->meta_parameters_set = true;
    this->auto_configure(opts);
}

bool TunedBatch::is_compatible(const MetaParameters &new_parameters) const {
    // a batch that already optimized with these exact parameters cannot be reused for the same parameters again
    if (*this->meta_parameters == new_parameters) {
        return false;
    }

    // if the current parameters do the bare minimum, we can always continue with new parameters
    if (this->meta_parameters->accept_starting_trees) {
        return true;
    }

    // if settings of the SPR rounds do not match, and we already completed some SPR rounds,
    // the new parameters cannot replace the current ones
    if (this->num_fast_spr_performed > 0) {
        if (this->meta_parameters->keep_top_k_topol != new_parameters.keep_top_k_topol) {
            return false;
        }

        if (this->meta_parameters->num_fast_spr > new_parameters.num_fast_spr) {
            return false;
        }
    }

    if (this->num_slow_spr_performed > 0) {
        if (this->meta_parameters->keep_top_k_topol != new_parameters.keep_top_k_topol) {
            return false;
        }

        // if we already completed some slow rounds, but the other parameter wants to do more fast rounds,
        // we reject, because order matters
        if (this->meta_parameters->num_fast_spr != new_parameters.num_fast_spr) {
            return false;
        }

        if (this->meta_parameters->num_slow_spr > new_parameters.num_slow_spr) {
            return false;
        }
    }

    // if the way the model is obtained doesn't match, the new parameters cannot replace the current ones
    if (this->initial_model_optimized && this->meta_parameters->early_commit != new_parameters.early_commit) {
        return false;
    }
    if (this->initial_model_optimized && this->meta_parameters->skip_model != new_parameters.skip_model) {
        return false;
    }

    return true;
}

void TunedBatch::backup_models(ModelMap &target) const {
    for (size_t part_id = 0; part_id < this->msa->part_count(); ++part_id) {
        // all threads have the same model, so backup from the first thread is sufficient
        assign(target[part_id], this->batch_trees[0][part_id].value(), part_id);
    }
}

void TunedBatch::assign_batch_models(const ModelMap &other) {
    initial_model = other;
}

void TunedBatch::finalize() {
    LOG_DEBUG << "Finalized " << name << "." << std::endl;

    // delete corax allocations
    this->batch_trees.clear();
}

unsigned int TunedBatch::elapsed_wall_time() const {
    return this->wall_time + this->au_wall_time;
}

unsigned int TunedBatch::get_plausible_tree_count() const {
    return this->plausible_tree_count;
}

Tree TunedBatch::get_tree(const unsigned int index) const {
    auto guard = std::lock_guard(*this->topology_access.get());

    if (this->tree_topologies.empty()) {
        throw RaxmlException("cannot obtain trees from non-optimized batch");
    }

    return this->tree_topologies[index];
}

std::vector<double> TunedBatch::get_tree_likelihoods() {
    auto result = std::vector<double>(this->batch_trees.size());

    for (unsigned int i = 0; i < this->batch_trees.size(); i++) {
        auto loglh = 0.0;
        for (auto &part: this->batch_trees[i]) {
            loglh += part.value().loglh();
        }
        result[i] = loglh;
    }

    return result;
}

doubleVector TunedBatch::get_p_values() const {
    return this->p_values;
}

void TunedBatch::get_plausible_trees(std::vector<Tree> &buffer) const {
    // access to both p-values and topology backups has to be guarded
    auto guard = std::lock_guard(*this->topology_access.get());

    for (unsigned int i = 0; i < this->tree_topologies.size(); ++i) {
        if (this->get_p_values()[this->reference_persite_loglh.size() + i] > SIGNIFICANCE_LEVEL) {
            buffer.push_back(this->tree_topologies[i]);
        }
    }
}
