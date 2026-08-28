#include "TunedBatch.hpp"

#include <chrono>
#include "Bandit.hpp"
#include "SharedBatchResources.hpp"
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
    for (const auto id: this->exclusive_assignment->at(context.get_group_thread_id(worker_id, thread_id))) {
        (*this->batch_start_trees)[id] = generate_tree(instance, StartingTree::parsimony, seeds[id], false);
        this->num_trees_generated.fetch_add(1);
    }

    // barrier so we dont start building tree-info objects without finished trees (since the thread assignment changes)
    context.enter_barrier();

    // create context for tree inference and assign the initial model
    for (const auto id: this->coarse_assignments.at(worker_id)) {
        if (meta_parameters->model_override.has_value()) {
            const auto model = Model(*meta_parameters->model_override);
            this->batch_trees[id][thread_id].emplace(opts, this->batch_start_trees->at(id), *this->msa,
                                                     *this->tip_msa_idmap, this->part_assignments->at(thread_id), &model);
        } else {
            this->batch_trees[id][thread_id].emplace(opts, this->batch_start_trees->at(id), *this->msa,
                                                     *this->tip_msa_idmap, this->part_assignments->at(thread_id));
            assign_models(batch_trees[id][thread_id].value(), *this->initial_model);
        }
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

void TunedBatch::optimize_nni(const Options &opts, SharedBatchResources &resources, const TaskGroup &context,
                              const unsigned int worker_id, const unsigned int thread_id) {
    const auto &tree_ids = this->coarse_assignments.at(worker_id);

    if (meta_parameters->nni_round) {
        for (const auto tree_id: tree_ids) {
            nni_round_.do_optimize(batch_trees[tree_id][thread_id], tree_id, opts, context, resources, worker_id, thread_id);
        }
    }
}

void TunedBatch::optimize_topology(const Options &opts, const TaskGroup &context, SharedBatchResources &resources,
                                   const unsigned int worker_id, const unsigned int thread_id) {
    const auto &tree_ids = this->coarse_assignments.at(worker_id);

    for (const auto tree_id: tree_ids) {
        fixed_spr_.do_optimize(batch_trees[tree_id][thread_id], tree_id, opts, context, resources, worker_id, thread_id);
    }
}

void TunedBatch::optimize_parameters(const Options &opts, SharedBatchResources &resources, const TaskGroup &context,
                                     const unsigned int worker_id, const unsigned int thread_id, double epsilon,
                                     const bool model, const bool branches, const bool force) {
    const auto &tree_ids = this->coarse_assignments.at(worker_id);

    for (const auto tree_id : tree_ids) {
        model_opt_.do_optimize(batch_trees[tree_id][thread_id], tree_id, opts, context, resources, worker_id, thread_id);
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
        if (meta_parameters->fallback_fast_raxml) {
            const auto &tree_ids = coarse_assignments.at(worker_id);
            for (const auto tree_id: tree_ids) {
                fast_raxml_.do_optimize(batch_trees[tree_id][thread_id], tree_id, opts, context, resources, worker_id, thread_id);
            }
        } else {
            // do initial model and branch length optimization
            if (!this->initial_model_optimized) {
                this->optimize_parameters(opts, resources, context, worker_id, thread_id, 3.0);

                // barrier required so initial_model_optimized isn't set before all threads optimized model
                context.enter_barrier();

                if (context.is_group_leader(worker_id, thread_id)) {
                    this->initial_model_optimized = true;
                }
            }

            // pre-optimize
            this->optimize_nni(opts, resources, context, worker_id, thread_id);

            // apply constraint
            if (meta_parameters->constrain) {
                // TODO remove explicit assignment here and move it to strategy construction
                if (context.is_group_leader(worker_id, thread_id)) {
                    constrain_.partition_assignments = *part_assignments;
                }
                context.enter_barrier();

                const auto my_trees = coarse_assignments.at(worker_id);
                for (const auto tree_id: my_trees) {
                    constrain_.do_optimize(this->batch_trees[tree_id][thread_id], tree_id, opts, context, resources, worker_id, thread_id);
                }
            }

            // compute all required SPR rounds
            this->optimize_topology(opts, context, resources, worker_id, thread_id);
        }
    }

    if (context.is_group_leader(worker_id, thread_id)) {
        LOG_INFO_TS << this->name << ": total batch time after heuristics: " << this->wall_time << "ms." << std::endl;
    }

    perform_plausibility_check(opts, resources, resources.is_initialized(context), context, worker_id, thread_id);
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

        for (const auto &pa: part_assignments->at(thread_id)) {
            thread_partition_view[pa.part_id] = tree_likelihood_vec[pa.part_id].data() + pa.start;
        }

        // calculate site likelihoods for the assigned sub-partitions
        batch_trees[tree_id][thread_id]->persite_loglh(thread_partition_view);
    }

    // replace with group barrier
    context.enter_barrier();

    // next, change the parallelization scheme to avoid splitting trees between workers. If we have more workers than
    // trees, this sucks, but currently AU doesn't support per-partition parallelization because that would require
    // synchronizing accesses to the bootstrap replicate likelihood sums.
    // we therefore use as many workers as possible with one thread each now.
    const std::vector<size_t> &tree_ids =
            initialized
                ? exclusive_assignment->at(context.get_group_thread_id(worker_id, thread_id))
                : au_assignment.at(context.get_group_thread_id(worker_id, thread_id));
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

void TunedBatch::perform_plausibility_check(const Options &opts, SharedBatchResources &resources,
                                            const bool initialized,
                                            const TaskGroup &context, const unsigned int worker_id,
                                            const unsigned int thread_id) {
    auto &au_test = resources.get_screening_test(context);

    if (context.is_group_leader(worker_id, thread_id)) {
        if (!initialized) {
            au_test.allocate_test_statistics(false);
        }
        au_test.replace_persite_loglh(reference_persite_loglh.size(), batch_persite_logh);
    }

    // TODO should we backup the less optimized model or just accept that we overspecify the model

    // reset model to original for AU test
    if (meta_parameters->model_override) {
        for (auto &tree_id: coarse_assignments.at(worker_id)) {
            batch_trees[tree_id][thread_id].emplace(opts, batch_trees[tree_id][thread_id]->tree(), *msa, *tip_msa_idmap,
                                                    part_assignments->at(thread_id));
        }
    }

    const auto begin = std::chrono::steady_clock::now();
    this->optimize_parameters(opts, resources, context, worker_id, thread_id, 0.1, true, true, true);

    this->perform_au_test(au_test, initialized, context, worker_id, thread_id);

    // no barrier required, since batch leader is the one who finishes the AU test
    if (context.is_group_leader(worker_id, thread_id)) {
        this->plausible_tree_count = 0;
        this->p_values.clear();

        auto first = au_test.get_p_values().begin();
        auto reference_p_count = 0;

        // count how many reference trees are plausible
        for (const auto last = au_test.get_p_values().begin() + reference_persite_loglh.size(); first != last; ++first) {
            if (*first > SIGNIFICANCE_LEVEL) {
                reference_p_count += 1;
            }
        }
        LOG_DEBUG_TS << "AU Test found " << reference_p_count << " plausible trees in the reference set." << std::endl;
        if (reference_p_count == 0) {
            LOG_WARN << "Warning: treeset search found strictly better tree than ML search. Plausible treeset no longer plausible." << std::endl;
        }

        // count how many inferred trees are plausible
        for (const auto last = au_test.get_p_values().end(); first != last; ++first) {
            this->p_values.push_back(*first);
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

void TunedBatch::update_meta_parameters(const shared_ptr<MetaParameters> &new_parameters) {
    // TODO the shared pointer is pre-shared with the strategies. But this means we must not re-assign it.
    *this->meta_parameters = *new_parameters;
    this->meta_parameters_set = true;
}

bool TunedBatch::is_compatible(const MetaParameters &new_parameters) const {
    // a batch that already optimized with these exact parameters cannot be reused for the same parameters again
    if (*this->meta_parameters == new_parameters) {
        return false;
    }

    // if the current parameters do the bare minimum, we can always continue with new parameters
    if (this->meta_parameters->accept_starting_trees) {
        return true;
        // however if we already did something, the other set need not do the bare minimum.
    } else if (new_parameters.accept_starting_trees) {
        return false;
    }

    // TODO: this method needs to be part of heuristics
    return false;

    // do not reuse batch if it was created with a different model
    // if ((this->num_fast_spr_performed > 0 || this->num_slow_spr_performed > 0) && this->meta_parameters->model_override
    //     != new_parameters.model_override) {
    //     return false;
    // }

    // if settings of the SPR rounds do not match, and we already completed some SPR rounds,
    // the new parameters cannot replace the current ones
    // if (this->num_fast_spr_performed > 0) {
    //     if (this->meta_parameters->keep_top_k_topol != new_parameters.keep_top_k_topol) {
    //         return false;
    //     }
    //     if (this->meta_parameters->max_adaptive_radius != new_parameters.max_adaptive_radius) {
    //         return false;
    //     }
    //
    //     if (this->meta_parameters->num_fast_spr > new_parameters.num_fast_spr) {
    //         return false;
    //     }
    // }
    //
    // if (this->num_slow_spr_performed > 0) {
    //     if (this->meta_parameters->keep_top_k_topol != new_parameters.keep_top_k_topol) {
    //         return false;
    //     }
    //     if (this->meta_parameters->max_adaptive_radius != new_parameters.max_adaptive_radius) {
    //         return false;
    //     }
    //
    //     // if we already completed some slow rounds, but the other parameter wants to do more fast rounds,
    //     // we reject, because order matters
    //     if (this->meta_parameters->num_fast_spr != new_parameters.num_fast_spr) {
    //         return false;
    //     }
    //
    //     if (this->meta_parameters->num_slow_spr > new_parameters.num_slow_spr) {
    //         return false;
    //     }
    // }

    // if the way the model is obtained doesn't match, the new parameters cannot replace the current ones
    // if (this->initial_model_optimized && this->meta_parameters->early_commit != new_parameters.early_commit) {
    //     return false;
    // }
    // if (this->initial_model_optimized && this->meta_parameters->skip_model != new_parameters.skip_model) {
    //     return false;
    // }
    //
    // return true;
}

void TunedBatch::backup_models(ModelMap &target) const {
    for (size_t thread_id = 0; thread_id < this->batch_trees[0].size(); ++thread_id) {
        for (size_t part_id: this->batch_trees[0][thread_id].value().parts_master()) {
            assign(target[part_id], this->batch_trees[0][thread_id].value(), part_id);
        }
    }
}

void TunedBatch::assign_batch_models(const ModelMap &other) {
    initial_model = make_shared<ModelMap>(other);
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

void TunedBatch::get_plausible_trees(std::vector<Tree> &buffer) const {
    // access to both p-values and topology backups has to be guarded
    auto guard = std::lock_guard(*this->topology_access.get());

    for (unsigned int i = 0; i < this->tree_topologies.size(); ++i) {
        if (this->p_values[i] > SIGNIFICANCE_LEVEL) {
            buffer.push_back(this->tree_topologies[i]);
        }
    }
}
