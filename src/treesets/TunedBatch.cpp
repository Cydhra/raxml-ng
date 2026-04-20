#include "TunedBatch.hpp"

#include "../coraxlib/src/corax/optimize/opt_generic.h"
#include "../loadbalance/CoarseLoadBalancer.hpp"
#include <chrono>
#include "Bandit.hpp"

using namespace std::placeholders;

/**
 * Ratio of plausible sets a batch needs to achieve to consider the thread parameters sufficient for inference
 */
constexpr double ACCEPT_TUNING_THRESHOLD = 0.9;

/**
 * Parallel kernel of the AU test bootstrapping, given to pthreads as their main function.
 * @param tester AuTest instance
 * @param assignment_list assignment of trees to workers
 */
void parallel_au_bootstrap(AuTest &tester, const CoarseAssignmentList &assignment_list) {
    const unsigned int worker_id = ParallelContext::local_group_id();
    const unsigned int thread_id = ParallelContext::local_thread_id();
    const unsigned int threads_per_worker = ParallelContext::threads_per_group();
    const unsigned int virtual_worker_id = worker_id * threads_per_worker + thread_id;
    auto &tree_ids = assignment_list.at(virtual_worker_id);

    const auto slice_start = *tree_ids.begin();

    tester.run_bootstrap(tree_ids.size(), slice_start);

    // todo replace with group barrier
    ParallelContext::global_barrier();
}

void TunedBatch::mark_p_values_dirty() {
    this->au_test_dirty = true;
}

unsigned int TunedBatch::get_batch_size() const {
    return this->batch_start_trees->size();
}

void TunedBatch::generate_starting_trees(RaxmlInstance &instance, const Options &opts) {
    const unsigned int thread_id = ParallelContext::local_thread_id();
    const unsigned int worker_id = ParallelContext::local_group_id();
    const unsigned int group_worker_id = worker_id * num_threads_per_worker() + thread_id;
    const bool thread_leader = thread_id == 0 && worker_id == 0;

    if (thread_leader) {
        this->mark_p_values_dirty();
    }

    // time measurement
    const auto begin = std::chrono::steady_clock::now();

    // generate ascending seeds from a starting point to allow coordinating batch seeds reproducibly.
    intVector seeds(this->get_batch_size());
    std::iota(seeds.begin(), seeds.end(), this->starting_seed);

    // generate trees from seeds
    for (const auto id: this->exclusive_assignment.at(group_worker_id)) {
        (*this->batch_start_trees)[id] = generate_tree(instance, StartingTree::parsimony, seeds[id], false);
    }

    // barrier so we dont start building tree-info objects without finished trees (since the thread assignment changes)
    // TODO replace with task group barrier
    ParallelContext::global_barrier();

    // create context for tree inference and assign the initial model
    for (const auto id: this->coarse_assignments.at(worker_id)) {
        this->batch_trees[id][thread_id].emplace(opts, this->batch_start_trees->at(id), *this->msa,
                                                        this->tip_msa_idmap, this->part_assignments[thread_id]);
        assign_models(batch_trees[id][thread_id].value(), this->initial_model);
    }

    if (thread_leader) {
        const auto end = std::chrono::steady_clock::now();

        const unsigned int elapsed = static_cast<unsigned int>(std::chrono::duration_cast<
            std::chrono::milliseconds>(end - begin).count());
        this->wall_time += elapsed;

        LOG_INFO_TS << this->name << ": total batch time after generating starting trees: " << this->wall_time << "ms."
                <<
                std::endl;
    }
}

void TunedBatch::optimize_topology(const Options &opts) {
    const unsigned int thread_id = ParallelContext::local_thread_id();
    const unsigned int worker_id = ParallelContext::local_group_id();
    const auto &tree_ids = this->coarse_assignments.at(worker_id);
    const bool batch_leader = worker_id == 0 && thread_id == 0;

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

        if (batch_leader) {
            auto round_name = fast ? "FAST" : "SLOW";

            LOG_INFO_TS << this->name << ": Optimizing topology (" << num_rounds << " of " << total_rounds << " total "
                    <<
                    round_name << " spr rounds)" << std::endl;

            // make sure the spr-params are set correctly for fast/slow rounds
            this->auto_configure(opts);

            // make sure the AU test is invalidated
            this->mark_p_values_dirty();
        }

        // Todo replace with group barrier
        ParallelContext::global_barrier(); // required to propagate auto-configuration
        const auto begin = std::chrono::steady_clock::now();

        // run optimization kernel
        for (const auto tree_id: tree_ids) {
            for (unsigned int spr_round = rounds_performed; spr_round < total_rounds; ++spr_round) {
                batch_trees[tree_id][thread_id].value().spr_round(spr_params);
                batch_trees[tree_id][thread_id].value().optimize_branches(1.0, 1);
            }

            LOG_WORKER_TS(LogLevel::debug) << "performed " << (total_rounds - rounds_performed)
                    << (spr_params.ntopol_keep < 20 ? " GREEDY" : " FAST") << " spr rounds (radius: " << spr_params.radius_min
                    << ") for tree search #" << (tree_id + 1) << std::endl;
        }

        if (fast) {
            current_spr_fast = this->meta_parameters->num_fast_spr;
        } else {
            current_spr_slow = this->meta_parameters->num_slow_spr;
        }

        // update the TunedBatch status
        if (batch_leader) {
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

void TunedBatch::optimize_parameters(double epsilon, const bool model, const bool branches, const bool force) {
    const unsigned int thread_id = ParallelContext::local_thread_id();
    const unsigned int worker_id = ParallelContext::local_group_id();
    const auto &tree_ids = this->coarse_assignments.at(worker_id);
    const bool batch_leader = worker_id == 0 && thread_id == 0;

    if (batch_leader) {
        this->mark_p_values_dirty();
    }

    const auto opt_model = model && (!this->meta_parameters->skip_model || force);
    const auto opt_branches = branches;

    if (!force && meta_parameters->early_commit) {
        // force hyper-optimization if this batch is early-committing
        epsilon = 0.1;
    }

    if (opt_model && opt_branches) {
        if (batch_leader) {
            LOG_INFO_TS << this->name << ": Optimizing all params (eps: " << epsilon << ")" << std::endl;
        }

        // run all parameters optimization
        for (const auto tree_id: tree_ids) {
            batch_trees[tree_id][thread_id].value().optimize_params(CORAX_OPT_PARAM_ALL, epsilon);
        }
    } else if (opt_model) {
        if (batch_leader) {
            LOG_INFO_TS << this->name << ": Optimizing model (eps: " << epsilon << ")" << std::endl;
        }

        // run model optimization
        for (const auto tree_id: tree_ids) {
            batch_trees[tree_id][thread_id].value().optimize_model(epsilon);
        }
    } else if (branches) {
        if (batch_leader) {
            LOG_INFO_TS << this->name << ": Optimizing branches (eps: " << epsilon << ")" << std::endl;
        }

        // run model optimization
        for (const auto tree_id: tree_ids) {
            batch_trees[tree_id][thread_id].value().optimize_params(CORAX_OPT_PARAM_BRANCHES_ITERATIVE, epsilon);
        }
    }

    if (batch_leader) {
        LOG_INFO_TS << this->name << ": Model Opt complete (eps: " << epsilon << ")" << std::endl;
    }
}

void TunedBatch::optimize(RaxmlInstance &instance, const Options &opts) {
    const unsigned int thread_id = ParallelContext::local_thread_id();
    const unsigned int worker_id = ParallelContext::local_group_id();
    const bool batch_leader = worker_id == 0 && thread_id == 0;

    if (!meta_parameters_set) {
        throw RaxmlException("TunedBatch has not been configured with meta heuristics");
    }

    if (!this->start_trees_generated()) {
        this->generate_starting_trees(instance, opts);
    }

    if (!meta_parameters->accept_starting_trees) {
        // do initial model and branch length optimization
        if (!this->initial_model_optimized) {
            this->optimize_parameters(3.0);

            if (batch_leader) {
                this->initial_model_optimized = true;
            }
        }

        // compute all required SPR rounds
        this->optimize_topology(opts);
    }

    if (batch_leader) {
        LOG_INFO_TS << this->name << ": total batch time after heuristics: " << this->wall_time << "ms." << std::endl;
    }

    perform_plausibility_check();

    if (batch_leader) {
        auto guard = std::lock_guard(*this->topology_access.get());

        // backup tree topologies so we can get the plausible trees on demand
        for (auto &batch_tree: this->batch_trees) {
            this->tree_topologies.push_back(batch_tree.at(0).value().tree());
        }
    }
}

void TunedBatch::perform_au_test() {
    const unsigned int thread_id = ParallelContext::local_thread_id();
    const unsigned int worker_id = ParallelContext::local_group_id();
    const bool batch_leader = worker_id == 0 && thread_id == 0;

    if (!this->au_test_dirty) {
        return;
    }

    if (batch_leader) {
        this->au_test->reset_test_statistics();
    }

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
    ParallelContext::global_barrier();

    // next, change the parallelization scheme to avoid splitting trees between workers. If we have more workers than
    // trees, this sucks, but currently AU doesn't support per-partition parallelization because that would require
    // synchronizing accesses to the bootstrap replicate likelihood sums.
    // we therefore use as many workers as possible with one thread each now.
    parallel_au_bootstrap(*au_test, au_assignment);

    if (batch_leader) {
        this->au_test->finalize_test_statistics();
        this->au_test->calculate_p_values();

        // mark AU test as valid
        this->au_test_dirty = false;
    }
}

void TunedBatch::perform_plausibility_check() {
    const unsigned int thread_id = ParallelContext::local_thread_id();
    const unsigned int worker_id = ParallelContext::local_group_id();
    const bool batch_leader = worker_id == 0 && thread_id == 0;

    // TODO should we backup the less optimized model or just accept that we overspecify the model
    this->optimize_parameters(0.1, true, true, true);

    this->perform_au_test();

    // no barrier required, since batch leader is the one who finishes the AU test
    if (batch_leader) {
        this->plausible_tree_count = 0;
        auto first = this->au_test->get_p_values().begin() + reference_persite_loglh.size();
        for (const auto last = this->au_test->get_p_values().end(); first != last; ++first) {
            if (*first > SIGNIFICANCE_LEVEL) {
                this->plausible_tree_count += 1;
            }
        }

        LOG_INFO_TS << "AU test found " << plausible_tree_count << " plausible trees for " << this->name << "." <<
                std::endl;
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
    this->au_test->free_test_statistics();

    // delete corax allocations
    this->batch_trees.clear();
}

unsigned int TunedBatch::elapsed_wall_time() const {
    return this->wall_time;
}

unsigned int TunedBatch::get_plausible_tree_count() const {
    if (au_test->is_finished() && !au_test_dirty) {
        return this->plausible_tree_count;
    }

    throw new RaxmlException("current p-values are dirty");
}

bool TunedBatch::start_trees_generated() const {
    // generating the starting trees will initialize the TreeInfo objects in this->batch_trees, which is otherwise empty.
    // note that this method is not thread save, because it assumes the vector is either initialized fully, or not at all,
    // meaning a partial initialization during starting tree generation will lead to unpredictable behavior
    return this->batch_trees.front().front().has_value();
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

std::vector<double> &TunedBatch::get_p_values() const {
    return this->au_test->get_p_values();
}
