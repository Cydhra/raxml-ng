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
    auto &tree_ids = assignment_list.at(worker_id);

    const auto slice_start = *tree_ids.begin();

    tester.run_bootstrap(tree_ids.size(), slice_start);
    ParallelContext::global_barrier();
}

/**
 * Helper function to perform fine-grained parallelization on a given function.
 */
void fine_grained_parallel(const CoarseAssignmentList &assignment_list,
                           const std::function<void(unsigned int, unsigned int)> &kernel) {
    const unsigned int worker_id = ParallelContext::local_group_id();
    const unsigned int thread_id = ParallelContext::local_proc_id();
    auto &tree_ids = assignment_list.at(worker_id);

    for (const unsigned int tree_id: tree_ids) {
        kernel(thread_id, tree_id);
    }
}

/**
 * Convert a std::function that takes a thread_id and a tree_id into a kernel for a p-thread.
 *
 * @param assignment_list coarse load balance assignment for the trees in the batch
 * @param kernel the function that is called for each tree in the assignment
 * @return a functor that acts as a main function for a p-thread
 */
std::function<void()> make_kernel(const CoarseAssignmentList &assignment_list,
                                  const std::function<void(unsigned int, unsigned int)> &kernel) {
    return std::bind(fine_grained_parallel, assignment_list, kernel);
}

/**
 * Parallel kernel for the per-site log-likelihood calculation, given to make_kernel to create a pthread-main
 */
void sitelh_kernel(const PartitionedMSA &msa,
                   const PartitionAssignmentList &partition_assignment,
                   std::vector<std::vector<doubleVector> > &persite_loglh,
                   std::vector<std::vector<TreeInfo> > &batch_trees,
                   const unsigned int thread_id,
                   const unsigned int tree_id) {
    // collect the sub-partitions for the local worker
    auto &partitions = partition_assignment.at(thread_id);
    auto &tree_likelihood_vec = persite_loglh[tree_id];
    std::vector<double *> thread_partition_view(msa.part_count(), nullptr);

    for (const auto &pa: partitions) {
        thread_partition_view[pa.part_id] = tree_likelihood_vec[pa.part_id].data() + pa.start;
    }

    // calculate site likelihoods for the assigned sub-partitions
    batch_trees[tree_id][thread_id].persite_loglh(thread_partition_view);
}

/**
 * Parallel kernel for SPR rounds, given to make_kernel to create a pthread-main
 */
void spr_kernel(std::vector<std::vector<TreeInfo> > &batch_trees,
                spr_round_params &spr_params,
                const unsigned int num_spr_performed,
                const unsigned int target_num_spr,
                const unsigned int thread_id,
                const unsigned int tree_id) {
    for (unsigned int spr_round = num_spr_performed; spr_round < target_num_spr; ++spr_round) {
        batch_trees[tree_id][thread_id].spr_round(spr_params);
        batch_trees[tree_id][thread_id].optimize_branches(1.0, 1);
    }

    LOG_WORKER_TS(LogLevel::debug) << "performed " << (target_num_spr - num_spr_performed)
            << (spr_params.ntopol_keep < 20 ? " GREEDY" : " FAST") << " spr rounds (radius: " << spr_params.radius_min
            << ") for tree search #" << (tree_id + 1) << std::endl;
}

/**
 * Parallel kernel for model optimization, given to make_kernel to create a pthread-main
 */
void model_opt_kernel(std::vector<std::vector<TreeInfo> > &batch_trees,
                      const double epsilon,
                      const unsigned int thread_id,
                      const unsigned int tree_id) {
    batch_trees[tree_id][thread_id].optimize_model(epsilon);
}

/**
 * Parallel kernel for branch length optimization, given to make_kernel to create a pthread-main
 */
void blo_kernel(std::vector<std::vector<TreeInfo> > &batch_trees,
                const double epsilon,
                const unsigned int thread_id,
                const unsigned int tree_id) {
    batch_trees[tree_id][thread_id].optimize_params(CORAX_OPT_PARAM_BRANCHES_ITERATIVE, epsilon);
}

void TunedBatch::mark_p_values_dirty() {
    this->au_test_dirty = true;
}

unsigned int TunedBatch::get_batch_size() const {
    return this->batch_start_trees->size();
}

void TunedBatch::generate_starting_trees(RaxmlInstance &instance, const Options &opts, LoadBalancer &load_balancer,
                                         const IDVector &tip_msa_idmap) {
    this->mark_p_values_dirty();

    const auto begin = std::chrono::steady_clock::now();
    intVector seeds(this->get_batch_size());
    // generate ascending seeds from a starting point to allow coordinating batch seeds reproducibly.
    std::iota(seeds.begin(), seeds.end(), this->starting_seed);

    auto tree_builder = std::bind(thread_start_trees,
                                  std::ref(instance),
                                  std::ref(*this->batch_start_trees),
                                  StartingTree::parsimony,
                                  std::cref(seeds),
                                  0,
                                  false);

    // infer starting trees
    const auto tree_workers = min(this->num_threads, this->get_batch_size());
    ParallelContext::init_pthreads_custom(opts, tree_builder, tree_workers, tree_workers);
    tree_builder();
    ParallelContext::finalize_threads();

    // load balance using the current thread assignment
    PartitionAssignment part_sizes;

    /* init list of partition sizes */
    for (unsigned int i = 0; i < this->msa->part_list().size(); ++i) {
        auto pinfo = &this->msa->part_list()[i];
        part_sizes.assign_sites(i, 0, pinfo->length(), pinfo->model().clv_entry_size());
    }

    const auto threads_per_worker = this->num_threads_per_worker();
    this->part_assignments = load_balancer.get_all_assignments(part_sizes, threads_per_worker);

    // step 3: create context for tree inference
    for (unsigned int tree_id = 0; tree_id < this->get_batch_size(); ++tree_id) {
        this->batch_trees.emplace_back();
        for (unsigned int local_thread_id = 0; local_thread_id < threads_per_worker; ++local_thread_id) {
            this->batch_trees[tree_id].emplace_back(opts, this->batch_start_trees->at(tree_id), *this->msa,
                                                    tip_msa_idmap, this->part_assignments[local_thread_id]);
        }
    }

    const auto end = std::chrono::steady_clock::now();
    const unsigned int elapsed = static_cast<unsigned int>(std::chrono::duration_cast<
        std::chrono::milliseconds>(end - begin).count());
    this->wall_time += elapsed;

    LOG_INFO_TS << this->name << ": total batch time after generating starting trees: " << this->wall_time << "ms." <<
            std::endl;
}

void TunedBatch::optimize_topology(const Options &opts) {
    // make sure debug outputs are initialized
    unsigned long int total_moves, increasing_moves;

    // escaping doesn't matter because SPR params are only ever used in this function, and are always overwritten
    // ReSharper disable CppDFALocalValueEscapesFunction
    this->spr_params.total_moves = &total_moves;
    this->spr_params.increasing_moves = &increasing_moves;
    // ReSharper restore CppDFALocalValueEscapesFunction

    while (this->meta_parameters->num_fast_spr > this->num_fast_spr_performed || this->meta_parameters->num_slow_spr >
           this->num_slow_spr_performed) {
        const auto fast = this->meta_parameters->num_fast_spr > this->num_fast_spr_performed;
        auto num_rounds = fast
                              ? meta_parameters->num_fast_spr - num_fast_spr_performed
                              : meta_parameters->num_slow_spr - num_slow_spr_performed;
        auto total_rounds = fast ? meta_parameters->num_fast_spr : meta_parameters->num_slow_spr;
        auto round_name = fast ? "FAST" : "SLOW";

        LOG_INFO_TS << this->name << ": Optimizing topology (" << num_rounds << " of " << total_rounds << " total " <<
                round_name << " spr rounds)" << std::endl;

        // make sure the spr-params are set correctly for fast/slow rounds
        this->auto_configure(opts);

        // make sure the AU test is invalidated
        this->mark_p_values_dirty();

        const auto begin = std::chrono::steady_clock::now();
        const auto spr_worker = make_kernel(
            std::ref(this->coarse_assignments),
            std::bind(spr_kernel,
                      std::ref(this->batch_trees),
                      std::ref(this->spr_params),
                      this->num_fast_spr_performed,
                      this->meta_parameters->num_fast_spr,
                      _1, _2));
        ParallelContext::init_pthreads_custom(opts, spr_worker, num_threads, num_workers);
        spr_worker();
        ParallelContext::finalize_threads();

        const auto end = std::chrono::steady_clock::now();
        const unsigned int elapsed = static_cast<unsigned int>(std::chrono::duration_cast<
            std::chrono::milliseconds>(end - begin).count());
        this->wall_time += elapsed;

        // TODO this only works if checkpoints cannot recover tree states. When checkpointing is added, this mechanism needs
        //  to be changed
        if (fast) {
            num_fast_spr_performed = this->meta_parameters->num_fast_spr;
        } else {
            num_slow_spr_performed = this->meta_parameters->num_slow_spr;
        }
    }
}

void TunedBatch::optimize(const Options &opts) {
    if (!meta_parameters_set) {
        throw RaxmlException("TunedBatch has not been configured with meta heuristics");
    }

    // do initial model and branch length optimization
    if (!this->initial_model_optimized) {
        this->optimize_parameters(opts, 3.0);
        this->initial_model_optimized = true;
    }

    // compute all required SPR rounds
    this->optimize_topology(opts);

    LOG_INFO_TS << this->name << ": total batch time after optimization: " << this->wall_time << "ms." << std::endl;
}

unsigned int TunedBatch::perform_au_test(const Options &opts) {
    if (!this->au_test_dirty) {
        return this->plausible_tree_count;
    }

    this->au_test->reset_test_statistics();

    // compute per-site log-likelihood in parallel
    const auto sitelh_worker = make_kernel(
        std::ref(this->coarse_assignments),
        std::bind(sitelh_kernel,
                  std::ref(*this->msa),
                  std::ref(this->part_assignments),
                  std::ref(this->batch_persite_logh),
                  std::ref(this->batch_trees),
                  _1, _2)
    );
    ParallelContext::init_pthreads_custom(opts, sitelh_worker, num_threads, num_workers);
    sitelh_worker();
    ParallelContext::finalize_threads();

    // next, change the parallelization scheme to avoid splitting trees between workers. If we have more workers than
    // trees, this sucks, but currently AU doesn't support per-partition parallelization because that would require
    // synchronizing accesses to the bootstrap replicate likelihood sums.
    // we therefore use as many workers as possible with one thread each now.
    const unsigned int total_trees_au = reference_persite_loglh.size() + this->get_batch_size();
    const unsigned int max_assigned_workers = min(total_trees_au, this->num_threads);
    ContiguousCoarseLoadBalancer load_balancer;
    CoarseAssignment tree_ids(total_trees_au);
    std::iota(tree_ids.begin(), tree_ids.end(), 0);

    const auto assignment = load_balancer.get_all_assignments(tree_ids, max_assigned_workers);

    auto au_worker = std::bind(parallel_au_bootstrap, std::ref(*this->au_test), std::ref(assignment));
    ParallelContext::init_pthreads_custom(opts, au_worker, max_assigned_workers, max_assigned_workers);
    au_worker();

    this->au_test->finalize_test_statistics();
    this->au_test->calculate_p_values();

    // TODO: there is a bug here that forces us to detach, find it.
    ParallelContext::finalize_threads(true);

    // mark AU test as valid
    this->au_test_dirty = false;

    // count plausible trees
    this->plausible_tree_count = 0;
    auto first = this->au_test->get_p_values().begin() + reference_persite_loglh.size();
    for (const auto last = this->au_test->get_p_values().end(); first != last; ++first) {
        if (*first > 0.05) {
            this->plausible_tree_count += 1;
        }
    }

    LOG_WORKER_TS(LogLevel::progress) << "AU test found " << plausible_tree_count << " plausible trees." << std::endl;
    return plausible_tree_count;
}

unsigned int TunedBatch::perform_plausibility_check(const Options &opts) {
    // we need to save the model backup, for two reasons: we do not want to perform tree search on hyper-optimized
    // models to allow for shallower likelihood curves of slightly suboptimal models.
    // Further, multiple calls to is_plausible must not optimize the hyper-optimized model with low episolon again
    // to avoid numerical oscillation.
    this->backup_models();
    this->optimize_parameters(opts, 0.1, true, true, true);
    const unsigned int plausible_trees = this->perform_au_test(opts);
    this->load_batch_models();
    return plausible_trees;
}

void TunedBatch::optimize_parameters(const Options &opts, const double epsilon, const bool model, const bool branches,
                                     const bool force) {
    this->mark_p_values_dirty();

    if (model && (!this->meta_parameters->skip_model || force)) {
        LOG_INFO_TS << this->name << ": Optimizing model (eps: " << epsilon << ")" << std::endl;

        // optimize model and branch lengths, such that we get accurate site likelihoods.
        const auto model_worker = make_kernel(
            std::ref(this->coarse_assignments),
            std::bind(model_opt_kernel,
                      std::ref(this->batch_trees),
                      epsilon,
                      _1, _2)
        );
        ParallelContext::init_pthreads_custom(opts, model_worker, num_threads, num_workers);
        model_worker();
        ParallelContext::finalize_threads();
    } else {
        load_batch_models();
    }

    if (branches) {
        LOG_INFO_TS << this->name << ": Optimizing branches (eps: " << epsilon << ")" << std::endl;
        // optimize branches
        const auto branch_worker = make_kernel(
            std::ref(this->coarse_assignments),
            std::bind(blo_kernel,
                      std::ref(this->batch_trees),
                      epsilon,
                      _1, _2)
        );
        ParallelContext::init_pthreads_custom(opts, branch_worker, num_threads, num_workers);
        branch_worker();
        ParallelContext::finalize_threads();
    }
}

void TunedBatch::update_meta_parameters(const Options &opts, const shared_ptr<MetaParameters> &new_parameters) {
    this->meta_parameters = new_parameters;
    this->meta_parameters_set = true;
    this->auto_configure(opts);
}

bool TunedBatch::is_compatible(const MetaParameters &new_parameters) const {
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
    if (this->initial_model_optimized && this->meta_parameters->skip_model != new_parameters.skip_model) {
        return false;
    }

    return true;
}

void TunedBatch::backup_models() {
    for (unsigned int i = 0; i < this->get_batch_size(); ++i) {
        for (size_t part_id = 0; part_id < this->msa->part_count(); ++part_id) {
            // all threads have the same model, so backup from the first thread is sufficient
            assign(this->batch_models[i][part_id], batch_trees[i][0], part_id);
        }
    }
}

void TunedBatch::load_batch_models() {
    for (unsigned int i = 0; i < this->get_batch_size(); ++i) {
        for (unsigned int local_thread_id = 0; local_thread_id < this->num_threads_per_worker(); ++local_thread_id) {
            assign_models(batch_trees[i][local_thread_id], this->batch_models[i]);
        }
    }
}

void TunedBatch::assign_batch_models(const TunedBatch &other) {
    for (unsigned int i = 0; i < this->get_batch_size(); ++i) {
        for (size_t part_id = 0; part_id < this->msa->part_count(); ++part_id) {
            // all threads have the same model, so backup from the first thread is sufficient
            assign(this->batch_models[i][part_id], other.batch_trees[i][0], part_id);
        }
    }

    // if the batch trees are already generated, assign them the models
    if (this->batch_trees.size() > 0) {
        this->load_batch_models();
    }
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
    // Conversely, the batch_start_trees vector always contains the tree objects, whether they have been generated or not.
    return this->batch_trees.size() == this->batch_start_trees->size();
}

Tree TunedBatch::get_tree(const unsigned int index) const {
    return this->batch_trees.at(index).at(0).tree();
}
