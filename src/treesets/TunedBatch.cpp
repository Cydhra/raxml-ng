#include "TunedBatch.hpp"

#include "../coraxlib/src/corax/optimize/opt_generic.h"
#include "../loadbalance/CoarseLoadBalancer.hpp"

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
    unsigned int worker_id = ParallelContext::local_thread_id();
    auto &tree_ids = assignment_list.at(worker_id);

    const auto slice_start = *tree_ids.begin();

    tester.run_bootstrap(tree_ids.size(), slice_start);
    ParallelContext::global_barrier();
}

/**
 * Count the plausible trees in a range of p-values given through two iterators
 * @param first starting p-value (inclusive)
 * @param last end iterator state (exclusive)
 */
template<class Iter>
static unsigned int count_plausible_trees(Iter first, Iter last) {
    unsigned int unrejected_trees = 0;
    for (; first != last; ++first) {
        if (*first > 0.05) {
            unrejected_trees += 1;
        }
    }

    return unrejected_trees;
}

unsigned int TunedBatch::get_batch_size() const {
    return this->batch_start_trees->size();
}

void TunedBatch::generate_starting_trees(RaxmlInstance &instance, const Options &opts, LoadBalancer &load_balancer,
                                         const IDVector &tip_msa_idmap) {
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
    auto tree_workers = min(this->num_threads, this->get_batch_size());
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

    this->part_assignment = load_balancer.get_all_assignments(part_sizes, this->num_threads_per_worker());

    // step 3: create context for tree inference
    for (unsigned int i = 0; i < this->get_batch_size(); ++i) {
        this->batch_trees.push_back(TreeInfo(opts, this->batch_start_trees->at(i), *this->msa, tip_msa_idmap,
                                             part_sizes));
    }
}

void TunedBatch::infer_batch(RaxmlInstance &instance, const Options &opts, LoadBalancer &load_balancer,
                             const IDVector &tip_msa_idmap) {
    LOG_INFO_TS << "Running inference batch [BLO: " << !this->light_spr << ", MO: " << !this->skip_model << ", SPR: " <<
            this->target_num_spr << "] with " << this->num_threads << " threads." << std::endl;

    // do initial model and branch length optimization
    if (num_spr_performed == 0) {
        this->optimize_all_parameters(3.0);
    }

    if (this->light_spr) {
        this->spr_params.ntopol_keep = 1;
    }

    // TODO parallelize
    for (unsigned int i = 0; i < this->get_batch_size(); ++i) {
        for (unsigned int spr_round = num_spr_performed; spr_round < this->target_num_spr; ++spr_round) {
            LOG_PROGRESS(this->batch_trees[i].loglh()) << (light_spr ? "GREEDY" : "FAST") << " spr round " << (spr_round + 1) << " (radius: " << spr_params.radius_min << ") for treesearch #" << (i + 1) << std::endl;
            this->batch_trees[i].spr_round(this->spr_params);
        }
    }

    // TODO this only works if checkpoints cannot recover tree states. When checkpointing is added, this mechanism needs
    //  to be changed
    num_spr_performed = this->target_num_spr;
}

unsigned int TunedBatch::perform_au_test(const Options &opts) {
    LOG_INFO_TS << "Running AU test for batch [BLO: " << !this->light_spr << ", MO: "
            << !this->skip_model << ", SPR: " << this->target_num_spr << "] with "
            << this->num_threads << " threads." << std::endl;

    this->au_test->reset_test_statistics();

    // TODO paralellelize (for per-site lnl calculation only)
    // first, calculate per-site loglikelihoods of the batch trees
    for (unsigned int i = 0; i < this->get_batch_size(); ++i) {
        // collect the sub-partitions for the local worker
        // auto& thread_assignment = part_assignment->at(ParallelContext::local_proc_id());
        auto &tree_likelihood_vec = batch_persite_logh[i];
        std::vector<double *> thread_partition_view(msa->part_count(), nullptr);

        // TODO mind thread assignment
        // for (const auto& pa: thread_assignment)
        // thread_partition_view[pa.part_id] = tree_likelihood_vec[pa.part_id].data() + pa.start;

        for (unsigned int part = 0; part < msa->part_count(); part++) {
            thread_partition_view[part] = tree_likelihood_vec[part].data();
        }

        // calculate site likelihoods for the assigned sub-partitions
        batch_trees[i].persite_loglh(thread_partition_view);
    }

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

    const unsigned int plausible_trees = count_plausible_trees(
        this->au_test->get_p_values().begin() + reference_persite_loglh.size(), this->au_test->get_p_values().end());
    LOG_WORKER_TS(LogLevel::progress) << "AU test found " << plausible_trees << " plausible trees." << std::endl;
    return plausible_trees;
}

bool TunedBatch::is_plausible(const Options &opts) {
    // we need to save the model backup, for two reasons: we do not want to perform tree search on hyper-optimized
    // models to allow for shallower likelihood curves of slightly suboptimal models.
    // Further, multiple calls to is_plausible must not optimize the hyper-optimized model with low episolon again
    // to avoid numerical oscillation.
    this->save_model_backup();
    this->optimize_all_parameters(0.1, true);
    const unsigned int plausible_trees = this->perform_au_test(opts);
    this->restore_model_backup();
    return plausible_trees >= static_cast<unsigned int>(static_cast<double>(this->get_batch_size()) * ACCEPT_TUNING_THRESHOLD);
}

void TunedBatch::optimize_all_parameters(const double epsilon, const bool force) {
    // TODO parallelize over trees
    for (unsigned int i = 0; i < this->get_batch_size(); ++i) {
        if (!this->skip_model || force) {
            // optimize model and branch lengths, such that we get accurate site likelihoods.
            batch_trees[i].optimize_model(epsilon);
        } else {
            restore_model_backup();
        }

        // optimize branches
        batch_trees[i].optimize_params(CORAX_OPT_PARAM_BRANCHES_ITERATIVE, epsilon);
    }
}

void TunedBatch::save_model_backup() {
    for (unsigned int i = 0; i < this->get_batch_size(); ++i) {
        for (size_t part_id = 0; part_id < this->msa->part_count(); ++part_id) {
            assign(this->batch_model_backup[i][part_id], batch_trees[i], part_id);
        }
    }
}

void TunedBatch::restore_model_backup() {
    for (unsigned int i = 0; i < this->get_batch_size(); ++i) {
        assign_models(batch_trees[i], this->batch_model_backup[i]);
    }
}

void TunedBatch::inherit_model(const TunedBatch &other) {
    for (unsigned int i = 0; i < this->get_batch_size(); ++i) {
        for (size_t part_id = 0; part_id < this->msa->part_count(); ++part_id) {
            assign(this->batch_model_backup[i][part_id], other.batch_trees[i], part_id);
        }
    }
}
