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
    ParallelContext::init_pthreads_custom(opts, tree_builder, this->num_threads, this->num_threads);
    tree_builder();
    ParallelContext::finalize_threads();

    // load balance using the current thread assignment
    PartitionAssignment part_sizes;

    /* init list of partition sizes */
    for (unsigned int i = 0; i < this->msa->part_list().size(); ++i) {
        auto pinfo = &this->msa->part_list()[i];
        part_sizes.assign_sites(i, 0, pinfo->length(), pinfo->model().clv_entry_size());
    }

    this->part_assignment.
            reset(new PartitionAssignmentList(load_balancer.get_all_assignments(part_sizes, num_threads)));

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
    this->optimize_all_parameters(3.0);

    if (this->light_spr) {
        this->spr_params.ntopol_keep = 1;
    }

    // TODO parallelize
    for (unsigned int i = 0; i < this->get_batch_size(); ++i) {
        for (unsigned int spr_round = num_spr_performed; spr_round < this->target_num_spr; ++spr_round) {
            LOG_PROGRESS(this->batch_trees[i].loglh()) << (light_spr ? "GREEDY" : "FAST") << " spr round " << spr_round << " (radius: " << spr_params.radius_min << ")" << std::endl;
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
    const unsigned int max_assigned_workers = min(this->get_batch_size(), this->num_threads);
    ContiguousCoarseLoadBalancer load_balancer;
    CoarseAssignment tree_ids(reference_persite_loglh.size() + batch_persite_logh.size());
    std::iota(tree_ids.begin(), tree_ids.end(), 0);

    const auto assignment = load_balancer.get_all_assignments(tree_ids, num_workers);

    auto au_worker = std::bind(parallel_au_bootstrap, std::ref(*this->au_test), std::ref(assignment));
    ParallelContext::init_pthreads_custom(opts, au_worker, max_assigned_workers, max_assigned_workers);
    au_worker();

    this->au_test->finalize_test_statistics();
    this->au_test->calculate_p_values();

    LOG_INFO_TS << "AU Test finished" << endl;

    // TODO: there is a bug here that forces us to detach, find it.
    ParallelContext::finalize_threads(true);

    const unsigned int plausible_trees = count_plausible_trees(
        this->au_test->get_p_values().begin() + reference_persite_loglh.size(), this->au_test->get_p_values().end());
    return plausible_trees;
}

bool TunedBatch::is_plausible(const Options &opts) {
    this->optimize_all_parameters(0.1, true);
    const unsigned int plausible_trees = this->perform_au_test(opts);
    return plausible_trees >= static_cast<unsigned int>(static_cast<double>(this->get_batch_size()) * ACCEPT_TUNING_THRESHOLD);
}

void TunedBatch::optimize_all_parameters(const double epsilon, const bool force) {
    // TODO parallelize over trees
    for (unsigned int i = 0; i < this->get_batch_size(); ++i) {
        if (!this->skip_model || force) {
            // optimize model and branch lengths, such that we get accurate site likelihoods.
            // TODO if they already are optimized from previous AU tests, do not re-optimize to avoid oscillation
            batch_trees[i].optimize_model(epsilon);
        } else {
            // TODO load model from backup
        }

        // optimize branches
        batch_trees[i].optimize_params(CORAX_OPT_PARAM_BRANCHES_ITERATIVE, epsilon);
    }
}
