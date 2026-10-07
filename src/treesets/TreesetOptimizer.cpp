#include "TreesetOptimizer.hpp"
#include "batch/TunedBatch.hpp"
#include "../Optimizer.hpp"
#include "mab/DynamicMAB.hpp"

void TreesetOptimizer::initialize_bandits() {
    auto starting_trees = make_shared<MultiArmedBandit<MetaParameters> >();
    starting_trees->emplace_back("Parsimony", MetaParameters());
    starting_trees->emplace_back("Split-Parsimony", MetaParameters().with_treesource(SplitParsimony));
    starting_trees->emplace_back("Split-Greedy", MetaParameters().with_treesource(SplitGreedy));

    const auto adaptive_radius = pythia_score >= 0.0
                                     ? Optimizer::adaptive_radius(pythia_score)
                                     : DEFAULT_ADAPTIVE_RADIUS;
    const auto small_radius = static_cast<unsigned int>(max(static_cast<signed int>(adaptive_radius) - 5, 5));
    const auto very_small_radius = static_cast<unsigned int>(max(static_cast<signed int>(adaptive_radius) - 10, 5));


    auto nni_mab = make_shared<MultiArmedBandit<MetaParameters> >();
    nni_mab->emplace_back("NNI,DoModel", MetaParameters().with_nni_round(true).with_first_model(true));
    nni_mab->emplace_back("NNI,NoModel", MetaParameters().with_nni_round(true));
    this->mab.register_new_successor(1, "NNI", std::move(nni_mab));

    auto light_mab = make_shared<MultiArmedBandit<MetaParameters> >();
    light_mab->emplace_back("Greedy,NoModel,2spr",
                            MetaParameters().with_topk(1).with_first_model(false).with_fast_rounds(2).with_radius(
                                adaptive_radius));
    light_mab->emplace_back("Fast,DoModel,2spr",
                            MetaParameters().with_first_model(true).with_fast_rounds(2).with_radius(adaptive_radius));
    light_mab->emplace_back("Fast,NoModel,2spr",
                            MetaParameters().with_first_model(false).with_fast_rounds(2).with_radius(adaptive_radius));
    light_mab->emplace_back("Greedy,DoModel,2spr",
                        MetaParameters().with_topk(1).with_first_model(true).with_fast_rounds(2).with_radius(
                            adaptive_radius));
    this->mab.register_new_successor(4, "Light", std::move(light_mab));

    // low radius heuristics
    if (small_radius < adaptive_radius) {
        auto low_mab = make_shared<MultiArmedBandit<MetaParameters> >();
        low_mab->emplace_back("Greedy,2spr,low",
                              MetaParameters().with_topk(1).with_first_model(true).with_fast_rounds(2).with_radius(
                                  small_radius));
        low_mab->emplace_back("Slow,2spr,low",
                              MetaParameters().with_first_model(true).with_slow_rounds(2).with_radius(small_radius));
        low_mab->emplace_back("Fast,2spr,low",
                              MetaParameters().with_first_model(true).with_fast_rounds(2).with_radius(small_radius));
        if (very_small_radius < small_radius) {
            low_mab->emplace_back("Fast,2spr,v-low",
                                  MetaParameters().with_first_model(true).with_fast_rounds(2).
                                  with_radius(very_small_radius));
        }
        this->mab.register_new_successor(2, "LowRadius", std::move(low_mab));
    }

    auto constrained_mab = make_shared<MultiArmedBandit<MetaParameters> >();
    constrained_mab->emplace_back("Greedy,DoModel,2spr,NNI,Constrained",
                                  MetaParameters().with_topk(1).with_first_model(true).with_fast_rounds(2).
                                  with_radius(adaptive_radius).with_nni_round(true).with_constrain(true));
    constrained_mab->emplace_back("Fast,DoModel,2spr,NNI,Constrained",
                                  MetaParameters().with_first_model(true).with_fast_rounds(2).
                                  with_radius(adaptive_radius).with_nni_round(true).with_constrain(true));
    if (small_radius < adaptive_radius) {
        constrained_mab->emplace_back("Fast,DoModel,2spr,NNI,Constrained,low",
                                  MetaParameters().with_first_model(true).with_fast_rounds(2).
                                  with_radius(small_radius).with_nni_round(true).with_constrain(true));
    }
    this->mab.register_new_successor(2, "Constrained", std::move(constrained_mab));

    auto dynamic_mab = make_shared<MultiArmedBandit<MetaParameters> >();
    if (small_radius < adaptive_radius) {
        dynamic_mab->emplace_back("Greedy,DoModel,Dynamic,low",
                              MetaParameters().with_topk(1).with_first_model(true).with_radius(small_radius).
                              with_dynamic_spr(true));
    }
    dynamic_mab->emplace_back("Fast,DoModel,Dynamic",
                              MetaParameters().with_first_model(true).with_dynamic_spr(true));

    if (small_radius < adaptive_radius) {
        dynamic_mab->emplace_back("Fast,DoModel,Dynamic,low",
                                  MetaParameters().with_first_model(true).with_radius(small_radius).
                                  with_dynamic_spr(true));
    }
    dynamic_mab->emplace_back("Greedy,DoModel,Dynamic",
                              MetaParameters().with_topk(1).with_first_model(true).with_dynamic_spr(true));

    this->mab.register_new_successor(5, "Dynamic", std::move(dynamic_mab));

    // fallbacks
    auto fallback_fast_mab = make_shared<MultiArmedBandit<MetaParameters> >();
    fallback_fast_mab->emplace_back("Fast-Raxml", MetaParameters().with_fallback_fast_raxml(true).with_final_model(false));
    this->mab.register_new_successor(6, "Fallback-Fast", std::move(fallback_fast_mab));

    auto fallback_adaptive_mab = make_shared<MultiArmedBandit<MetaParameters> >();
    fallback_adaptive_mab->emplace_back("Adaptive-Raxml", MetaParameters().with_fallback_adaptive(true).with_final_model(false));
    this->mab.register_new_successor(7, "Fallback-Adaptive", std::move(fallback_adaptive_mab));

    // second-level MAB
    this->mab.register_new_arm("Starting Trees", starting_trees);
}

bool TreesetOptimizer::run_batch(TunedBatch &batch, const TaskGroup &context, const unsigned int worker_id,
                                 const unsigned int thread_id) {
    try {
        batch.optimize(instance, opts, shared_batch_resources, context, worker_id, thread_id);
    } catch (BanditFailedException &e) {
        LOG_INFO << "Batch " << batch.get_name() << " failed because: " << e.message() << ". Disabling bandit." <<
                std::endl;

        // inform the batch queue that the batch has been inferred
        this->batch_queue.finish_batch(batch);

        return false;
    }


    if (context.is_group_leader(worker_id, thread_id)) {
        // inform the mab about the results
        this->mab.take_measurement(batch);

        // inform the batch queue that the batch has been inferred
        this->batch_queue.finish_batch(batch);

        if (this->batch_queue.num_plausible_trees() > this->target_tree_count || this->batch_queue.view_batches().size()
            >= 250) {
            pool.shutdown();
        }
    }

    return true;
}

BatchTask TreesetOptimizer::next_work_unit() {
    const auto &parameters = this->mab.select_next_bandit();
    auto &current_batch = this->batch_queue.select_next_batch(parameters, pool.workers_per_task(),
                                                              pool.threads_per_task());
    current_batch.update_meta_parameters(parameters);

    BatchTask runner = [this, &parameters, &current_batch](TaskGroup &context, const unsigned int worker_id,
                                              const unsigned int thread_id) {
        if (!this->run_batch(current_batch, context, worker_id, thread_id)) {
            this->mab.disable_bandit(parameters);
        }
    };

    return runner;
}

void TreesetOptimizer::initialize_resources(const SmartBarrier &global_barrier) {
    LOG_INFO_TS << "Generating " << NUM_DONOR_TREE << " parsimony trees upfront." << std::endl;
    shared_batch_resources.get_parsimony().ensure(instance, global_barrier, 1, ParallelContext::local_group_id(), ParallelContext::local_thread_id(), NUM_DONOR_TREE);

    LOG_INFO_TS << "Generating " << target_tree_count << " split-informed parsimony trees." << std::endl;
    shared_batch_resources.get_parsimony_split_source().generate(instance, global_barrier, ParallelContext::num_groups(), ParallelContext::local_group_id(), target_tree_count);

    LOG_INFO_TS << "Generating " << target_tree_count << " split-greedy trees." << std::endl;
    shared_batch_resources.get_greedy_split_source().generate(instance, global_barrier, ParallelContext::num_groups(), ParallelContext::local_group_id(), target_tree_count);
}

void TreesetOptimizer::run() {
    LOG_INFO << std::endl;
    LOG_INFO_TS << "Treeset: Inferring at least " << this->target_tree_count <<
            " plausible trees while optimizing throughput." << std::endl;

    auto global_barrier = SmartBarrier(pool.num_threads_total());
    auto initializing_function = [this, &global_barrier]() {
        this->initialize_resources(global_barrier);
    };

    // infer 300 parsimony trees as donors
    ParallelContext::init_pthreads_custom(opts, initializing_function, pool.num_threads_total(), pool.num_threads_total());
    initializing_function();
    ParallelContext::finalize_threads();
    LOG_INFO_TS << "Starting bandit algorithm." << std::endl;

    // initialize all bandit arms, and add the parsimony arm to the top-level MAB.
    this->initialize_bandits();

    pool.work(opts);

    LOG_INFO_TS << "Inferred " << this->batch_queue.num_plausible_trees() << " plausible trees in " << this->batch_queue
            .num_batches() <<
            " batches." << std::endl;
}

std::vector<Tree> TreesetOptimizer::get_all_trees() const {
    auto full_set = std::vector<Tree>();

    for (auto &batch: batch_queue.view_batches()) {
        if (batch.has_trees()) {
            for (unsigned int tree_id = 0; tree_id < batch.get_batch_size(); ++tree_id) {
                full_set.push_back(batch.get_tree(tree_id));
            }
        }
    }

    return full_set;
}


std::vector<Tree> TreesetOptimizer::get_plausible_trees() const {
    auto plausible_set = std::vector<Tree>();

    for (auto &batch: batch_queue.view_batches()) {
        if (batch.has_trees()) {
            batch.get_plausible_trees(plausible_set);
        }
    }

    return plausible_set;
}
