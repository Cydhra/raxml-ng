#include "TreesetOptimizer.hpp"
#include "batch/TunedBatch.hpp"
#include "../Optimizer.hpp"
#include "mab/DynamicMAB.hpp"

void TreesetOptimizer::initialize_bandits() {
    auto parsimony = make_shared<MultiArmedBandit<MetaParameters>>();
    parsimony->emplace_back("Parsimony", MetaParameters(1, false, 0, 0, true, false));

    const auto adaptive_radius = pythia_score >= 0.0
                                     ? Optimizer::adaptive_radius(pythia_score)
                                     : DEFAULT_ADAPTIVE_RADIUS;
    const auto small_radius = static_cast<unsigned int>(max(static_cast<signed int>(adaptive_radius) - 5, 5));
    const auto very_small_radius = static_cast<unsigned int>(max(static_cast<signed int>(adaptive_radius) - 10, 5));


    auto nni_mab = make_shared<MultiArmedBandit<MetaParameters>>();
    nni_mab->emplace_back("NNI,DoModel", MetaParameters(20, true, 0, 0, false, adaptive_radius, true));
    nni_mab->emplace_back("NNI,NoModel", MetaParameters(20, false, 0, 0, false, adaptive_radius, true));
    this->mab.register_new_successor(1, "NNI", std::move(nni_mab));

    auto light_mab = make_shared<MultiArmedBandit<MetaParameters>>();
    light_mab->emplace_back("Greedy,DoModel,2spr", MetaParameters(1, true, 2, 0, false, adaptive_radius));
    light_mab->emplace_back("Greedy,NoModel,2spr", MetaParameters(1, false, 2, 0, false, adaptive_radius));
    light_mab->emplace_back("Fast,DoModel,2spr", MetaParameters(20, true, 2, 0, false, adaptive_radius));
    light_mab->emplace_back("Fast,NoModel,2spr", MetaParameters(20, false, 2, 0, false, adaptive_radius));
    this->mab.register_new_successor(4, "Light", std::move(light_mab));

    // low radius heuristics
    auto low_mab = make_shared<MultiArmedBandit<MetaParameters>>();
    low_mab->emplace_back("Greedy,2spr,low",
                                MetaParameters(1, true, 2, 0, false, small_radius));
    low_mab->emplace_back("Fast,2spr,low",
                                MetaParameters(20, true, 2, 0, false, small_radius));
    if (very_small_radius < small_radius) {
        low_mab->emplace_back("Fast,2spr,v-low",
                          MetaParameters(20, true, 2, 0, false, very_small_radius));
    }

    low_mab->emplace_back("Slow,2spr,low",
                          MetaParameters(20, true, 0, 2, false, small_radius));
    this->mab.register_new_successor(2, "LowRadius", std::move(low_mab));

    auto constrained_mab = make_shared<MultiArmedBandit<MetaParameters>>();
    constrained_mab->emplace_back("Fast,DoModel,2spr,NNI,Constrained",
                                        MetaParameters(20, true, 2, 0, false, adaptive_radius, true, true));
    constrained_mab->emplace_back("Greedy,DoModel,2spr,NNI,Constrained",
                                        MetaParameters(1, true, 2, 0, false, adaptive_radius, true, true));
    constrained_mab->emplace_back("Fast,DoModel,2spr,NNI,Constrained,low",
                                            MetaParameters(20, true, 2, 0, false, small_radius, true, true));
    this->mab.register_new_successor(2, "Constrained", std::move(constrained_mab));


    auto dynamic_mab = make_shared<MultiArmedBandit<MetaParameters>>();
    dynamic_mab->emplace_back("Greedy,NoModel,Dynamic,low",
                                    MetaParameters(1, true, 0, 0, false, small_radius, false, false, std::nullopt,
                                                   false, true));
    dynamic_mab->emplace_back("Fast,DoModel,Dynamic,low",
                                    MetaParameters(20, true, 0, 0, false, small_radius, false, false, std::nullopt,
                                                   false, true));
    dynamic_mab->emplace_back("Greedy,NoModel,Dynamic",
                                    MetaParameters(1, true, 0, 0, false, adaptive_radius, false, false, std::nullopt,
                                                   false, true));
    dynamic_mab->emplace_back("Fast,DoModel,Dynamic",
                                    MetaParameters(20, true, 0, 0, false, adaptive_radius, false, false, std::nullopt,
                                                   false, true));
    this->mab.register_new_successor(5, "Dynamic", std::move(dynamic_mab));

    // fallbacks
    auto fallback_fast_mab = make_shared<MultiArmedBandit<MetaParameters>>();
    fallback_fast_mab->emplace_back("Fast-Raxml",
                                          MetaParameters(20, false, 0, 0, false, 20, false, false, std::nullopt,
                                                         true));
    this->mab.register_new_successor(6, "Fallback", std::move(fallback_fast_mab));

    // second-level MAB
    this->mab.register_new_arm("Starting Trees", std::move(parsimony));
}

void TreesetOptimizer::run_batch(
    TunedBatch &batch, TaskGroup &context, unsigned int worker_id,
    unsigned int thread_id) {
    batch.optimize(instance, opts, shared_batch_resources, context, worker_id, thread_id);

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
}

BatchTask TreesetOptimizer::next_work_unit() {
    const auto &parameters = this->mab.select_next_bandit();
    auto &current_batch = this->batch_queue.select_next_batch(parameters, pool.workers_per_task(),
                                                              pool.threads_per_task());
    current_batch.update_meta_parameters(parameters);

    BatchTask runner = [this, &current_batch](TaskGroup &context, const unsigned int worker_id,
                                                                     const unsigned int thread_id) {
        this->run_batch(current_batch, context, worker_id, thread_id);
    };

    return runner;
}

void TreesetOptimizer::run() {
    LOG_INFO << std::endl;
    LOG_INFO_TS << "Treeset: Inferring at least " << this->target_tree_count <<
            " plausible trees while optimizing throughput." << std::endl;

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
        for (unsigned int tree_id = 0; tree_id < batch.get_batch_size(); ++tree_id) {
            full_set.push_back(batch.get_tree(tree_id));
        }
    }

    return full_set;
}


std::vector<Tree> TreesetOptimizer::get_plausible_trees() const {
    auto plausible_set = std::vector<Tree>();

    for (auto &batch: batch_queue.view_batches()) {
        batch.get_plausible_trees(plausible_set);
    }

    return plausible_set;
}
