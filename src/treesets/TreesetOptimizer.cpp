#include "TreesetOptimizer.hpp"
#include "TunedBatch.hpp"
#include "../Optimizer.hpp"

constexpr unsigned int MIN_PAUSE_BETWEEN_MODIFICATIONS = 6;

void TreesetOptimizer::initialize_bandits() {
    this->parsimony->emplace_back("Parsimony", MetaParameters(1, false, 0, 0, true, false));

    const auto adaptive_radius = pythia_score >= 0.0
                                     ? Optimizer::adaptive_radius(pythia_score)
                                     : DEFAULT_ADAPTIVE_RADIUS;

    // init default bandits
    this->nni_mab->emplace_back("NNI,DoModel", MetaParameters(20, false, 0, 0, false, false, adaptive_radius, true));
    this->nni_mab->emplace_back("NNI,NoModel", MetaParameters(20, true, 0, 0, false, false, adaptive_radius, true));

    this->light_mab->emplace_back("Greedy,DoModel,2spr", MetaParameters(1, false, 2, 0, false, false, adaptive_radius));
    this->light_mab->emplace_back("Greedy,NoModel,2spr", MetaParameters(1, true, 2, 0, false, false, adaptive_radius));

    this->light_mab->emplace_back("Fast,DoModel,2spr", MetaParameters(20, false, 2, 0, false, false, adaptive_radius));
    this->light_mab->emplace_back("Fast,NoModel,2spr", MetaParameters(20, true, 2, 0, false, false, adaptive_radius));

    // low radius heuristics
    this->low_mab->emplace_back("Greedy,2spr,low",
                                MetaParameters(1, false, 2, 0, false, false,
                                               max(adaptive_radius - 5, static_cast<unsigned int>(5))));
    this->low_mab->emplace_back("Fast,2spr,low",
                                MetaParameters(20, false, 2, 0, false, false,
                                               max(adaptive_radius - 5, static_cast<unsigned int>(5))));
    this->low_mab->emplace_back("Fast,2spr,v-low",
                                MetaParameters(20, false, 2, 0, false, false,
                                               max(adaptive_radius - 10, static_cast<unsigned int>(5))));
    this->low_mab->emplace_back("Slow,2spr,low",
                                MetaParameters(20, false, 0, 2, false, false,
                                               max(adaptive_radius - 5, static_cast<unsigned int>(5))));

    this->constrained_mab->emplace_back("Fast,DoModel,2spr,NNI,Constrained",
                                        MetaParameters(20, false, 2, 0, false, false, adaptive_radius, true, true));
    this->constrained_mab->emplace_back("Fast,NoModel,2spr,NNI,Constrained",
                                        MetaParameters(20, true, 2, 0, false, false, adaptive_radius, true, true));
    this->constrained_mab->emplace_back("Greedy,NoModel,2spr,NNI,Constrained",
                                        MetaParameters(1, true, 2, 0, false, false, adaptive_radius, true, true));
    this->constrained_mab->emplace_back("Greedy,DoModel,2spr,NNI,Constrained",
                                        MetaParameters(1, false, 2, 0, false, false, adaptive_radius, true, true));

    // fallbacks
    this->fallback_fast_mab->emplace_back("Fast-Raxml",
                                          MetaParameters(20, true, 0, 0, false, false, 20, false, false, std::nullopt,
                                                         true));

    // set up successors
    this->successors.emplace_back(make_tuple(1, "NNI", this->nni_mab));
    this->successors.emplace_back(make_tuple(2, "Light", this->light_mab));
    this->successors.emplace_back(make_tuple(3, "Constrained", this->constrained_mab));
    this->successors.emplace_back(make_tuple(4, "LowRadius", this->low_mab));
    this->successors.emplace_back(make_tuple(5, "Fallback", this->fallback_fast_mab));

    // second-level MAB
    this->hierarchical_mab.emplace_back("Starting Trees", parsimony);
}

void TreesetOptimizer::run_batch(Bandit<std::shared_ptr<MultiArmedBandit<MetaParameters> > > &mab,
                                 Bandit<MetaParameters> &bandit,
                                 TunedBatch &batch, TaskGroup &context, unsigned int worker_id,
                                 unsigned int thread_id) {
    batch.optimize(instance, opts, shared_batch_resources, context, worker_id, thread_id);

    if (context.is_group_leader(worker_id, thread_id)) {
        // take measurements
        mab.get_parameters()->get()->take_measurement(bandit, batch, true);
        this->hierarchical_mab.take_measurement(mab, batch, true);

        this->check_mab_modification();

        // inform the batch queue that the batch has been inferred
        this->batch_queue.finish_batch(batch);

        if (this->batch_queue.num_plausible_trees() > this->target_tree_count || this->batch_queue.view_batches().size()
            >= 250) {
            pool.shutdown();
        }
    }
}

BatchTask TreesetOptimizer::next_work_unit() {
    auto &mab = this->hierarchical_mab.select_next_bandit();
    auto &current_bandit = mab.get_parameters().get()->get()->select_next_bandit();
    auto &current_batch = this->batch_queue.select_next_batch(*current_bandit.get_parameters(), pool.workers_per_task(),
                                                              pool.threads_per_task());
    current_batch.update_meta_parameters(current_bandit.get_parameters());

    BatchTask runner = [this, &mab, &current_bandit, &current_batch](TaskGroup &context, const unsigned int worker_id,
                                                                     const unsigned int thread_id) {
        this->run_batch(mab, current_bandit, current_batch, context, worker_id, thread_id);
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

    shared_batch_resources.get_profiling().print_report();
}

void TreesetOptimizer::check_mab_modification() {
    // TODO implement a proper heuristic here. For now, we check if the current arm exceeds 75% success per batch,
    //  and if not, we add arms according to a pre-defined mapping.

    // check if we have more than a trivial amount of data, if the success rate is low, and we had at least 4 batches since the last modification
    if (last_mab_modification + MIN_PAUSE_BETWEEN_MODIFICATIONS < hierarchical_mab.get_iterations_completed() &&
        hierarchical_mab.get_best_bandit().num_samples() >= 4 && hierarchical_mab.get_best_bandit().
        get_expected_tree_rate() < 0.75) {
        const auto current_level = hierarchical_mab.num_bandits();

        // add all bandits of the current level
        for (auto it = this->successors.begin(); it != this->successors.end(); it += 1) {
            if (std::get<0>(*it) <= current_level) {
                if (!hierarchical_mab.has_bandit(std::get<1>(*it))) {
                    LOG_WORKER_TS(LogLevel::info) << std::endl << "Adding bandit " << std::get<1>(*it) <<
                            " to algorithm." << std::endl;
                    hierarchical_mab.emplace_back(std::get<1>(*it), std::get<2>(*it));
                    last_mab_modification = hierarchical_mab.get_iterations_completed();
                }
            }
        }
    }
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
