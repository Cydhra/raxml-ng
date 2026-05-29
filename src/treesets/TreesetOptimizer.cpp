#include "TreesetOptimizer.hpp"
#include "TunedBatch.hpp"
#include "../Optimizer.hpp"

void TreesetOptimizer::initialize_bandits() {
    this->parsimony->emplace_back("Parsimony", MetaParameters(1, false, 0, 0, true, false));

    const auto adaptive_radius = pythia_score >= 0.0
                                     ? Optimizer::adaptive_radius(pythia_score)
                                     : DEFAULT_ADAPTIVE_RADIUS;

    // init default bandits
    this->nni_mab->emplace_back("NNI,DoModel", MetaParameters(20, false, 0, 0, false, false, adaptive_radius, true));
    this->nni_mab->emplace_back("NNI,NoModel", MetaParameters(20, true, 0, 0, false, false, adaptive_radius, true));

    this->light_mab->emplace_back("Greedy,DoModel,2spr", MetaParameters(1, false, 2, 0, false, false, adaptive_radius));
    this->light_mab->emplace_back("Greedy,DoModel,4spr", MetaParameters(1, false, 4, 0, false, false, adaptive_radius));
    this->light_mab->emplace_back("Greedy,NoModel,2spr", MetaParameters(1, true, 2, 0, false, false, adaptive_radius));

    this->light_mab->emplace_back("Fast,DoModel,2spr", MetaParameters(20, false, 2, 0, false, false, adaptive_radius));
    this->light_mab->emplace_back("Fast,DoModel,4spr", MetaParameters(20, false, 4, 0, false, false, adaptive_radius));
    this->light_mab->emplace_back("Fast,NoModel,2spr", MetaParameters(20, true, 2, 0, false, false, adaptive_radius));

    // heavy heuristics
    this->heavy_mab->emplace_back("Slow,2spr", MetaParameters(20, false, 0, 2, false, false, adaptive_radius));
    this->heavy_mab->emplace_back("Mixed,2+2spr", MetaParameters(20, false, 2, 2, false, false, adaptive_radius));
    this->heavy_mab->emplace_back("Mixed,4+2spr", MetaParameters(20, false, 4, 2, false, false, adaptive_radius));

    // early commitment
    this->commitment_mab->emplace_back("Commit,Greedy,DoModel,2spr",
                                       MetaParameters(1, false, 2, 0, false, true, adaptive_radius));
    this->commitment_mab->emplace_back("Commit,Greedy,DoModel,4spr",
                                       MetaParameters(1, false, 4, 0, false, true, adaptive_radius));
    this->commitment_mab->emplace_back("Commit,Fast,DoModel,2spr",
                                       MetaParameters(20, false, 2, 0, false, true, adaptive_radius));
    this->commitment_mab->emplace_back("Commit,Fast,DoModel,4spr",
                                       MetaParameters(20, false, 4, 0, false, true, adaptive_radius));

    // fallbacks
    this->fallback_fast_mab->emplace_back("Fast-Raxml", MetaParameters(20, true, 0, 0, false, false, 20, false, false, std::nullopt, true));

    // set up successors
    this->successors.emplace_back(make_tuple(1, "NNI", this->nni_mab));
    this->successors.emplace_back(make_tuple(1, "Light", this->light_mab));

    this->successors.emplace_back(make_tuple(3, "Commitment", this->commitment_mab));
    this->successors.emplace_back(make_tuple(4, "Heavy", this->heavy_mab));

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

        this->check_mab_modification(this->hierarchical_mab.get_best_bandit());

        // inform the batch queue that the batch has been inferred
        this->batch_queue.finish_batch(batch);

        if (this->batch_queue.num_plausible_trees() > this->target_tree_count || this->batch_queue.view_batches().size() >= 250) {
            pool.shutdown();
        }
    }
}

BatchTask TreesetOptimizer::next_work_unit() {
    auto &mab = this->hierarchical_mab.select_next_bandit();
    auto &current_bandit = mab.get_parameters().get()->get()->select_next_bandit();
    auto &current_batch = this->batch_queue.select_next_batch(*current_bandit.get_parameters(), pool.workers_per_task(),
                                                              pool.threads_per_task());
    current_batch.update_meta_parameters(opts, current_bandit.get_parameters());

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

void TreesetOptimizer::check_mab_modification(
    const Bandit<shared_ptr<MultiArmedBandit<MetaParameters> > > &current_arm) {
    // TODO implement a proper heuristic here. For now, we check if the current arm exceeds 75% success per batch,
    //  and if not, we add arms according to a pre-defined mapping.

    // check if we have more than a trivial amount of data, if the success rate is low, and we had at least 4 batches per bandit
    if (current_arm.num_samples() >= 4 && current_arm.get_expected_tree_rate() < 0.75 && hierarchical_mab.num_bandits() * 4 <= batch_queue.num_batches()) {
        const auto current_level = hierarchical_mab.num_bandits();

        // add all bandits of the current level
        for (auto it = this->successors.begin(); it != this->successors.end(); it += 1) {
            if (std::get<0>(*it) <= current_level) {
                if (!hierarchical_mab.has_bandit(std::get<1>(*it))) {
                    LOG_INFO << "Adding bandit " << std::get<1>(*it) << " to algorithm." << std::endl;
                    hierarchical_mab.emplace_back(std::get<1>(*it), std::get<2>(*it));
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
