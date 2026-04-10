#include "TreesetOptimizer.hpp"
#include "TunedBatch.hpp"

void TreesetOptimizer::prepare_initial_batches() {
    auto &first_batch = this->batch_queue.generate_batches(1)[0];
    this->light_mab->take_measurement(starting_tree_bandit(), first_batch, false);
    this->batch_queue.backup_batch_model(first_batch);

    const auto starter_batches = this->batch_queue.generate_batches(INITIAL_VARIANCE_WEIGHT - 1);

    // take a few measurements, but no more than necessary to have reasonable values for mean and variance
    for (unsigned int index = 0; index < INITIAL_VARIANCE_WEIGHT - 1; index++) {
        auto &batch = starter_batches[index];
        this->light_mab->take_measurement(starting_tree_bandit(), batch, false);
    }
}

void TreesetOptimizer::initialize_bandits() {
    // TODO replace this by a general elimination rule
    if (this->starting_tree_bandit().get_expected_tree_rate() >= 0.9) {
        LOG_INFO << "Starting trees are so successful, no ML optimization is necessary." << std::endl;
        batch_queue.max_reuse_attempts = 0;
    } else {
        // init default bandits
        this->light_mab->emplace_back("Greedy,DoModel,2spr", MetaParameters(1, false, 2, 0, false, false));
        this->light_mab->emplace_back("Greedy,DoModel,4spr", MetaParameters(1, false, 4, 0, false, false));
        this->light_mab->emplace_back("Greedy,NoModel,2spr", MetaParameters(1, true, 2, 0, false, false));

        this->light_mab->emplace_back("Fast,DoModel,2spr", MetaParameters(20, false, 2, 0, false, false));
        this->light_mab->emplace_back("Fast,DoModel,4spr", MetaParameters(20, false, 4, 0, false, false));
        this->light_mab->emplace_back("Fast,NoModel,2spr", MetaParameters(20, true, 2, 0, false, false));

        // heavy heuristics
        this->heavy_mab->emplace_back("Slow,2spr", MetaParameters(20, false, 0, 2, false, false));
        this->heavy_mab->emplace_back("Mixed,2+2spr", MetaParameters(20, false, 2, 2, false, false));
        this->heavy_mab->emplace_back("Mixed,4+2spr", MetaParameters(20, false, 4, 2, false, false));

        // early commitment
        this->commitment_mab->emplace_back("Commit,Greedy,DoModel,2spr", MetaParameters(1, false, 2, 0, false, true));
        this->commitment_mab->emplace_back("Commit,Greedy,DoModel,4spr", MetaParameters(1, false, 4, 0, false, true));
        this->commitment_mab->emplace_back("Commit,Fast,DoModel,2spr", MetaParameters(20, false, 2, 0, false, true));
        this->commitment_mab->emplace_back("Commit,Fast,DoModel,4spr", MetaParameters(20, false, 4, 0, false, true));

        // second-level MAB
        this->hierarchical_mab.emplace_back("Light", this->light_mab);
        this->hierarchical_mab.emplace_back("Heavy", this->heavy_mab);
        this->hierarchical_mab.emplace_back("Committing", this->commitment_mab);
    }
}


void TreesetOptimizer::run() {
    LOG_INFO << std::endl;
    LOG_INFO_TS << "Treeset: Inferring at least " << this->target_tree_count <<
            " plausible trees while optimizing throughput." << std::endl;

    this->prepare_initial_batches();
    LOG_INFO << std::endl;
    LOG_INFO << "Expected throughput of starting trees: " << (
                this->starting_tree_bandit().get_mean_throughput() * 1000.0) <<
            " trees per second at a mean success rate of "
            << (this->starting_tree_bandit().get_expected_tree_rate() * 100.0) << "%." << std::endl;

    this->initialize_bandits();

    while (this->batch_queue.num_plausible_trees() < this->target_tree_count) {
        auto &mab = this->hierarchical_mab.select_next_bandit();
        auto &current_bandit = mab.get_parameters().get()->get()->select_next_bandit();
        auto &current_batch = this->batch_queue.select_next_batch(*current_bandit.get_parameters());

        current_batch.update_meta_parameters(opts, current_bandit.get_parameters());
        current_batch.optimize_main(instance, opts);
        mab.get_parameters()->get()->take_measurement(current_bandit, current_batch, true);
        this->hierarchical_mab.take_measurement(mab, current_batch, true);
        this->batch_queue.finish_batch(current_batch);

        // check if we would exceed the final tree count if we finalized the current batch immediately
        if (this->batch_queue.num_plausible_trees() > this->target_tree_count) {
            // TODO we should also count the unfinished batches
            break;
        }
    }

    LOG_INFO_TS << "Inferred " << this->batch_queue.num_plausible_trees() << " plausible trees in " << this->batch_queue.num_batches() <<
            " batches." << std::endl;
}

std::vector<Tree> TreesetOptimizer::get_all_trees() const {
    auto full_set = std::vector<Tree>();

    for (auto &batch : batch_queue.view_batches()) {
        for (unsigned int tree_id = 0; tree_id < batch.get_batch_size(); ++tree_id) {
            full_set.push_back(batch.get_tree(tree_id));
        }
    }

    return full_set;
}


std::vector<Tree> TreesetOptimizer::get_plausible_trees() const {
    auto plausible_set = std::vector<Tree>();

    for (auto &batch : batch_queue.view_batches()) {
        auto &p_values = batch.get_p_values();
        for (unsigned int tree_id = 0; tree_id < batch.get_batch_size(); ++tree_id) {
            if (p_values[tree_id + 16] > SIGNIFICANCE_LEVEL) {
                plausible_set.push_back(batch.get_tree(tree_id));
            }
        }
    }

    return plausible_set;
}
