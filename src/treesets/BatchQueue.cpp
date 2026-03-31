#include "BatchQueue.hpp"

int recommended_thread_count() {
    // TODO add parameters to options containing the max thread count, which we just assign to the single worker per rank
    return 8;
}

int recommended_worker_count() {
    // TODO this is only true for the tuning phase
    return 1;
}

TunedBatch *BatchQueue::generate_batches(const unsigned int n) {
    const std::string name_prefix = "Batch";
    const auto tree_gen = std::make_shared<MetaParameters>(1, false, 0, 0, true, false);

    const auto start_index = this->batches.size();

    for (size_t i = 0; i < n; i++) {
        std::string batch_name = name_prefix + std::to_string(this->batches.size());
        this->batches.emplace_back(batch_name,
                                   generate_seed_for_trees(this->batch_size),
                                   this->batch_size,
                                   recommended_thread_count(),
                                   recommended_worker_count(),
                                   msa,
                                   persite_loglh);

        // mark the batch as unfinished
        this->unfinished.emplace(batch_name);

        auto &current_batch = this->batches[this->batches.size() - 1];

        // TODO schedule the batches in parallel if enough threads are available
        current_batch.update_meta_parameters(opts, tree_gen);
        current_batch.generate_starting_trees(this->instance, opts, load_balancer, tip_msa_idmap);

        // assign the prepared model. If we have no model backed up yet, this is initialized with the default model,
        // so nothing will break
        current_batch.assign_batch_models(*this->backup_model);
    }

    return this->batches.data() + start_index;
}

void BatchQueue::finalize_batch(TunedBatch &batch) {
    LOG_INFO_TS << "Finalized " << batch.get_name() << " with " << batch.get_plausible_tree_count() << " plausible trees." << std::endl;
    this->unfinished.erase(batch.get_name());
    batch.finalize();
    this->total_plausible_trees += batch.get_plausible_tree_count();
}

TunedBatch &BatchQueue::select_next_batch(const MetaParameters &current_parameters) {
    TunedBatch *selected_batch = nullptr;

    // check if there exists a batch that is compatible
    for (auto &batch : this->batches) {
        if (unfinished.find(batch.get_name()) != unfinished.end()) {
            if (in_flight.find(batch.get_name()) != in_flight.end()) {
                continue;
            }

            if (batch.is_compatible(current_parameters)) {
                selected_batch = &batch;
                batch.reuse_attempts = 0;
                break;
            } else {
                batch.reuse_attempts += 1;

                if (batch.reuse_attempts >= this->max_reuse_attempts) {
                    this->finalize_batch(batch);
                }
            }
        }
    }

    if (!selected_batch) {
        generate_batches(1);
        selected_batch = &this->batches[this->batches.size() - 1];
    }

    this->in_flight.emplace(selected_batch->get_name());
    return *selected_batch;
}

void BatchQueue::finish_batch(TunedBatch &batch) {
    this->backup_batch_model(batch);

    if (batch.get_plausible_tree_count() > this->batch_size / 2) {
        this->finalize_batch(batch);
    }

    this->in_flight.erase(batch.get_name());
}

// ReSharper disable once CppMemberFunctionMayBeConst (confusing contract due to inner mutability)
void BatchQueue::backup_batch_model(const TunedBatch &batch) {
    batch.backup_models(*this->backup_model);
}
