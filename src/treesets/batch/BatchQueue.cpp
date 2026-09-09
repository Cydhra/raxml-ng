#include "BatchQueue.hpp"

/**
 * Backup a model from the given batch, iff a lock guard is provided.
 *
 * @param batch a tuned batch with a model that should be backed up
 * @param backup_model the target reference where to store the backup
 */
void guarded_backup_batch_model(const TunedBatch &batch, ModelMap &backup_model, std::lock_guard<std::mutex> const &) {
    batch.backup_models(backup_model);
}

TunedBatch &BatchQueue::generate_batch(const unsigned int num_workers, const unsigned int num_threads, TreeList inject_trees, AggressiveSourceFamily source, bool pin_initial_ml_model) {
    const std::string name_prefix =
        source == AggressiveSourceFamily::seed_greedy
            ? "AggressiveSeedGreedyBatch"
            : source == AggressiveSourceFamily::constrained_parsimony
                ? "AggressiveConstrainedParsimonyBatch"
                : "Batch";

    // lock the mutex for the batch queue
    const std::lock_guard<std::mutex> lock(batch_mutex);

    auto batch_name_index = this->batches.size();

    // place new batches at the end of the queue, and mark them as unfinished
    std::string batch_name = name_prefix + std::to_string(batch_name_index);
    this->batches.emplace_back(batch_name,
                               msa,
                               tip_msa_idmap,
                               persite_loglh,
                               generate_seed_for_trees(this->batch_size),
                               this->batch_size,
                               num_threads,
                               num_workers,
                               load_balancer);

    auto &batch = this->batches.back();

    if (!inject_trees.empty()) {
        batch.set_starting_trees(std::move(inject_trees));
    }

    if (source != AggressiveSourceFamily::none) {
        batch.set_aggressive_source(source);
        in_flight.emplace(batch.get_name());
    }

    // mark the batch as unfinished
    this->unfinished.emplace(batch.get_name());

    // assign the prepared model. If we have no model backed up yet, this is initialized with the default model,
    // so nothing will break. This requires that the batch mutex is locked
    if (pin_initial_ml_model) {
        batch.assign_batch_models(initial_ml_model);
    } else {
        batch.assign_batch_models(*backup_model);
    }

    // return (which drops the mutex guard)
    return this->batches[batch_name_index];
}

void BatchQueue::finalize_batch(TunedBatch &batch) {
    LOG_WORKER_TS(LogLevel::info) << "Finalized " << batch.get_name() << " with " << batch.get_plausible_tree_count() <<
            " plausible trees." << std::endl;
    this->unfinished.erase(batch.get_name());
    batch.finalize();
    this->finalized_plausible_trees += batch.get_plausible_tree_count();

    unsigned int unfinished_plausible = 0;
    for (auto &batch_ref: this->batches) {
        if (this->unfinished.find(batch_ref.get_name()) != this->unfinished.cend()) {
            unfinished_plausible += batch_ref.get_plausible_tree_count();
        }
    }

    this->unfinished_plausible_trees = unfinished_plausible;
}

TunedBatch &BatchQueue::select_next_batch(const MetaParameters &current_parameters, const unsigned int num_workers,
                                          const unsigned int num_threads) {
    TunedBatch *selected_batch = nullptr;

    batch_mutex.lock();
    // check if there exists a batch that is compatible
    for (auto &batch: this->batches) {
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
        // unlock mutex to allow generation of batches without keeping the queue locked, and because generate_batches
        // will attempt to lock it again when the batch is added to the vector.
        batch_mutex.unlock();
        selected_batch = &generate_batch(num_workers, num_threads, {}, AggressiveSourceFamily::none, current_parameters.pin_initial_ml_model);

        // relock to add batch to in-flight set
        batch_mutex.lock();
    }

    this->in_flight.emplace(selected_batch->get_name());
    batch_mutex.unlock();

    return *selected_batch;
}

void BatchQueue::finish_batch(TunedBatch &batch, bool pin_initial_ml_model) {
    const std::lock_guard<std::mutex> lock(batch_mutex);
    if (!pin_initial_ml_model && !batch.is_aggressive_source()) {
        guarded_backup_batch_model(batch, *this->backup_model, lock);
    }

    if (batch.is_aggressive_source() || batch.get_plausible_tree_count() > this->batch_size / 2) {
        this->finalize_batch(batch);
    }

    this->in_flight.erase(batch.get_name());
}

// ReSharper disable once CppMemberFunctionMayBeConst (confusing contract due to inner mutability)
void BatchQueue::backup_batch_model(const TunedBatch &batch) {
    const std::lock_guard<std::mutex> lock(batch_mutex);
    guarded_backup_batch_model(batch, *this->backup_model, lock);
}
