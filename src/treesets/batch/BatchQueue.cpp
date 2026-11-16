#include "BatchQueue.hpp"

TunedBatch &BatchQueue::generate_batch(const unsigned int num_workers, const unsigned int num_threads, const std::lock_guard<std::mutex> &) {
    const std::string name_prefix = "Batch";

    // lock the mutex for the batch queue
    const auto batch_name_index = this->batches.size();

    // place new batches at the end of the queue, and mark them as unfinished
    std::string batch_name = name_prefix + std::to_string(batch_name_index);
    this->batches.emplace_back(batch_name,
                               msa,
                               tip_msa_idmap,
                               persite_loglh,
                               this->batch_size,
                               num_threads,
                               num_workers,
                               load_balancer);

    const auto &batch = this->batches.back();

    // mark the batch as unfinished and reserved
    this->in_flight.emplace(batch.get_name());
    this->unfinished.emplace(batch.get_name());

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

    // lock the batch mutex to avoid comodification of the list
    const std::lock_guard lock(batch_mutex);

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
        selected_batch = &generate_batch(num_workers, num_threads, lock);
    } else {
        this->in_flight.emplace(selected_batch->get_name());
    }

    return *selected_batch;
}

void BatchQueue::finish_batch(TunedBatch &batch) {
    const std::lock_guard lock(batch_mutex);

    if (batch.get_plausible_tree_count() > this->batch_size / 2) {
        this->finalize_batch(batch);
    }

    this->in_flight.erase(batch.get_name());
}
