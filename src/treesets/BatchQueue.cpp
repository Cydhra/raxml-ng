#include "BatchQueue.hpp"

int recommended_thread_count() {
    // TODO add parameters to options containing the max thread count, which we just assign to the single worker per rank
    return 8;
}

int recommended_worker_count() {
    // TODO this is only true for the tuning phase
    return 1;
}

void BatchQueue::generate_batches(const unsigned int n) {
    const std::string name_prefix = "Batch";
    const auto tree_gen = std::make_shared<MetaParameters>(1, false, 0, 0, true, false);

    for (size_t i = 0; i < n; i++) {
        std::string batch_name = name_prefix + std::to_string(this->batches.size());
        this->batches.emplace_back(batch_name,
                                   generate_seed_for_trees(this->batch_size),
                                   this->batch_size,
                                   recommended_thread_count(),
                                   recommended_worker_count(),
                                   msa,
                                   persite_loglh);
        auto &current_batch = this->batches[this->batches.size() - 1];

        // TODO schedule the batches in parallel if enough threads are available
        current_batch.update_meta_parameters(opts, tree_gen);
        current_batch.generate_starting_trees(this->instance, opts, load_balancer, tip_msa_idmap);

        // if this isn't the very first tree, inherit model parameters from previous trees. Since we run the AU test
        // on the first tree, we have parameters optimized once
        if (this->batches.size() > 1) {
            current_batch.assign_batch_models(*this->backup_model);
        }
    }
}

void BatchQueue::advance_batch_cursor(const unsigned int n) {
    const auto target_index = this->batch_cursor + n;

    // if we have not enough batches to move the cursor to that index, generate the missing ones
    if (this->batches.size() < target_index) {
        LOG_DEBUG << "Warning: cursor moved " << (target_index - this->batches.size()) <<
                " batches past the end of the queue. Why are we generating batches that we will finalize instantly?" <<
                std::endl;
        generate_batches(target_index - this->batches.size());
    }

    for (unsigned int batch = this->batch_cursor; batch < target_index; ++batch) {
        // perform AU test to get plausible tree count. If the AU test was already executed, it will transparently
        // return the result of the previous run and not do any work
        const auto batch_plausible_trees = this->batches[batch].perform_plausibility_check(this->opts);
        this->total_plausible_trees += batch_plausible_trees;

        // finalize batch
        this->batches[batch].finalize();
    }

    LOG_DEBUG_TS << "Finalized " << this->total_plausible_trees << " plausible trees." << std::endl;
    this->batch_cursor += n;
    assert(this->batch_cursor <= this->batches.size());
}

TunedBatch &BatchQueue::select_next_batch(const MetaParameters &current_parameters) {
    advance_batch_cursor(1);

    // if the cursor now surpasses the queue, fill it up
    if (this->batch_cursor >= this->batches.size()) {
        generate_batches(this->batches.size() - this->batch_cursor + 1);
    }

    LOG_DEBUG << "Batch cursor: " << this->batch_cursor << std::endl;
    return this->batches[this->batch_cursor];
}

// ReSharper disable once CppMemberFunctionMayBeConst (confusing contract due to inner mutability)
void BatchQueue::backup_batch_model(const TunedBatch &batch) {
    batch.backup_models(*this->backup_model);
}
