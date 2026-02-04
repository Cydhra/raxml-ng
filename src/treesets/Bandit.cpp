#include "Bandit.hpp"
#include "TunedBatch.hpp"

void Bandit::apply_parameters(const Options &opts, TunedBatch &batch) const {
    batch.update_meta_parameters(opts, this->parameters);
}

void Bandit::take_measurement(const TunedBatch &batch) {
    LOG_INFO_TS << "[" << this->name << "]: Takes measurement: " << (
                static_cast<double>(batch.get_plausible_tree_count()) / static_cast<double>(batch.elapsed_wall_time()) *
                1000.0)
            << " trees per second." << std::endl;
    this->samples.emplace_back(batch.elapsed_wall_time(), batch.get_plausible_tree_count(), batch.get_batch_size());
}

void Bandit::initialize_variance(const double variance, const unsigned int weight) {
    this->estimated_variance = variance;
    this->estimated_variance_weight = weight;
}

double Bandit::get_mean_throughput() const {
    double expectation = 0.0;

    for (auto &sample: this->samples) {
        // the cost function is TIME / #TREES (time spent per plausible tree), but the throughput (which is our reward
        // function) is the inverse of that: 1 / (TIME / #TREES)) = #TREES / TIME, with a reward of 0 if no plausible
        // trees are found
        expectation += static_cast<double>(sample.plausible_trees) / static_cast<double>(sample.time_spent);
    }

    return expectation / this->samples.size();
}

double Bandit::get_mean_success() const {
    double success = 0.0;
    for (auto &sample: this->samples) {
        success += static_cast<double>(sample.plausible_trees) / static_cast<double>(sample.batch_size);
    }

    return success / this->samples.size();
}

double Bandit::get_variance() const {
    const double mean = get_mean_throughput();
    double variance_sum = 0.0;

    for (auto &sample: this->samples) {
        const double sample_throughput = static_cast<double>(sample.plausible_trees) / static_cast<double>(sample.
                                             time_spent);
        variance_sum += (mean - sample_throughput) * (mean - sample_throughput);
    }

    return variance_sum / this->samples.size();
}

MetaParameters &Bandit::get_parameters() const {
    return *this->parameters;
}
