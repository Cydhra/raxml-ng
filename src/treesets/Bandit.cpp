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

bool Bandit::is_worse_than(const Bandit &other, const unsigned int total_samples) const {
    // each bandit needs to be sampled at least once
    if (this->samples.empty() || other.samples.empty()) {
        return false;
    }

    const auto mean_success = this->get_mean_success();
    const auto variance = this->get_variance();

    // as defined by 10.1016/0196-8858(85)90002-8, formula 4.13 with the choice of `a_(n,i)` = `(log n) / i`,
    // where `n` is the total number of samples, and `i` is the number of samples drawn for this bandit
    const auto upper_confidence = mean_success + variance * sqrt(2.0 * log(static_cast<double>(total_samples)) / static_cast<double>(this->samples.size()));

    return other.get_mean_success() > upper_confidence;
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
    double variance_sum = this->estimated_variance;

    // calculate participation of the estimator, gradually replacing it with actual measurements
    const auto weight = this->estimated_variance_weight - min(this->estimated_variance_weight,
                                                        static_cast<unsigned int>(this->samples.size()));

    variance_sum *= weight;

    for (auto &sample: this->samples) {
        const double sample_throughput = static_cast<double>(sample.plausible_trees) / static_cast<double>(sample.
                                             time_spent);
        variance_sum += (mean - sample_throughput) * (mean - sample_throughput);
    }

    return variance_sum / (this->samples.size() + weight);
}

MetaParameters &Bandit::get_parameters() const {
    return *this->parameters;
}

string Bandit::get_name() const {
    return this->name;
}
