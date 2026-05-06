#include "TreesetProfiling.hpp"

void TreesetProfiling::start_measurement(const TunedBatch &batch, const InferencePhase &phase) {
    for (const auto &profiler : this->registered_profilers) {
        profiler->start_measurement(batch, phase);
    }
}

void TreesetProfiling::finish_measurement(const TunedBatch &batch, const InferencePhase &phase) {
    for (const auto &profiler : this->registered_profilers) {
        profiler->finish_measurement(batch, phase);
    }
}

void TreesetProfiling::print_report() const {
    LOG_INFO << "Fast SPR round timings: [";
    for (int i = 0; i < 4; ++i) {
        LOG_INFO << "\t" << i << ": " << fast_spr_profiler->get_mean_wall_time(i) << ";";
    }
    LOG_INFO << "\t]" << std::endl;

    LOG_INFO << "Slow SPR round timings: [";
    for (int i = 0; i < 4; ++i) {
        LOG_INFO << "\t" << i << ": " << slow_spr_profiler->get_mean_wall_time(i) << ";";
    }
    LOG_INFO << "\t]" << std::endl;

    LOG_INFO << "Greedy SPR round timings: [";
    for (int i = 0; i < 4; ++i) {
        LOG_INFO << "\t" << i << ": " << greedy_spr_profiler->get_mean_wall_time(i) << ";";
    }
    LOG_INFO << "\t]" << std::endl;

    LOG_INFO << "Model Optimizer round timings:\t" << model_opt_profiler->get_mean_wall_time() << ";" << std::endl;
    LOG_INFO << "Branch Optimizer round timings:\t" << branch_opt_profiler->get_mean_wall_time() << ";" << std::endl;
}
