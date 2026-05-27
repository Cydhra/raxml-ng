#include "TreesetProfiling.hpp"

#include "../io/file_io.hpp"

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

    LOG_INFO << "NNI Round timings:\t" << nni_profiler->get_mean_wall_time() << ";" << std::endl;
    LOG_INFO << "Model Optimizer round timings:\t" << model_opt_profiler->get_mean_wall_time() << ";" << std::endl;
    LOG_INFO << "Branch Optimizer round timings:\t" << branch_opt_profiler->get_mean_wall_time() << ";" << std::endl;
}

void TreesetProfiling::write_report_and_reset(string path) {
    std::ofstream file(path);
    for (int i = 0; i < 4; ++i) {
        file << "Fast SPR " << i << "\t" << fast_spr_profiler->get_mean_wall_time(i) << std::endl;
    }
    fast_spr_profiler->clear();

    for (int i = 0; i < 4; ++i) {
        file << "Slow SPR " << i << "\t" << slow_spr_profiler->get_mean_wall_time(i) << std::endl;
    }
    slow_spr_profiler->clear();

    for (int i = 0; i < 4; ++i) {
        file << "Greedy SPR " << i << "\t" << greedy_spr_profiler->get_mean_wall_time(i) << std::endl;
    }
    greedy_spr_profiler->clear();

    file << "NNI" << "\t" << nni_profiler->get_mean_wall_time() << std::endl;
    nni_profiler->clear();

    file << "BLO" << "\t" << branch_opt_profiler->get_mean_wall_time() << std::endl;
    branch_opt_profiler->clear();

    file << "MO" << "\t" << model_opt_profiler->get_mean_wall_time() << std::endl;
    model_opt_profiler->clear();

    file << "Success" << "\t" << success_profiler->get_average_count() << std::endl;
    success_profiler->clear();

    file << "Throughput" << "\t" << throughput_profiler->get_average_count() << std::endl;
    throughput_profiler->clear();

    file.close();
}
