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
