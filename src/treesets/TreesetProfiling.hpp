#ifndef RAXML_TREESETPROFILER_HPP_
#define RAXML_TREESETPROFILER_HPP_

#include <memory>
#include <variant>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "../types.hpp"
#include "TunedBatch.hpp"

struct SprRound {
    unsigned int round;
};

struct FastSprRound : SprRound {
};

struct SlowSprRound : SprRound {
};

struct GreedySprRound : SprRound {
};

struct ParameterOptimization {
    double epsilon;
};

struct ModelOptimization : ParameterOptimization {
};

struct BranchOptimization : ParameterOptimization {
};

struct CompleteInference {
};

struct NNIOptimization {
};

struct RaxmlFastOptimization {

};

typedef std::variant<FastSprRound, SlowSprRound, GreedySprRound, NNIOptimization, ModelOptimization, BranchOptimization, CompleteInference, RaxmlFastOptimization> InferencePhase;

class Profiler {
public:
    std::mutex m;

    virtual ~Profiler() = default;

    virtual void start_measurement(const TunedBatch &, const InferencePhase &) {
    }

    virtual void finish_measurement(const TunedBatch &, const InferencePhase &) {
    }

    virtual void clear() = 0;
};

template<class SprType>
class WallTimeProfiler : public Profiler {
protected:
    std::unordered_map<std::string, unsigned int> start_times{};

public:
    WallTimeProfiler() : Profiler() {
    }

    void start_measurement(const TunedBatch &batch, const InferencePhase &phase) override {
        if (std::holds_alternative<SprType>(phase)) {
            std::lock_guard lock(m);
            this->start_times[batch.get_name()] = batch.elapsed_wall_time();
        }
    }
};

template<class SprType>
class SimpleTimeProfiler : public WallTimeProfiler<SprType> {
protected:
    std::vector<unsigned int> samples{};

public:
    void finish_measurement(const TunedBatch &batch, const InferencePhase &phase) override {
        if (std::holds_alternative<SprType>(phase)) {
            std::lock_guard lock(this->m);
            const auto time = batch.elapsed_wall_time() - this->start_times[batch.get_name()];
            samples.push_back(time);
        }
    }

    void clear() override {
        std::lock_guard lock(this->m);
        samples.clear();
        this->start_times.clear();
    }

    /**
     * Get the average wall time spent on this phase
     * @return the average time for optimization of this phase
     */
    double get_mean_wall_time() {
        std::lock_guard lock(this->m);
        const auto &measurements = samples;

        if (measurements.empty()) {
            return -1.0;
        }

        double average = 0.0;
        for (const auto time: measurements) {
            average += static_cast<double>(time);
        }
        average /= measurements.size();

        return average;
    }
};

template<class SprType>
class SprRoundProfiler : public WallTimeProfiler<SprType> {
    std::unordered_map<unsigned int, std::vector<unsigned int> > samples{};

public:
    SprRoundProfiler() : WallTimeProfiler<SprType>() {
    }

    void finish_measurement(const TunedBatch &batch, const InferencePhase &phase) override {
        if (std::holds_alternative<SprType>(phase)) {
            std::lock_guard lock(this->m);
            const auto time = batch.elapsed_wall_time() - this->start_times[batch.get_name()];

            if (std::is_same_v<SprType, FastSprRound>) {
                samples[std::get<FastSprRound>(phase).round].push_back(time);
            } else if (std::is_same_v<SprType, SlowSprRound>) {
                samples[std::get<SlowSprRound>(phase).round].push_back(time);
            } else if (std::is_same_v<SprType, GreedySprRound>) {
                samples[std::get<GreedySprRound>(phase).round].push_back(time);
            } else {
                throw RaxmlException("SprRoundProfiler has not been implemented for this type");
            }
        }
    }

    void clear() override {
        std::lock_guard lock(this->m);
        samples.clear();
        this->start_times.clear();
    }

    /**
     * Get the average walltime spent on the n'th spr round.
     * @param round_num the number of the SPR round in question: the first spr round generally takes longer than the
     * second one, so the rounds are measured individually.
     * @return the average time for SPR rounds of this number, or -1.0 if no measurements have been taken for that number.
     */
    double get_mean_wall_time(const unsigned int round_num) {
        std::lock_guard lock(this->m);
        const auto &measurements = samples[round_num];

        if (measurements.empty()) {
            return -1.0;
        }

        double average = 0.0;
        for (const auto time: measurements) {
            average += static_cast<double>(time);
        }
        average /= measurements.size();

        return average;
    }
};

/**
 * Singleton class responsible for handling all profilers that take measurements of running batches. The
 * batch calls back into this class whenever progress has been made to allow profilers to take measurements.
 */
class TreesetProfiling {
protected:
    std::vector<std::shared_ptr<Profiler> > registered_profilers;

public:
    std::shared_ptr<SprRoundProfiler<FastSprRound> > fast_spr_profiler = std::make_shared<SprRoundProfiler<
        FastSprRound> >();
    std::shared_ptr<SprRoundProfiler<SlowSprRound> > slow_spr_profiler = std::make_shared<SprRoundProfiler<
        SlowSprRound> >();
    std::shared_ptr<SprRoundProfiler<GreedySprRound> > greedy_spr_profiler = std::make_shared<SprRoundProfiler<
        GreedySprRound> >();
    std::shared_ptr<SimpleTimeProfiler<ModelOptimization> > model_opt_profiler = std::make_shared<SimpleTimeProfiler<
        ModelOptimization> >();
    std::shared_ptr<SimpleTimeProfiler<BranchOptimization> > branch_opt_profiler = std::make_shared<SimpleTimeProfiler<
        BranchOptimization> >();
    std::shared_ptr<SimpleTimeProfiler<NNIOptimization> > nni_profiler = std::make_shared<SimpleTimeProfiler<
        NNIOptimization> >();


    TreesetProfiling() {
        registered_profilers.push_back(fast_spr_profiler);
        registered_profilers.push_back(slow_spr_profiler);
        registered_profilers.push_back(greedy_spr_profiler);
        registered_profilers.push_back(model_opt_profiler);
        registered_profilers.push_back(branch_opt_profiler);
        registered_profilers.push_back(nni_profiler);
    }

    /**
     * Notify all profilers that a new phase of optimization is starting.
     * This method reads the wall-time measurement of the input batch, so it should reflect the current combined
     * wall-time spent before the newly started phase.
     */
    void start_measurement(const TunedBatch &batch, const InferencePhase &phase);

    /**
     * Notify all profilers that a previously started phase is now finished.
     * This method relies on the fact that the wall-time of the TunedBatch is updated, so the caller must make sure
     * the wall-time measurements in TunedBatch are updated before calling this method.
     */
    void finish_measurement(const TunedBatch &batch, const InferencePhase &phase);

    void print_report() const;

    void write_report_and_reset(string path);
};

#endif //RAXML_TREESETPROFILER_HPP_
