#include "TreesetOptimizer.hpp"
#include "TunedBatch.hpp"
#include "../Optimizer.hpp"
#include <regex>

void create_mab(std::deque<MultiArmedBandit<MetaParameters> > &target_list, const std::string &name,
                const MetaParameters &parameters) {
    target_list.emplace_back();
    target_list[target_list.size() - 1].emplace_back(name, parameters);
}

void TreesetOptimizer::initialize_bandits() {
    create_mab(this->benchmark_mabs, "Parsimony", MetaParameters(1, false, 0, 0, true, false));

    const auto adaptive_radius = pythia_score >= 0.0
                                     ? Optimizer::adaptive_radius(pythia_score)
                                     : DEFAULT_ADAPTIVE_RADIUS;

    // constrained
    create_mab(this->benchmark_mabs, "Constrained,Fast,NNI,2spr",
               MetaParameters(20, false, 2, 0, false, false, adaptive_radius, true, true));
    create_mab(this->benchmark_mabs, "Constrained,Fast,NNI,4spr",
               MetaParameters(20, false, 4, 0, false, false, adaptive_radius, true, true));
    create_mab(this->benchmark_mabs, "Constrained,Fast,2spr",
               MetaParameters(20, false, 2, 0, false, false, adaptive_radius, false, true));

    // model downgrade
    create_mab(this->benchmark_mabs, "Fast,2spr,JC",
               MetaParameters(20, false, 2, 0, false, false, adaptive_radius, false, false, "JC"));
    create_mab(this->benchmark_mabs, "Greedy,2spr,JC",
               MetaParameters(1, false, 2, 0, false, false, adaptive_radius, false, false, "JC"));
    create_mab(this->benchmark_mabs, "Fast,2spr,GTR",
               MetaParameters(20, false, 2, 0, false, false, adaptive_radius, false, false, "GTR"));
    create_mab(this->benchmark_mabs, "Greedy,2spr,GTR",
               MetaParameters(1, false, 2, 0, false, false, adaptive_radius, false, false, "GTR"));
    create_mab(this->benchmark_mabs, "Constrained,Fast,JC,2spr",
               MetaParameters(20, false, 2, 0, false, false, adaptive_radius, false, true, "JC"));

    // init default bandits
    create_mab(this->benchmark_mabs, "NNI,DoModel",
               MetaParameters(20, false, 0, 0, false, false, adaptive_radius, true));
    create_mab(this->benchmark_mabs, "NNI,NoModel",
               MetaParameters(20, true, 0, 0, false, false, adaptive_radius, true));

    create_mab(this->benchmark_mabs, "Greedy,DoModel,2spr",
               MetaParameters(1, false, 2, 0, false, false, adaptive_radius));
    create_mab(this->benchmark_mabs, "Greedy,DoModel,4spr",
               MetaParameters(1, false, 4, 0, false, false, adaptive_radius));
    create_mab(this->benchmark_mabs, "Greedy,NoModel,2spr",
               MetaParameters(1, true, 2, 0, false, false, adaptive_radius));

    create_mab(this->benchmark_mabs, "Fast,DoModel,2spr",
               MetaParameters(20, false, 2, 0, false, false, adaptive_radius));
    create_mab(this->benchmark_mabs, "Fast,DoModel,4spr",
               MetaParameters(20, false, 4, 0, false, false, adaptive_radius));
    create_mab(this->benchmark_mabs, "Fast,NoModel,2spr",
               MetaParameters(20, true, 2, 0, false, false, adaptive_radius));

    // heavy heuristics
    create_mab(this->benchmark_mabs, "Slow,2spr", MetaParameters(20, false, 0, 2, false, false, adaptive_radius));
    create_mab(this->benchmark_mabs, "Mixed,2+2spr", MetaParameters(20, false, 2, 2, false, false, adaptive_radius));
    create_mab(this->benchmark_mabs, "Mixed,4+2spr", MetaParameters(20, false, 4, 2, false, false, adaptive_radius));
    create_mab(this->benchmark_mabs, "Mixed,4+4spr", MetaParameters(20, false, 4, 4, false, false, adaptive_radius));
    create_mab(this->benchmark_mabs, "Constrained,Mixed,4+2spr,NNI",
               MetaParameters(20, false, 4, 2, false, false, adaptive_radius, true, true));

    // early commitment
    create_mab(this->benchmark_mabs, "Commit,Greedy,DoModel,2spr",
               MetaParameters(1, false, 2, 0, false, true, adaptive_radius));
    create_mab(this->benchmark_mabs, "Commit,Fast,DoModel,4spr",
               MetaParameters(20, false, 4, 0, false, true, adaptive_radius));
}

constexpr unsigned int SAMPLES_PER_BANDIT = 10;

std::string get_report_filename(std::string output_prefix, std::string bandit_name) {
    auto file_name = string();
    bandit_name = std::regex_replace(bandit_name, std::regex("[,+]"), "_");
    file_name.append(output_prefix).append(".").append(bandit_name).append(".txt");
    return file_name;
}

void TreesetOptimizer::run_batch(Bandit<MetaParameters> &bandit,
                                 TunedBatch &batch, TaskGroup &context, unsigned int worker_id,
                                 unsigned int thread_id) {
    batch.optimize(instance, opts, shared_batch_resources, context, worker_id, thread_id);

    if (context.is_group_leader(worker_id, thread_id)) {
        // take measurements
        bandit.take_measurement(batch);

        // inform the batch queue that the batch has been inferred
        this->batch_queue.finish_batch(batch);

        if (bandit.num_samples() >= SAMPLES_PER_BANDIT) {
            // serialize work unit by writing a report from the profiler into a special file (one per bandit). This allows
            // restoring progress (tell snakemake to keep files). Remember to clear the profiler after the report.
            const auto file_name = get_report_filename(opts.outfile_prefix,
                                                       this->benchmark_mabs[bandit_cursor].get_bandit(0).get_name());
            shared_batch_resources.get_profiling().write_report_and_reset(file_name);

            LOG_INFO << "Wrote results of bandit [" << this->benchmark_mabs[bandit_cursor].get_bandit(0).get_name() <<
                    "] to file " << file_name << std::endl;

            // Warning: this only works if we have a single TaskGroup, so we
            // if this was the last bandit
            if (this->bandit_cursor == this->benchmark_mabs.size() - 1) {
                pool.shutdown();
            } else {
                bandit_cursor += 1;
            }
        }
    }
}

BatchTask TreesetOptimizer::next_work_unit() {
    auto *current_mab = &this->benchmark_mabs[bandit_cursor];
    auto file_name = get_report_filename(opts.outfile_prefix, current_mab->get_bandit(0).get_name());

    // check if we already have a report file for this bandit, so we can skip redoing it
    while (std::filesystem::exists(file_name)) {
        LOG_INFO << "Skipping bandit [" << current_mab->get_bandit(0).get_name() <<
                "] because we already benchmarked it" << std::endl;
        bandit_cursor += 1;

        if (bandit_cursor == this->benchmark_mabs.size()) {
            pool.shutdown();
            return [](TaskGroup &context, const unsigned int worker_id,
                                                               const unsigned int thread_id) {
                // do nothing
            };
        }

        current_mab = &this->benchmark_mabs[bandit_cursor];
        file_name = get_report_filename(opts.outfile_prefix, current_mab->get_bandit(0).get_name());
    }


    auto &current_bandit = current_mab->select_next_bandit();
    auto &current_batch = this->batch_queue.select_next_batch(*current_bandit.get_parameters(), pool.workers_per_task(),
                                                              pool.threads_per_task());
    current_batch.update_meta_parameters(opts, current_bandit.get_parameters());

    BatchTask runner = [this, &current_bandit, &current_batch](TaskGroup &context, const unsigned int worker_id,
                                                               const unsigned int thread_id) {
        this->run_batch(current_bandit, current_batch, context, worker_id, thread_id);
    };

    return runner;
}

void TreesetOptimizer::run() {
    LOG_INFO << std::endl;
    LOG_INFO_TS << "Treeset: Inferring at least " << this->target_tree_count <<
            " plausible trees while optimizing throughput." << std::endl;

    // initialize all bandit arms, and add the parsimony arm to the top-level MAB.
    this->initialize_bandits();

    pool.work(opts);

    LOG_INFO_TS << "Inferred " << this->batch_queue.num_plausible_trees() << " plausible trees in " << this->batch_queue
            .num_batches() <<
            " batches." << std::endl;

    shared_batch_resources.get_profiling().print_report();
}

void TreesetOptimizer::check_mab_modification(
    const Bandit<shared_ptr<MultiArmedBandit<MetaParameters> > > &) {
    // TODO implement a proper heuristic here. For now, we check if the current arm exceeds 75% success per batch,
    //  and if not, we add arms according to a pre-defined mapping.
}

std::vector<Tree> TreesetOptimizer::get_all_trees() const {
    auto full_set = std::vector<Tree>();

    for (auto &batch: batch_queue.view_batches()) {
        for (unsigned int tree_id = 0; tree_id < batch.get_batch_size(); ++tree_id) {
            full_set.push_back(batch.get_tree(tree_id));
        }
    }

    return full_set;
}


std::vector<Tree> TreesetOptimizer::get_plausible_trees() const {
    auto plausible_set = std::vector<Tree>();

    for (auto &batch: batch_queue.view_batches()) {
        batch.get_plausible_trees(plausible_set);
    }

    return plausible_set;
}
