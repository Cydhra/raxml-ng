#ifndef RAXML_NG_AGGRESSIVESOURCE_HPP
#define RAXML_NG_AGGRESSIVESOURCE_HPP

#include <utility>
#include <memory>

#include "ParsimonySource.hpp"
#include "TreeSource.hpp"
#include "../util/EbgFilter.hpp"

using namespace std;

Tree generate_parsimony_tree(const RaxmlInstance &instance,
                             int random_seed,
                             bool bootstrap,
                             const Tree &constraint_tree);

struct RaxmlInstance;

class SplitSamplerSource : public TreeSource {
public:
    explicit SplitSamplerSource(const std::shared_ptr<ParsimonySource> &donor, const int seed,
                                TreeList initial_ml_trees,
                                Tree baseline_tree) : filter(EbgFilter(baseline_tree, initial_ml_trees)),
                                                      baseline_tree(std::move(baseline_tree)),
                                                      initial_ml_trees(std::move(initial_ml_trees)),
                                                      donor_tree_source(donor), seed(seed) {
    }

    void ensure(const RaxmlInstance &instance, const SmartBarrier &barrier, unsigned int threads_per_worker,
                unsigned int worker_id, unsigned int thread_id, unsigned int required_trees) override;

    [[nodiscard]] double amortized_time(unsigned int batch_size) const override;

protected:
    EbgFilter filter;

    /**
     * Best ML tree from reference set to use for labels
     */
    const Tree baseline_tree;
    const TreeList initial_ml_trees;


    // build trees from these
    struct CandidateSplit {
        std::vector<corax_split_base_t> words;
        unsigned int ml_frequency = 0;
        unsigned int donor_frequency = 0;
    };

    bool remember_topology(const Tree &candidate);

    static Split topology_key(const Tree &tree);

    static bool has_majority_split(
        const SplitList &donor_topologies,
        const std::vector<std::size_t> &donor_ids,
        std::size_t words_per_split);

    static SplitList extract_splits(const Tree &tree, bool normalize);

    virtual TreeList generate_candidates(
        const RaxmlInstance &instance,
        const ConstTreeRange &donor_pool,
        const SplitList &donor_splits,
        unsigned int requested_candidates,
        unsigned long round_seed) = 0;

    Tree materialize_candidate(const std::vector<CandidateSplit> &selected) const;

    std::shared_ptr<ParsimonySource> donor_tree_source;

    int seed;

    std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();

    std::set<Split> seen_topologies;

    int sampled_batches = 0;
};


#endif //RAXML_NG_AGGRESSIVESOURCE_HPP
