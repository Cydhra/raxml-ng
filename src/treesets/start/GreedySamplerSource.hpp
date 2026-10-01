#ifndef RAXML_NG_GREEDYSAMPLERSOURCE_HPP
#define RAXML_NG_GREEDYSAMPLERSOURCE_HPP

#include <memory>
#include "SplitSamplerSource.hpp"

/**
 * A split extracted from the donor pool, ranked by its frequency
 */
struct CandidateSplit {
    Split words;
    unsigned int ml_frequency = 0;
    unsigned int donor_frequency = 0;
};

class GreedySamplerSource : public SplitSamplerSource {
public:
    GreedySamplerSource(const std::shared_ptr<ParsimonySource> &donor, const std::shared_ptr<EbgGate> &gate,
        const int seed, const TreeList &initial_ml_trees, const Tree &baseline_tree)
        : SplitSamplerSource(donor, gate, seed, initial_ml_trees, baseline_tree) {
    }

protected:
    TreeList generate_candidates(
        const RaxmlInstance &instance,
        const ConstTreeRange &donor_pool,
        const SplitList &donor_splits,
        unsigned int requested_candidates,
        unsigned long round_seed) override;

    Tree materialize_candidate(const std::vector<CandidateSplit> &selected) const;
};


#endif //RAXML_NG_GREEDYSAMPLERSOURCE_HPP
