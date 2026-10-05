#ifndef RAXML_SPLITSAMPLERSOURCE_HPP_
#define RAXML_SPLITSAMPLERSOURCE_HPP_

#include <memory>
#include "ResampleSource.hpp"

/**
 * A split extracted from the donor pool, ranked by its frequency
 */
struct CandidateSplit {
    Split words;
    unsigned int ml_frequency = 0;
    unsigned int donor_frequency = 0;
};

class SplitSamplerSource : public ResampleSource {
public:
    SplitSamplerSource(const std::shared_ptr<ParsimonySource> &donor, const std::shared_ptr<EbgGate> &gate,
        const int seed, const TreeList &initial_ml_trees, const Tree &baseline_tree)
        : ResampleSource(donor, gate, seed, initial_ml_trees, baseline_tree) {
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


#endif //RAXML_SPLITSAMPLERSOURCE_HPP_
