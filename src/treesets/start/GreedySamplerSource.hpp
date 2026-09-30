#ifndef RAXML_NG_GREEDYSAMPLERSOURCE_HPP
#define RAXML_NG_GREEDYSAMPLERSOURCE_HPP

#include <memory>
#include "SplitSamplerSource.hpp"

class GreedySamplerSource : public SplitSamplerSource {
public:
    GreedySamplerSource(const std::shared_ptr<ParsimonySource> &donor, const int seed, const TreeList &initial_ml_trees,
        const Tree &baseline_tree)
        : SplitSamplerSource(donor, seed, initial_ml_trees, baseline_tree) {
    }

protected:
    TreeList generate_candidates(
        const RaxmlInstance &instance,
        const ConstTreeRange &donor_pool,
        const SplitList &donor_splits,
        unsigned int requested_candidates,
        unsigned long round_seed) override;
};


#endif //RAXML_NG_GREEDYSAMPLERSOURCE_HPP
