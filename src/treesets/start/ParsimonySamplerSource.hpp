#ifndef RAXML_NG_PARSIMONYSAMPLERSOURCE_HPP
#define RAXML_NG_PARSIMONYSAMPLERSOURCE_HPP

#include <memory>
#include "SplitSamplerSource.hpp"

class ParsimonySamplerSource : public SplitSamplerSource {
public:
    ParsimonySamplerSource(const std::shared_ptr<ParsimonySource> &donor, const std::shared_ptr<EbgGate> &gate,
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
};


#endif //RAXML_NG_PARSIMONYSAMPLERSOURCE_HPP
