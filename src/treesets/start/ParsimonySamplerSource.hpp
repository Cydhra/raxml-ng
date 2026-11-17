#ifndef RAXML_NG_PARSIMONYSAMPLERSOURCE_HPP
#define RAXML_NG_PARSIMONYSAMPLERSOURCE_HPP

#include <memory>
#include "ResampleSource.hpp"

class ParsimonySamplerSource : public ResampleSource {
public:
    ParsimonySamplerSource(const std::shared_ptr<ParsimonySource> &donor, const std::shared_ptr<EbgGate> &gate,
        const int seed, const TreeList &initial_ml_trees, const Tree &baseline_tree)
        : ResampleSource(donor, gate, seed, initial_ml_trees, baseline_tree) {
    }

protected:
    /**
     * Generate additional starting trees using existing starting trees as informative
     *
     * @param instance RaxmlInstance to generate (constrained) parsimony trees
     * @param donor_pool pool of starting trees to use as donors for topological constraints
     * @param donor_splits
     * @param requested_candidates how many trees to generate
     * @param donor_offset offset within the donor list, different for each worker to ensure unique trees
     * @param base_seed base seed for parsimony tree generation. incremented by the donor_offset and generated tree number
     */
    TreeList generate_candidates(
        const RaxmlInstance &instance,
        const ConstTreeRange &donor_pool,
        const SplitList &donor_splits,
        unsigned int requested_candidates,
        unsigned int donor_offset,
        unsigned long base_seed) override;
};


#endif //RAXML_NG_PARSIMONYSAMPLERSOURCE_HPP
