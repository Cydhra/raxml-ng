#include "StartTreeHeuristicFactory.hpp"

#include "Parsimony.hpp"
#include "ProvidedStartTrees.hpp"
#include "../AggressiveCandidateManager.hpp"

std::optional<TreeList> StartTreeHeuristicFactory::prepare(
    const MetaParameters &parameters, const unsigned int batch_size) const {
    if (parameters.uses_aggressive_starting_trees()) {
        if (!candidate_manager)
            return std::nullopt;

        auto trees = candidate_manager->take_batch(parameters.starting_tree_source, batch_size);
        if (trees.size() != batch_size)
            return std::nullopt;
        return trees;
    }

    return candidate_manager
               ? candidate_manager->take_reusable_parsimony(batch_size)
               : TreeList{};
}

std::unique_ptr<StartTreeHeuristic> StartTreeHeuristicFactory::build(
    const MetaParameters &parameters,
    std::string batch_name,
    const unsigned int batch_size,
    const unsigned long starting_seed,
    TreeList prepared_trees) const {
    if (parameters.uses_aggressive_starting_trees()) {
        return make_unique<ProvidedStartTrees>(
            std::move(batch_name), std::move(prepared_trees), starting_seed);
    }

    return make_unique<Parsimony>(
        std::move(batch_name), batch_size, starting_seed, nullptr,
        std::move(prepared_trees));
}

std::string StartTreeHeuristicFactory::batch_name(
    const MetaParameters &parameters, const std::size_t batch_index) const {
    const char *prefix = "Batch";
    if (parameters.starting_tree_source == StartingTreeSource::seed_greedy)
        prefix = "AggressiveSeedGreedyBatch";
    else if (parameters.starting_tree_source == StartingTreeSource::constrained_parsimony)
        prefix = "AggressiveConstrainedParsimonyBatch";
    return std::string(prefix) + std::to_string(batch_index);
}
