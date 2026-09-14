#ifndef RAXML_STARTTREEHEURISTICFACTORY_HPP_
#define RAXML_STARTTREEHEURISTICFACTORY_HPP_

#include <optional>

#include "StartTreeHeuristic.hpp"
#include "../MetaParameters.hpp"

class AggressiveCandidateManager;

/**
 * Creates starting-tree strategies without exposing their concrete
 * implementations to the batch queue.
 */
class StartTreeHeuristicFactory {
public:
    explicit StartTreeHeuristicFactory(AggressiveCandidateManager *candidate_manager)
        : candidate_manager(candidate_manager) {}

    /**
     * Prepare any trees needed by a strategy. An empty optional means that the
     * requested source is exhausted; an empty TreeList is a valid ordinary
     * parsimony preparation.
     */
    std::optional<TreeList> prepare(const MetaParameters &parameters,
                                    unsigned int batch_size) const;

    std::unique_ptr<StartTreeHeuristic> build(const MetaParameters &parameters,
                                              std::string batch_name,
                                              unsigned int batch_size,
                                              unsigned long starting_seed,
                                              TreeList prepared_trees) const;

    std::string batch_name(const MetaParameters &parameters,
                           std::size_t batch_index) const;

private:
    AggressiveCandidateManager *candidate_manager;
};

#endif
