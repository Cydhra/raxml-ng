#ifndef RAXML_PROVIDEDSTARTTREES_HPP_
#define RAXML_PROVIDEDSTARTTREES_HPP_

#include "StartTreeHeuristic.hpp"

/** A complete, precomputed batch exposed through the start-tree strategy API. */
class ProvidedStartTrees final : public StartTreeHeuristic {
public:
    ProvidedStartTrees(std::string batch_name, TreeList trees, unsigned int starting_seed)
        : StartTreeHeuristic(std::move(batch_name), trees.size(), starting_seed, nullptr),
          trees(std::move(trees)) {}

    void do_generate(Tree &tree, unsigned int tree_id, const RaxmlInstance &instance,
                     const TaskGroup &context, SharedBatchResources &resources,
                     unsigned int worker_id, unsigned int thread_id) override;

private:
    TreeList trees;
};

#endif
