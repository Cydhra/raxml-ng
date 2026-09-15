#include "ProvidedStartTrees.hpp"

void ProvidedStartTrees::do_generate(Tree &tree, const unsigned int tree_id, const RaxmlInstance &,
                                     const TaskGroup &, SharedBatchResources &,
                                     unsigned int, unsigned int) {
    tree = std::move(trees.at(tree_id));
}
