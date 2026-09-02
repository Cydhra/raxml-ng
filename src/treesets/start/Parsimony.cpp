#include "Parsimony.hpp"

// forward declaration of generate_tree in main.cpp to make it accessible. If the function in main.cpp
// changes signature, just update this declaration as well.
Tree generate_tree(const RaxmlInstance &instance, StartingTree type, int random_seed, bool bootstrap);

void Parsimony::do_generate(Tree &tree, const unsigned int tree_id, const RaxmlInstance &instance, const TaskGroup &,
                            SharedBatchResources &, unsigned int, unsigned int) {
    tree = generate_tree(instance, StartingTree::parsimony, static_cast<int>(seeds[tree_id]), false);
}
