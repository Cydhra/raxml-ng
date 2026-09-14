#ifndef RAXML_NG_PARSIMONY_HPP
#define RAXML_NG_PARSIMONY_HPP
#include "StartTreeHeuristic.hpp"

class Parsimony : public StartTreeHeuristic {
public:
    Parsimony(std::string batch_name, const unsigned num_trees, const unsigned starting_seed,
        std::unique_ptr<StartTreeHeuristic> inner, TreeList reusable_trees = {})
        : StartTreeHeuristic(std::move(batch_name), num_trees, starting_seed, std::move(inner)),
          reusable_trees(std::move(reusable_trees)) {
    }

    void do_generate(Tree &tree, unsigned tree_id, const RaxmlInstance &instance,
                     const TaskGroup &context, SharedBatchResources &resources, unsigned worker_id, unsigned thread_id) override;

private:
    TreeList reusable_trees;
};


#endif //RAXML_NG_PARSIMONY_HPP
