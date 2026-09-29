#ifndef RAXML_NG_TREEREPOSITORY_HPP
#define RAXML_NG_TREEREPOSITORY_HPP

#include "../../Tree.hpp"

class TreeRepository {
public:
    void append_candidate(Tree candidate);

    TreeList take_candidate_batch(unsigned int batch_size);

    std::size_t candidate_count() const;

protected:
    std::deque<Tree> candidates;

    std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();
};


#endif //RAXML_NG_TREEREPOSITORY_HPP
