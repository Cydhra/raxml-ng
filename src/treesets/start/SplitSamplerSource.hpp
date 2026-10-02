#ifndef RAXML_NG_AGGRESSIVESOURCE_HPP
#define RAXML_NG_AGGRESSIVESOURCE_HPP

#include <utility>
#include <memory>

#include "ParsimonySource.hpp"
#include "TreeSource.hpp"
#include "../util/EbgGate.hpp"

using namespace std;

Tree generate_parsimony_tree(const RaxmlInstance &instance,
                             int random_seed,
                             bool bootstrap,
                             const Tree &constraint_tree);

struct RaxmlInstance;

constexpr size_t NUM_DONOR_TREE = 300;

class SplitSamplerSource : public TreeSource {
public:
    explicit SplitSamplerSource(const std::shared_ptr<ParsimonySource> &donor,
                                const std::shared_ptr<EbgGate> &gate,
                                const int seed,
                                TreeList initial_ml_trees,
                                Tree baseline_tree) : gate(gate),
                                                      baseline_tree(std::move(baseline_tree)),
                                                      initial_ml_trees(std::move(initial_ml_trees)),
                                                      donor_tree_source(donor), seed(seed) {
    }

    bool ensure(const RaxmlInstance &instance, const SmartBarrier &barrier, unsigned int threads_per_worker,
                unsigned int worker_id, unsigned int thread_id, unsigned int required_trees) override;

    /**
     * Generate trees in the source. This is not implemented in `ensure` to avoid refilling it.
     */
    void generate(const RaxmlInstance &instance, const SmartBarrier &barrier, unsigned int threads_per_worker,
                  unsigned int worker_id, unsigned int thread_id, unsigned int num_trees);

    [[nodiscard]] double amortized_time(unsigned int batch_size) const override;

protected:
    /**
     * EBG-gate that deactivates the tree source if the generated trees cannot reproduce the expected bootstrap
     * distribution of split frequencies.
     */
    shared_ptr<EbgGate> gate;

    /**
     * Best ML tree from reference set to use for labels
     */
    const Tree baseline_tree;

    /**
     * All initially inferred ML trees
     */
    const TreeList initial_ml_trees;

    /**
     * Tree source which yields a number of donor trees required for statistics in the implementing heuristic.
     */
    std::shared_ptr<ParsimonySource> donor_tree_source;

    /**
     * The starting seed for generating trees. Seed is increased by 1 for each tree.
     */
    int seed;

    /**
     * Mutex for inserting topologies intp seen_topologies
     */
    std::unique_ptr<std::mutex> duplicate_filter_mutex = std::make_unique<std::mutex>();

    /**
     * All topologies previously encountered to avoid duplicate trees generated.
     */
    std::set<Split> seen_topologies;

    /**
     * How often a call to ensure() created trees. This variable is mutex-guarded.
     */
    int sampled_batches = 0;

    /**
     * Walltime measurements of tree generation for later bandit time estimation.
     */
    std::unique_ptr<std::atomic_uint> cumulative_wall_time = std::make_unique<std::atomic_uint>(0);

    bool is_unique(const Tree &candidate);

    /**
     * Get a hash key for a topology.
     * @param tree tree topology
     * @return a bitvector consisting of all appended splits in the given tree, serving as a hash key
     */
    static std::vector<corax_split_base_t> topology_key(const Tree &tree);

    static bool has_majority_split(
        const SplitList &donor_topologies,
        const std::vector<std::size_t> &donor_ids,
        std::size_t words_per_split);

    static SplitList extract_splits(const Tree &tree, bool normalize);

    /**
     * Generate new candidates using the pool of donor trees. Implementations of this class may implement different
     * heuristics to generate trees.
     *
     * @param instance RaxmlInstance required for parsimony, if the subclass does Parsimony
     * @param donor_pool pool of trees used as donors for statistics required by implementing classes
     * @param donor_splits all splits of the donor trees
     * @param requested_candidates how many candidates are requested. Implementing classes may generate less trees.
     *                             If the number of generated trees is smaller than the minimum number required from
     *                             the ensure() call, the tree source deactivates itself.
     * @param round_seed seed for generating trees if required. Seed should be increased by 1 per tree.
     * @return A list of generated candidate trees
     */
    virtual TreeList generate_candidates(
        const RaxmlInstance &instance,
        const ConstTreeRange &donor_pool,
        const SplitList &donor_splits,
        unsigned int requested_candidates,
        unsigned long round_seed) = 0;
};


#endif //RAXML_NG_AGGRESSIVESOURCE_HPP
