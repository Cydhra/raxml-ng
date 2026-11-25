#ifndef RAXML_SHAREDBATCHRESOURCES_HPP_
#define RAXML_SHAREDBATCHRESOURCES_HPP_

#include "../../pool/Threadpool.hpp"
#include "../../topology/AuTest.hpp"
#include "../../Optimizer.hpp"
#include "../start/SplitSamplerSource.hpp"
#include "../start/ParsimonySource.hpp"
#include "../start/ParsimonySamplerSource.hpp"
/**
 * Shallow replication counts for a faster AU test.
 */
const uintVector SHALLOW_REPS = {500, 500, 500, 500, 500, 500, 500, 500, 500, 500};

constexpr unsigned int DEFAULT_FACTOR = 1;

/**
 * We share AUTest instances between batches, one per TaskGroup of the threadpool, to save on resource and reuse
 * the computed likelihood values for the reference trees.
 */
class SharedBatchResources {
public:
    SharedBatchResources(const unsigned int num_task_groups,
                         const unsigned int workers_per_group,
                         const unsigned int total_threads,
                         const Options &opts,
                         const TreeList &initial_ml_trees,
                         const Tree &baseline_tree,
                         const std::shared_ptr<PartitionedMSA> &msa,
                         const IDVector &tip_msa_idmap,
                         const std::vector<std::vector<doubleVector> > &reference_persite_loglh,
                         const unsigned int batch_size,
                         unsigned int seed,
                         const double pythia_score) : msa(msa),
                                                      tip_msa_idmap(make_shared<IDVector>(tip_msa_idmap)),
                                                      reference_ml_trees(make_shared<TreeList>(initial_ml_trees)),
                                                      reference_persite_loglh(
                                                          make_shared<std::vector<std::vector<doubleVector> > >(
                                                              reference_persite_loglh)),
                                                      gate(make_shared<EbgGate>(
                                                          opts, baseline_tree, this->reference_ml_trees)),
                                                      parsimony_source(make_shared<ParsimonySource>(seed)),
                                                      constrained_parsimony_sampling(
                                                          make_shared<ParsimonySamplerSource>(
                                                              parsimony_source, gate, seed, initial_ml_trees,
                                                              baseline_tree)),
                                                      greedy_split_sampling(
                                                          make_shared<SplitSamplerSource>(
                                                              parsimony_source, gate, seed, initial_ml_trees,
                                                              baseline_tree)) {
        // prepare a dummy matrix with empty vectors to correctly initialize the AU-Test. These dummy vectors
        // will be replaced by the TunedBatch instance before the AU test is called.
        std::vector<std::vector<doubleVector> > batch_loglh_dummy(batch_size);
        for (unsigned int i = 0; i < batch_size; ++i) {
            batch_loglh_dummy[i] = std::vector<doubleVector>(reference_persite_loglh[0].size());
        }

        for (unsigned int group = 0; group < num_task_groups; ++group) {
            screening_au_tests.emplace_back(msa, reference_persite_loglh, batch_loglh_dummy, AU_DEFAULT_SCALES,
                                            SHALLOW_REPS, seed);
            au_test_factors.emplace_back(DEFAULT_FACTOR);
            au_initialized.emplace_back(false);
        }

        // configure options for the raxml-fast optimizer
        fast_options = make_shared<Options>(opts);
        fast_options->topology_opt_method = TopologyOptMethod::simplified;
        fast_options->stopping_rule = StoppingRule::kh_mult;
        fast_options->nofiles_mode = true;
        fast_options->num_searches = batch_size;

        adaptive_options = make_shared<Options>(opts);
        adaptive_options->topology_opt_method = TopologyOptMethod::adaptive;
        adaptive_options->stopping_rule = StoppingRule::none;
        adaptive_options->nofiles_mode = true;
        adaptive_options->num_searches = batch_size;

        // initialize an optimizer and checkpoint manager for raxml-fast.
        fast_optimizer = make_shared<Optimizer>(*fast_options);
        adaptive_optimizer = make_shared<Optimizer>(*adaptive_options);
        checkpoint_manager = make_shared<CheckpointManager>(*fast_options);
        adaptive_checkpoint_manager = make_shared<CheckpointManager>(*adaptive_options);

        fast_stop = make_shared<KHStoppingTest>(msa,
                                                workers_per_group * num_task_groups,
                                                total_threads,
                                                true,
                                                opts.random_seed,
                                                opts.lh_epsilon);

        // unfortunately this method wants a tree. Please do not ask why it wants that, it doesn't deserve the tree.
        // But we have to comply, so we give it one of the reference trees since
        // any tree that conforms to the MSA will do.
        checkpoint_manager->init_checkpoints(baseline_tree, msa->models(), num_task_groups * workers_per_group);

        adaptive_checkpoint_manager->pythia_score(pythia_score);
        adaptive_checkpoint_manager->
                init_checkpoints(baseline_tree, msa->models(), num_task_groups * workers_per_group);

        max_clade_diameter = find_constraint_diameter();
        LOG_INFO << "Maximum Clade Diameter: " << max_clade_diameter << std::endl;
    }

    /**
     * The target factor that the au_test_factors should have. Since thread groups are responsible with allocating
     * and freeing their AuTest instance, this factor can be increased from the outside to make thread groups increase
     * their AU test size at the next convenient time.
     */
    std::unique_ptr<atomic_uint> target_au_test_factor = make_unique<atomic_uint>(DEFAULT_FACTOR);

    /**
     * How many of the reference trees should be plausible.
     */
    std::unique_ptr<atomic_int> expected_plausible_reference_trees = make_unique<atomic_int>(-1);

    /**
     * @return the current run's MSA
     */
    [[nodiscard]] PartitionedMSA &get_msa() const {
        return *msa;
    }

    [[nodiscard]] std::shared_ptr<PartitionedMSA> share_msa() const {
        return msa;
    }

    /**
     * @return the current run's MSA's tip ID vector
     */
    [[nodiscard]] IDVector &get_tip_msa_idmap() const {
        return *tip_msa_idmap;
    }

    [[nodiscard]] std::vector<std::vector<doubleVector> > get_reference_persite_loglh() const {
        return *reference_persite_loglh;
    }

    /**
     * @return An Optimizer instance pre-configured to run `RAxML-ng --fast` inference
     */
    [[nodiscard]] Optimizer &get_fast_optimizer() const {
        return *fast_optimizer;
    }

    /**
     * @return An Optimizer instance pre-configured to run `RAxML-ng --adaptive` inference
     */
    [[nodiscard]] Optimizer &get_adaptive_optimizer() const {
        return *adaptive_optimizer;
    }

    /**
     * @return The KH stopping criterion instance pre-configured to run `RAxML-ng --fast` inference
     */
    std::shared_ptr<KHStoppingTest> get_fast_stop_criterion() {
        return fast_stop;
    }

    /**
     * @return an instance of the CheckpointManager that is configured to work for raxml fast inference
     */
    [[nodiscard]] CheckpointManager &get_fast_cm() const {
        return *checkpoint_manager;
    }

    /**
     * @return an instance of the CheckpointManager that is configured to work for raxml adaptive inference
     */
    [[nodiscard]] CheckpointManager &get_adaptive_cm() const {
        return *adaptive_checkpoint_manager;
    }

    /**
     * Get an AuTest instance exclusively for the caller batch.
     *
     * @param context the parallel context for the threads that require an au test instance
     * @return An AuTest instance that is reserved for the caller TaskGroup. Call is_initialized() to find out whether
     * the reference tree bootstrap values have been generated already.
     */
    AuTest &get_au_test(const TaskGroup &context) {
        return screening_au_tests.at(context.group_id());
    }

    /**
     * The batch leader of any batch may ask if the AU Test instance obtained by get_au_test
     * @param context the parallel context for the calling batch leader
     * @return true, if the AuTest instance returned for the calling thread has been initialized before.
     */
    [[nodiscard]] bool is_au_initialized(const TaskGroup &context) const {
        return au_initialized.at(context.group_id()) > 0;
    }

    /**
     * Set the AU Test instance of the calling batch leader to initialized. Later batches of the same task group no longer
     * have to compute bootstrap values for the reference trees.
     *
     * @param context the parallel context for the calling batch leader
     */
    void set_au_initialized(const TaskGroup &context) {
        this->au_initialized[context.group_id()] = 1;
    }

    /**
     * @return the factor currently used for AU test replicates
     */
    [[nodiscard]] unsigned int get_au_factor(const TaskGroup &context) const {
        return this->au_test_factors.at(context.group_id());
    }

    /**
     * Update the current au-test factor to a new value.
     */
    void set_au_factor(const TaskGroup &context, const unsigned int factor) {
        this->au_test_factors[context.group_id()] = factor;
    }

    [[nodiscard]] TreeSource &get_parsimony_source() const {
        return *parsimony_source;
    }

    [[nodiscard]] ResampleSource &get_parsimony_resample_source() const {
        return *constrained_parsimony_sampling;
    }

    [[nodiscard]] ResampleSource &get_greedy_split_source() const {
        return *greedy_split_sampling;
    }

    /**
     * Maximum diameter of clades that cannot be resolved. Estimated value.
     */
    [[nodiscard]] double get_clade_diameter() const {
        return max_clade_diameter;
    }

protected:
    /* ***********************
     * MSA and Input trees
     * ***********************/

    /**
     * The input MSA for the tree inference.
     */
    std::shared_ptr<PartitionedMSA> msa;

    /**
     * Vector mapping sequence IDs to the MSA.
     */
    std::shared_ptr<IDVector> tip_msa_idmap;

    /**
     * The set of reference ML trees inferred by the standard RAxML-ng pipeline.
     */
    std::shared_ptr<TreeList> reference_ml_trees;

    /**
     * Per-site log-likelihoods of the reference trees already inferred before the treeset heuristic kicked in.
     * These cannot change, and constitute the first part of the AU test input.
     */
    std::shared_ptr<std::vector<std::vector<doubleVector> > > reference_persite_loglh;

    /* ***********************
     * AU TEST
     * ***********************/

    /**
     * Shared AU test instances, one for each thread group. These are initialized with drastically reduced replication
     * counts to be able to be used for fast screening. If the implementation detects instability, the AU tests are
     * made more expensive.
     */
    std::vector<AuTest> screening_au_tests;

    /**
     * Factors for the SHALLOW_REPS constants of AU test replicate counts.
     */
    std::vector<unsigned int> au_test_factors;

    /**
     * Whether the corresponding Au Test has been used before, initializing the bootstrap values of the reference trees.
     * The flags are not atomic, because only the group leader of each batch is supposed to ask and only for its own
     * context.
     */
    std::vector<uint8_t> au_initialized;

    /* ***********************
     * Optimizer
     * ***********************/

    /**
     * A copy of the CLI Options instance, with every option required to run raxml-fast forcibly enabled.
     * This instance is shared with fast_optimizer and local_fast_cm, but they do not own it, so we store it here.
     */
    shared_ptr<Options> fast_options;

    /**
     * A copy of the CLI Options instance, with every option required to run raxml-adaptive forcibly enabled.
     * This instance is shared with adaptive_optimizer and local_fast_cm, but they do not own it, so we store it here.
     */
    shared_ptr<Options> adaptive_options;

    /**
     * An instance of the kh stopping criterion used for raxml-fast. It is used in the optimizer but not owned by it,
     * so we store it here. One for all workers is sufficient, RAxML-ng stores a single one in RaxmlInstance as well.
     */
    std::shared_ptr<KHStoppingTest> fast_stop;

    /**
     * Batch-local instance of the standard RAxML optimizer. This is instanced for the raxml-fast fallback bandit.
     * The options passed to this optimizer are forced to perform the --fast heuristic.
     */
    std::shared_ptr<Optimizer> fast_optimizer;

    /**
     * Batch-local instance of the standard RAxML optimizer. This is instanced for the raxml-adaptive fallback bandit.
     * The options passed to this optimizer are forced to perform the standard adaptive heuristic.
     */
    std::shared_ptr<Optimizer> adaptive_optimizer;

    /**
     * Batch-local instance of the standard RAxML checkpoint manager configured for raxml-fast.
     * This is instanced for the raxml-fast fallback bandit.
     */
    shared_ptr<CheckpointManager> checkpoint_manager;

    /**
     * Batch-local instance of the standard RAxML checkpoint manager configured for raxml-fast.
     * This is instanced for the raxml-fast fallback bandit.
     */
    shared_ptr<CheckpointManager> adaptive_checkpoint_manager;

    /* ***********************
     * Tree sources
     * ***********************/

    /**
     * EBG gate to reject this source if the generated trees do not approximate the expected distribution of splits.
     */
    shared_ptr<EbgGate> gate;

    /**
     * Global source of parsimony trees, such that trees can be reused for heuristics other than ML.
     */
    shared_ptr<ParsimonySource> parsimony_source;

    /**
     * Global source of parsimony trees built from a constrained and optimized with parsimony SPR.
     */
    shared_ptr<ParsimonySamplerSource> constrained_parsimony_sampling;

    /**
     * Global source of trees built from compatible splits from a set of reference trees.
     */
    shared_ptr<SplitSamplerSource> greedy_split_sampling;

    /* ***********************
     * Hyperparameters
     * ***********************/

    /**
     * The maximum diameter of unresolvable clades.
     */
    double max_clade_diameter = 0.0;

private:
    [[nodiscard]] double find_constraint_diameter() const {
        // generate patristic matrices of all reference trees
        TreeList collapsed_trees;
        doubleVector non_zero_branch_lengths;

        for (auto &tree: *reference_ml_trees) {
            for (auto len: tree.brlens()) {
                if (len > CORAX_OPT_MIN_BRANCH_LEN)
                    non_zero_branch_lengths.push_back(len);
            }

            collapsed_trees.push_back(tree);
            collapsed_trees.back().collapse_short_branches(CORAX_OPT_MIN_BRANCH_LEN);
        }

        if (non_zero_branch_lengths.empty()) {
            return 0.0;
        }

        // identify upper limit of clade diameter: median of non-minimal branch lengths
        sort(non_zero_branch_lengths.begin(), non_zero_branch_lengths.end());
        double diameter = non_zero_branch_lengths.at(non_zero_branch_lengths.size() / 2);

        // calculate split sets
        vector<corax_split_set_t *> split_sets;
        for (size_t tree_id = 0; tree_id < reference_ml_trees->size(); tree_id++) {
            auto &tree = collapsed_trees.at(tree_id);
            const auto s_set = corax_utree_splitset_create(&tree.pll_utree());
            split_sets.push_back(s_set);
        }

        // get the tip ids from a split. Set or unset side to get the 1 or 0 side.
        auto split_ids = [](const corax_split_set_t *sset, const unsigned int id, const bool side) {
            uintVector tips;
            corax_split_t split = sset->splits[id];
            for (size_t i = 0; i < sset->tip_count; ++i) {
                const bool in = split[i / sset->split_size] & (1u << (i % sset->split_size));
                if (in == side) tips.push_back(i);
            }
            return tips;
        };

        // get the diameter of a set of tips. Aborts when the maximum diameter exceeds limit.
        auto clade_diameter = [](const uintVector &tip_ids, const size_t num_tips, const doubleVector &distances,
                                 const double limit) {
            auto diam = 0.0;
            for (size_t x = 0; x < tip_ids.size(); x++) {
                for (size_t y = x + 1; y < tip_ids.size(); y++) {
                    diam = max(diam, distances[tip_ids[x] * num_tips + tip_ids[y]]);

                    if (diam > limit) {
                        return diam;
                    }
                }
            }

            return diam;
        };

        // for each split, take the smaller tree and test whether it disagrees with any other reference tree.
        // If so, make the maximum diameter smaller (we take steps of MIN_BRANCH_LENGTH).
        for (size_t tree_id = 0; tree_id < reference_ml_trees->size(); tree_id++) {
            const auto s_set = split_sets.at(tree_id);
            doubleVector patristic_matrix = reference_ml_trees->at(tree_id).patristic_distances();

            for (size_t split_id = 0; split_id < s_set->split_count; split_id++) {
                auto first_side = split_ids(s_set, split_id, true);
                auto second_side = split_ids(s_set, split_id, false);

                const auto clade_diam = min(
                    clade_diameter(first_side, msa->taxon_count(), patristic_matrix, diameter),
                    clade_diameter(second_side, msa->taxon_count(), patristic_matrix, diameter)
                );

                // if clade diameter is smaller or equal, check if it violates a split
                if (diameter - clade_diam > -1E-9) {
                    for (size_t other_tree = tree_id + 1; other_tree < split_sets.size(); other_tree++) {
                        const auto other_set = split_sets.at(other_tree);

                        for (size_t i = 0; i < other_set->split_count; i++) {
                            if (!corax_utree_split_compatible(s_set->splits[split_id], other_set->splits[i],
                                                              s_set->split_len, s_set->tip_count)) {
                                LOG_DEBUG << "Reject Diameter " << diameter << std::endl;
                                diameter = clade_diam - CORAX_OPT_MIN_BRANCH_LEN;
                                // diameter must be smaller than this one
                                goto break_check;
                            }
                        }
                    }
                }
            break_check:
                continue;
            }
        }

        return diameter;
    }
};

#endif //RAXML_SHAREDBATCHRESOURCES_HPP_
