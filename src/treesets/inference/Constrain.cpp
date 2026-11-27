#include "Constrain.hpp"

#include "../batch/SharedBatchResources.hpp"

static std::optional<Tree> get_reverse_backbone(const TreeInfo &tree) {
    // obtain the topology with negative branch lengths to invert the order of branch lengths.
    Tree constraint = tree.tree();

    // threshold for the reverse-backbone
    constexpr auto cutoff_threshold = RAXML_BRLEN_MIN;

    const auto tip_list = constraint.tip_labels_list();
    const auto tip_id_map = constraint.tip_ids();
    NameList remove_list;

    for (auto &label: tip_list) {
        const auto tip_id = tip_id_map.at(label);
        assert(CORAX_UTREE_IS_TIP(constraint.pll_utree().nodes[tip_id]));
        if (constraint.pll_utree().nodes[tip_id]->length > cutoff_threshold + CORAX_ONE_EPSILON) {
            remove_list.push_back(label);
        }
    }

    // at least 4 tips need to remain in the dataset
    if (remove_list.size() >= tip_list.size() - 3) {
        return std::nullopt;
    }

    // if we aren't removing any tips, the constraint is too restrictive
    if (remove_list.size() < 3) {
        return std::nullopt;
    }

    constraint.remove_tips(remove_list);

    // TODO verify that we do not obtain a constraint containing all or no branches, but only a certain portion are in
    //  the constraint.

    constraint.collapse_long_branches(cutoff_threshold * 2);

    return constraint;
}

namespace {
    struct Clade {
        double diameter;
        uintVector taxa;

        explicit Clade(const double diameter) : diameter(diameter) {
        }
    };

    struct State {
        /** number of taxa of the full binary tree */
        size_t num_taxa;

        /** maximum diameter a constrained clade is allowed to have */
        double max_diameter;

        /** patristic distances of the full binary tree */
        doubleVector patristic;

        /** pointer to tree root to recognize it easily during traversal */
        corax_unode_t *root;

        /** number of internal nodes required to keep the locked clades locked. */
        size_t locked_internal_nodes = 0;

        /** list of taxa which are included in the constraint tree */
        uintVector keep_taxa;

        /** pointers to tree nodes root to a locked clade */
        std::vector<corax_unode_t *> subtrees;

        State(corax_unode_t *root, const size_t num_taxa, const double max_diameter,
              doubleVector patristic) : num_taxa(num_taxa),
                                        max_diameter(max_diameter),
                                        patristic(std::move(patristic)),
                                        root(root) {
        }

        /**
         * Lock a subtree in the final tree constraint if it is not too small.
         *
         * @param clade clade data
         * @param node tree node which is the root of the locked subtree
         */
        void lock_clade(const Clade *clade, corax_unode_t *node) {
            // ignore clades which are less than 3 taxa
            if (clade->taxa.size() < 3) {
                return;
            }

            for (auto tax: clade->taxa) {
                this->keep_taxa.push_back(tax);
            }

            this->subtrees.push_back(node);
            this->locked_internal_nodes += clade->taxa.size() - 1;
        }
    };

}

static double clade_diameter(const uintVector &tip_ids, const size_t num_tips, const doubleVector &distances) {
    auto diam = 0.0;
    for (size_t x = 0; x < tip_ids.size(); x++) {
        for (size_t y = x + 1; y < tip_ids.size(); y++) {
            diam = max(diam, distances[tip_ids[x] * num_tips + tip_ids[y]]);
        }
    }

    return diam;
}

static int identify_clades(corax_unode_t *node, void *data) {
    const auto state = static_cast<State *>(data);

    if (node == state->root) {
        // TODO: if only two clades are to be locked, we can still try merging them, which will help if the root
        //  is close to the leaves of those clades. If all three clades are to be locked, this isn't worth it, as
        //  the tree is already strongly constrained at that point.

        const auto left_node = node->next->back;
        const auto left = static_cast<Clade *>(left_node->data);
        left_node->data = nullptr;

        const auto right_node = node->next->next->back;
        const auto right = static_cast<Clade *>(right_node->data);
        right_node->data = nullptr;

        const auto back_node = node->back;
        const auto back = static_cast<Clade *>(back_node->data);
        back_node->data = nullptr;

        if (right) {
            state->lock_clade(right, right_node);
        }

        if (left) {
            state->lock_clade(left, left_node);
        }

        if (back) {
            state->lock_clade(back, back_node);
        }
    } else if (node->next) {
        const auto left_node = node->next->back;
        const auto left = static_cast<Clade *>(left_node->data);
        left_node->data = nullptr;

        const auto right_node = node->next->next->back;
        const auto right = static_cast<Clade *>(right_node->data);
        right_node->data = nullptr;

        // make sure we have an empty node unless we explicitly specify something
        node->data = nullptr;

        if (!left && right) {
            assert(right->diameter < state->max_diameter);
            state->lock_clade(right, right_node);
            delete right;
            return 0;
        }

        if (!right && left) {
            assert(left->diameter < state->max_diameter);
            state->lock_clade(left, left_node);
            delete left;
            return 0;
        }

        if (right && left) {
            // calculate union of child clades
            auto new_clade_taxa = left->taxa;
            for (auto tax: right->taxa) {
                new_clade_taxa.push_back(tax);
            }

            // if the union of these clades exceeds the maximum, we need to finish the clades of the children
            if (const auto new_clade_diameter = clade_diameter(new_clade_taxa, state->num_taxa, state->patristic);
                new_clade_diameter > state->max_diameter || new_clade_taxa.size() > state->num_taxa / 2) {

                assert(left->diameter < state->max_diameter);
                assert(right->diameter < state->max_diameter);

                state->lock_clade(left, left_node);
                state->lock_clade(right, right_node);

                delete left;
                delete right;
            } else {
                // combine clades to new encompassing clade and delete unused child data
                left->diameter = new_clade_diameter;
                left->taxa = new_clade_taxa;
                node->data = left;

                delete right;
            }
        }
    } else {
        // tip node
        auto clade = make_unique<Clade>(0);
        clade->taxa.push_back(node->node_index);
        node->data = clade.release();
    }

    return 0;
}

static std::optional<Tree> lock_clades(const TreeInfo &tree, const SharedBatchResources &resources) {
    const auto base_tree = tree.tree();
    auto utree = corax_utree_clone(&base_tree.pll_utree());
    const auto num_taxa = utree->tip_count;
    State state(utree->vroot, num_taxa, resources.get_clade_diameter(), base_tree.patristic_distances());

    // collect subtrees that we want to leave constrained as is
    corax_utree_traverse_apply(utree->vroot, nullptr, nullptr, identify_clades, &state);

    // do not lock the entire tree or leave it unconstrained
    if (state.keep_taxa.size() < 3) {
        return nullopt;
    }

    assert(state.subtrees.size() > 0);

    // build a new root node from the first three clades
    corax_unode_t *new_root = nullptr;

    // keep an entry into the old graph. We cannot use the old vroot, because it could be contracted during a prune.
    corax_unode_t *old_graph_entry;

    if (state.subtrees.size() == 1) {
        corax_utree_bisect(state.subtrees.front(), &new_root, &old_graph_entry);
    } else {
        const auto free_taxa = num_taxa - state.keep_taxa.size();

        auto wire_into_root = [](corax_unode_t **root, corax_unode_t *single_node) {
            if (!*root) {
                *root = single_node;
                single_node->next = single_node;
            } else {
                corax_unode_t *old_next = (*root)->next;
                (*root)->next = single_node;
                single_node->next = old_next;
            }
        };

        // move all remaining constrained subtrees to the new root
        for (size_t clade_id = 0; clade_id < state.subtrees.size(); clade_id++) {
            const auto clade = state.subtrees[clade_id];

            // if the remaining tree is a star, i.e. there are exactly three remaining clades or free taxa
            if (free_taxa + state.subtrees.size() - clade_id == 3) {
                const auto star = clade->back;
                old_graph_entry = nullptr; // nothing will be left, nothing to destroy

                // identify potential clades
                corax_unode_t *remaining_clades[3] = {star, star->next, star->next->next};
                assert(star->next->next->next == star);

                if (star->label) free(star->label);  // inner label shared by the ring

                // identify which of the nodes we keep, destroy the rest
                std::vector<corax_unode_t *> keep;
                for (auto m: remaining_clades) {
                    m->label = nullptr;
                    if (!m->back->next) {          // free taxon: delete tip (and its label) and this unode
                        corax_unode_destroy(m->back, nullptr);
                        free(m);
                    } else {
                        keep.push_back(m); // has to be a part of subtrees array
                    }
                }

                // wire the remaining nodes into the polytomy.
                // They aren't full rings, since we detached them from the root, so there is nothing to destroy
                for (const auto dangling_node : keep) {
                    wire_into_root(&new_root, dangling_node);
                }

                break; // we wired everything from the root, escape the loop
            }

            // if there is more remaining tree than a star, prune the clade and add it to the root
            old_graph_entry = corax_utree_prune(clade->back);

            // store pointer to the pruned ring
            const auto dangling_node = clade->back;
            const auto old_ring = dangling_node->next;

            // rewire the connected node of the dangling ring into the polytomy
            wire_into_root(&new_root, dangling_node);

            assert(old_ring->back == NULL);
            assert(old_ring->next->back == NULL);
            assert(old_ring->next->next == dangling_node);
            // rewire ring to exclude the rewired node
            old_ring->next->next = old_ring;

            // free the left-over ring. The one node we keep, we already extracted
            corax_unode_destroy(old_ring, nullptr);
        }

        // in case we connected exactly two nodes, our new root has only two parts to its ring, which is invalid.
        // Delete it. Otherwise, count the new artificial root as an extra internal node
        size_t ring = 1;
        for (auto u = new_root->next; u != new_root; u = u->next) ++ring;

        if (ring == 2) {
            corax_unode_t *root_n1 = new_root, *root_n2 = new_root->next;
            corax_unode_t *clade1 = root_n1->back, *clade2 = root_n2->back;
            corax_utree_connect_nodes(clade1, clade2, root_n1->length + root_n2->length);
            free(root_n1);
            free(root_n2);
            new_root = clade1;
        } else {
            state.locked_internal_nodes += 1;  // the root polytomy is one extra inner node
        }
    }

    // delete leftover tree
    if (old_graph_entry) corax_utree_graph_destroy(old_graph_entry, nullptr);

    // assign new internal indices for nodes and tips
    corax_utree_reset_template_indices(new_root, state.keep_taxa.size());

    // call wrap_tree to the new root and get a new tree item
    const auto constraint = corax_utree_wraptree_multi(new_root, state.keep_taxa.size(),
                                                       state.locked_internal_nodes);

    return Tree(*constraint);
}

void Constrain::do_optimize(std::optional<TreeInfo> &tree, const unsigned int tree_id, const Options &opts,
                            const TaskGroup &context, SharedBatchResources &, const unsigned int worker_id,
                            const unsigned int thread_id) {


    auto constraint = lock_clades(tree.value(), *resources);
    if (!constraint) {
        return;
    }

    // sort the tip ids so the constrained ids come first:
    NameIdMap new_label_id_map;
    IDVector new_tip_msa_map;
    new_tip_msa_map.resize(resources->get_msa().taxon_count());
    const auto cons_name_map = constraint->tip_ids();
    size_t seq_id = 0;
    size_t cons_tip_id = 0;
    size_t free_tip_id = constraint->num_tips();
    for (const auto &tip_name: resources->get_msa().taxon_names()) {
        const auto tip_id = cons_name_map.count(tip_name) ? cons_tip_id++ : free_tip_id++;
        new_label_id_map[tip_name] = tip_id;
        new_tip_msa_map[tip_id] = seq_id++;
    }
    assert(cons_tip_id == constraint->num_tips());
    assert(free_tip_id == new_tip_msa_map.size());
    assert(new_label_id_map.size() == resources->get_msa().taxon_count());

    if (context.is_group_leader(worker_id, thread_id) && tree_id == 0) {
        LOG_INFO_TS << "Constraining " << cons_tip_id << " taxa for tree #" << tree_id << "." << std::endl;
    }

    auto topology = tree->tree();
    topology.reset_tip_ids(new_label_id_map);
    constraint->reset_tip_ids(new_label_id_map);

    ModelMap target;
    for (auto part_info: part_assignments->at(thread_id)) {
        assign(target[part_info.part_id], *tree, part_info.part_id);
    }

    tree.emplace(TreeInfo(opts, topology, resources->get_msa(), new_tip_msa_map, part_assignments->at(thread_id)));
    assign_models(*tree, target);

    tree->set_topology_constraint(*constraint);
    assert(constraint->compatible(tree->tree()));

    // make sure the trees aren't used until all constraints are applied
    ParallelContext::barrier();
}
