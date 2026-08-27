#include "Constrain.hpp"

std::optional<Tree> get_reverse_backbone(TreeInfo &tree) {
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

    constraint.collapse_long_branches(cutoff_threshold);

    return constraint;
}

void Constrain::do_optimize(std::optional<TreeInfo> &tree, const Options &opts, const TaskGroup &, SharedBatchResources &, unsigned int, unsigned int thread_id) {
    auto constraint = get_reverse_backbone(tree.value());

    if (!constraint) {
        return;
    }

    // sort the tip ids so the constrained ids come first:
    NameIdMap new_label_id_map;
    IDVector new_tip_msa_map;
    new_tip_msa_map.resize(msa->taxon_count());
    auto cons_name_map = constraint->tip_ids();
    size_t seq_id = 0;
    size_t cons_tip_id = 0;
    size_t free_tip_id = constraint->num_tips();
    for (const auto &tip_name: msa->taxon_names()) {
        auto tip_id = cons_name_map.count(tip_name) ? cons_tip_id++ : free_tip_id++;
        new_label_id_map[tip_name] = tip_id;
        new_tip_msa_map[tip_id] = seq_id++;
    }
    assert(cons_tip_id == constraint->num_tips());
    assert(free_tip_id == new_tip_msa_map.size());
    assert(new_label_id_map.size() == msa->taxon_count());

    auto topology = tree->tree();
    topology.reset_tip_ids(new_label_id_map);
    constraint->reset_tip_ids(new_label_id_map);

    tree.emplace(TreeInfo(opts, topology, *msa, new_tip_msa_map, partition_assignments.at(thread_id)));
    tree->set_topology_constraint(*constraint);
    assert(constraint->compatible(tree->tree()));

    // make sure the trees aren't used until all constraints are applied
    ParallelContext::barrier();
}
