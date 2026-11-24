// Construction engine of persforest: plain C++17, no Python dependency.
//
// The phases mirror the former Python engine (persforest._forest.build,
// reduction and compute_barcode): the reverse sweep creates leaves, birth
// roots, merges and updates; the reduction collapses parent-child pairs within
// the tolerance; roots are normalized to one child each; bars follow the elder
// rule, visiting leaves from the latest created. Node IDs during construction
// are creation indices, so every parent has a larger ID than its children.
#include "engine.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace persforest {
namespace {

constexpr std::int64_t kNone = -1;

// Nodes in creation order. Each sweep node has at most one diff entry: the
// top cell of a leaf, or the facet of a merge or update; roots have none.
struct Nodes {
    std::vector<double> value;
    std::vector<std::int64_t> parent;
    std::vector<std::int64_t> diff_cell;
    std::vector<std::int8_t> diff_sign;
    std::vector<std::uint8_t> is_birth_root;

    std::int64_t add(double node_value, std::int64_t cell, std::int8_t sign, bool birth_root) {
        value.push_back(node_value);
        parent.push_back(kNone);
        diff_cell.push_back(cell);
        diff_sign.push_back(sign);
        is_birth_root.push_back(birth_root ? 1 : 0);
        return static_cast<std::int64_t>(value.size()) - 1;
    }
};

// Union-find over nodes, used only to find the current top of a leaf's tree.
// Every link points to a newly created node, so the forest stays acyclic.
std::int64_t find_top(std::vector<std::int64_t>& up, std::int64_t node) {
    while (up[node] != node) {
        up[node] = up[up[node]];
        node = up[node];
    }
    return node;
}

void check_input(const CellEvents& events) {
    if (events.n_events < 0 || events.n_top < 0 || events.n_cells < events.n_top || events.facets_per_top < 0) {
        throw std::invalid_argument("inconsistent cell filtration sizes");
    }
    if (events.n_events > 0 && (events.cells == nullptr || events.values == nullptr)) {
        throw std::invalid_argument("missing event arrays");
    }
    if (events.n_top > 0 && (events.top_orientation == nullptr ||
                             (events.facets_per_top > 0 && events.top_facets == nullptr))) {
        throw std::invalid_argument("missing top-cell arrays");
    }
}

// Reverse sweep: the complement of the growing complex, run backwards, merges.
Nodes sweep(const CellEvents& events) {
    const std::int64_t n_top = events.n_top;
    const std::int64_t k = events.facets_per_top;
    // Leaves created by the (at most two) top cells of each facet.
    std::vector<std::int64_t> slot_a(events.n_cells - n_top, kNone);
    std::vector<std::int64_t> slot_b(events.n_cells - n_top, kNone);
    Nodes nodes;
    std::vector<std::int64_t> up;
    nodes.value.reserve(events.n_events);
    up.reserve(events.n_events);

    auto add_node = [&](double value, std::int64_t cell, std::int8_t sign, bool birth_root) {
        std::int64_t id = nodes.add(value, cell, sign, birth_root);
        up.push_back(id);
        return id;
    };
    // Top of the tree of a leaf, or kNone if that tree has already been closed by a birth root.
    auto active_top = [&](std::int64_t leaf) {
        if (leaf == kNone) return kNone;
        std::int64_t top = find_top(up, leaf);
        return nodes.is_birth_root[top] ? kNone : top;
    };

    for (std::int64_t e = events.n_events - 1; e >= 0; --e) {
        const std::int64_t cell = events.cells[e];
        const double value = events.values[e];
        if (cell < 0 || cell >= events.n_cells) {
            throw std::invalid_argument("cell ID " + std::to_string(cell) + " out of range");
        }
        if (cell < n_top) {
            const std::int8_t orientation = events.top_orientation[cell];
            const std::int64_t leaf = add_node(value, cell, orientation, false);
            for (std::int64_t j = 0; j < k; ++j) {
                const std::int64_t facet = events.top_facets[cell * k + j];
                if (facet < n_top || facet >= events.n_cells) {
                    throw std::invalid_argument("facet ID " + std::to_string(facet) + " out of range");
                }
                const std::int64_t slot = facet - n_top;
                if (slot_a[slot] == kNone) {
                    slot_a[slot] = leaf;
                } else if (slot_b[slot] == kNone) {
                    slot_b[slot] = leaf;
                } else {
                    throw std::invalid_argument("facet ID " + std::to_string(facet) +
                                                " has more than two top cells; the complex is not embedded");
                }
            }
            continue;
        }
        const std::int64_t slot = cell - n_top;
        std::int64_t a = active_top(slot_a[slot]);
        std::int64_t b = active_top(slot_b[slot]);
        if (a == kNone) std::swap(a, b);
        if (a == kNone) continue;  // no active cycle contains the facet
        if (b == kNone) {
            // One active cycle contains the facet once: its class is born here.
            const std::int64_t root = add_node(value, kNone, 0, true);
            nodes.parent[a] = root;
            up[a] = root;
        } else if (a != b) {
            // Two cycles merge across the facet.
            const std::int64_t merged = add_node(value, cell, 0, false);
            nodes.parent[a] = merged;
            nodes.parent[b] = merged;
            up[a] = merged;
            up[b] = merged;
        } else {
            // The same cycle contains the facet in both orientations and drops it.
            const std::int64_t updated = add_node(value, cell, 0, false);
            nodes.parent[a] = updated;
            up[a] = updated;
        }
    }
    return nodes;
}

}  // namespace

ForestArrays build_forest(const CellEvents& events, bool reduce, double filtration_tol) {
    check_input(events);
    Nodes nodes = sweep(events);
    const std::int64_t n_swept = static_cast<std::int64_t>(nodes.value.size());

    // Reduction: a child within the tolerance above its parent is merged into
    // it. Parents have larger IDs, so descending IDs resolve chains top-down.
    std::vector<std::int64_t> representative(n_swept);
    for (std::int64_t v = n_swept - 1; v >= 0; --v) {
        const std::int64_t p = nodes.parent[v];
        representative[v] = v;
        if (!reduce || p == kNone) continue;
        if (nodes.value[v] < nodes.value[p]) {
            throw std::invalid_argument("invalid parent-child filtration order");
        }
        if (nodes.value[v] <= nodes.value[p] + filtration_tol) representative[v] = representative[p];
    }

    std::vector<double> value(nodes.value);
    std::vector<std::int64_t> parent(n_swept, kNone);
    std::vector<std::uint8_t> alive(n_swept, 0);
    for (std::int64_t v = 0; v < n_swept; ++v) {
        if (representative[v] != v) continue;
        alive[v] = 1;
        if (nodes.parent[v] != kNone) parent[v] = representative[nodes.parent[v]];
    }
    std::vector<std::int64_t> n_children(n_swept, 0);
    for (std::int64_t v = 0; v < n_swept; ++v) {
        if (alive[v] && parent[v] != kNone) ++n_children[parent[v]];
    }
    if (reduce) {
        // A root whose children all collapsed into it carries no class.
        for (std::int64_t v = 0; v < n_swept; ++v) {
            if (alive[v] && parent[v] == kNone && n_children[v] == 0) alive[v] = 0;
        }
        // Normalize roots: every child of a root gets a root of its own (new,
        // larger IDs); the first child keeps the existing root.
        std::vector<std::int64_t> first_child(n_swept, kNone);
        for (std::int64_t v = 0; v < n_swept; ++v) {
            if (!alive[v] || parent[v] == kNone || parent[parent[v]] != kNone) continue;
            const std::int64_t root = parent[v];
            if (first_child[root] == kNone) {
                first_child[root] = v;
                continue;
            }
            value.push_back(value[root]);
            parent.push_back(kNone);
            alive.push_back(1);
            n_children.push_back(1);
            --n_children[root];
            parent[v] = static_cast<std::int64_t>(value.size()) - 1;
        }
    }
    const std::int64_t n_nodes = static_cast<std::int64_t>(value.size());

    // Bars by the elder rule: leaves from the latest created; at a merge, the
    // first arriving branches end there and the last one continues.
    ForestArrays out;
    std::vector<std::int64_t> covered(n_nodes, 0);
    std::vector<std::int64_t> stop_node;
    std::vector<std::int64_t> walk;
    out.progression_offsets.push_back(0);
    for (std::int64_t leaf = n_nodes - 1; leaf >= 0; --leaf) {
        if (!alive[leaf] || n_children[leaf] != 0) continue;
        if (parent[leaf] == kNone) throw std::logic_error("persistence forest leaf without parent");
        walk.assign(1, leaf);
        std::int64_t p = parent[leaf];
        while (parent[p] != kNone) {
            if (covered[p] < n_children[p] - 1) {
                ++covered[p];
                break;
            }
            walk.push_back(p);
            ++covered[p];
            p = parent[p];
        }
        out.bar_birth.push_back(value[p]);
        out.bar_death.push_back(value[leaf]);
        stop_node.push_back(p);
        out.progression_nodes.insert(out.progression_nodes.end(), walk.rbegin(), walk.rend());
        out.progression_offsets.push_back(static_cast<std::int64_t>(out.progression_nodes.size()));
    }
    const std::int64_t n_bars = static_cast<std::int64_t>(out.bar_birth.size());

    // A bar splits off the bar whose progression contains its stop node.
    std::vector<std::int64_t> owner(n_nodes, kNone);
    for (std::int64_t b = 0; b < n_bars; ++b) {
        for (std::int64_t i = out.progression_offsets[b]; i < out.progression_offsets[b + 1]; ++i) {
            owner[out.progression_nodes[i]] = b;
        }
    }
    // Trees are numbered by their roots in ID order; parents have larger IDs.
    std::vector<std::int64_t> tree(n_nodes, kNone);
    std::int64_t n_trees = 0;
    for (std::int64_t v = 0; v < n_nodes; ++v) {
        if (alive[v] && parent[v] == kNone) tree[v] = n_trees++;
    }
    for (std::int64_t v = n_nodes - 1; v >= 0; --v) {
        if (alive[v] && parent[v] != kNone) tree[v] = tree[parent[v]];
    }
    out.bar_parent.resize(n_bars);
    out.bar_tree.resize(n_bars);
    for (std::int64_t b = 0; b < n_bars; ++b) {
        const std::int64_t stop = stop_node[b];
        out.bar_parent[b] = parent[stop] == kNone ? kNone : owner[stop];
        out.bar_tree[b] = tree[stop];
    }

    // Depth-first post-order: roots and children by ascending ID.
    std::vector<std::int64_t> child_offsets(n_nodes + 1, 0);
    for (std::int64_t v = 0; v < n_nodes; ++v) {
        if (alive[v] && parent[v] != kNone) ++child_offsets[parent[v] + 1];
    }
    for (std::int64_t v = 0; v < n_nodes; ++v) child_offsets[v + 1] += child_offsets[v];
    std::vector<std::int64_t> children(child_offsets[n_nodes]);
    {
        std::vector<std::int64_t> fill(child_offsets.begin(), child_offsets.end() - 1);
        for (std::int64_t v = 0; v < n_nodes; ++v) {
            if (alive[v] && parent[v] != kNone) children[fill[parent[v]]++] = v;
        }
    }
    std::vector<std::int64_t> post(n_nodes, kNone);
    std::vector<std::int64_t> first(n_nodes, kNone);
    std::int64_t counter = 0;
    std::vector<std::pair<std::int64_t, std::int64_t>> stack;
    for (std::int64_t root = 0; root < n_nodes; ++root) {
        if (!alive[root] || parent[root] != kNone) continue;
        stack.emplace_back(root, child_offsets[root]);
        first[root] = counter;
        while (!stack.empty()) {
            auto& [node, next] = stack.back();
            if (next < child_offsets[node + 1]) {
                const std::int64_t child = children[next++];
                first[child] = counter;
                stack.emplace_back(child, child_offsets[child]);
            } else {
                post[node] = counter++;
                stack.pop_back();
            }
        }
    }
    const std::int64_t n_out = counter;

    out.node_value.resize(n_out);
    out.node_parent.resize(n_out);
    out.node_first.resize(n_out);
    for (std::int64_t v = 0; v < n_nodes; ++v) {
        if (post[v] == kNone) continue;
        out.node_value[post[v]] = value[v];
        out.node_parent[post[v]] = parent[v] == kNone ? kNone : post[parent[v]];
        out.node_first[post[v]] = first[v];
    }
    for (std::int64_t& node : out.progression_nodes) node = post[node];

    // Diffs of collapsed nodes belong to their representative; roots carry none.
    auto diff_owner = [&](std::int64_t u) {
        if (nodes.diff_cell[u] == kNone) return kNone;
        const std::int64_t owner_node = representative[u];
        if (!alive[owner_node] || parent[owner_node] == kNone) return kNone;
        return post[owner_node];
    };
    out.diff_offsets.assign(n_out + 1, 0);
    for (std::int64_t u = 0; u < n_swept; ++u) {
        const std::int64_t target = diff_owner(u);
        if (target != kNone) ++out.diff_offsets[target + 1];
    }
    for (std::int64_t v = 0; v < n_out; ++v) out.diff_offsets[v + 1] += out.diff_offsets[v];
    out.diff_cells.resize(out.diff_offsets[n_out]);
    out.diff_signs.resize(out.diff_offsets[n_out]);
    {
        std::vector<std::int64_t> fill(out.diff_offsets.begin(), out.diff_offsets.end() - 1);
        for (std::int64_t u = 0; u < n_swept; ++u) {
            const std::int64_t target = diff_owner(u);
            if (target == kNone) continue;
            const std::int64_t slot = fill[target]++;
            out.diff_cells[slot] = nodes.diff_cell[u];
            out.diff_signs[slot] = nodes.diff_sign[u];
        }
    }
    return out;
}

}  // namespace persforest
