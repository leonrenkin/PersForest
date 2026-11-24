// Construction engine of persforest: plain C++17, no Python dependency.
//
// The engine runs the reverse sweep of a cell filtration (top cells and their
// facets, integer cell IDs), optionally reduces the forest, extracts the
// elder-rule bars and returns everything as flat arrays.
#pragma once

#include <cstdint>
#include <vector>

namespace persforest {

// Version of the array interface between the engine and the Python layer.
// Increase it whenever inputs or outputs of the engine change.
constexpr int kEngineInterfaceVersion = 2;

// Cell filtration as consumed by the sweep. IDs 0 .. n_top - 1 are top cells,
// IDs n_top .. n_cells - 1 their facets. All pointers are borrowed.
struct CellEvents {
    const std::int64_t* cells = nullptr;      // event cell IDs, forward filtration order
    const double* values = nullptr;           // filtration value of each event
    std::int64_t n_events = 0;
    std::int64_t n_top = 0;
    std::int64_t n_cells = 0;
    const std::int64_t* top_facets = nullptr;  // n_top x facets_per_top, row-major
    std::int64_t facets_per_top = 0;
    const std::int8_t* top_orientation = nullptr;  // +1 or -1 per top cell
};

// The forest as flat arrays.
//
// Nodes are numbered in depth-first post-order, so the subtree of node v is the
// contiguous range [node_first[v], v]. Each node owns a diff: the entries
// diff_cells[diff_offsets[v] .. diff_offsets[v + 1]). A top cell (ID < n_top)
// enters the representative with orientation diff_signs; a facet (ID >= n_top,
// sign 0) leaves it in both orientations. The representative of node v consists
// of all diffs in its subtree.
//
// Bars are listed in extraction order. A bar's progression lists its nodes by
// ascending value: from the node just above its birth to the leaf at its death.
// bar_parent is the bar it splits off (-1 at a root); bar_tree numbers the roots.
struct ForestArrays {
    std::vector<double> node_value;
    std::vector<std::int64_t> node_parent;  // -1 for roots
    std::vector<std::int64_t> node_first;
    std::vector<std::int64_t> diff_offsets;
    std::vector<std::int64_t> diff_cells;
    std::vector<std::int8_t> diff_signs;
    std::vector<double> bar_birth;
    std::vector<double> bar_death;
    std::vector<std::int64_t> bar_parent;
    std::vector<std::int64_t> bar_tree;
    std::vector<std::int64_t> progression_offsets;
    std::vector<std::int64_t> progression_nodes;
};

// Build the forest. With reduce, parent-child pairs whose values differ by at
// most filtration_tol are collapsed, roots left without children are removed,
// and roots with several children are split into one root per child.
//
// Throws std::invalid_argument for inputs violating the engine's assumptions
// (IDs out of range, a facet with more than two top cells, decreasing values).
ForestArrays build_forest(const CellEvents& events, bool reduce, double filtration_tol);

}  // namespace persforest
