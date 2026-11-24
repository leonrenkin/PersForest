// Event order of full cubical grids: plain C++17, no Python dependency.
#pragma once

#include <cstdint>
#include <vector>

namespace persforest {

// Sort the top cells and facets of a full cubical grid by (rank, dimension,
// doubled coordinates in row-major order) and return their IDs in that order.
//
// Cell IDs follow persforest.filtrations.cubical: the top cells (shape
// top_shape), then the facets normal to axis 0, 1, ..., each block in
// row-major order; facets normal to axis a form a block of shape
// top_shape + e_a. cell_rank[id] is the position of the cell's filtration
// value among the distinct values, 0 .. n_ranks - 1. A counting sort over the
// cells in row-major order of their doubled coordinates gives the order of a
// lexicographic sort, ties included, in linear time.
//
// Throws std::invalid_argument for an inconsistent grid shape or cell count,
// or a rank out of range.
std::vector<std::int64_t> cubical_event_order(const std::vector<std::int64_t>& top_shape,
                                              const std::int64_t* cell_rank, std::int64_t n_cells,
                                              std::int64_t n_ranks);

}  // namespace persforest
