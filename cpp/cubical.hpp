// Events of full cubical grids: plain C++17, no Python dependency.
#pragma once

#include <cstdint>
#include <vector>

namespace persforest {

// Top cells and facets of a cubical grid in filtration order, numbered as the
// construction engine reads them best: cell IDs follow the event order, top
// cells 0 .. n_top - 1 first, then facets n_top .. n_cells - 1.
struct CubicalEvents {
    std::vector<std::int64_t> event_cells;     // cell IDs in filtration order
    std::vector<double> event_values;          // filtration value of each event
    std::vector<std::int32_t> cell_positions;  // per cell ID: row-major index of its doubled coordinates
    std::vector<std::int64_t> top_facets;      // n_top x 2d: per axis the upper, then the lower facet
};

// Sort the top cells and facets of a full cubical grid by (rank, dimension,
// doubled coordinates in row-major order) and number them in that order.
//
// The inputs index the cells block by block: the top cells (shape
// top_shape), then the facets normal to axis 0, 1, ..., each block in
// row-major order; facets normal to axis a form a block of shape
// top_shape + e_a. cell_rank is the position of each cell's filtration value
// among the distinct values, 0 .. n_ranks - 1, and cell_value its value. A
// counting sort over the cells in row-major order of their doubled
// coordinates gives the order of a lexicographic sort, ties included, in
// linear time.
//
// Throws std::invalid_argument for an inconsistent grid shape or cell count,
// a rank out of range, or more than 2^31 - 1 cells or doubled-grid positions
// (32-bit IDs, as in the construction engine).
CubicalEvents cubical_events(const std::vector<std::int64_t>& top_shape, const std::int32_t* cell_rank,
                             const double* cell_value, std::int64_t n_cells, std::int64_t n_ranks);

}  // namespace persforest
