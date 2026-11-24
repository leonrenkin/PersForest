// Alpha filtrations of point clouds in R^2 and R^3 from geogram's Delaunay
// triangulation: plain C++17, no Python dependency.
#pragma once

#include <cstdint>
#include <vector>

namespace persforest {

// Top simplices and their facets with alpha values, in the layout consumed by
// the engine and SimplicialFiltration. Cell IDs follow the event order: the
// top simplices get IDs 0 .. n_top - 1 in the order in which they enter, the
// facets IDs n_top .. n_cells - 1 likewise.
struct AlphaFiltration {
    std::int64_t n_top = 0;
    std::vector<std::int64_t> top_vertices;    // n_top x (dim + 1), increasing vertex IDs per row
    std::vector<std::int64_t> facet_vertices;  // (n_cells - n_top) x dim, increasing vertex IDs per row
    std::vector<std::int64_t> top_facets;      // n_top x (dim + 1); position j omits vertex j of the row
    std::vector<std::int8_t> top_orientation;  // sign of det(p[v_k] - p[v_0]), k = 1 .. dim, per top
    std::vector<std::int64_t> event_cells;     // cell IDs in forward filtration order
    std::vector<double> event_values;          // alpha values in radius units
    // Numbers of radii and Gabriel tests that needed exact arithmetic.
    std::int64_t n_exact_radii = 0;
    std::int64_t n_exact_gabriel_tests = 0;
};

// Alpha filtration of the points (n_points x dim, row-major, dim 2 or 3)
// restricted to top simplices and facets.
//
// The complex is the Delaunay triangulation. Degenerate configurations are
// resolved by geogram's symbolic perturbation in lexicographic order of the
// coordinates, so the cells do not depend on the order of the points. Of
// identical points, the first is a vertex and the others are not. With fewer
// than dim + 1 affinely independent points there are no top simplices and the
// result is empty.
//
// Values, in radius units: a top simplex enters at its circumradius. A facet
// that is attached (a vertex opposite it in an adjacent top simplex lies
// strictly inside its smallest circumsphere) enters with the earliest
// adjacent top simplex; any other facet at the radius of its smallest
// circumsphere, but never after an adjacent top simplex. Events are ordered by
// value, then facets before top simplices, then by increasing vertex tuple.
// Orientations and Gabriel tests are exact; radii are accurate to about
// 1e-12 relative or better (see alpha.cpp).
//
// Throws std::invalid_argument for another dimension, non-finite
// coordinates, nonzero coordinates smaller than 2^-100 times the largest
// absolute coordinate (exact arithmetic could underflow), too many points or
// cells for 32-bit IDs, and radii beyond the double range.
AlphaFiltration alpha_filtration(const double* points, std::int64_t n_points, int dim);

}  // namespace persforest
