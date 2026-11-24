// Simplicial complexes embedded in R^d: plain C++17, no Python dependency.
#pragma once

#include <cstdint>
#include <vector>

namespace persforest {

// Orientation of each simplex relative to the ambient frame: +1 or -1, the sign
// of the determinant of its edge vectors p[v_k] - p[v_0], k = 1 .. d, after
// dividing them by their largest absolute entry.
//
// points is n_points x dim (row-major), simplices n_simplices x (dim + 1)
// vertex IDs (row-major, any vertex order). The scaling makes the rule
// invariant under uniform scaling of the points; rejecting scaled determinants
// of magnitude <= 1e-12 is a conservative numerical cutoff, not an exact
// degeneracy predicate.
//
// Throws std::invalid_argument for a vertex ID out of range and for the first
// simplex whose edge vectors are not finite and nonzero or whose scaled
// determinant has magnitude <= 1e-12; the message names the simplex by its
// vertex IDs.
std::vector<std::int8_t> simplex_orientations(const double* points, std::int64_t n_points, std::int64_t dim,
                                              const std::int64_t* simplices, std::int64_t n_simplices);

// Facet IDs of each top simplex, n_top x (dim + 1) row-major. tops
// (n_top x (dim + 1)) and facets (n_facets x dim) are rows of strictly
// increasing vertex IDs below n_points; facet j of a top simplex omits its
// vertex j, as in simplicial_boundary, and has ID n_top + its row in facets.
// Facets are bucketed by their first vertex and sorted within each bucket, so
// the lookup takes O((n_facets + n_top dim) log n_facets) time whatever the
// vertex degrees.
//
// Throws std::invalid_argument for a row that is not strictly increasing or
// has a vertex out of range, for a repeated facet, and for a facet of a top
// simplex that is not among the facets.
std::vector<std::int64_t> simplicial_top_facets(const std::int64_t* tops, std::int64_t n_top,
                                                const std::int64_t* facets, std::int64_t n_facets,
                                                std::int64_t dim, std::int64_t n_points);

}  // namespace persforest
