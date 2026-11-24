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

}  // namespace persforest
