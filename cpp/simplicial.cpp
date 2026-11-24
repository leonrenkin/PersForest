// Simplicial complexes embedded in R^d: plain C++17, no Python dependency.
#include "simplicial.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace persforest {
namespace {

std::string simplex_name(const std::int64_t* vertices, std::int64_t n_vertices) {
    std::string name = "(";
    for (std::int64_t k = 0; k < n_vertices; ++k) {
        if (k > 0) name += ", ";
        name += std::to_string(vertices[k]);
    }
    return name + ")";
}

// Determinant of the n x n row-major matrix m (overwritten): closed forms for
// n <= 3, Gaussian elimination with partial pivoting above.
double determinant(double* m, std::int64_t n) {
    if (n == 1) return m[0];
    if (n == 2) return m[0] * m[3] - m[1] * m[2];
    if (n == 3) {
        return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
               m[2] * (m[3] * m[7] - m[4] * m[6]);
    }
    double det = 1.0;
    for (std::int64_t col = 0; col < n; ++col) {
        std::int64_t pivot = col;
        for (std::int64_t row = col + 1; row < n; ++row) {
            if (std::abs(m[row * n + col]) > std::abs(m[pivot * n + col])) pivot = row;
        }
        if (m[pivot * n + col] == 0.0) return 0.0;
        if (pivot != col) {
            for (std::int64_t k = 0; k < n; ++k) std::swap(m[pivot * n + k], m[col * n + k]);
            det = -det;
        }
        const double diagonal = m[col * n + col];
        det *= diagonal;
        for (std::int64_t row = col + 1; row < n; ++row) {
            const double factor = m[row * n + col] / diagonal;
            for (std::int64_t k = col + 1; k < n; ++k) m[row * n + k] -= factor * m[col * n + k];
        }
    }
    return det;
}

}  // namespace

std::vector<std::int8_t> simplex_orientations(const double* points, std::int64_t n_points, std::int64_t dim,
                                              const std::int64_t* simplices, std::int64_t n_simplices) {
    if (dim < 1 || n_points < 0 || n_simplices < 0) throw std::invalid_argument("invalid simplex array shape");
    const std::int64_t width = dim + 1;
    for (std::int64_t k = 0; k < n_simplices * width; ++k) {
        if (simplices[k] < 0 || simplices[k] >= n_points) throw std::invalid_argument("vertex ID out of range");
    }
    std::vector<std::int8_t> orientation(static_cast<std::size_t>(n_simplices));
    std::vector<double> m(static_cast<std::size_t>(dim * dim));
    for (std::int64_t s = 0; s < n_simplices; ++s) {
        const std::int64_t* vertices = simplices + s * width;
        const double* origin = points + vertices[0] * dim;
        double scale = 0.0;
        for (std::int64_t row = 0; row < dim; ++row) {
            const double* p = points + vertices[row + 1] * dim;
            for (std::int64_t col = 0; col < dim; ++col) {
                const double entry = p[col] - origin[col];
                m[row * dim + col] = entry;
                scale = std::fmax(scale, std::abs(entry));
            }
        }
        double det = 0.0;
        if (std::isfinite(scale) && scale > 0.0) {
            for (double& entry : m) entry /= scale;
            det = determinant(m.data(), dim);
        }
        if (!(std::isfinite(det) && std::abs(det) > 1e-12)) {
            throw std::invalid_argument("Cannot determine orientation reliably for simplex " +
                                        simplex_name(vertices, width) +
                                        ": degenerate, numerically near-degenerate, or non-finite edge vectors.");
        }
        orientation[static_cast<std::size_t>(s)] = det > 0.0 ? 1 : -1;
    }
    return orientation;
}

}  // namespace persforest
