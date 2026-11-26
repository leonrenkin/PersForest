// Alpha and Delaunay-Cech filtrations of point clouds in R^2 and R^3 from
// geogram's Delaunay triangulation: plain C++17, no Python dependency.
//
// Numerical setting (u = 2^-53 is the unit roundoff, gamma_k = k u / (1 - k u)).
// This file must be compiled without fused multiply-add contraction.
//
// Scaling. The points are multiplied by the power of two that puts the largest
// absolute coordinate into [1/2, 1), which is exact, and every nonzero scaled
// coordinate must be at least 2^-101. Scaled coordinates are then multiples of
// 2^-153, so a polynomial of degree k <= 7 in coordinate differences is a
// multiple of 2^-153k >= 2^-1071: geogram's expansion arithmetic (expansion_nt,
// also behind its predicates) neither underflows nor overflows and is exact.
// Values are scaled back at the end, exactly for normal finite results. A cell much smaller than the
// cloud (all edge components below 2^-60) is in addition rescaled by a power
// of two for the double formulas, so they do not underflow either.
//
// Radii (values). With edge vectors a, b (and c) from the first vertex, a
// triangle's squared circumradius is |a|^2 |b|^2 |c|^2 / (4 |a x b|^2), c the
// third edge, and a tetrahedron's |m|^2 / (4 det^2) with det = a.(b x c) and
// m = |a|^2 (b x c) + |b|^2 (c x a) + |c|^2 (a x b). A first-order forward
// error analysis bounds the relative error of the double evaluation by about
// (8 sqrt(3) K + 25) u for a triangle, K = |a||b| / |a x b|, and by about
// (16 sqrt(3) K_m + 14 sqrt(3) K_d + 30) u for a tetrahedron, K_d =
// |a||b||c| / |det| and K_m = (|a|^2|b||c| + |b|^2|c||a| + |c|^2|a||b|) / |m|.
// The cancellations in the cross products and det are what K measures. Below
// kCond = 100 this is under 1e-12 for squared radii; above, numerator and
// denominator are computed exactly. Overlapping error ranges are resolved
// before event ordering by rounding the exact radius to nearest, ties to even.
//
// Orientation (decision). The sign of the edge-vector determinant (|a x b| in
// 2D, det in 3D) has absolute error at most gamma_7 sqrt(3) |a||b||c| (gamma_4
// |a||b| in 2D), far below |det| whenever the condition above holds; otherwise
// it is computed exactly.
//
// Ball tests (decisions): alpha's Gabriel tests (strictly inside a smallest
// sphere?) and the containment tests behind minimal enclosing balls (inside or
// on?) are signs of the same polynomials and use a semi-static filter. For a
// polynomial G evaluated in doubles from the rounded coordinate differences
// with at most d roundings on any path, |G~ - G| <= gamma_d A / (1 - u)^d,
// where A~ is the same expression evaluated in doubles with every operand
// replaced by its absolute value and every difference by a sum (Higham,
// Accuracy and Stability of Numerical Algorithms, ch. 3). Here d <= 11, so
// 1e-14 A~ bounds the error with a margin of 8; kTiny = 2^-1000 covers
// underflow in intermediate products, whose absolute error is below 2^-1050
// for these magnitudes (all differences below 2). A sign the filter cannot
// certify is computed exactly; G has degree 6, so that never underflows (see
// Scaling).
#include "delaunay.hpp"

#include <Delaunay_psm.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace persforest {
namespace {

using Exact = GEO::expansion_nt;
using GEO::index_t;

constexpr double kCond = 100.0;
constexpr double kBallRelative = 1e-14;
constexpr double kTiny = 0x1p-1000;
constexpr double kMinScaledCoordinate = 0x1p-101;
constexpr double kLocalThreshold = 0x1p-60;
// Allow for both radius errors, with margin for the bounds above.
constexpr double kRadiusOverlap = 2e-12;

// ---------------------------------------------------------------------------
// geogram setup. geogram keeps global state (thread manager, random generator,
// symbolic perturbation mode), so the builder runs under one lock.

std::mutex& geogram_mutex() {
    static std::mutex mutex;
    return mutex;
}

void initialize_geogram() {
    static std::once_flag once;
    std::call_once(once, [] {
        // No signal handlers, no floating-point traps, no console output.
        GEO::initialize(GEO::GEOGRAM_INSTALL_NONE);
        GEO::Logger::instance()->set_quiet(true);
        // Perturb degenerate configurations by coordinates, not by addresses,
        // so the cells do not depend on the order of the points.
        GEO::PCK::set_SOS_mode(GEO::PCK::SOS_LEXICO);
    });
}

// ---------------------------------------------------------------------------
// Small vector helpers. 2D vectors have z = 0.

struct Vec {
    double x = 0.0, y = 0.0, z = 0.0;
};

template <int D>
Vec difference(const double* p, const double* q) {
    if constexpr (D == 2) {
        return {p[0] - q[0], p[1] - q[1], 0.0};
    } else {
        return {p[0] - q[0], p[1] - q[1], p[2] - q[2]};
    }
}

double dot(const Vec& a, const Vec& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec cross(const Vec& a, const Vec& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
Vec absolute(const Vec& a) { return {std::abs(a.x), std::abs(a.y), std::abs(a.z)}; }
// Cross product with every difference replaced by a sum; for the error bounds.
Vec absolute_cross(const Vec& a, const Vec& b) {
    return {a.y * b.z + a.z * b.y, a.z * b.x + a.x * b.z, a.x * b.y + a.y * b.x};
}
Vec scaled(const Vec& a, double s) { return {a.x * s, a.y * s, a.z * s}; }
Vec sum(const Vec& a, const Vec& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
double max_abs(const Vec& a) { return std::max({std::abs(a.x), std::abs(a.y), std::abs(a.z)}); }

// Exponent k such that x 2^k lies in [1/2, 1) for 0 < x < kLocalThreshold,
// else 0: the local rescaling of tiny cells.
int local_exponent(double x) {
    if (!(x > 0.0 && x < kLocalThreshold)) return 0;
    int e = 0;
    std::frexp(x, &e);
    return -e;
}

struct ExactVec {
    Exact x, y, z;
};

// p - q exactly; z = 0 in 2D.
template <int D>
ExactVec exact_difference(const double* p, const double* q) {
    auto component = [](double a, double b) { return Exact(Exact::DIFF, a, b); };
    if constexpr (D == 2) {
        return {component(p[0], q[0]), component(p[1], q[1]), Exact(0.0)};
    } else {
        return {component(p[0], q[0]), component(p[1], q[1]), component(p[2], q[2])};
    }
}

Exact dot(const ExactVec& a, const ExactVec& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
ExactVec cross(const ExactVec& a, const ExactVec& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
ExactVec scaled(const ExactVec& a, const Exact& s) { return {a.x * s, a.y * s, a.z * s}; }
ExactVec sum(const ExactVec& a, const ExactVec& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }

struct ExactSquaredRadius {
    Exact numerator;
    Exact denominator;
    int exponent = 0;  // the ratio is multiplied by 2^exponent

    double estimate() const {
        int numerator_exponent, denominator_exponent;
        const double numerator_fraction = std::frexp(numerator.estimate(), &numerator_exponent);
        const double denominator_fraction = std::frexp(denominator.estimate(), &denominator_exponent);
        return std::ldexp(numerator_fraction / denominator_fraction,
                          exponent + numerator_exponent - denominator_exponent);
    }
};

void scale_expansion(Exact& x, int exponent) {
    for (index_t i = 0; i < x.length(); ++i) {
        x.rep()[i] = std::ldexp(x.component(i), exponent);
    }
}

// Compare the squared radius with the square of the exact midpoint of two
// adjacent doubles. Scale the two expansions to a common exponent before
// subtracting. Their bit spans are at most 8*153 plus a few bits for sums, so
// a leading exponent of 320 leaves room for even their lowest components.
GEO::Sign compare_midpoint(const ExactSquaredRadius& squared, double low, double high) {
    int midpoint_exponent;
    std::frexp(high, &midpoint_exponent);
    const Exact midpoint =
        (Exact(std::ldexp(low, -midpoint_exponent)) + std::ldexp(high, -midpoint_exponent)) * 0.5;
    Exact rhs = squared.denominator * (midpoint * midpoint);
    Exact lhs = squared.numerator;
    const int rhs_exponent = 2 * midpoint_exponent;
    const int common_exponent = std::max(
        std::ilogb(lhs.component(lhs.length() - 1)) + squared.exponent,
        std::ilogb(rhs.component(rhs.length() - 1)) + rhs_exponent);
    scale_expansion(lhs, 320 + squared.exponent - common_exponent);
    scale_expansion(rhs, 320 + rhs_exponent - common_exponent);
    return lhs.compare(rhs);
}

std::uint64_t double_bits(double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
}

double round_radius(const ExactSquaredRadius& squared) {
    double radius = std::sqrt(squared.estimate());
    while (true) {
        const double low = std::nextafter(radius, 0.0);
        const bool odd = double_bits(radius) & 1;
        const GEO::Sign lower = compare_midpoint(squared, low, radius);
        if (lower == GEO::NEGATIVE || (lower == GEO::ZERO && odd)) {
            radius = low;
            continue;
        }
        const double high = std::nextafter(radius, std::numeric_limits<double>::infinity());
        const GEO::Sign upper = compare_midpoint(squared, radius, high);
        if (upper != GEO::POSITIVE && !(upper == GEO::ZERO && odd)) return radius;
        radius = high;
    }
}

struct Counters {
    std::int64_t exact_radii = 0;
    std::int64_t exact_ball_tests = 0;
};

// ---------------------------------------------------------------------------
// Geometry of cells; vertices in increasing ID order.

struct TopGeometry {
    double squared_radius;
    std::int8_t orientation;
};

[[noreturn]] void flat_cell() {
    throw std::logic_error("Delaunay filtration: geogram returned a flat cell");
}

template <int D>
ExactSquaredRadius exact_triangle(const double* const* p, std::int8_t* orientation = nullptr) {
    const ExactVec a = exact_difference<D>(p[1], p[0]), b = exact_difference<D>(p[2], p[0]),
                   c = exact_difference<D>(p[2], p[1]);
    const ExactVec normal = cross(a, b);
    const Exact denominator = dot(normal, normal) * 4.0;
    if (denominator.sign() == GEO::ZERO) flat_cell();
    if (orientation != nullptr) *orientation = normal.z.sign() == GEO::POSITIVE ? 1 : -1;
    return {dot(a, a) * dot(b, b) * dot(c, c), denominator};
}

ExactSquaredRadius exact_tetrahedron(const double* const* p, std::int8_t* orientation = nullptr) {
    const ExactVec a = exact_difference<3>(p[1], p[0]), b = exact_difference<3>(p[2], p[0]),
                   c = exact_difference<3>(p[3], p[0]);
    const ExactVec bc = cross(b, c);
    const Exact det = dot(a, bc);
    const GEO::Sign sign = det.sign();
    if (sign == GEO::ZERO) flat_cell();
    if (orientation != nullptr) *orientation = sign == GEO::POSITIVE ? 1 : -1;
    ExactVec m = sum(sum(scaled(bc, dot(a, a)), scaled(cross(c, a), dot(b, b))), scaled(cross(a, b), dot(c, c)));
    // The numerator is degree eight. Scale before squaring to keep its
    // lowest terms normal. Leave room in the denominator for a midpoint square.
    scale_expansion(m.x, 128);
    scale_expansion(m.y, 128);
    scale_expansion(m.z, 128);
    return {dot(m, m), det * det * 0x1p66, 64 - 256};
}

TopGeometry triangle_top(const double* const* p, Counters& counters) {
    Vec a = difference<2>(p[1], p[0]), b = difference<2>(p[2], p[0]), c = difference<2>(p[2], p[1]);
    const int k = local_exponent(std::max({max_abs(a), max_abs(b), max_abs(c)}));
    if (k != 0) {
        const double s = std::ldexp(1.0, k);
        a = scaled(a, s), b = scaled(b, s), c = scaled(c, s);
    }
    const double det = a.x * b.y - a.y * b.x;
    const double aa = dot(a, a), bb = dot(b, b), cc = dot(c, c);
    if (det != 0.0 && std::sqrt(aa * bb) <= kCond * std::abs(det)) {
        return {std::ldexp(aa * bb * cc / (4.0 * det * det), -2 * k), static_cast<std::int8_t>(det > 0.0 ? 1 : -1)};
    }
    ++counters.exact_radii;
    std::int8_t orientation;
    const double squared = exact_triangle<2>(p, &orientation).estimate();
    return {squared, orientation};
}

TopGeometry tetrahedron_top(const double* const* p, Counters& counters) {
    Vec a = difference<3>(p[1], p[0]), b = difference<3>(p[2], p[0]), c = difference<3>(p[3], p[0]);
    const int k = local_exponent(std::max({max_abs(a), max_abs(b), max_abs(c)}));
    if (k != 0) {
        const double s = std::ldexp(1.0, k);
        a = scaled(a, s), b = scaled(b, s), c = scaled(c, s);
    }
    const Vec bc = cross(b, c), ca = cross(c, a), ab = cross(a, b);
    const double det = dot(a, bc);
    const double aa = dot(a, a), bb = dot(b, b), cc = dot(c, c);
    const Vec m = sum(sum(scaled(bc, aa), scaled(ca, bb)), scaled(ab, cc));
    const double la = std::sqrt(aa), lb = std::sqrt(bb), lc = std::sqrt(cc);
    const double mm = dot(m, m);
    const double terms = aa * lb * lc + bb * lc * la + cc * la * lb;
    if (det != 0.0 && la * lb * lc <= kCond * std::abs(det) && terms <= kCond * std::sqrt(mm)) {
        return {std::ldexp(mm / (4.0 * det * det), -2 * k), static_cast<std::int8_t>(det > 0.0 ? 1 : -1)};
    }
    ++counters.exact_radii;
    std::int8_t orientation;
    const double squared = exact_tetrahedron(p, &orientation).estimate();
    return {squared, orientation};
}

// Squared radius of the smallest sphere through an edge: well conditioned.
template <int D>
double edge_squared_radius(const double* p0, const double* p1) {
    const Vec a = difference<D>(p1, p0);
    return dot(a, a) / 4.0;
}

// Squared radius of the smallest sphere through a triangle in R^3 (3D facet).
double triangle_squared_radius(const double* const* p, Counters& counters) {
    Vec a = difference<3>(p[1], p[0]), b = difference<3>(p[2], p[0]), c = difference<3>(p[2], p[1]);
    const int k = local_exponent(std::max({max_abs(a), max_abs(b), max_abs(c)}));
    if (k != 0) {
        const double s = std::ldexp(1.0, k);
        a = scaled(a, s), b = scaled(b, s), c = scaled(c, s);
    }
    const Vec n = cross(a, b);
    const double nn = dot(n, n), aa = dot(a, a), bb = dot(b, b), cc = dot(c, c);
    if (nn > 0.0 && std::sqrt(aa * bb) <= kCond * std::sqrt(nn)) {
        return std::ldexp(aa * bb * cc / (4.0 * nn), -2 * k);
    }
    ++counters.exact_radii;
    return exact_triangle<3>(p).estimate();
}

// Side of q relative to the smallest sphere through the edge p0 p1 (its
// diametral sphere): the sign of G = (q - p0).(q - p1), negative inside.
// Degree 2, d <= 4 roundings.
template <int D>
GEO::Sign edge_ball_side(const double* p0, const double* p1, const double* q, Counters& counters) {
    const Vec w = difference<D>(q, p0), v = difference<D>(q, p1);
    const double g = dot(w, v);
    const double bound = dot(absolute(w), absolute(v));
    if (std::abs(g) > kBallRelative * bound + kTiny) return g < 0.0 ? GEO::NEGATIVE : GEO::POSITIVE;
    ++counters.exact_ball_tests;
    return dot(exact_difference<D>(q, p0), exact_difference<D>(q, p1)).sign();
}

// Side of q relative to the smallest sphere through the triangle p[0] p[1] p[2]
// in R^3, negative inside. With a = p1 - p0, b = p2 - p0, w = q - p0, n = a x b
// and m = |a|^2 (b x n) + |b|^2 (n x a) (twice |n|^2 times the circumcenter
// offset), the sign of G = |n|^2 |w|^2 - w.m; degree 6, d = 11 roundings.
GEO::Sign triangle_sphere_side(const double* const* p, const double* q, Counters& counters) {
    Vec a = difference<3>(p[1], p[0]), b = difference<3>(p[2], p[0]), w = difference<3>(q, p[0]);
    const int k = local_exponent(std::max({max_abs(a), max_abs(b), max_abs(w)}));
    if (k != 0) {
        const double s = std::ldexp(1.0, k);
        a = scaled(a, s), b = scaled(b, s), w = scaled(w, s);
    }
    const Vec n = cross(a, b);
    const double aa = dot(a, a), bb = dot(b, b), ww = dot(w, w);
    const Vec m = sum(scaled(cross(b, n), aa), scaled(cross(n, a), bb));
    const double g = dot(n, n) * ww - dot(w, m);
    const Vec abs_a = absolute(a), abs_b = absolute(b), abs_n = absolute_cross(abs_a, abs_b);
    const Vec abs_m = sum(scaled(absolute_cross(abs_b, abs_n), aa), scaled(absolute_cross(abs_n, abs_a), bb));
    const double bound = dot(abs_n, abs_n) * ww + dot(absolute(w), abs_m);
    if (std::abs(g) > kBallRelative * bound + kTiny) return g < 0.0 ? GEO::NEGATIVE : GEO::POSITIVE;
    ++counters.exact_ball_tests;
    const ExactVec ea = exact_difference<3>(p[1], p[0]), eb = exact_difference<3>(p[2], p[0]),
                   ew = exact_difference<3>(q, p[0]);
    const ExactVec en = cross(ea, eb);
    const ExactVec em = sum(scaled(cross(eb, en), dot(ea, ea)), scaled(cross(en, ea), dot(eb, eb)));
    return (dot(en, en) * dot(ew, ew) - dot(ew, em)).sign();
}

// Minimal enclosing balls (Delaunay-Cech values). The minimal enclosing ball
// of a simplex is the smallest sphere through some face (center in the face's
// affine hull) that contains every vertex, so its radius is the least radius
// among these candidates that contain the simplex (closed balls). Every
// candidate containing the simplex encloses it and is at least as large.

// The support of a minimal enclosing ball is a face whose circumsphere
// contains the simplex and whose circumcenter belongs to that face.
constexpr std::uint8_t vertex_mask(int count) {
    return static_cast<std::uint8_t>((1 << count) - 1);
}

// A triangle's minimal ball is its diametral edge ball at a right or obtuse
// angle, or its circumsphere. Support masks refer to positions in vertex order.
template <int D>
std::uint8_t triangle_support(const double* const* p, Counters& counters) {
    for (int k = 0; k < 3; ++k) {
        const int i = k == 0 ? 1 : 0, j = k == 2 ? 1 : 2;  // the edge opposite vertex k, increasing
        if (edge_ball_side<D>(p[i], p[j], p[k], counters) != GEO::POSITIVE) {
            return static_cast<std::uint8_t>(vertex_mask(3) ^ (1 << k));
        }
    }
    return vertex_mask(3);
}

std::uint8_t tetrahedron_support(const double* const* p, Counters& counters) {
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            int others[2], n_others = 0;
            for (int k = 0; k < 4; ++k) {
                if (k != i && k != j) others[n_others++] = k;
            }
            if (edge_ball_side<3>(p[i], p[j], p[others[0]], counters) != GEO::POSITIVE &&
                edge_ball_side<3>(p[i], p[j], p[others[1]], counters) != GEO::POSITIVE) {
                return static_cast<std::uint8_t>((1 << i) | (1 << j));
            }
        }
    }
    for (int omitted = 0; omitted < 4; ++omitted) {
        const double* face[3];
        for (int k = 0, f = 0; k < 4; ++k) {
            if (k != omitted) face[f++] = p[k];
        }
        if (triangle_sphere_side(face, p[omitted], counters) != GEO::POSITIVE &&
            triangle_support<3>(face, counters) == vertex_mask(3)) {
            return static_cast<std::uint8_t>(vertex_mask(4) ^ (1 << omitted));
        }
    }
    return vertex_mask(4);
}

template <int D>
double enclosing_squared(const double* const* p, int count, std::uint8_t support, double circumradius,
                         Counters& counters) {
    if (support == vertex_mask(count)) return circumradius;
    const double* ball[3];
    int size = 0;
    for (int j = 0; j < count; ++j) {
        if (support & (1 << j)) ball[size++] = p[j];
    }
    return size == 2 ? edge_squared_radius<D>(ball[0], ball[1]) : triangle_squared_radius(ball, counters);
}

// ---------------------------------------------------------------------------
// Input preparation.

// Scale the points by the power of two 2^-exponent that puts the largest
// absolute coordinate into [1/2, 1); returns false if all coordinates are zero.
bool scale_points(const double* points, std::size_t n_values, std::vector<double>& out, int& exponent) {
    double largest = 0.0;
    for (std::size_t i = 0; i < n_values; ++i) {
        if (!std::isfinite(points[i])) throw std::invalid_argument("point coordinates must be finite");
        largest = std::max(largest, std::abs(points[i]));
    }
    if (largest == 0.0) return false;
    std::frexp(largest, &exponent);
    out.resize(n_values);
    for (std::size_t i = 0; i < n_values; ++i) {
        const double x = std::ldexp(points[i], -exponent);
        if (points[i] != 0.0 && std::abs(x) < kMinScaledCoordinate) {
            throw std::invalid_argument(
                "point coordinates span too many orders of magnitude for the Delaunay filtration: every nonzero "
                "coordinate must be at least 2^-101 after scaling the largest absolute coordinate into [1/2, 1)");
        }
        out[i] = x;
    }
    return true;
}

// Are there dim + 1 affinely independent points? Searched like geogram's
// parallel 3D triangulation does, which does not return without them.
bool spans_space(const double* points, index_t n, int dim) {
    auto point = [&](index_t i) { return points + std::size_t(i) * dim; };
    index_t i1 = 1;
    while (i1 < n && std::equal(point(0), point(0) + dim, point(i1))) ++i1;
    if (i1 >= n) return false;
    index_t i2 = i1 + 1;
    if (dim == 2) {
        while (i2 < n && GEO::PCK::orient_2d(point(0), point(i1), point(i2)) == GEO::ZERO) ++i2;
        return i2 < n;
    }
    while (i2 < n && GEO::PCK::points_are_colinear_3d(point(0), point(i1), point(i2))) ++i2;
    if (i2 >= n) return false;
    index_t i3 = i2 + 1;
    while (i3 < n && GEO::PCK::orient_3d(point(0), point(i1), point(i2), point(i3)) == GEO::ZERO) ++i3;
    return i3 < n;
}

struct PointKey {
    std::array<double, 3> coordinates;
    bool operator==(const PointKey& other) const { return coordinates == other.coordinates; }
};

struct PointKeyHash {
    std::size_t operator()(const PointKey& key) const {
        std::size_t hash = 0;
        for (double x : key.coordinates) {
            std::uint64_t bits;
            const double normalized = x + 0.0;  // -0.0 and 0.0 are the same point
            std::memcpy(&bits, &normalized, sizeof bits);
            hash = hash * 0x100000001b3ULL ^ static_cast<std::size_t>(bits ^ (bits >> 29));
        }
        return hash;
    }
};

// geogram inserts one of several identical points (which one depends on its
// insertion order); the others are vertices of no cell. Rename the inserted
// one to the first occurrence, so the choice is predictable.
void use_first_of_identical_points(std::vector<index_t>& cell_vertices, const double* points, index_t n, int dim) {
    std::vector<std::uint8_t> used(n, 0);
    for (index_t v : cell_vertices) used[v] = 1;
    auto key = [&](index_t v) {
        PointKey k{{0.0, 0.0, 0.0}};
        std::copy(points + std::size_t(v) * dim, points + std::size_t(v + 1) * dim, k.coordinates.begin());
        return k;
    };
    std::unordered_map<PointKey, index_t, PointKeyHash> first_unused;
    for (index_t v = 0; v < n; ++v) {
        if (!used[v]) first_unused.try_emplace(key(v), v);  // ascending v: keeps the smallest
    }
    if (first_unused.empty()) return;
    std::unordered_map<index_t, index_t> renamed;
    std::size_t matched = 0;
    for (index_t v = 0; v < n; ++v) {
        if (!used[v]) continue;
        const auto found = first_unused.find(key(v));
        if (found == first_unused.end()) continue;
        ++matched;
        if (found->second < v) renamed.emplace(v, found->second);
    }
    if (matched != first_unused.size()) {
        throw std::logic_error("Delaunay filtration: a point is missing from geogram's triangulation");
    }
    if (renamed.empty()) return;
    for (index_t& v : cell_vertices) {
        const auto found = renamed.find(v);
        if (found != renamed.end()) v = found->second;
    }
}

// ---------------------------------------------------------------------------

struct Event {
    double value;
    std::uint32_t index;  // cell or facet index
    bool top;
    std::uint8_t support;
};

// Stable LSD radix sort for nonnegative doubles: unsigned bit order is numeric
// order. Keep events in triangulation order within each value until refinement.
void sort_by_radius(std::vector<Event>& events) {
    constexpr int kDigitBits = 11, kPasses = 6;  // 6 x 11 bits cover 64
    constexpr std::size_t kBuckets = std::size_t(1) << kDigitBits;
    const std::size_t n = events.size();
    std::vector<std::size_t> counts(kPasses * kBuckets, 0);
    for (const Event& event : events) {
        const std::uint64_t bits = double_bits(event.value);
        for (int pass = 0; pass < kPasses; ++pass) ++counts[pass * kBuckets + ((bits >> (pass * kDigitBits)) & (kBuckets - 1))];
    }
    std::vector<Event> buffer(n);
    std::vector<Event>* source = &events;
    std::vector<Event>* target = &buffer;
    for (int pass = 0; pass < kPasses; ++pass) {
        std::size_t* count = &counts[pass * kBuckets];
        if (std::find(count, count + kBuckets, n) != count + kBuckets) continue;  // one digit for all
        std::size_t offset = 0;
        for (std::size_t b = 0; b < kBuckets; ++b) offset += std::exchange(count[b], offset);
        for (const Event& event : *source) {
            (*target)[count[(double_bits(event.value) >> (pass * kDigitBits)) & (kBuckets - 1)]++] = event;
        }
        std::swap(source, target);
    }
    if (source != &events) events.swap(buffer);
}

// The radius is that of the sphere through this increasing vertex tuple.
// Trailing sentinels distinguish supports of different dimensions.
using BallSupport = std::array<index_t, 4>;

constexpr std::size_t kRadiusCacheSize = 128;

template <int D>
struct CachedRadius {
    std::array<double, 3 * D + 1> offsets{};
    double radius = 0.0;
};

template <int D>
double exact_radius(const double* points, const BallSupport& support, CachedRadius<D>* cache) {
    const double* p[4];
    int size = 0;
    while (size < 4 && support[size] != GEO::NO_INDEX) {
        p[size] = points + std::size_t(support[size]) * D;
        ++size;
    }
    const auto compute = [&] {
        if (size == 2) {
            const ExactVec edge = exact_difference<D>(p[1], p[0]);
            return round_radius({dot(edge, edge), Exact(4.0)});
        }
        return size == 3 ? round_radius(exact_triangle<D>(p)) : round_radius(exact_tetrahedron(p));
    };
    if (cache == nullptr) return compute();

    // Large ties on regular lattices contain many translated copies of the
    // same ball support. Reuse a radius only if the edge offsets are exact.
    std::array<double, 3 * D + 1> offsets{};
    offsets.back() = size;
    std::uint64_t hash = size;
    for (int i = 1; i < size; ++i) {
        for (int axis = 0; axis < D; ++axis) {
            double offset, error;
            GEO::two_diff(p[i][axis], p[0][axis], offset, error);
            if (error != 0.0) return compute();
            offsets[(i - 1) * D + axis] = offset;
            const std::uint64_t bits = double_bits(offset + 0.0);
            hash = hash * 0x100000001b3ULL ^ (bits >> 32) ^ bits;
        }
    }
    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdULL;
    const auto slot = static_cast<std::size_t>((hash ^ (hash >> 33)) % kRadiusCacheSize);
    auto& entry = cache[slot];
    if (entry.offsets != offsets) entry = {offsets, compute()};
    return entry.radius;
}

// Recompute only the runs in which radius error ranges can overlap. Exact
// rounding gives equal mathematical radii the same value without merging
// distinguishable nearby radii. Most ties just share a support and can take
// the fast path, e.g. an alpha facet that enters with its top coface.
template <int D, typename GetSupport>
void refine_radii(std::vector<Event>& events, const double* points, GetSupport get_support) {
    // Not `near`: <windows.h>, included by geogram, defines it as an empty macro.
    const auto overlapping = [](double left, double right) { return right - left <= kRadiusOverlap * right; };
    const auto less = [](const Event& x, const Event& y) { return x.value < y.value; };
    std::array<CachedRadius<D>, kRadiusCacheSize> cache;
    for (auto begin = events.begin(); begin != events.end();) {
        auto end = begin + 1;
        while (end != events.end() && overlapping((end - 1)->value, end->value)) ++end;
        if (end - begin > 1) {
            const BallSupport first = get_support(*begin);
            const auto distinct = std::find_if(begin + 1, end, [&](const Event& event) {
                return get_support(event) != first;
            });
            if (distinct != end) {
                auto* cache_entries = end - begin >= 16 ? cache.data() : nullptr;
                BallSupport last = first;
                double radius = exact_radius<D>(points, last, cache_entries);
                for (auto item = begin; item != end; ++item) {
                    const BallSupport support = get_support(*item);
                    if (support != last) {
                        radius = exact_radius<D>(points, support, cache_entries);
                        last = support;
                    }
                    item->value = radius;
                }
                if (!std::is_sorted(begin, end, less)) std::sort(begin, end, less);
            }
        }
        begin = end;
    }
}

enum class Values { alpha, delaunay_cech };

template <int D>
DelaunayFiltration build(const double* points, index_t n, int exponent, Values values) {
    constexpr int kWidth = D + 1;  // vertices per top simplex
    DelaunayFiltration result;
    if (!spans_space(points, n, D)) return result;

    // Triangulate and copy the cells' vertices and neighbours.
    std::vector<index_t> raw_vertices, neighbours;
    index_t n_cells = 0;
    {
        const char* algorithm = D == 2 ? "BDEL2d" : (GEO::DelaunayFactory::has_creator("PDEL") ? "PDEL" : "BDEL");
        GEO::Delaunay_var triangulation = GEO::Delaunay::create(GEO::coord_index_t(D), algorithm);
        if (triangulation.is_null()) throw std::runtime_error("Delaunay filtration: geogram has no Delaunay algorithm");
        GEO::Numeric::random_reset(-1);  // the same insertion order on every call
        triangulation->set_vertices(n, points);
        n_cells = triangulation->nb_cells();
        raw_vertices.resize(std::size_t(n_cells) * kWidth);
        neighbours.resize(std::size_t(n_cells) * kWidth);
        for (index_t c = 0; c < n_cells; ++c) {
            for (index_t j = 0; j < kWidth; ++j) {
                raw_vertices[std::size_t(c) * kWidth + j] = triangulation->cell_vertex(c, j);
                neighbours[std::size_t(c) * kWidth + j] = triangulation->cell_adjacent(c, j);
            }
        }
    }
    use_first_of_identical_points(raw_vertices, points, n, D);

    std::size_t n_hull = 0;
    for (index_t neighbour : neighbours) n_hull += neighbour == GEO::NO_INDEX;
    const std::size_t n_facets = (std::size_t(n_cells) * kWidth + n_hull) / 2;
    if (std::size_t(n_cells) + n_facets > std::size_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Delaunay filtration: too many cells for 32-bit cell IDs");
    }

    // Top simplices: sorted vertices, the sorted position of each geogram
    // vertex, value (circumradius or minimal enclosing ball) and
    // orientation of the sorted simplex.
    Counters counters;
    std::vector<index_t> vertices(raw_vertices.size());
    std::vector<std::uint8_t> position(raw_vertices.size());
    std::vector<Event> events;
    events.reserve(std::size_t(n_cells) + n_facets);
    std::vector<std::int8_t> orientation(n_cells);
    for (index_t c = 0; c < n_cells; ++c) {
        const std::size_t base = std::size_t(c) * kWidth;
        std::array<index_t, kWidth> order;
        for (int j = 0; j < kWidth; ++j) order[j] = index_t(j);
        std::sort(order.begin(), order.end(),
                  [&](index_t x, index_t y) { return raw_vertices[base + x] < raw_vertices[base + y]; });
        const double* p[kWidth];
        for (int j = 0; j < kWidth; ++j) {
            vertices[base + j] = raw_vertices[base + order[j]];
            position[base + order[j]] = static_cast<std::uint8_t>(j);
            p[j] = points + std::size_t(vertices[base + j]) * D;
        }
        TopGeometry top;
        if constexpr (D == 2) {
            top = triangle_top(p, counters);
        } else {
            top = tetrahedron_top(p, counters);
        }
        double squared = top.squared_radius;
        std::uint8_t support = vertex_mask(kWidth);
        if (values == Values::delaunay_cech) {
            if constexpr (D == 2) {
                support = triangle_support<2>(p, counters);
            } else {
                support = tetrahedron_support(p, counters);
            }
            squared = enclosing_squared<D>(p, kWidth, support, squared, counters);
        }
        events.push_back({std::sqrt(squared), c, true, support});
        orientation[c] = top.orientation;
    }
    raw_vertices = std::vector<index_t>();

    // Facets, each once: from the cell with the smaller index, or the only one.
    std::vector<index_t> facet_vertices;
    facet_vertices.reserve(n_facets * D);
    std::vector<index_t> attached_top;
    if (values == Values::alpha) attached_top.reserve(n_facets);
    std::vector<std::uint32_t> cell_facet(std::size_t(n_cells) * kWidth);  // by sorted position
    for (index_t c = 0; c < n_cells; ++c) {
        const std::size_t base = std::size_t(c) * kWidth;
        for (int local = 0; local < kWidth; ++local) {
            const index_t other = neighbours[base + local];
            if (other != GEO::NO_INDEX && other < c) continue;
            const auto facet = static_cast<std::uint32_t>(events.size() - n_cells);
            const int j = position[base + local];
            cell_facet[base + j] = facet;
            const double* p[D];
            for (int i = 0, k = 0; i < kWidth; ++i) {
                if (i == j) continue;
                facet_vertices.push_back(vertices[base + i]);
                p[k++] = points + std::size_t(vertices[base + i]) * D;
            }
            // Is the vertex opposite the facet in a top simplex strictly inside the
            // facet's smallest sphere (alpha's Gabriel test)?
            auto inside = [&](index_t opposite) {
                const double* q = points + std::size_t(opposite) * D;
                if constexpr (D == 2) {
                    return edge_ball_side<2>(p[0], p[1], q, counters) == GEO::NEGATIVE;
                } else {
                    return triangle_sphere_side(p, q, counters) == GEO::NEGATIVE;
                }
            };
            index_t other_opposite = GEO::NO_INDEX;
            if (other != GEO::NO_INDEX) {
                const std::size_t other_base = std::size_t(other) * kWidth;
                int other_local = 0;
                while (other_local < kWidth && neighbours[other_base + other_local] != c) ++other_local;
                if (other_local == kWidth) throw std::logic_error("Delaunay filtration: inconsistent neighbours");
                const int other_j = position[other_base + other_local];
                cell_facet[other_base + other_j] = facet;
                other_opposite = vertices[other_base + other_j];
            }
            if (values == Values::alpha) {
                // The dual Voronoi edge lies on the facet's normal line.
                // If its smallest sphere contains an opposite vertex, the
                // coface on that side gives the closest feasible center.
                const index_t attached = inside(vertices[base + j]) ? c :
                    (other_opposite != GEO::NO_INDEX && inside(other_opposite) ? other : GEO::NO_INDEX);
                attached_top.push_back(attached);
                if (attached != GEO::NO_INDEX) {
                    events.push_back({events[attached].value, facet, false, vertex_mask(kWidth)});
                    continue;
                }
            }
            double own;
            if constexpr (D == 2) {
                own = edge_squared_radius<2>(p[0], p[1]);
            } else {
                own = triangle_squared_radius(p, counters);
            }
            std::uint8_t support = vertex_mask(D);
            if constexpr (D == 3) {
                if (values == Values::delaunay_cech) {
                    support = triangle_support<3>(p, counters);
                    own = enclosing_squared<3>(p, D, support, own, counters);
                }
            }
            events.push_back({std::sqrt(own), facet, false, support});
        }
    }
    neighbours = std::vector<index_t>();
    position = std::vector<std::uint8_t>();

    // Refine in scaled radius units so even extreme input scales can be
    // compared before conversion to the output's double range.
    sort_by_radius(events);
    refine_radii<D>(events, points, [&](const Event& event) {
        const index_t top = event.top ? event.index :
            (values == Values::alpha ? attached_top[event.index] : GEO::NO_INDEX);
        const index_t* row = top != GEO::NO_INDEX ? &vertices[std::size_t(top) * kWidth] :
            &facet_vertices[std::size_t(event.index) * D];
        const int width = top != GEO::NO_INDEX ? kWidth : D;
        BallSupport support;
        support.fill(GEO::NO_INDEX);
        for (int j = 0, k = 0; j < width; ++j) {
            if (event.support & (1 << j)) support[k++] = row[j];
        }
        return support;
    });
    for (Event& event : events) {
        event.value = std::ldexp(event.value, exponent);
        if (!std::isfinite(event.value)) {
            throw std::invalid_argument("Delaunay filtration: a radius exceeds the double range; rescale the points");
        }
    }

    // Order ties by dimension, then increasing vertex tuple. Scaling can
    // introduce ties when the output radius is subnormal.
    const auto tie_less = [&](const Event& x, const Event& y) {
        if (x.top != y.top) return y.top;
        const index_t* vx = x.top ? &vertices[std::size_t(x.index) * kWidth] : &facet_vertices[std::size_t(x.index) * D];
        const index_t* vy = y.top ? &vertices[std::size_t(y.index) * kWidth] : &facet_vertices[std::size_t(y.index) * D];
        const int width = x.top ? kWidth : D;
        return std::lexicographical_compare(vx, vx + width, vy, vy + width);
    };
    for (auto begin = events.begin(); begin != events.end();) {
        auto end = begin + 1;
        while (end != events.end() && end->value == begin->value) ++end;
        if (end - begin > 1) std::sort(begin, end, tie_less);
        begin = end;
    }

    // Number cells in event order, top simplices first.
    const std::int64_t n_top = n_cells;
    std::vector<std::int64_t> top_id(n_cells), facet_id(events.size() - std::size_t(n_cells));
    result.n_top = n_top;
    result.event_cells.resize(events.size());
    result.event_values.resize(events.size());
    std::int64_t next_top = 0, next_facet = 0;
    for (std::size_t e = 0; e < events.size(); ++e) {
        const Event& event = events[e];
        std::int64_t id;
        if (event.top) {
            id = top_id[event.index] = next_top++;
        } else {
            facet_id[event.index] = next_facet;
            id = n_top + next_facet++;
        }
        result.event_cells[e] = id;
        result.event_values[e] = event.value;
    }
    result.top_vertices.resize(std::size_t(n_top) * kWidth);
    result.top_facets.resize(std::size_t(n_top) * kWidth);
    result.top_orientation.resize(std::size_t(n_top));
    for (index_t c = 0; c < n_cells; ++c) {
        const std::size_t row = std::size_t(top_id[c]) * kWidth, base = std::size_t(c) * kWidth;
        for (int j = 0; j < kWidth; ++j) {
            result.top_vertices[row + j] = vertices[base + j];
            result.top_facets[row + j] = n_top + facet_id[cell_facet[base + j]];
        }
        result.top_orientation[std::size_t(top_id[c])] = orientation[c];
    }
    result.facet_vertices.resize(facet_id.size() * D);
    for (std::size_t f = 0; f < facet_id.size(); ++f) {
        std::copy(&facet_vertices[f * D], &facet_vertices[f * D] + D, &result.facet_vertices[std::size_t(facet_id[f]) * D]);
    }
    result.n_exact_radii = counters.exact_radii;
    result.n_exact_ball_tests = counters.exact_ball_tests;
    return result;
}

}  // namespace

namespace {

DelaunayFiltration delaunay_filtration(const double* points, std::int64_t n_points, int dim, Values values) {
    if (dim != 2 && dim != 3) throw std::invalid_argument("the compiled Delaunay filtrations support R^2 and R^3 only");
    if (n_points < 0 || n_points > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("Delaunay filtration: unsupported number of points");
    }
    std::vector<double> scaled_points;
    int exponent = 0;
    if (!scale_points(points, std::size_t(n_points) * dim, scaled_points, exponent)) return {};
    const std::lock_guard<std::mutex> lock(geogram_mutex());
    initialize_geogram();
    const auto n = static_cast<index_t>(n_points);
    return dim == 2 ? build<2>(scaled_points.data(), n, exponent, values)
                    : build<3>(scaled_points.data(), n, exponent, values);
}

}  // namespace

DelaunayFiltration alpha_filtration(const double* points, std::int64_t n_points, int dim) {
    return delaunay_filtration(points, n_points, dim, Values::alpha);
}

DelaunayFiltration delaunay_cech_filtration(const double* points, std::int64_t n_points, int dim) {
    return delaunay_filtration(points, n_points, dim, Values::delaunay_cech);
}

}  // namespace persforest
