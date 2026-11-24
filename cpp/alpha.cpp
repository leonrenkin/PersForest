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
// Values are scaled back exactly at the end. A cell much smaller than the
// cloud (all edge components below 2^-60) is in addition rescaled by a power
// of two before its radius or ball test is computed, so the double formulas
// do not underflow either.
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
// denominator are computed exactly and rounded (a few ulps).
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
#include "alpha.hpp"

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

// (p - q) 2^k exactly; z = 0 in 2D.
template <int D>
ExactVec exact_difference(const double* p, const double* q, int k) {
    auto component = [k](double a, double b) { return Exact(std::ldexp(a, k)) - Exact(std::ldexp(b, k)); };
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
    const ExactVec ea = exact_difference<2>(p[1], p[0], k), eb = exact_difference<2>(p[2], p[0], k),
                   ec = exact_difference<2>(p[2], p[1], k);
    const Exact edet = ea.x * eb.y - ea.y * eb.x;
    const GEO::Sign sign = edet.sign();
    if (sign == GEO::ZERO) flat_cell();
    const Exact numerator = dot(ea, ea) * dot(eb, eb) * dot(ec, ec);
    const Exact denominator = edet * edet * 4.0;
    return {std::ldexp(numerator.estimate() / denominator.estimate(), -2 * k),
            static_cast<std::int8_t>(sign == GEO::POSITIVE ? 1 : -1)};
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
    const ExactVec ea = exact_difference<3>(p[1], p[0], k), eb = exact_difference<3>(p[2], p[0], k),
                   ec = exact_difference<3>(p[3], p[0], k);
    const ExactVec ebc = cross(eb, ec);
    const Exact edet = dot(ea, ebc);
    const GEO::Sign sign = edet.sign();
    if (sign == GEO::ZERO) flat_cell();
    const ExactVec em =
        sum(sum(scaled(ebc, dot(ea, ea)), scaled(cross(ec, ea), dot(eb, eb))), scaled(cross(ea, eb), dot(ec, ec)));
    const Exact denominator = edet * edet * 4.0;
    return {std::ldexp(dot(em, em).estimate() / denominator.estimate(), -2 * k),
            static_cast<std::int8_t>(sign == GEO::POSITIVE ? 1 : -1)};
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
    const ExactVec ea = exact_difference<3>(p[1], p[0], k), eb = exact_difference<3>(p[2], p[0], k),
                   ec = exact_difference<3>(p[2], p[1], k);
    const ExactVec en = cross(ea, eb);
    const Exact denominator = dot(en, en) * 4.0;
    if (denominator.sign() == GEO::ZERO) flat_cell();
    const Exact numerator = dot(ea, ea) * dot(eb, eb) * dot(ec, ec);
    return std::ldexp(numerator.estimate() / denominator.estimate(), -2 * k);
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
    return dot(exact_difference<D>(q, p0, 0), exact_difference<D>(q, p1, 0)).sign();
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
    const ExactVec ea = exact_difference<3>(p[1], p[0], 0), eb = exact_difference<3>(p[2], p[0], 0),
                   ew = exact_difference<3>(q, p[0], 0);
    const ExactVec en = cross(ea, eb);
    const ExactVec em = sum(scaled(cross(eb, en), dot(ea, ea)), scaled(cross(en, ea), dot(eb, eb)));
    return (dot(en, en) * dot(ew, ew) - dot(ew, em)).sign();
}

// Minimal enclosing balls (Delaunay-Cech values). The minimal enclosing ball
// of a simplex is the smallest sphere through some face (center in the face's
// affine hull) that contains every vertex, so its radius is the least radius
// among these candidates that contain the simplex (closed balls). Every
// candidate containing the simplex encloses it and is at least as large.

// Squared radius of the minimal enclosing ball of a triangle in R^D, given the
// squared radius of its smallest sphere: an edge's diametral ball if it
// contains the opposite vertex (an obtuse or right angle there), else that sphere.
template <int D>
double triangle_enclosing_squared(const double* const* p, double sphere_squared, Counters& counters) {
    double best = sphere_squared;
    for (int k = 0; k < 3; ++k) {
        const int i = k == 0 ? 1 : 0, j = k == 2 ? 1 : 2;  // the edge opposite vertex k, increasing
        if (edge_ball_side<D>(p[i], p[j], p[k], counters) != GEO::POSITIVE) {
            best = std::min(best, edge_squared_radius<D>(p[i], p[j]));
        }
    }
    return best;
}

// The same for a tetrahedron, given its squared circumradius. An edge ball that
// contains the other two vertices has half the diameter as radius and is the
// minimal ball; otherwise the minimal ball is the circumsphere or the smallest
// sphere of a face triangle that contains the fourth vertex.
double tetrahedron_enclosing_squared(const double* const* p, double sphere_squared, Counters& counters) {
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            int others[2], n_others = 0;
            for (int k = 0; k < 4; ++k) {
                if (k != i && k != j) others[n_others++] = k;
            }
            if (edge_ball_side<3>(p[i], p[j], p[others[0]], counters) != GEO::POSITIVE &&
                edge_ball_side<3>(p[i], p[j], p[others[1]], counters) != GEO::POSITIVE) {
                return edge_squared_radius<3>(p[i], p[j]);
            }
        }
    }
    double best = sphere_squared;
    for (int omitted = 0; omitted < 4; ++omitted) {
        const double* face[3];
        for (int k = 0, f = 0; k < 4; ++k) {
            if (k != omitted) face[f++] = p[k];
        }
        if (triangle_sphere_side(face, p[omitted], counters) != GEO::POSITIVE) {
            best = std::min(best, triangle_squared_radius(face, counters));
        }
    }
    return best;
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
        if (x != 0.0 && std::abs(x) < kMinScaledCoordinate) {
            throw std::invalid_argument(
                "point coordinates span too many orders of magnitude for the Delaunay filtration: every nonzero "
                "coordinate must be at least 2^-100 (about 8e-31) times the largest absolute coordinate");
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
};

// Sort events by value with a stable LSD radix sort on the bits of the values
// (for nonnegative doubles, unsigned integer order is numeric order), then each
// run of equal values with tie_less. In general position nearly all runs have
// one or two events, so this is much faster than a comparison sort.
template <typename TieLess>
void sort_events(std::vector<Event>& events, TieLess tie_less) {
    constexpr int kDigitBits = 11, kPasses = 6;  // 6 x 11 bits cover 64
    constexpr std::size_t kBuckets = std::size_t(1) << kDigitBits;
    auto key = [](const Event& event) {
        const double value = event.value + 0.0;  // never -0.0
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof bits);
        return bits;
    };
    const std::size_t n = events.size();
    std::vector<std::size_t> counts(kPasses * kBuckets, 0);
    for (const Event& event : events) {
        const std::uint64_t bits = key(event);
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
        for (const Event& event : *source) (*target)[count[(key(event) >> (pass * kDigitBits)) & (kBuckets - 1)]++] = event;
        std::swap(source, target);
    }
    if (source != &events) events.swap(buffer);
    for (std::size_t begin = 0; begin < n;) {
        std::size_t end = begin + 1;
        while (end < n && events[end].value == events[begin].value) ++end;
        if (end - begin > 1) std::sort(events.begin() + std::ptrdiff_t(begin), events.begin() + std::ptrdiff_t(end), tie_less);
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
    // vertex, squared value (circumradius or minimal enclosing ball) and
    // orientation of the sorted simplex.
    Counters counters;
    std::vector<index_t> vertices(raw_vertices.size());
    std::vector<std::uint8_t> position(raw_vertices.size());
    std::vector<double> top_squared(n_cells);
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
        top_squared[c] = top.squared_radius;
        if (values == Values::delaunay_cech) {
            if constexpr (D == 2) {
                top_squared[c] = triangle_enclosing_squared<2>(p, top.squared_radius, counters);
            } else {
                top_squared[c] = tetrahedron_enclosing_squared(p, top.squared_radius, counters);
            }
        }
        orientation[c] = top.orientation;
    }
    raw_vertices = std::vector<index_t>();

    // Facets, each once: from the cell with the smaller index, or the only one.
    std::vector<index_t> facet_vertices;
    facet_vertices.reserve(n_facets * D);
    std::vector<double> facet_squared;
    facet_squared.reserve(n_facets);
    std::vector<std::uint32_t> cell_facet(std::size_t(n_cells) * kWidth);  // by sorted position
    for (index_t c = 0; c < n_cells; ++c) {
        const std::size_t base = std::size_t(c) * kWidth;
        for (int local = 0; local < kWidth; ++local) {
            const index_t other = neighbours[base + local];
            if (other != GEO::NO_INDEX && other < c) continue;
            const auto facet = static_cast<std::uint32_t>(facet_squared.size());
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
            double earliest_top = top_squared[c];
            index_t other_opposite = GEO::NO_INDEX;
            if (other != GEO::NO_INDEX) {
                const std::size_t other_base = std::size_t(other) * kWidth;
                int other_local = 0;
                while (other_local < kWidth && neighbours[other_base + other_local] != c) ++other_local;
                if (other_local == kWidth) throw std::logic_error("Delaunay filtration: inconsistent neighbours");
                const int other_j = position[other_base + other_local];
                cell_facet[other_base + other_j] = facet;
                earliest_top = std::min(earliest_top, top_squared[other]);
                other_opposite = vertices[other_base + other_j];
            }
            double own;
            if constexpr (D == 2) {
                own = edge_squared_radius<2>(p[0], p[1]);
            } else {
                own = triangle_squared_radius(p, counters);
            }
            double value;
            if (values == Values::alpha) {
                const bool attached =
                    inside(vertices[base + j]) || (other_opposite != GEO::NO_INDEX && inside(other_opposite));
                value = attached ? earliest_top : own;
            } else if constexpr (D == 2) {
                value = own;  // an edge's minimal enclosing ball is its diametral ball
            } else {
                value = triangle_enclosing_squared<3>(p, own, counters);
            }
            // The clamp keeps a facet from entering after a top simplex it bounds;
            // mathematically it holds already, so it only absorbs rounding.
            facet_squared.push_back(std::min(value, earliest_top));
        }
    }
    neighbours = std::vector<index_t>();
    position = std::vector<std::uint8_t>();

    // Values in radius units, scaled back exactly.
    auto radius = [exponent](double squared) {
        const double value = std::ldexp(std::sqrt(squared), exponent);
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Delaunay filtration: a radius exceeds the double range; rescale the points");
        }
        return value;
    };
    std::vector<Event> events;
    events.reserve(std::size_t(n_cells) + facet_squared.size());
    for (index_t c = 0; c < n_cells; ++c) events.push_back({radius(top_squared[c]), c, true});
    for (std::size_t f = 0; f < facet_squared.size(); ++f) {
        events.push_back({radius(facet_squared[f]), static_cast<std::uint32_t>(f), false});
    }
    top_squared = std::vector<double>();
    facet_squared = std::vector<double>();

    // Order: value, facets before tops, increasing vertex tuple.
    sort_events(events, [&](const Event& x, const Event& y) {
        if (x.top != y.top) return y.top;
        const index_t* vx = x.top ? &vertices[std::size_t(x.index) * kWidth] : &facet_vertices[std::size_t(x.index) * D];
        const index_t* vy = y.top ? &vertices[std::size_t(y.index) * kWidth] : &facet_vertices[std::size_t(y.index) * D];
        const int width = x.top ? kWidth : D;
        return std::lexicographical_compare(vx, vx + width, vy, vy + width);
    });

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
