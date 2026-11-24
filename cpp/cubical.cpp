// Event order of full cubical grids: plain C++17, no Python dependency.
#include "cubical.hpp"

#include <cstddef>
#include <stdexcept>

namespace persforest {
namespace {

// Call visit(cell_id, is_top) for every top cell and facet in row-major order
// of doubled coordinates. A row fixes all doubled coordinates but the last.
// If they are all odd, top cells (odd last coordinate) alternate with facets
// normal to the last axis (even); if exactly one, at axis a, is even, the odd
// last coordinates are facets normal to a and the even ones lower-dimensional
// cells. Within a row, the IDs of each block advance by one per step.
template <typename Visit>
void for_each_cell_row_major(const std::vector<std::int64_t>& top_shape,
                             const std::vector<std::int64_t>& block_offsets, Visit&& visit) {
    const std::size_t d = top_shape.size();
    const std::int64_t last = top_shape[d - 1];
    std::vector<std::int64_t> x(d - 1, 0);  // doubled coordinates of the row
    // ID of the row's first cell in the block normal to `axis` (d: the top cells).
    auto row_start = [&](std::size_t axis) {
        std::int64_t index = 0;
        for (std::size_t a = 0; a + 1 < d; ++a) {
            index = index * (top_shape[a] + (a == axis ? 1 : 0)) + x[a] / 2;
        }
        return block_offsets[axis == d ? 0 : axis + 1] + index * (last + (axis == d - 1 ? 1 : 0));
    };
    for (;;) {
        std::size_t n_even = 0;
        std::size_t even_axis = 0;
        for (std::size_t a = 0; a + 1 < d; ++a) {
            if (x[a] % 2 == 0) {
                ++n_even;
                even_axis = a;
            }
        }
        if (n_even == 0) {
            const std::int64_t top = row_start(d);
            const std::int64_t facet = row_start(d - 1);
            for (std::int64_t j = 0; j < last; ++j) {
                visit(facet + j, false);
                visit(top + j, true);
            }
            visit(facet + last, false);
        } else if (n_even == 1) {
            const std::int64_t facet = row_start(even_axis);
            for (std::int64_t j = 0; j < last; ++j) visit(facet + j, false);
        }
        // Next row: advance the doubled coordinates 0 .. 2 n_a of all axes but the last.
        std::size_t a = d - 1;
        for (;;) {
            if (a == 0) return;
            --a;
            if (++x[a] <= 2 * top_shape[a]) break;
            x[a] = 0;
        }
    }
}

}  // namespace

std::vector<std::int64_t> cubical_event_order(const std::vector<std::int64_t>& top_shape,
                                              const std::int64_t* cell_rank, std::int64_t n_cells,
                                              std::int64_t n_ranks) {
    const std::size_t d = top_shape.size();
    if (d == 0 || n_ranks < 0) throw std::invalid_argument("invalid cubical grid");
    std::int64_t n_top = 1;
    for (const std::int64_t extent : top_shape) {
        if (extent < 1) throw std::invalid_argument("cubical grid extents must be positive");
        n_top *= extent;
    }
    // Blocks: top cells, then the facets normal to each axis.
    std::vector<std::int64_t> block_offsets(d + 2, 0);
    block_offsets[1] = n_top;
    for (std::size_t a = 0; a < d; ++a) {
        block_offsets[a + 2] = block_offsets[a + 1] + n_top / top_shape[a] * (top_shape[a] + 1);
    }
    if (block_offsets[d + 1] != n_cells) throw std::invalid_argument("cell count does not match the grid shape");

    // Counting sort: bucket 2 r holds the facets of rank r, bucket 2 r + 1 its
    // top cells (faces precede cofaces among equal values).
    std::vector<std::int64_t> bucket_start(2 * n_ranks + 1, 0);
    for_each_cell_row_major(top_shape, block_offsets, [&](std::int64_t cell, bool is_top) {
        const std::int64_t rank = cell_rank[cell];
        if (rank < 0 || rank >= n_ranks) throw std::invalid_argument("cell rank out of range");
        ++bucket_start[2 * rank + (is_top ? 1 : 0) + 1];
    });
    for (std::int64_t b = 0; b < 2 * n_ranks; ++b) bucket_start[b + 1] += bucket_start[b];
    std::vector<std::int64_t> order(n_cells);
    for_each_cell_row_major(top_shape, block_offsets, [&](std::int64_t cell, bool is_top) {
        order[bucket_start[2 * cell_rank[cell] + (is_top ? 1 : 0)]++] = cell;
    });
    return order;
}

}  // namespace persforest
