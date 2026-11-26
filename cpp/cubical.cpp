// Events of full cubical grids: plain C++17, no Python dependency.
#include "cubical.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace persforest {
namespace {

// Call visit(cell, is_top, position) for every top cell and facet in
// row-major order of doubled coordinates; cell is the input index and
// position the row-major index in the doubled grid. A row fixes all doubled
// coordinates but the last. If they are all odd, top cells (odd last
// coordinate) alternate with facets normal to the last axis (even); if
// exactly one, at axis a, is even, the odd last coordinates are facets normal
// to a and the even ones lower-dimensional cells. Within a row, the input
// indices of each block advance by one per step.
template <typename Visit>
void for_each_cell_row_major(const std::vector<std::int64_t>& top_shape,
                             const std::vector<std::int64_t>& block_offsets, Visit&& visit) {
    const std::size_t d = top_shape.size();
    const std::int64_t last = top_shape[d - 1];
    std::vector<std::int64_t> x(d - 1, 0);  // doubled coordinates of the row
    // Input index of the row's first cell in the block normal to `axis` (d: the top cells).
    auto row_start = [&](std::size_t axis) {
        std::int64_t index = 0;
        for (std::size_t a = 0; a + 1 < d; ++a) {
            index = index * (top_shape[a] + (a == axis ? 1 : 0)) + x[a] / 2;
        }
        return block_offsets[axis == d ? 0 : axis + 1] + index * (last + (axis == d - 1 ? 1 : 0));
    };
    std::int64_t position = 0;  // doubled-grid index of the row's first entry
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
                visit(facet + j, false, position + 2 * j);
                visit(top + j, true, position + 2 * j + 1);
            }
            visit(facet + last, false, position + 2 * last);
        } else if (n_even == 1) {
            const std::int64_t facet = row_start(even_axis);
            for (std::int64_t j = 0; j < last; ++j) visit(facet + j, false, position + 2 * j + 1);
        }
        position += 2 * last + 1;
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

CubicalEvents cubical_events(const std::vector<std::int64_t>& top_shape, const std::int32_t* cell_rank,
                             const double* cell_value, std::int64_t n_cells, std::int64_t n_ranks) {
    const std::size_t d = top_shape.size();
    if (d == 0 || n_ranks < 0) throw std::invalid_argument("invalid cubical grid");
    // Cells are addressed by their position in the doubled grid, whose size
    // prod(2 n + 1) must fit the 32-bit IDs. It exceeds the number of cells
    // (and of top cells), so checking it before each multiplication also
    // keeps n_top from overflowing. Thin grids in many dimensions reach the
    // limit with few cells: shape (1,) * 20 has 41 cells but 3^20 positions.
    constexpr std::int64_t max_positions = std::numeric_limits<std::int32_t>::max();
    std::int64_t n_top = 1;
    std::int64_t n_positions = 1;
    for (const std::int64_t extent : top_shape) {
        if (extent < 1) throw std::invalid_argument("cubical grid extents must be positive");
        if (extent > max_positions / 2 || n_positions > max_positions / (2 * extent + 1)) {
            throw std::invalid_argument(
                "cubical grids with more than " + std::to_string(max_positions) +
                " doubled-grid positions (the product of 2 n + 1 over the extents n) are not supported "
                "(32-bit IDs); this includes thin grids in many dimensions");
        }
        n_top *= extent;
        n_positions *= 2 * extent + 1;
    }
    // Input blocks: top cells, then the facets normal to each axis.
    std::vector<std::int64_t> block_offsets(d + 2, 0);
    block_offsets[1] = n_top;
    for (std::size_t a = 0; a < d; ++a) {
        block_offsets[a + 2] = block_offsets[a + 1] + n_top / top_shape[a] * (top_shape[a] + 1);
    }
    if (block_offsets[d + 1] != n_cells) throw std::invalid_argument("cell count does not match the grid shape");

    // Counting sort by (rank, dimension): the events of rank r are its facets,
    // then its top cells. Count both per rank, then turn the counts into the
    // numbers of facets and top cells of lower ranks.
    std::vector<std::int32_t> facets_below(n_ranks + 1, 0);
    std::vector<std::int32_t> tops_below(n_ranks + 1, 0);
    for_each_cell_row_major(top_shape, block_offsets, [&](std::int64_t cell, bool is_top, std::int64_t) {
        const std::int32_t rank = cell_rank[cell];
        if (rank < 0 || rank >= n_ranks) throw std::invalid_argument("cell rank out of range");
        ++(is_top ? tops_below : facets_below)[rank + 1];
    });
    for (std::int64_t r = 0; r < n_ranks; ++r) {
        facets_below[r + 1] += facets_below[r];
        tops_below[r + 1] += tops_below[r];
    }
    // Number the cells in event order, top cells first: the k-th top cell is
    // event k + (facets of rank <= r), the k-th facet event k + (top cells of
    // rank < r).
    CubicalEvents out;
    out.event_cells.resize(n_cells);
    out.event_values.resize(n_cells);
    out.cell_positions.resize(n_cells);
    std::vector<std::int32_t> id_at_position(n_positions, -1);
    {
        std::vector<std::int32_t> next_facet(facets_below.begin(), facets_below.end() - 1);
        std::vector<std::int32_t> next_top(tops_below.begin(), tops_below.end() - 1);
        for_each_cell_row_major(top_shape, block_offsets, [&](std::int64_t cell, bool is_top, std::int64_t position) {
            const std::int32_t rank = cell_rank[cell];
            std::int64_t id;
            std::int64_t event;
            if (is_top) {
                id = next_top[rank]++;
                event = id + facets_below[rank + 1];
            } else {
                const std::int64_t k = next_facet[rank]++;
                id = n_top + k;
                event = k + tops_below[rank];
            }
            out.event_cells[event] = id;
            out.event_values[event] = cell_value[cell];
            out.cell_positions[id] = static_cast<std::int32_t>(position);
            id_at_position[position] = static_cast<std::int32_t>(id);
        });
    }
    facets_below = std::vector<std::int32_t>();
    tops_below = std::vector<std::int32_t>();

    // Facets of each top cell, in the order of cubical_boundary: along each
    // axis the upper, then the lower facet, one doubled step away.
    std::vector<std::int64_t> stride(d, 1);
    for (std::size_t a = d - 1; a > 0; --a) stride[a - 1] = stride[a] * (2 * top_shape[a] + 1);
    out.top_facets.resize(n_top * 2 * d);
    for (std::int64_t t = 0; t < n_top; ++t) {
        const std::int64_t position = out.cell_positions[t];
        std::int64_t* facets = out.top_facets.data() + t * 2 * static_cast<std::int64_t>(d);
        for (std::size_t a = 0; a < d; ++a) {
            facets[2 * a] = id_at_position[position + stride[a]];
            facets[2 * a + 1] = id_at_position[position - stride[a]];
        }
    }
    return out;
}

}  // namespace persforest
