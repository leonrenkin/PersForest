// Python bindings of the construction engine (nanobind). Only this file knows about Python.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "cubical.hpp"
#include "engine.hpp"

namespace nb = nanobind;

namespace {

template <typename T>
using InArray = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename T>
using OutArray = nb::ndarray<nb::numpy, T, nb::ndim<1>>;

// Hand a vector to NumPy without copying; the capsule frees it with the array.
template <typename T>
OutArray<T> to_numpy(std::vector<T>&& values) {
    auto* owned = new std::vector<T>(std::move(values));
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return OutArray<T>(owned->data(), {owned->size()}, owner);
}

nb::dict build_forest(InArray<std::int64_t> event_cells, InArray<double> event_values, std::int64_t n_top,
                      std::int64_t n_cells,
                      nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> top_facets,
                      InArray<std::int8_t> top_orientation, bool reduce, double filtration_tol) {
    if (event_cells.shape(0) != event_values.shape(0)) {
        throw std::invalid_argument("event_cells and event_values differ in length");
    }
    if (static_cast<std::int64_t>(top_facets.shape(0)) != n_top ||
        static_cast<std::int64_t>(top_orientation.shape(0)) != n_top) {
        throw std::invalid_argument("top_facets and top_orientation need one row per top cell");
    }
    persforest::CellEvents events;
    events.cells = event_cells.data();
    events.values = event_values.data();
    events.n_events = static_cast<std::int64_t>(event_cells.shape(0));
    events.n_top = n_top;
    events.n_cells = n_cells;
    events.top_facets = top_facets.data();
    events.facets_per_top = static_cast<std::int64_t>(top_facets.shape(1));
    events.top_orientation = top_orientation.data();

    persforest::ForestArrays forest;
    {
        nb::gil_scoped_release release;
        forest = persforest::build_forest(events, reduce, filtration_tol);
    }
    nb::dict out;
    out["node_value"] = to_numpy(std::move(forest.node_value));
    out["node_parent"] = to_numpy(std::move(forest.node_parent));
    out["node_first"] = to_numpy(std::move(forest.node_first));
    out["diff_offsets"] = to_numpy(std::move(forest.diff_offsets));
    out["diff_cells"] = to_numpy(std::move(forest.diff_cells));
    out["diff_signs"] = to_numpy(std::move(forest.diff_signs));
    out["bar_birth"] = to_numpy(std::move(forest.bar_birth));
    out["bar_death"] = to_numpy(std::move(forest.bar_death));
    out["bar_parent"] = to_numpy(std::move(forest.bar_parent));
    out["bar_tree"] = to_numpy(std::move(forest.bar_tree));
    out["progression_offsets"] = to_numpy(std::move(forest.progression_offsets));
    out["progression_nodes"] = to_numpy(std::move(forest.progression_nodes));
    return out;
}

nb::dict cubical_events(std::vector<std::int64_t> top_shape, InArray<std::int32_t> cell_rank,
                        InArray<double> cell_value, std::int64_t n_ranks) {
    if (cell_rank.shape(0) != cell_value.shape(0)) {
        throw std::invalid_argument("cell_rank and cell_value differ in length");
    }
    persforest::CubicalEvents events;
    {
        nb::gil_scoped_release release;
        events = persforest::cubical_events(top_shape, cell_rank.data(), cell_value.data(),
                                            static_cast<std::int64_t>(cell_rank.shape(0)), n_ranks);
    }
    nb::dict out;
    out["event_cells"] = to_numpy(std::move(events.event_cells));
    out["event_values"] = to_numpy(std::move(events.event_values));
    out["cell_positions"] = to_numpy(std::move(events.cell_positions));
    out["top_facets"] = to_numpy(std::move(events.top_facets));
    return out;
}

}  // namespace

NB_MODULE(_engine, m) {
    m.doc() = "Compiled construction engine of persforest (internal).";
    m.attr("INTERFACE_VERSION") = persforest::kEngineInterfaceVersion;
    m.def("build_forest", &build_forest, nb::arg("event_cells"), nb::arg("event_values"), nb::arg("n_top"),
          nb::arg("n_cells"), nb::arg("top_facets"), nb::arg("top_orientation"), nb::arg("reduce"),
          nb::arg("filtration_tol"),
          "Run the reverse sweep, optional reduction and elder-rule bar extraction; return flat arrays.");
    m.def("cubical_events", &cubical_events, nb::arg("top_shape"), nb::arg("cell_rank"), nb::arg("cell_value"),
          nb::arg("n_ranks"),
          "Order and number the top cells and facets of a cubical grid; return events, positions and facets.");
}
