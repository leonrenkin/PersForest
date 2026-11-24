// Python bindings of the construction engine (nanobind). Only this file knows about Python.
#include <nanobind/nanobind.h>

#include "engine.hpp"

namespace nb = nanobind;

NB_MODULE(_engine, m) {
    m.doc() = "Compiled construction engine of persforest (internal).";
    m.attr("INTERFACE_VERSION") = persforest::kEngineInterfaceVersion;
}
