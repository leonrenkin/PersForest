// Construction engine of persforest: plain C++17, no Python dependency.
#pragma once

namespace persforest {

// Version of the array interface between the engine and the Python layer.
// Increase it whenever inputs or outputs of the engine change.
constexpr int kEngineInterfaceVersion = 1;

}  // namespace persforest
