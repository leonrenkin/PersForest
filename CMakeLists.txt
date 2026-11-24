cmake_minimum_required(VERSION 3.18...3.31)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# The engine is plain C++17 (cpp/engine.*, cpp/cubical.*); bindings.cpp is the
# only file that knows about Python. No fast-math: filtration ties and
# tolerances rely on exact IEEE comparisons. NOMINSIZE keeps -O3: nanobind's
# default -Os slows the engine by about 20%.
nanobind_add_module(_engine NB_STATIC NOMINSIZE cpp/bindings.cpp cpp/engine.cpp cpp/cubical.cpp)

install(TARGETS _engine LIBRARY DESTINATION persforest)
