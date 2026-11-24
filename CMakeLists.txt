cmake_minimum_required(VERSION 3.18...3.31)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(Threads REQUIRED)

# geogram's Delaunay module, vendored unmodified (BSD-3-Clause; see
# cpp/third_party/geogram/README.md).
set(GEOGRAM_DIR cpp/third_party/geogram)
set(GEOGRAM_SOURCES ${GEOGRAM_DIR}/Delaunay_psm.cpp)

# The engine is plain C++17 (cpp/engine.*, cpp/cubical.*, cpp/simplicial.*,
# cpp/delaunay.*); bindings.cpp is the only file that knows about Python. No
# fast-math: filtration ties and tolerances rely on exact IEEE comparisons.
# NOMINSIZE keeps -O3: nanobind's default -Os slows the engine by about 20%.
nanobind_add_module(_engine NB_STATIC NOMINSIZE cpp/bindings.cpp cpp/engine.cpp cpp/cubical.cpp
                    cpp/simplicial.cpp cpp/delaunay.cpp ${GEOGRAM_SOURCES})

# GEO_STATIC_LIBS keeps geogram's API macros empty (no dllimport on Windows).
# The header is a system include so its warnings stay out of our files.
target_compile_definitions(_engine PRIVATE GEO_STATIC_LIBS)
target_include_directories(_engine SYSTEM PRIVATE ${GEOGRAM_DIR})
target_link_libraries(_engine PRIVATE Threads::Threads)

# geogram's exact predicates need separately rounded products: no fused
# multiply-add contraction in its source or in any file including its header
# (Clang on arm64 contracts by default). Warnings are silenced for the vendored
# source only.
if(MSVC)
  set(GEOGRAM_EXACT_FLAGS /fp:precise)
  set(GEOGRAM_VENDORED_FLAGS /w)
else()
  set(GEOGRAM_EXACT_FLAGS -ffp-contract=off)
  set(GEOGRAM_VENDORED_FLAGS -w)
endif()
set_source_files_properties(${GEOGRAM_SOURCES} PROPERTIES
                            COMPILE_OPTIONS "${GEOGRAM_EXACT_FLAGS};${GEOGRAM_VENDORED_FLAGS}")
set_source_files_properties(cpp/delaunay.cpp PROPERTIES COMPILE_OPTIONS "${GEOGRAM_EXACT_FLAGS}")

install(TARGETS _engine LIBRARY DESTINATION persforest)
