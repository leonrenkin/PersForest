# geogram Delaunay module (vendored)

`Delaunay_psm.h`, `Delaunay_psm.cpp` and `LICENSE` are copied unmodified from
geogram's Delaunay pluggable software module (PSM), a standalone extract of
[geogram](https://github.com/BrunoLevy/geogram) by Bruno Levy (Inria):

- Repository: <https://github.com/BrunoLevy/geogram.psm.Delaunay>
- Commit: `9d4b56b9c1b0ae419baf4ec1425cfa6da769cd31` (2026-09-30)
- License: BSD-3-Clause, see `LICENSE`. Its notice ships with every
  distribution of persforest (`license-files` in `pyproject.toml`).

SHA-256 of the copied files:

```
a9d292ef24a8fd9e2353cf9fa3e232473eabc25c06241ab7b96586cfefcfb328  Delaunay_psm.h
8b93e9ad904f7f035112e453187bf24ca5d192deb902bb8bb8e2e266fa8c0591  Delaunay_psm.cpp
5071ceca467d28e6fefb5f0dba4066abfd03d8b8b5b3af968d0aa7754c5e3f9c  LICENSE
```

## Build requirements

`CMakeLists.txt` compiles `Delaunay_psm.cpp` into the `_engine` module. The
settings below are needed by the module, not stylistic:

- Fused multiply-add contraction must be off (`-ffp-contract=off`) in this file
  and in every file that includes `Delaunay_psm.h`: the exact predicates and
  `expansion_nt` arithmetic rely on separately rounded products. Clang on arm64
  contracts by default. GCC has no pragma for this, so geogram checks at load
  time and aborts if it detects contraction. MSVC builds use `/fp:precise`,
  and geogram adds an `fp_contract(off)` pragma (not tested here).
- `GEO_STATIC_LIBS` is defined, so the API macros do not expand to
  `__declspec(dllimport)` on Windows.
- geogram uses its own threads (pthreads, or Windows threads), linked through
  `Threads::Threads`. OpenMP, TBB and the parallel STL stay disabled; they are
  enabled only by defining `GEO_OPENMP`, `GEO_TBB` or `GEO_PARALLEL_STL`.
- `-frounding-math`, which geogram's README uses with GCC, is not needed: the
  interval filters use round-to-nearest arithmetic (`intervalRN`).

## Updating

Replace the three files with those of a newer commit, update the commit and
hashes above, and rerun the builder tests (`test_delaunay_builder.py`), which
compare with GUDHI's `precision='exact'` mode and exact rational oracles. Do
not edit the vendored files; keep any adaptation in persforest's own sources.
