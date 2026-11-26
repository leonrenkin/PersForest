"""Test an installed wheel with the engine-focused modules of this directory.

Usage
-----
    python tests/tests-by-agents/wheel_smoke.py

cibuildwheel runs this against every built wheel, from a directory outside the
checkout. The script refuses to run against a ``persforest`` imported from the
checkout's ``src``. Modules whose fixtures need GUDHI are skipped where GUDHI
has no wheel (e.g. Linux aarch64); the skip is printed, not hidden.
"""
import importlib.metadata
import importlib.util
from pathlib import Path
import sys
import unittest

TESTS = Path(__file__).resolve().parent
CHECKOUT_SRC = TESTS.parents[1] / "src"

# Construction, engine output, ordering and core queries; no plotting.
WITHOUT_GUDHI = ["test_installation", "test_cubical_event_order", "test_orientation_numerics",
                 "test_bar_order_and_queries"]
WITH_GUDHI = ["test_engine_golden", "test_compiled_engine", "test_delaunay_builder", "test_cubical_grid",
              "test_simplicial_adapter", "test_filtration_ties", "test_additive_measurements",
              "test_public_api"]


def main() -> int:
    """Run the selected modules; return the process exit status."""
    import persforest
    package = Path(persforest.__file__).resolve().parent
    if CHECKOUT_SRC in package.parents:
        print(f"persforest was imported from the checkout ({package}), not from the wheel", file=sys.stderr)
        return 2
    print(f"Testing persforest {importlib.metadata.version('persforest')} from {package}")
    names = list(WITHOUT_GUDHI)
    if importlib.util.find_spec("gudhi") is None:
        print("GUDHI is not installed; skipping " + ", ".join(WITH_GUDHI))
    else:
        names += WITH_GUDHI
    sys.path.insert(0, str(TESTS))
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
