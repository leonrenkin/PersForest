"""Include the AI-generated suite in standard unittest discovery."""

from pathlib import Path
import unittest


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Load the complete agent suite whenever this bridge is selected."""
    suite_dir = Path(__file__).parent / "tests-by-agents"
    # Discovery makes fixture/oracle imports available from this non-package directory.
    # Use its own pattern even when outer discovery selects only this bridge.
    tests.addTests(loader.discover(
        str(suite_dir), pattern="test*.py", top_level_dir=str(suite_dir),
    ))
    return tests
