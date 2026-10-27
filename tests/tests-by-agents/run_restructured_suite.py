"""Run tests with explicit package selection and import-path guards.

Examples
--------
    .venv/bin/python tests/tests-by-agents/run_restructured_suite.py --implementation src

The installed mode must be invoked from outside the checkout. No package path is
injected in that mode. Child Python processes receive the same path assertions
through a temporary sitecustomize module.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Select an implementation, assert its origin, and run unittest discovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation', choices=('src', 'installed'), required=True)
    parser.add_argument('--expected-package', type=Path)
    parser.add_argument('--pattern', default='test*.py')
    parser.add_argument('--summary', type=Path)
    args = parser.parse_args()
    if args.implementation == 'installed':
        if args.expected_package is None:
            parser.error('installed mode requires --expected-package')
        if Path.cwd().resolve().is_relative_to(ROOT):
            parser.error('installed mode must run outside the checkout')
        expected = args.expected_package.resolve()
    else:
        expected = ROOT / 'src/persforest'
        sys.path.insert(0, str(expected.parent))

    import persforest
    assert Path(persforest.__file__).resolve().parent == expected, persforest.__file__
    print('VERIFIED PACKAGE:', persforest.__file__, flush=True)

    with tempfile.TemporaryDirectory(prefix='persforest-import-guard-') as directory:
        # Subprocess tests may change cwd. Preload the selected package before
        # user code so that checkout-root shadowing cannot go undetected.
        startup = f'''
import atexit
import os
from pathlib import Path
import sys
expected = Path({str(expected)!r})
if {args.implementation != 'installed'!r}:
    sys.path.insert(0, str(expected.parent))
import persforest
def check_imports():
    bad = [(name, module.__file__) for name, module in list(sys.modules.items())
           if (name == "persforest" or name.startswith("persforest."))
           and getattr(module, "__file__", None)
           and not Path(module.__file__).resolve().is_relative_to(expected)]
    if bad:
        print("WRONG PACKAGE ORIGIN:", bad, file=sys.stderr, flush=True)
        os._exit(97)
check_imports()
atexit.register(check_imports)
'''
        Path(directory, 'sitecustomize.py').write_text(startup)
        old_pythonpath = os.environ.get('PYTHONPATH')
        os.environ['PYTHONPATH'] = directory + (os.pathsep + old_pythonpath if old_pythonpath else '')
        try:
            if args.pattern == 'test*.py':
                suite = unittest.defaultTestLoader.discover(str(ROOT), pattern=args.pattern)
            else:
                suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests/tests-by-agents'), pattern=args.pattern)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
        finally:
            if old_pythonpath is None:
                os.environ.pop('PYTHONPATH', None)
            else:
                os.environ['PYTHONPATH'] = old_pythonpath
    modules = {name: str(Path(module.__file__).resolve()) for name, module in list(sys.modules.items())
               if (name == 'persforest' or name.startswith('persforest.'))
               and getattr(module, '__file__', None)}
    for name, path in modules.items():
        assert Path(path).is_relative_to(expected), (name, path)
    summary = dict(implementation=args.implementation, package=str(expected),
                   tests=result.testsRun, failures=[test.id() for test, _ in result.failures],
                   errors=[test.id() for test, _ in result.errors],
                   skipped=[(test.id(), reason) for test, reason in result.skipped], modules=modules)
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=2) + '\n')
    print('VERIFIED: all loaded persforest modules are under', expected, flush=True)
    sys.exit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
