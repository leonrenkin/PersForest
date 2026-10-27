"""Core installation works without visualization and notebook dependencies."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import persforest


class InstallationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'plot extra is not installed')
    def test_plotting_without_seaborn_or_pandas(self):
        package_parent = str(Path(persforest.__file__).resolve().parent.parent)
        code = f'''
import importlib.abc
import sys
import warnings
sys.path.insert(0, {package_parent!r})
class BlockDataframeLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'seaborn', 'pandas'}}:
            raise ModuleNotFoundError('Unneeded plotting dependency', name=fullname)
sys.meta_path.insert(0, BlockDataframeLibraries())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from persforest import PersistenceForest
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    forest = PersistenceForest([[0, 0], [3, 0], [0, 4]], filtration=[
        ([0], 0), ([1], 0), ([2], 0), ([0, 1], 1),
        ([0, 2], 1), ([1, 2], 2), ([0, 1, 2], 6)])
for coloring in ('bars', 'forest'):
    ax = forest.plot_barcode(coloring=coloring)
    ax.figure.canvas.draw()
plt.close('all')
assert not ({{'pandas', 'seaborn'}} & {{name.split('.')[0] for name in sys.modules}})
'''
        result = subprocess.run(
            [sys.executable, '-I', '-B', '-c', code],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_computation_without_optional_dependencies_and_plotting_install_hint(self):
        # A fresh interpreter avoids optional modules cached by the full suite.
        # Explicitly select the same installed or source package as the parent.
        package_parent = str(Path(persforest.__file__).resolve().parent.parent)
        code = f'''
import importlib.abc
import sys
import warnings
sys.path.insert(0, {package_parent!r})
optional = {{'matplotlib', 'seaborn', 'pandas', 'plotly', 'PIL',
            'nbformat', 'ipywidgets', 'anywidget', 'sklearn'}}
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in optional:
            raise ModuleNotFoundError('Optional dependency blocked', name=fullname)
sys.meta_path.insert(0, BlockOptional())
from persforest import PersistenceForest
from persforest.chains import SignedChain
from persforest.measurements.basic import signed_chain_edge_length
points = [[0, 0], [3, 0], [0, 4]]
filtration = [([0], 0), ([1], 0), ([2], 0), ([0, 1], 1),
              ([0, 2], 1), ([1, 2], 2), ([0, 1, 2], 6)]
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    forest = PersistenceForest(points, filtration=filtration)
bar, = forest.barcode
assert (bar.birth, bar.death) == (2, 6)
assert signed_chain_edge_length(bar.cycle_at_filtration_value(3), forest.point_cloud) == 12
family = forest.compute_measurement_landscapes(
    cycle_func=signed_chain_edge_length, max_k=1, x_grid=[2, 3, 4, 5, 6], label='length')
assert family.evaluate_on_grid().shape == (1, 5)
alpha = PersistenceForest(points)
assert alpha.simplex_tree.num_vertices() == 3
assert not (optional & {{name.split('.')[0] for name in sys.modules}})
for plot in (forest.plot_barcode, lambda: forest.plot_at_filtration(3),
             lambda: forest.plot_measurement_landscapes(label='length')):
    try:
        plot()
    except ImportError as error:
        assert 'persforest[plot]' in str(error), str(error)
    else:
        raise AssertionError('Plotting succeeded without its dependencies')
'''
        with self.subTest(python=sys.version):
            result = subprocess.run(
                [sys.executable, '-I', '-B', '-c', code],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
