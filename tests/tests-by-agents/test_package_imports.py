"""The public forest entry point and canonical implementation modules."""
import ast
import importlib.util
from pathlib import Path
import unittest
import persforest
from persforest import PersistenceForest
from persforest.forest import PersistenceForest as ForestImplementation

PACKAGE = Path(persforest.__file__).resolve().parent
REMOVED_MODULES = (
    'PersistenceForest', '_validation', 'color_scheme', 'cycle_rep_vectorisations',
    'forest_landscapes', 'forest_plotting', 'forest_plotting_geometry',
    'interior_activity_plotly', 'interior_activity_plotting',
    'simplicial_filtration_animation', 'simplicial_filtration_plotly',
    'simplicial_filtration_plotting',
)


class PackageImportTests(unittest.TestCase):
    def test_public_entry_point_exports_the_canonical_forest_class(self):
        self.assertIs(PersistenceForest, ForestImplementation)
        self.assertEqual(persforest.__all__, ['PersistenceForest'])
        forest = PersistenceForest([[0, 0], [1, 0], [0, 1]])
        self.assertIsInstance(forest, ForestImplementation)

    def test_legacy_import_shims_are_removed(self):
        for name in REMOVED_MODULES:
            with self.subTest(module=name):
                self.assertFalse((PACKAGE / (name + '.py')).exists())
                self.assertIsNone(importlib.util.find_spec('persforest.' + name))

    def test_domain_classes_have_one_canonical_definition(self):
        from persforest.chains import SignedChain
        self.assertEqual(SignedChain.__bases__, (object,))
        classes = {}
        for path in PACKAGE.rglob('*.py'):
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, ast.ClassDef):
                    self.assertNotIn(node.name, classes, (path, classes.get(node.name)))
                    classes[node.name] = path

    def test_animation_uses_implementation_modules_without_reexports(self):
        import persforest.plotting.animation as animation
        import persforest.plotting.landscapes as landscapes
        self.assertFalse(hasattr(animation, '_animate_filtration_generic'))
        self.assertFalse(hasattr(animation, 'animate_filtration'))
        self.assertFalse(hasattr(landscapes, 'animate_barcode_measurement_generic'))

