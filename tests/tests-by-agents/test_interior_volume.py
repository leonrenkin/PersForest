import math
import unittest

import numpy as np

from persforest import PersistenceForest
from persforest.chains import SignedChain
from persforest.measurements.basic import signed_chain_interior_volume
from persforest.landscapes.profiles import compute_measurement_profiles


class InteriorVolumeTests(unittest.TestCase):
    def test_simplex_volume_translation_and_scaling(self):
        for dim in (1, 2, 3, 4):
            points = np.vstack([np.zeros(dim), np.eye(dim)])
            chain = SignedChain(set(), interior_available=True,
                                interior={(tuple(range(dim + 1)), -1)})
            expected = 1 / math.factorial(dim)
            self.assertAlmostEqual(signed_chain_interior_volume(chain, points), expected)
            self.assertAlmostEqual(signed_chain_interior_volume(chain, 2 * points + 7),
                                   expected * 2**dim)

    def test_sum_triangle_areas_and_empty_interior(self):
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        chain = SignedChain(set(), interior_available=True,
                            interior={((0, 1, 2), 1), ((0, 2, 3), -1)})
        self.assertAlmostEqual(signed_chain_interior_volume(chain, points), 1.)
        chain.interior.clear()
        self.assertEqual(signed_chain_interior_volume(chain, points), 0.)

    def test_missing_interior_and_wrong_dimension(self):
        for available, interior in ((False, None), (False, set()), (True, None)):
            chain = SignedChain(set(), interior_available=available, interior=interior)
            with self.assertRaisesRegex(ValueError, "iter_bar_cycle_reps"):
                signed_chain_interior_volume(chain, np.eye(3))
        chain = SignedChain(set(), interior_available=True, interior={((0, 1), 1)})
        with self.assertRaisesRegex(ValueError, "full-dimensional"):
            signed_chain_interior_volume(chain, np.eye(3))

    def test_measurement_profiles_match_representatives(self):
        points = np.random.default_rng(42).random((30, 3))
        forest = PersistenceForest(points)
        profiles = compute_measurement_profiles(forest, signed_chain_interior_volume, "volume", signed=True)
        for bar in forest.barcode:
            ascending = list(reversed(list(forest.iter_bar_cycle_reps(bar))))
            np.testing.assert_array_equal(profiles[bar].starts, [r.active_start for r in ascending])
            np.testing.assert_array_equal(profiles[bar].ends, [r.active_end for r in ascending])
            np.testing.assert_allclose(profiles[bar].vals,
                                       [signed_chain_interior_volume(r, points) for r in ascending])


if __name__ == '__main__':
    unittest.main()
