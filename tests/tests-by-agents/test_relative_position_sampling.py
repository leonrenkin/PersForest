"""Representatives sampled at a relative position of each bar's half-open lifespan.

With ``reduce=False`` many bars have ``birth == death``: their lifespan
``[birth, death)`` is empty and no representative is active, so the sampling
methods skip them. A bar a single ulp long has exactly one filtration value,
its birth, at which a representative is active.
"""
import math
import unittest
import warnings
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from persforest import PersistenceForest
from persforest.barcode import _value_at_relative_position

ONE_ULP_DEATH = math.nextafter(1.0, 2.0)


def _key(chain):
    return (frozenset(chain.signed_simplices), frozenset(chain.interior), chain.active_start, chain.active_end)


def _forests_with_zero_length_bars():
    rng = np.random.default_rng(0)
    yield "alpha2", PersistenceForest(rng.random((60, 2)), reduce=False)
    yield "alpha3", PersistenceForest(rng.random((40, 3)), reduce=False)
    yield "cubical2", PersistenceForest.from_cubical_grid(rng.integers(0, 4, (12, 12)).astype(float), reduce=False)
    yield "cubical2-vertices", PersistenceForest.from_cubical_grid(
        rng.random((12, 12)), filtration_type="vertices", reduce=False)


def _one_ulp_triangle():
    """A triangle whose boundary closes at 1.0 and fills one ulp later."""
    filtration = [((0,), 0.), ((1,), 0.), ((2,), 0.), ((0, 1), 1.), ((0, 2), 1.), ((1, 2), 1.),
                  ((0, 1, 2), ONE_ULP_DEATH)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # custom filtrations are not checked
        return PersistenceForest([(0, 0), (3, 0), (0, 4)], filtration=filtration, reduce=False)


class RelativePositionSamplingTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_barcode_cycle_reps_skips_bars_with_empty_lifespan(self):
        for name, forest in _forests_with_zero_length_bars():
            bars = sorted(forest.barcode, key=lambda bar: bar.lifespan(), reverse=True)
            self.assertTrue(any(bar.birth == bar.death for bar in bars))
            positive = [bar for bar in bars if bar.birth < bar.death]
            for position in (0, 0.1, 0.5):
                with self.subTest(forest=name, position=position):
                    values = [bar.birth + bar.lifespan() * position for bar in positive]
                    reps = forest.barcode_cycle_reps(relative_position=position)
                    expected = [forest.cycle_for_bar_at(bar, value) for bar, value in zip(positive, values)]
                    self.assertEqual([_key(rep) for rep in reps], [_key(rep) for rep in expected])
                    for rep, value in zip(reps, values):
                        self.assertTrue(rep.active_start <= value < rep.active_end)

    def test_representative_plots_skip_bars_with_empty_lifespan(self):
        for name, forest in _forests_with_zero_length_bars():
            if forest.dim != 2:
                continue
            with self.subTest(forest=name):
                positive = {bar for bar in forest.barcode if bar.birth < bar.death}
                with mock.patch.object(forest, 'cycle_for_bar_at', wraps=forest.cycle_for_bar_at) as query:
                    forest.plot_barcode_cycle_reps(relative_position=0, show=False)
                queried = [call.args for call in query.call_args_list]
                self.assertEqual({bar for bar, _ in queried}, positive)
                self.assertEqual(len(queried), len(positive))
                self.assertTrue(all(value == bar.birth for bar, value in queried))

    def test_relative_position_one_is_rejected(self):
        forests = {"alpha2": PersistenceForest(np.random.default_rng(1).random((30, 2))),
                   "cubical2": PersistenceForest.from_cubical_grid(np.random.default_rng(1).random((8, 8)))}
        for name, forest in forests.items():
            for call in (lambda: forest.barcode_cycle_reps(relative_position=1),
                         lambda: forest.plot_barcode_cycle_reps(relative_position=1, show=False)):
                with self.subTest(forest=name), self.assertRaisesRegex(ValueError, r"\[0,1\)"):
                    call()

    def test_bar_one_ulp_long_is_sampled_at_its_birth(self):
        forest = _one_ulp_triangle()
        [bar] = forest.barcode
        self.assertEqual((bar.birth, bar.death), (1.0, ONE_ULP_DEATH))
        # 1.0 + 0.6 * ulp rounds up to the death, outside the half-open lifespan.
        self.assertEqual(bar.birth + bar.lifespan() * 0.6, bar.death)
        for position in (0, 0.4, 0.6, 0.999):
            with self.subTest(position=position):
                self.assertEqual(_value_at_relative_position(bar, position), 1.0)
                [rep] = forest.barcode_cycle_reps(relative_position=position)
                self.assertEqual(rep.signed_simplices, {((0, 1), 1), ((0, 2), -1), ((1, 2), 1)})
                self.assertEqual(rep.interior, {((0, 1, 2), 1)})
                self.assertEqual((rep.active_start, rep.active_end), (1.0, ONE_ULP_DEATH))
        with mock.patch.object(forest, 'cycle_for_bar_at', wraps=forest.cycle_for_bar_at) as query:
            forest.plot_barcode_cycle_reps(relative_position=0.6, show=False)
        self.assertEqual([call.args for call in query.call_args_list], [(bar, 1.0)])

    def test_value_is_none_for_empty_lifespan(self):
        forest = PersistenceForest(np.random.default_rng(0).random((60, 2)), reduce=False)
        zero = next(bar for bar in forest.barcode if bar.birth == bar.death)
        self.assertIsNone(_value_at_relative_position(zero, 0.5))


if __name__ == '__main__':
    unittest.main()
