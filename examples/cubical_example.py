"""Persistence forest of a 2D image: two rings give two long H1 bars."""
import numpy as np

from persforest import PersistenceForest
from persforest.measurements.cubical import cubical_chain_interior_volume, cubical_chain_volume

# Pixel values: distance to the nearer of two circles, plus seeded noise.
# Low values on the circles enter first and close two loops; the enclosed
# pixels enter later and fill them.
rng = np.random.default_rng(0)
y, x = np.mgrid[0:120, 0:160] * 0.05
rings = [((2.0, 2.5), 1.2), ((5.0, 3.5), 0.8)]
image = np.min([np.abs(np.hypot(x - cx, y - cy) - r) for (cx, cy), r in rings], axis=0)
image += 0.02 * rng.random(image.shape)

# Values on pixels (GUDHI's top-cell convention); 0.05 length units per pixel.
forest = PersistenceForest.from_cubical_grid(image, spacing=0.05)
print(f"{len(forest.barcode)} bars from {image.size} pixels "
      f"({forest.filtration_type}, storage={forest.storage!r}).")

for bar in forest.longest_bars(2):
    # Representative at birth: the closed ring, its length and enclosed area.
    rep = forest.cycle_for_bar_at(bar, bar.birth)
    print(f"bar [{bar.birth:.3f}, {bar.death:.3f}): boundary length "
          f"{cubical_chain_volume(rep.unsigned(), forest.geometry):.2f}, "
          f"enclosed area {cubical_chain_interior_volume(rep, forest.geometry):.2f}")

family = forest.compute_measurement_landscapes(
    cubical_chain_interior_volume, "enclosed area", max_k=2, x_grid=np.linspace(0, 1.3, 200))
print(f"Enclosed-area landscape array shape: {family.evaluate_on_grid().shape}")

forest.plot_barcode(min_bar_length=0.05)
