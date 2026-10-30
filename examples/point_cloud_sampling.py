import numpy as np

def sample_noisy_star(n, spikes=5, amplitude=0.5, noise_std=0.01, seed=None, radius = 0.5):
    """
    Sample noisy points from a smooth star-shaped curve in 2D.

    Parameters:
        n (int): Number of points to sample.
        spikes (int): Number of star spikes.
        amplitude (float): Spike strength; 0 corresponds to a circle.
        noise_std (float): Std. dev. of Gaussian noise.
        seed (int or None): Random seed.
        radius (float): Base radius of the star prior to modulation.

    Returns:
        np.ndarray: Array of shape (n, 2) with noisy (x, y) points.
    """
    if seed is not None:
        np.random.seed(seed)
    
    r0 = radius  # base radius (like circle with diameter 1)
    theta = np.random.uniform(0, 2*np.pi, n)

    # star shape: radial function
    r = r0 * (1 + amplitude * np.cos(spikes * theta))
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # add Gaussian noise
    x_noisy = x + np.random.normal(0, noise_std, n)
    y_noisy = y + np.random.normal(0, noise_std, n)
    
    return np.column_stack((x_noisy, y_noisy))


def sample_noisy_star_pointy(
    n,
    seed=0,
    spikes=5,
    r_outer=1.0,
    r_inner=0.4,
    noise_std=0.02
):
    """
    Sample noisy points along the boundary of a star with sharp tips.

    Parameters
    ----------
    n : int
        Number of points to sample.
    seed : int
        Random seed for reproducibility.
    spikes : int
        Number of arms/spikes of the star.
    r_outer : float
        Radius of the outer tips.
    r_inner : float
        Radius of the inner recesses.
    noise_std : float
        Standard deviation of Gaussian noise added to points.

    Returns
    -------
    points : ndarray, shape (n, 2)
        2D array of sampled star boundary points.
    """

    rng = np.random.default_rng(seed)

    # Angles for outer and inner points
    angles_outer = np.linspace(0, 2*np.pi, spikes, endpoint=False)
    angles_inner = angles_outer + np.pi / spikes

    # Boundary vertices alternating outer and inner points
    angles = np.empty(spikes * 2)
    radii = np.empty(spikes * 2)

    angles[0::2] = angles_outer
    angles[1::2] = angles_inner

    radii[0::2] = r_outer
    radii[1::2] = r_inner

    # Convert to x,y coordinates for the star vertices
    vertices = np.column_stack([radii * np.cos(angles),
                                radii * np.sin(angles)])

    # Compute the lengths of each edge segment for uniform boundary sampling
    diffs = np.diff(np.vstack([vertices, vertices[0]]), axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    cum_lengths = np.cumsum(seg_lengths)
    total_length = cum_lengths[-1]

    # Sample distances along the boundary
    s = rng.uniform(0, total_length, size=n)

    points = np.zeros((n, 2))

    # For each sampled distance, find its segment
    for i, dist in enumerate(s):
        seg_idx = np.searchsorted(cum_lengths, dist)
        if seg_idx == 0:
            prev_length = 0
        else:
            prev_length = cum_lengths[seg_idx - 1]

        t = (dist - prev_length) / seg_lengths[seg_idx]

        v0 = vertices[seg_idx]
        v1 = vertices[(seg_idx + 1) % len(vertices)]

        # Linear interpolation
        points[i] = (1 - t) * v0 + t * v1

    # Add Gaussian noise
    noise = rng.normal(scale=noise_std, size=points.shape)
    points += noise

    return points

def sample_noisy_circle(n, noise_std=0.05, seed=42, radius: float = 1):
    """
    Sample noisy points along a circle of the given radius.
    
    Parameters:
        n (int): Number of points to sample.
        noise_std (float): Standard deviation of Gaussian noise.
        seed (int or None): Random seed for reproducibility.
        radius (float): Circle radius.
    
    Returns:
        np.ndarray: Array of shape (n, 2) with noisy (x, y) points.
    """
    if seed is not None:
        np.random.seed(seed)
    
    r = radius  # radius of circle
    theta = np.random.uniform(0, 2 * np.pi, n)  # random angles
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # add Gaussian noise
    x_noisy = x + np.random.normal(0, noise_std, n)
    y_noisy = y + np.random.normal(0, noise_std, n)
    
    return np.column_stack((x_noisy, y_noisy))

def sample_noisy_ellipse(
    n: int,
    a: float = 1.5,
    b: float = 1.0,
    angle: float = 0.0,
    noise_std: float = 0.01,
    seed: int | None = 42,
    center: tuple[float, float] = (0.0, 0.0),
):
    """
    Sample n noisy points on the perimeter of an ellipse.

    The ellipse is defined by semi-axes a (x-axis) and b (y-axis), optionally
    rotated by 'angle' radians about its center, then perturbed by Gaussian noise.

    Parameters:
        n (int): Number of points to sample.
        a (float): Semi-major axis length along x before rotation.
        b (float): Semi-minor axis length along y before rotation.
        angle (float): Rotation of the ellipse in radians (counterclockwise).
        noise_std (float): Std. dev. of isotropic Gaussian noise added to (x, y).
        seed (int or None): Random seed for reproducibility.
        center (tuple): (cx, cy) center of the ellipse.

    Returns:
        np.ndarray: Array of shape (n, 2) with noisy (x, y) points.
    """
    if seed is not None:
        np.random.seed(seed)

    # Parameterize ellipse: (a cos t, b sin t), t uniform in [0, 2π)
    # (Note: uniform in parameter, not exactly uniform in arclength.)
    t = np.random.uniform(0, 2 * np.pi, n)
    x0 = a * np.cos(t)
    y0 = b * np.sin(t)

    # Rotate by 'angle'
    ca, sa = np.cos(angle), np.sin(angle)
    x = ca * x0 - sa * y0
    y = sa * x0 + ca * y0

    # Translate to center
    x += center[0]
    y += center[1]

    # Add Gaussian noise
    x_noisy = x + np.random.normal(0, noise_std, n)
    y_noisy = y + np.random.normal(0, noise_std, n)

    return np.column_stack((x_noisy, y_noisy))

def sample_noisy_circle_with_tendril(
    n: int,
    noise_std: float = 0.01,
    seed: int | None = 42,
    radius: float = 1.0,
    tendril_length: float = 0.7,
    tendril_fraction: float = 0.15,
    tendril_angle: float = 0.0,
    tendril_width_deg: float = 8.0,
    radial_jitter: float = 0.02,
):
    """
    Sample n noisy points on a circle plus a thin inward tendril.

    The base shape is a circle of given radius. A fraction of points are drawn
    along a narrow angular sector ('tendril') that extends radially inward from
    the circle toward the center.

    Parameters:
        n (int): Number of points to sample.
        noise_std (float): Std. dev. of isotropic Gaussian noise added to (x, y).
        seed (int or None): Random seed for reproducibility.
        radius (float): Circle radius.
        tendril_length (float): How far inward the tendril reaches (in radius units).
                                Effective radius along tendril goes from r to r - tendril_length.
        tendril_fraction (float): Fraction of points to allocate to the tendril (0–1).
        tendril_angle (float): Central angle (radians) where the tendril is located.
        tendril_width_deg (float): Angular spread (degrees) around tendril_angle.
        radial_jitter (float): Extra radial noise (std dev) applied to tendril points only.

    Returns:
        np.ndarray: Array of shape (n, 2) with noisy (x, y) points.
    """
    if seed is not None:
        np.random.seed(seed)

    # Split points between circle and tendril
    n_tendril = int(round(tendril_fraction * n))
    n_circle = n - n_tendril

    # --- Circle perimeter points ---
    theta_circle = np.random.uniform(0, 2 * np.pi, n_circle)
    x_c = radius * np.cos(theta_circle)
    y_c = radius * np.sin(theta_circle)

    # --- Tendril points ---
    # Narrow angular sector around tendril_angle
    width_rad = np.deg2rad(tendril_width_deg)
    theta_t = np.random.normal(loc=tendril_angle, scale=width_rad / 2.355, size=n_tendril)
    # (Use FWHM ≈ 2.355*σ so 'tendril_width_deg' roughly matches visible width.)

    # Radii move inward from r to r - tendril_length (thin path)
    t = np.random.uniform(0.0, 1.0, n_tendril)
    r_t = radius - tendril_length * t + np.random.normal(0, radial_jitter, n_tendril)
    r_t = np.clip(r_t, 0.0, radius)

    x_t = r_t * np.cos(theta_t)
    y_t = r_t * np.sin(theta_t)

    # Combine
    x = np.concatenate([x_c, x_t])
    y = np.concatenate([y_c, y_t])

    # Add overall Gaussian positional noise
    x_noisy = x + np.random.normal(0, noise_std, n)
    y_noisy = y + np.random.normal(0, noise_std, n)

    # Shuffle so tendril/circle points are mixed
    pts = np.column_stack((x_noisy, y_noisy))
    idx = np.random.permutation(n)
    return pts[idx]

def sample_uniform_points(n, dim, seed=15):
    """
    Sample n uniformly distributed points in a unit hypercube.

    Parameters
    ----------
    n : int
        Number of points to sample.
    dim : int
        Dimensionality of each point.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        An array of shape (n, dim) with uniformly sampled points in [0, 1]^dim.
    """
    rng = np.random.default_rng(seed)
    return rng.random((n, dim))

def sample_points_without_balls(
    n,
    dim,
    num_discs =30,
    radius_range = [0,0.05],
    seed=13,
    max_trials_factor=1000,
):
    """
    Sample points in [0,1]^dim while avoiding randomly placed balls.

    Parameters
    ----------
    n : int
        Number of points to sample (outside the balls).
    dim : int
        Dimension of the space (unit hypercube [0,1]^dim).
    num_discs : int
        Number of balls (hyperspheres) to sample.
    radius_range : tuple or list of length 2
        (a, b) range for radii. Each radius is sampled uniformly in [a, b].
    seed : int, optional
        Random seed for reproducibility.
    max_trials_factor : int, optional
        Safety factor to cap total number of candidate samples.
        Roughly, at most max_trials_factor * n candidate points will be drawn
        before giving up.

    Returns
    -------
    points : np.ndarray, shape (n, dim)
        Uniform points in [0,1]^dim excluding the sampled balls.

    Raises
    ------
    RuntimeError
        If not enough points can be sampled within the allowed number of trials.
    """
    rng = np.random.default_rng(seed)
    a, b = radius_range

    if not (0 <= a <= b):
        raise ValueError("radius_range must satisfy 0 <= a <= b.")

    # --- Sample balls (centers + radii) ---
    radii = rng.uniform(a, b, size=num_discs)

    if np.any(radii > 0.5):
        raise ValueError("Some radii are > 0.5, balls cannot fit inside [0,1]^d.")

    # Ensure balls are fully contained in [0,1]^d:
    # each center coordinate is in [r, 1 - r]
    centers = rng.uniform(
        low=radii[:, None],
        high=1.0 - radii[:, None],
        size=(num_discs, dim),
    )

    # --- Rejection sampling for points outside balls ---
    points = []
    total_trials = 0
    max_trials = max_trials_factor * n

    while len(points) < n and total_trials < max_trials:
        remaining = n - len(points)
        # Draw in reasonably large batches for efficiency
        batch_size = min(100000, max(5000, 5 * remaining))
        candidates = rng.random((batch_size, dim))  # uniform in [0,1]^d

        if num_discs > 0:
            # Compute squared distances from each candidate to each center
            diff = candidates[:, None, :] - centers[None, :, :]  # (batch, num_discs, d)
            dist2 = np.sum(diff * diff, axis=-1)                  # (batch, num_discs)
            inside_any = (dist2 <= radii[None, :] ** 2).any(axis=1)
            valid = candidates[~inside_any]
        else:
            valid = candidates  # no balls: all points are valid

        if valid.size > 0:
            points.append(valid[:remaining])

        total_trials += batch_size

    if len(points) == 0:
        raise RuntimeError(
            "Failed to sample any valid points. "
            "Balls may cover almost all of [0,1]^d or parameters are inconsistent."
        )

    points = np.concatenate(points, axis=0)
    if points.shape[0] < n:
        raise RuntimeError(
            f"Only sampled {points.shape[0]} valid points out of requested {n}. "
            "Try reducing num_discs, the radii, or increasing max_trials_factor."
        )

    return points[:n]

def sample_unit_sphere(n: int, dim: int, seed: int | None = 23) -> np.ndarray:
    """
    Sample n points uniformly from the unit sphere S^{dim-1} in R^dim.

    Parameters
    ----------
    n : int
        Number of points to sample.
    dim : int
        Dimension of the ambient space R^dim.
    seed : int or None
        Seed for reproducibility.

    Returns
    -------
    points : (n, dim) np.ndarray
        Array of n points on the unit sphere.
    """
    rng = np.random.default_rng(seed)

    x = rng.normal(size=(n, dim))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid divide-by-zero just in case
    return x / norms

def sample_noisy_sphere(
    n: int, 
    dim: int, 
    noise_std: float, 
    seed: int | None = None
) -> np.ndarray:
    """
    Sample n points from the dim-dimensional unit sphere and 
    perturb them with Gaussian noise N(0, noise_std^2 I).

    Parameters
    ----------
    n : int
        Number of points to sample.
    dim : int
        Dimension of the ambient space R^dim.
    noise_std : float
        Standard deviation of Gaussian noise.
    seed : int or None
        Seed for reproducibility.

    Returns
    -------
    noisy_points : (n, dim) np.ndarray
        Noisy points.
    """
    rng = np.random.default_rng(seed)

    # Sample clean points
    sphere_points = sample_unit_sphere(n, dim, seed)

    # Add Gaussian noise
    noise = rng.normal(0.0, noise_std, size=(n, dim))
    return sphere_points + noise


# Point clouds for measurement-landscape construction figures.


def _resample_polyline(
    vertices: np.ndarray,
    n_points: int,
    *,
    closed: bool,
) -> np.ndarray:
    """Resample a polyline at approximately uniform arclength spacing."""
    vertices = np.asarray(vertices, dtype=float)
    path = np.vstack((vertices, vertices[0])) if closed else vertices
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    distances = np.linspace(
        0.0,
        cumulative[-1],
        n_points,
        endpoint=not closed,
    )
    segment_indices = np.searchsorted(cumulative, distances, side="right") - 1
    segment_indices = np.minimum(segment_indices, len(segments) - 1)
    fractions = (
        (distances - cumulative[segment_indices]) / lengths[segment_indices]
    )
    return (
        path[segment_indices]
        + fractions[:, None] * segments[segment_indices]
    )


def _ellipse(
    center: tuple[float, float],
    semi_axes: tuple[float, float],
    n_points: int,
    *,
    angle: float = 0.0,
) -> np.ndarray:
    parameter = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
    points = np.column_stack(
        (semi_axes[0] * np.cos(parameter), semi_axes[1] * np.sin(parameter))
    )
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    points = points @ rotation.T + np.asarray(center)
    return _resample_polyline(points, n_points, closed=True)


def _superellipse(
    center: tuple[float, float],
    semi_axes: tuple[float, float],
    n_points: int,
    *,
    exponent: float,
    angle: float = 0.0,
) -> np.ndarray:
    parameter = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
    cosine = np.cos(parameter)
    sine = np.sin(parameter)
    power = 2.0 / exponent
    points = np.column_stack(
        (
            semi_axes[0] * np.sign(cosine) * np.abs(cosine) ** power,
            semi_axes[1] * np.sign(sine) * np.abs(sine) ** power,
        )
    )
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    points = points @ rotation.T + np.asarray(center)
    return _resample_polyline(points, n_points, closed=True)


def _pointy_star(
    center: tuple[float, float],
    outer_radius: float,
    inner_radius: float,
    n_points: int,
    *,
    spikes: int = 5,
    angle: float = 0.0,
) -> np.ndarray:
    vertex_angles = angle + np.arange(2 * spikes) * np.pi / spikes
    radii = np.empty(2 * spikes)
    radii[0::2] = outer_radius
    radii[1::2] = inner_radius
    vertices = np.column_stack(
        (radii * np.cos(vertex_angles), radii * np.sin(vertex_angles))
    )
    vertices += np.asarray(center)
    return _resample_polyline(vertices, n_points, closed=True)


def _add_noise(
    points: np.ndarray,
    noise: float,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return points + rng.normal(scale=noise, size=points.shape)


def sample_mixed_loops(*, noise: float = 0.004, seed: int = 11) -> np.ndarray:
    """Sample a separated circle, elongated ellipse, and pointy star."""
    circle = _ellipse((-1.0, -1.0), (0.55, 0.55), 52)
    circle = np.delete(
        circle,
        [1, 2],
        axis=0,
    )  # A controlled gap delays this birth.
    ellipse = _ellipse((0.0, 0.0), (0.82, 0.36), 58, angle=-0.25 * np.pi)
    star = _pointy_star(
        (1.0, 1.0),
        outer_radius=0.76,
        inner_radius=0.33,
        n_points=82,
        angle=-0.5 * np.pi,
    )
    return _add_noise(np.vstack((circle, ellipse, star)), noise, seed=seed)


def sample_compartmented_oval(
    *,
    noise: float = 0.003,
    seed: int = 12,
) -> np.ndarray:
    """Sample an oval split into three asymmetric compartments."""
    x_dense = np.linspace(-1.4, 1.4, 2048)
    normalized_x = x_dense / 1.4
    envelope = np.sqrt(np.maximum(0.0, 1.0 - normalized_x**2))

    top = _resample_polyline(
        np.column_stack((x_dense, 0.92 * envelope)),
        70,
        closed=False,
    )
    upper = _resample_polyline(
        np.column_stack(
            (
                x_dense,
                (1.0 - normalized_x**2)
                * (0.33 + 0.07 * np.sin(3.0 * np.pi * normalized_x)),
            )
        ),
        44,
        closed=False,
    )
    upper = np.delete(upper, [21, 22], axis=0)
    lower = _resample_polyline(
        np.column_stack(
            (
                x_dense,
                (1.0 - normalized_x**2)
                * (-0.20 - 0.10 * np.sin(2.0 * np.pi * normalized_x)),
            )
        ),
        40,
        closed=False,
    )
    bottom = _resample_polyline(
        np.column_stack((x_dense, -0.92 * envelope)),
        72,
        closed=False,
    )

    # The four paths share their two endpoints.
    points = np.vstack((top, upper[1:-1], lower[1:-1], bottom[1:-1]))
    return _add_noise(points, noise, seed=seed)


def sample_tendril_triplet(
    *,
    noise: float = 0.003,
    seed: int = 13,
) -> np.ndarray:
    """Sample a circle, pointy star, and circle with an inward tendril."""
    circle = _ellipse((-0.95, -0.95), (0.48, 0.48), 52)
    star = _pointy_star(
        (0.0, 0.0),
        outer_radius=0.67,
        inner_radius=0.29,
        n_points=78,
        angle=0.5 * np.pi,
    )
    tendril_loop = _ellipse((0.95, 0.95), (0.56, 0.56), 62)
    tendril = np.column_stack(
        (
            np.full(14, 0.95),
            np.linspace(0.39, 0.83, 14),
        )
    )
    points = np.vstack((circle, star, tendril_loop, tendril[1:]))
    return _add_noise(points, noise, seed=seed)


def sample_three_cavities(*, n_points: int = 1400, seed: int = 14) -> np.ndarray:
    """Sample a filled rectangle with circular, elliptical, and star cavities."""
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    while sum(len(batch) for batch in accepted) < n_points:
        points = rng.uniform((-1.65, -1.25), (1.65, 1.25), size=(n_points, 2))

        circle_hole = np.linalg.norm(points - (-0.85, 0.45), axis=1) < 0.36

        ellipse_points = points - (0.78, 0.45)
        cosine, sine = np.cos(0.25), np.sin(0.25)
        ellipse_x = cosine * ellipse_points[:, 0] + sine * ellipse_points[:, 1]
        ellipse_y = -sine * ellipse_points[:, 0] + cosine * ellipse_points[:, 1]
        ellipse_hole = (ellipse_x / 0.58) ** 2 + (ellipse_y / 0.27) ** 2 < 1.0

        star_points = points - (0.0, -0.58)
        angles = np.arctan2(star_points[:, 1], star_points[:, 0])
        star_radius = 0.31 * (1.0 + 0.42 * np.cos(5.0 * angles))
        star_hole = np.linalg.norm(star_points, axis=1) < star_radius

        accepted.append(points[~(circle_hole | ellipse_hole | star_hole)])

    return np.vstack(accepted)[:n_points]


def sample_circle_star_pair(
    *,
    noise: float = 0.003,
    seed: int = 15,
) -> np.ndarray:
    """Sample an overlapping circle and pointy star with three cavities."""
    separation = 0.74
    circle = _ellipse((-separation / 2.0, 0.0), (0.68, 0.68), 80)
    circle = np.delete(circle, [40, 41, 42, 43], axis=0)
    star = _pointy_star(
        (separation / 2.0, 0.0),
        outer_radius=0.80,
        inner_radius=0.34,
        n_points=60,
        angle=np.pi,
    )
    star = np.delete(star, [10, 11], axis=0)
    return _add_noise(np.vstack((circle, star)), noise, seed=seed)


def sample_circle_rounded_rectangle_intersection(
    *,
    noise: float = 0.003,
    seed: int = 17,
) -> np.ndarray:
    """Sample an intersecting circle and tilted rounded rectangle."""
    separation = 0.75
    circle = _ellipse((-separation / 2.0, 0.0), (0.68, 0.68), 76)
    rounded_rectangle = _superellipse(
        (separation / 2.0, 0.0),
        (0.92, 0.40),
        48,
        exponent=4.0,
        angle=0.30,
    )
    rounded_rectangle = np.delete(rounded_rectangle, [20, 21], axis=0)
    return _add_noise(
        np.vstack((circle, rounded_rectangle)),
        noise,
        seed=seed,
    )


def sample_mixed_bouquet(
    *,
    noise: float = 0.012,
    seed: int = 16,
) -> np.ndarray:
    """Sample three irregular, differently shaped petals at one junction."""

    def petal(
        direction: float,
        length: float,
        width: float,
        n_points: int,
        modulation: float,
    ) -> np.ndarray:
        parameter = np.linspace(0.0, 1.0, 2048)
        longitudinal = length * np.sin(np.pi * parameter)
        transverse = (
            width
            * np.sin(2.0 * np.pi * parameter)
            * (1.0 + modulation * np.cos(6.0 * np.pi * parameter))
        )
        tangent = np.array((np.cos(direction), np.sin(direction)))
        normal = np.array((-np.sin(direction), np.cos(direction)))
        vertices = (
            longitudinal[:, None] * tangent
            + transverse[:, None] * normal
        )
        return _resample_polyline(vertices, n_points + 1, closed=False)

    rounded = petal(np.pi, 1.35, 0.58, 72, 0.0)
    rounded = np.delete(rounded, [20, 21, 22], axis=0)
    elongated = petal(-np.pi / 3.0, 1.55, 0.26, 42, 0.05)
    pointed = petal(np.pi / 3.0, 0.95, 0.375, 84, 0.65)

    # Keep one shared junction; the rounded-petal gap makes its bar born last.
    points = np.vstack((rounded[:-1], elongated[1:-1], pointed[1:-1]))
    return _add_noise(points, noise, seed=seed)
