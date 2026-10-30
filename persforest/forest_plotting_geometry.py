"""Display-space geometry for persistence forests."""
from dataclasses import dataclass
import numpy as np
from matplotlib.path import Path


def _split(points, t):
    a = points[:-1] * (1-t) + points[1:] * t
    b = a[:-1] * (1-t) + a[1:] * t
    c = b[0] * (1-t) + b[1] * t
    return np.array([points[0], a[0], b[0], c]), np.array([c, b[1], a[2], points[-1]])


@dataclass
class ForestRoute:
    """Cubic transition followed, if necessary, by a vertical continuation."""
    controls: np.ndarray
    end: np.ndarray
    curved: bool = True

    def __getitem__(self, i):
        return (self.controls[0, 0], self.end[0], self.controls[0, 1],
                self.end[1], self.controls[-1, 1], float(self.curved))[i]

    def parameter(self, y):
        p = self.controls[:, 1]
        if y <= p[0]:
            return 0.0
        if y >= p[-1]:
            return 1.0
        target = float((y-p[0])/(p[-1]-p[0]))
        if not self.curved:
            return target
        c1, c2 = (p[1:3]-p[0])/(p[-1]-p[0])
        a, b, c = float(1+3*c1-3*c2), float(3*c2-6*c1), float(3*c1)
        low, high, t = 0., 1., target
        for _ in range(40):
            error = ((a*t+b)*t+c)*t-target
            if abs(error) <= 2e-14:
                return t
            if error < 0:
                low = t
            else:
                high = t
            derivative = (3*a*t+2*b)*t+c
            candidate = t-error/derivative if derivative > 1e-15 else -1.
            t = candidate if low < candidate < high else (low+high)/2
        return t

    def at(self, y):
        if y >= self[4]:
            return self[1], 0.
        if not hasattr(self, '_at_cache'):
            self._at_cache = {}
        if y in self._at_cache:
            return self._at_cache[y]
        t = self.parameter(y)
        p = self.controls
        value = (1-t)**3*p[0] + 3*(1-t)**2*t*p[1] + 3*(1-t)*t*t*p[2] + t**3*p[3]
        derivative = 3*((1-t)**2*(p[1]-p[0]) + 2*(1-t)*t*(p[2]-p[1]) + t*t*(p[3]-p[2]))
        slope = derivative[0]/derivative[1] if derivative[1] else np.inf
        result = float(value[0]), float(slope)
        if len(self._at_cache) < 256:
            self._at_cache[y] = result
        return result

    def section(self, low, high):
        """Exact de Casteljau restriction; markers and color cuts share geometry."""
        start, end = self.parameter(low), self.parameter(high)
        left, _ = _split(self.controls, end)
        _, middle = _split(left, start/end if end else 0.)
        return middle

    def path(self, low, high):
        start = (self.at(low)[0], low)
        vertices, codes = [start], [Path.MOVETO]
        if low < self[4] and high > low:
            controls = self.section(low, min(high, self[4]))
            if self.curved:
                vertices.extend(controls[1:])
                codes.extend([Path.CURVE4]*3)
            else:
                vertices.append(controls[-1])
                codes.append(Path.LINETO)
        if high > self[4] or high == low:
            vertices.append((self[1], high))
            codes.append(Path.LINETO)
        return Path(vertices, codes)

    def transform(self, scale, origin):
        return ForestRoute(self.controls / scale + origin,
                           self.end / scale + origin, self.curved)

    def polyline(self, tolerance=.15):
        """Flatten adaptively in points, for the visual clearance objective."""
        if hasattr(self, '_polyline_cache'):
            return self._polyline_cache
        result = [self.controls[0]]
        stack = [self.controls]
        while stack:
            p = stack.pop()
            chord = p[-1] - p[0]
            length = np.linalg.norm(chord)
            error = (max(abs(chord[0]*(q[1]-p[0, 1])-chord[1]*(q[0]-p[0, 0])) for q in p[1:-1]) / length
                     if length else np.max(np.linalg.norm(p-p[0], axis=1)))
            if error <= tolerance:
                result.append(p[-1])
            else:
                left, right = _split(p, .5)
                stack.extend([right, left])
        if self.end[1] > self[4]:
            result.append(self.end)
        self._polyline_cache = np.asarray(result)
        return self._polyline_cache


def route_x(route, y):
    return route.at(y)


def routes_intersect(left, right, shared_heights=()):
    """Conservative monotone-curve intersection test, without flattening.

    Subdivide overlapping height intervals until their exact x ranges separate.
    Ambiguous tangencies count as intersections, except at shared graph nodes.
    A numerical junction tolerance is necessary for coincident endpoint limits.
    """
    if (min(left[0], left[1]) > max(right[0], right[1]) or
            min(right[0], right[1]) > max(left[0], left[1])):
        return False
    low, high = max(left[2], right[2]), min(left[3], right[3])
    if high < low:
        return False
    span = max(abs(left[1]-left[0]), abs(right[1]-right[0]), high-low, 1.)
    eps = 1e-11 * span
    near_shared = lambda y: any(abs(y-h) <= 1e-6*span for h in shared_heights)
    if high == low:
        l = sorted((left[0], left[1])) if left[2] == left[3] else [left.at(low)[0]]*2
        r = sorted((right[0], right[1])) if right[2] == right[3] else [right.at(low)[0]]*2
        overlap = min(l[1], r[1])-max(l[0], r[0])
        return overlap > eps or (overlap >= -eps and not near_shared(low))
    cache_l, cache_r = {}, {}

    def points(y):
        if y not in cache_l:
            cache_l[y], cache_r[y] = left.at(y)[0], right.at(y)[0]
        return cache_l[y], cache_r[y]

    stack = [(low, high, 0)]
    while stack:
        a, b, depth = stack.pop()
        la, ra = points(a)
        lb, rb = points(b)
        da, db = la-ra, lb-rb
        if min(la, lb) > max(ra, rb) + eps or min(ra, rb) > max(la, lb) + eps:
            continue
        if da*db < 0 and abs(da) > eps and abs(db) > eps:
            return True
        if (abs(da) <= eps and not near_shared(a) or
                abs(db) <= eps and not near_shared(b)):
            return True
        if depth >= 26 or b-a <= eps:
            if not near_shared(a) and not near_shared(b):
                return True
            continue
        middle = (a+b)/2
        stack.extend([(a, middle, depth+1), (middle, b, depth+1)])
    return False


def _segment_distances(a, b, c, d):
    """All segment-pair distances, including intersections, in screen points."""
    u, v = b-a, d-c
    w = a[:, None, :] - c[None, :, :]
    cross = lambda x, y: x[..., 0]*y[..., 1]-x[..., 1]*y[..., 0]
    den = cross(u[:, None], v[None, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        t = cross(-w, v[None, :])/den
        s = cross(-w, u[:, None])/den
    hit = (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)

    def point_segment(p, q, r):
        delta = r-q
        denom = np.sum(delta*delta, axis=-1)
        fraction = np.clip(np.sum((p-q)*delta, axis=-1)/np.maximum(denom, 1e-30), 0, 1)
        return np.linalg.norm(p-q-fraction[..., None]*delta, axis=-1)

    distances = np.minimum.reduce([
        point_segment(a[:, None], c[None, :], d[None, :]),
        point_segment(b[:, None], c[None, :], d[None, :]),
        point_segment(c[None, :], a[:, None], b[:, None]),
        point_segment(d[None, :], a[:, None], b[:, None]),
    ])
    return np.where(hit, 0., distances)


def route_clearance(left, right, shared_points=(), junction_radius=6.):
    """Approximate Euclidean separation, excluding disks around shared nodes."""
    if left[0] == left[1] and right[0] == right[1] and not shared_points:
        vertical_gap = max(0., left[2]-right[3], right[2]-left[3])
        return float(np.hypot(left[0]-right[0], vertical_gap))
    polylines = []
    for route in (left, right):
        p = route.polyline()
        if not shared_points:
            polylines.append((p[:-1], p[1:]))
            continue
        clipped = []
        for a, b in zip(p, p[1:]):
            delta = b-a
            cuts = [0., 1.]
            length2 = float(delta @ delta)
            for shared in shared_points:
                offset = a-shared
                linear = 2*float(offset @ delta)
                constant = float(offset @ offset)-junction_radius**2
                discriminant = linear*linear-4*length2*constant
                if length2 > 0 and discriminant >= 0:
                    for t in ((-linear-np.sqrt(discriminant))/(2*length2),
                              (-linear+np.sqrt(discriminant))/(2*length2)):
                        if 0 < t < 1:
                            cuts.append(t)
            cuts.sort()
            for low, high in zip(cuts, cuts[1:]):
                midpoint = a + .5*(low+high)*delta
                if all(np.linalg.norm(midpoint-q) >= junction_radius for q in shared_points):
                    clipped.append((a+low*delta, a+high*delta))
        clipped = np.asarray(clipped).reshape(-1, 2, 2)
        polylines.append((clipped[:, 0], clipped[:, 1]))
    (a, b), (c, d) = polylines
    if not len(a) or not len(c):
        return np.inf
    # Bound temporary memory for long nearly vertical branches.
    minimum = np.inf
    for start in range(0, len(a), 128):
        minimum = min(minimum, float(_segment_distances(a[start:start+128], b[start:start+128], c, d).min()))
    return minimum


def _make_route(start, end, turn, style, angle, curvature, handle_limit=np.inf):
    dx, dy = abs(end[0]-start[0]), turn-start[1]
    if style == 'curved' and dx > 0 and dy > 0:
        radians = np.deg2rad(angle)
        sn, cs = np.sin(radians), max(0., np.cos(radians))
        strength = .1 + .3*curvature
        handle = strength*min(dx/sn, dy/max(cs, 1e-15), dy, handle_limit)
        p1 = start + np.array([np.sign(end[0]-start[0])*handle*sn, handle*cs])
        p2 = np.array([end[0], turn-strength*dy])
        controls = np.array([start, p1, p2, [end[0], turn]])
        return ForestRoute(controls, end, True)
    target = np.array([end[0], turn])
    return ForestRoute(np.array([start, start+(target-start)/3,
                                start+2*(target-start)/3, target]), end, False)


def edge_routes(paths, x, values, style, curvature, angle, scales, min_clearance, linewidth):
    """Mandatory intersection checks; clearance is a best-effort screen objective.

    Work in physical points. Structural endpoints may be too close to attain
    the requested gap without changing the filtration or enlarging the figure.
    Report those deficits instead of claiming an impossible spacing guarantee.
    """
    if not paths:
        return [], (), ()
    origin = np.array([min(x.values()), min(values.values())])
    scales = np.asarray(scales)
    starts = [(np.array([x[p[0]], values[p[0]]])-origin)*scales for p in paths]
    ends = [(np.array([x[p[-1]], values[p[-1]]])-origin)*scales for p in paths]
    clearance, handles = {}, {}
    for p, a, b in zip(paths, starts, ends):
        clearance[p[0]] = min(clearance.get(p[0], np.inf), b[1]-a[1])
        if b[0] != a[0]:
            handles[p[0]] = min(handles.get(p[0], np.inf), abs(b[0]-a[0]))
    routes = [_make_route(a, b, a[1]+.5*clearance[p[0]], style, angle, curvature,
                          handles.get(p[0], np.inf)) for p, a, b in zip(paths, starts, ends)]
    bounds = np.array([(min(a[0], b[0]), max(a[0], b[0]), a[1], b[1]) for a, b in zip(starts, ends)])
    target_gap = min_clearance + linewidth
    junction_radius = max(6., 3*target_gap)
    neighbors = []
    for i, p in enumerate(paths):
        box = bounds[i]
        nearby = ((bounds[:, 0] <= box[1]+target_gap) &
            (bounds[:, 1] >= box[0]-target_gap) & (bounds[:, 2] <= box[3]+target_gap) &
            (bounds[:, 3] >= box[2]-target_gap))
        # Consecutive pieces of a straight trunk are one visual stroke, not
        # separate branches to push apart (important for uncompressed chains).
        if box[0] == box[1]:
            same_trunk = ((bounds[:, 0] == box[0]) & (bounds[:, 1] == box[1]) &
                          ((bounds[:, 3] <= box[2]) | (bounds[:, 2] >= box[3])))
            nearby &= ~same_trunk
        candidates = np.flatnonzero(nearby)
        pairs = []
        for j in candidates:
            if j == i:
                continue
            shared = {p[0], p[-1]}.intersection((paths[j][0], paths[j][-1]))
            points = tuple((np.array([x[n], values[n]])-origin)*scales for n in shared)
            junctions = list(points)
            # Nearby attachments on the same trunk form a junction region,
            # even if intermediate critical nodes separate them in the graph.
            # Exempt this unavoidable convergence only from the spacing target.
            for a in (starts[i], ends[i]):
                for b in (starts[j], ends[j]):
                    if a[0] == b[0] and abs(a[1]-b[1]) < junction_radius:
                        junctions.extend((a, b))
            pairs.append((j, points, tuple(junctions)))
        neighbors.append(pairs)

    gap_cache = {}
    def pair_gap(candidate, other, junctions):
        key = (id(candidate), id(other))
        # Keep object references with the entry: Python may otherwise reuse
        # the id of a rejected temporary candidate.
        cached = gap_cache.get(key)
        if cached is not None and cached[0] is candidate and cached[1] is other:
            return cached[2]
        gap = route_clearance(candidate, other, junctions, junction_radius)
        gap_cache[key] = (candidate, other, gap)
        return gap

    def score(i, candidate, intersections_only=False):
        gap = np.inf
        for j, shared, junctions in neighbors[i]:
            other = routes[j]
            if routes_intersect(candidate, other, tuple(p[1] for p in shared)):
                return -1.
            if not intersections_only:
                gap = min(gap, pair_gap(candidate, other, junctions))
        return min(target_gap, gap)

    # A conservative corridor is planar for strictly increasing events. Never
    # draw a remaining intersection, including degenerate horizontal overlaps.
    # When curves crowd inside the initial corridor, a linear corridor is safe.
    for i in range(len(paths)):
        if score(i, routes[i], intersections_only=True) < 0:
            routes[i] = _make_route(starts[i], ends[i], routes[i][4], 'routed', angle, curvature)
    order = sorted(range(len(paths)), key=lambda i: (-starts[i][1], abs(ends[i][0]-starts[i][0]), i))
    if style != 'routed':
        for i in order:
            if starts[i][0] == ends[i][0] or starts[i][1] == ends[i][1]:
                continue
            baseline = routes[i]
            best_score = score(i, baseline)
            # Prefer full-height curves, then successively earlier transitions.
            # Every candidate must pass intersection checks, regardless of gap.
            low = baseline[4]
            for fraction in (1., .8, .6, .4, .2, 0.):
                turn = low + fraction*(ends[i][1]-low)
                trial = _make_route(starts[i], ends[i], turn, style, angle, curvature)
                candidate_score = score(i, trial)
                if candidate_score >= 0 and candidate_score >= best_score - .05:
                    routes[i], best_score = trial, candidate_score
                    if best_score >= target_gap:
                        break
    deficits = []
    for i, pairs in enumerate(neighbors):
        for j, shared, junctions in pairs:
            if j <= i:
                continue
            if routes_intersect(routes[i], routes[j], tuple(p[1] for p in shared)):
                raise ValueError(f'Cannot draw paths {paths[i]} and {paths[j]} without intersections. '
                                 'Reduce equal-filtration nodes or filter the affected branches.')
            gap = pair_gap(routes[i], routes[j], junctions)
            if gap + .1 < target_gap:
                deficits.append((i, j, max(0., gap-linewidth)))
    adjusted = tuple(paths[i] for i, route in enumerate(routes)
                     if route[0] != route[1] and route[4] < route[3])
    return [r.transform(scales, origin) for r in routes], adjusted, tuple(deficits)
