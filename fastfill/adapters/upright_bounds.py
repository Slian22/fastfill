"""Yaw-frame bounds of a centered source box (Scan2CAD and SpatialGen only)."""
import math


def finite_vector(values, length):
    if not isinstance(values, (list, tuple)) or len(values) != length:
        return False
    try:
        return all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) for v in values)
    except OverflowError:
        return False


def yaw_extents(size, linear, yaw_deg):
    """Return abs(Q(yaw).T @ linear) @ size, or None for unusable geometry.

    `size` is the full local box size; `linear` maps its axes into world space.
    Scale must occur exactly once between these two inputs. Reflections and
    shear are retained. The output frame uses the exported yaw, BEFORE the
    adapters' semantic-front quarter turn in convert_room.
    """
    if not finite_vector(size, 3) or min(size) <= 0 or not finite_vector([yaw_deg], 1):
        return None
    if not (isinstance(linear, (list, tuple)) and len(linear) == 3
            and all(finite_vector(row, 3) for row in linear)):
        return None
    norms = [math.hypot(*(linear[i][j] for i in range(3))) for j in range(3)]
    if not finite_vector(norms, 3) or min(norms) == 0:
        return None
    a, b, c = [[linear[i][j] / norms[j] for j in range(3)] for i in range(3)]
    determinant = (a[0] * (b[1] * c[2] - b[2] * c[1])
                   - a[1] * (b[0] * c[2] - b[2] * c[0])
                   + a[2] * (b[0] * c[1] - b[1] * c[0]))
    if abs(determinant) <= 1e-12:
        return None
    yaw = math.radians(yaw_deg)
    co, si = math.cos(yaw), math.sin(yaw)
    projected = [[co * linear[0][j] + si * linear[1][j] for j in range(3)],
                 [-si * linear[0][j] + co * linear[1][j] for j in range(3)],
                 linear[2]]
    extents = [sum(abs(v) * s for v, s in zip(row, size)) for row in projected]
    return extents if finite_vector(extents, 3) and min(extents) > 0 else None
