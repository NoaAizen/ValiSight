"""Uncertainty-preserving projection of a radar point into the thermal plane.

A radar point projected into the thermal image must carry an uncertainty
ellipse, not collapse to a single pixel. On the IWR1843 sigma_el ~= 12 deg; at
30 m that is ~6.4 m of vertical uncertainty — the size of a whole building.
A projection that returns one pixel silently erases that.

``project_with_covariance(point, cov_in) -> Projection`` propagates the input
covariance through the pinhole projection (Jacobian by central differences, so
there is nothing hand-derived to get wrong) and also reports the physical
vertical uncertainty in metres, which ``elevation_uncertainty_is_honest`` then
checks against human height.
"""
import math
from collections import namedtuple

# Thermal sensor (FLIR Lepton-class): 160x120, ~57 deg horizontal FOV.
THERMAL_W = 160
THERMAL_H = 120
THERMAL_HFOV_DEG = 57.0
FOCAL_PX = (THERMAL_W / 2.0) / math.tan(math.radians(THERMAL_HFOV_DEG / 2.0))
CX = THERMAL_W / 2.0
CY = THERMAL_H / 2.0

# IWR1843 elevation angular sigma, and the height a real target must not be
# smaller than for the uncertainty to be physically honest.
SIGMA_EL_DEG = 12.0
HUMAN_HEIGHT_M = 1.7

Projection = namedtuple("Projection", ["pixel", "cov_px", "vertical_sigma_m"])


def spherical_cov(sigma_az_deg, sigma_el_deg, sigma_rng_m):
    """Diagonal radar covariance in (az_rad, el_rad, range_m)."""
    az = math.radians(sigma_az_deg)
    el = math.radians(sigma_el_deg)
    return [[az * az, 0.0, 0.0],
            [0.0, el * el, 0.0],
            [0.0, 0.0, sigma_rng_m * sigma_rng_m]]


def _pixel(az, el, rng):
    """Pinhole projection of a spherical point to (u, v) pixels."""
    x = rng * math.cos(el) * math.sin(az)
    y = rng * math.sin(el)
    z = rng * math.cos(el) * math.cos(az)
    u = CX + FOCAL_PX * x / z
    v = CY - FOCAL_PX * y / z
    return u, v


def _matmul(a, b):
    rows, inner, cols = len(a), len(b), len(b[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            aik = a[i][k]
            if aik == 0.0:
                continue
            for j in range(cols):
                out[i][j] += aik * b[k][j]
    return out


def _transpose(m):
    return [list(col) for col in zip(*m)]


def _jacobian(point, h=(1e-6, 1e-6, 1e-4)):
    """2x3 Jacobian d(u,v)/d(az,el,rng) by central differences."""
    az, el, rng = point
    base = [az, el, rng]
    cols = []
    for i in range(3):
        step = h[i]
        plus = list(base)
        minus = list(base)
        plus[i] += step
        minus[i] -= step
        up, vp = _pixel(*plus)
        um, vm = _pixel(*minus)
        cols.append(((up - um) / (2 * step), (vp - vm) / (2 * step)))
    # cols[i] = (du/di, dv/di); assemble as 2x3
    return [[cols[0][0], cols[1][0], cols[2][0]],
            [cols[0][1], cols[1][1], cols[2][1]]]


def project_with_covariance(point, cov_in):
    """Project a radar spherical ``point=(az_rad, el_rad, range_m)`` with input
    covariance ``cov_in`` (3x3 in az,el,range) into the thermal image.

    Returns a ``Projection`` with the pixel, the 2x2 pixel covariance (the
    uncertainty ellipse), and the physical vertical uncertainty in metres —
    ``range * sigma_el`` — which is what makes the elevation blur legible.
    """
    az, el, rng = point
    u, v = _pixel(az, el, rng)
    j = _jacobian(point)
    cov_px = _matmul(_matmul(j, cov_in), _transpose(j))
    sigma_el = math.sqrt(max(cov_in[1][1], 0.0))
    vertical_sigma_m = rng * sigma_el
    return Projection(pixel=(u, v), cov_px=cov_px,
                      vertical_sigma_m=vertical_sigma_m)


def elevation_uncertainty_is_honest(projection, min_height_m=HUMAN_HEIGHT_M):
    """Contract check: the projected vertical uncertainty must not be smaller
    than a human. A sub-human value means the elevation spread (sigma_el) was
    collapsed to a point — the silent single-pixel bug.

    Returns ``(ok, reason)``.
    """
    vs = projection.vertical_sigma_m
    if vs < min_height_m:
        return (False,
                "vertical uncertainty %.2f m is smaller than human height "
                "%.2f m: the projection collapsed the sigma_el elevation "
                "spread to (near) a point, silently erasing structure-scale "
                "uncertainty. Return the ellipse, not a pixel." %
                (vs, min_height_m))
    return (True,
            "vertical uncertainty %.2f m >= %.2f m — elevation blur preserved"
            % (vs, min_height_m))
