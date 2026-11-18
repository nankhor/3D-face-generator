"""Named regions on the head, derived from landmark positions.

Weights are in [0, 1] per vertex. They drive which data each part of the head
trusts during fitting (scan on the open face, face model and jaw lines under the
beard) and what Claude can adjust by name ("make the jaw 3% narrower").
"""

from __future__ import annotations

import numpy as np

from . import landmarks as L
from .mesh import smooth_vertex_values

REGION_NAMES = ["face", "beard", "jaw", "chin", "neck", "scalp", "ears", "nose", "eyes", "brows",
                "mouth", "cheeks", "forehead"]

# The jaw curve: from near the right ear, under the chin, to near the left ear.
JAW_CURVE = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397,
             288, 361, 323, 454]

GROUP_RADIUS = {"nose": 0.018, "eyes": 0.014, "brows": 0.014, "mouth": 0.016, "cheeks": 0.025,
                "forehead": 0.03, "chin": 0.025}


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd rule point-in-polygon for 2D points."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    px, py = poly[:, 0], poly[:, 1]
    j = len(poly) - 1
    for i in range(len(poly)):
        cond = ((py[i] > y) != (py[j] > y)) & (
            x < (px[j] - px[i]) * (y - py[i]) / np.where(py[j] - py[i] == 0, 1e-12, py[j] - py[i]) + px[i])
        inside ^= cond
        j = i
    return inside


def _dist_to_polyline3(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    t = np.clip(np.einsum("nmk,mk->nm", pts[:, None] - a[None], ab) / np.maximum((ab * ab).sum(1), 1e-18), 0, 1)
    c = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None] - c, axis=2).min(axis=1)


def compute_regions(V: np.ndarray, F: np.ndarray, lm: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute soft region weights for every vertex of the head."""
    need = [2, 4, 10, 152, 234, 454, 33, 263, 168]
    missing = [i for i in need if i not in lm]
    if missing:
        raise ValueError(f"head is missing landmarks {missing}; annotate the template first")
    regions: dict[str, np.ndarray] = {}
    x0 = 0.5 * (lm[33][0] + lm[263][0])
    half_width = 0.5 * abs(lm[454][0] - lm[234][0])
    y_sub = lm[2][1]
    y_chin = lm[152][1]
    y_top = lm[10][1]
    z_ear = 0.5 * (lm[234][2] + lm[454][2])
    scale = max(np.linalg.norm(lm[33] - lm[263]), 1e-6) / 0.09  # ~1 for an average adult

    # face: inside the face oval seen from the front, on the front of the head
    oval = np.array([lm[i][:2] for i in L.FACE_OVAL if i in lm])
    inside = _points_in_polygon(V[:, :2], oval)
    front = _smoothstep((V[:, 2] - (z_ear - 0.01 * scale)) / (0.02 * scale))
    regions["face"] = inside.astype(float) * front

    # beard: below the nose, in front of the ears, down to the upper neck
    below_nose = _smoothstep((y_sub + 0.004 * scale - V[:, 1]) / (0.01 * scale))
    in_front = _smoothstep((V[:, 2] - (z_ear - 0.015 * scale)) / (0.02 * scale))
    above_neck = _smoothstep((V[:, 1] - (y_chin - 0.07 * scale)) / (0.02 * scale))
    lateral = _smoothstep((half_width * 1.05 - np.abs(V[:, 0] - x0)) / (0.015 * scale))
    regions["beard"] = below_nose * in_front * above_neck * lateral

    # jaw: near the jaw curve
    jaw_pts = np.array([lm[i] for i in JAW_CURVE if i in lm])
    d_jaw = _dist_to_polyline3(V, jaw_pts)
    regions["jaw"] = 1.0 - _smoothstep(d_jaw / (0.025 * scale))

    # neck and scalp
    regions["neck"] = _smoothstep((y_chin - 0.015 * scale - V[:, 1]) / (0.02 * scale)) * (1 - regions["beard"])
    above_brow = _smoothstep((V[:, 1] - (y_top - 0.015 * scale)) / (0.02 * scale))
    behind = _smoothstep(((z_ear - 0.03 * scale) - V[:, 2]) / (0.02 * scale)) * _smoothstep(
        (V[:, 1] - (y_sub - 0.01 * scale)) / (0.02 * scale))
    regions["scalp"] = np.clip(np.maximum(above_brow, behind) - regions["face"], 0, 1)

    # ears: lateral extremes between eye and nose height, around the ear plane
    lat = _smoothstep((np.abs(V[:, 0] - x0) - 0.85 * half_width) / (0.01 * scale))
    band = _smoothstep((V[:, 1] - (y_sub - 0.03 * scale)) / (0.01 * scale)) * _smoothstep(
        ((lm[168][1] + 0.015 * scale) - V[:, 1]) / (0.01 * scale))
    depth = _smoothstep((V[:, 2] - (z_ear - 0.05 * scale)) / (0.01 * scale)) * _smoothstep(
        ((z_ear + 0.015 * scale) - V[:, 2]) / (0.01 * scale))
    regions["ears"] = lat * band * depth

    # feature groups by distance to their landmarks
    for name, ids in L.GROUPS.items():
        if name == "jaw":
            continue
        pts = np.array([lm[i] for i in ids if i in lm])
        if len(pts) == 0:
            regions[name] = np.zeros(len(V))
            continue
        d = np.min(np.linalg.norm(V[:, None] - pts[None], axis=2), axis=1)
        r = GROUP_RADIUS.get(name, 0.02) * scale
        regions[name] = 1.0 - _smoothstep(d / r)

    for k in regions:
        regions[k] = np.clip(smooth_vertex_values(regions[k], F, iterations=3), 0, 1)
    return regions
