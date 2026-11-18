"""Non-rigid fitting of the template head to the person.

The template's skin is deformed so that:

* its landmarks land on the person's triangulated landmarks,
* its surface lies on the scan (open face) and on the face-model fit (under the
  beard, neck and back of the head),
* its jaw curve projects onto the jaw lines the person drew on the photos,

while the Laplacian of the displacement stays small, so the template's own detail
(eyelids, lips, ears) and topology survive. All solves are sparse linear least
squares; the closest points and line matches are refreshed each iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import landmarks as L
from .geometry import Camera, closest_on_polyline, normalize
from .head import HeadModel, embed_points
from .mesh import SurfaceQuery, barycentric_points, face_normals, uniform_laplacian, vertex_normals
from .regions import JAW_CURVE, compute_regions


@dataclass
class TargetSurface:
    """A surface the head should lie on, with a per-head-vertex trust weight."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    weights: np.ndarray  # (N_head,) in [0, 1]


@dataclass
class LineView:
    """Jaw lines drawn on one photo."""

    photo: str
    camera: Camera
    polylines: list[np.ndarray]  # each (K, 2) pixels


@dataclass
class FitSettings:
    iterations: int = 12
    stiffness_start: float = 40.0
    stiffness_end: float = 3.0
    landmark_weight: float = 6.0
    surface_weight: float = 1.0
    line_weight: float = 4.0
    rest_weight: float = 1e-3
    max_surface_distance: float = 0.015  # metres
    max_line_distance: float = 0.06  # fraction of the image's longer side
    min_normal_agreement: float = 0.2


@dataclass
class FitReport:
    landmark_rms_mm: float = 0.0
    surface_rms_mm: dict[str, float] = field(default_factory=dict)
    line_rms_px: dict[str, float] = field(default_factory=dict)
    iterations: int = 0

    def to_dict(self) -> dict:
        return {
            "landmark_rms_mm": round(self.landmark_rms_mm, 2),
            "surface_rms_mm": {k: round(v, 2) for k, v in self.surface_rms_mm.items()},
            "line_rms_px": {k: round(v, 1) for k, v in self.line_rms_px.items()},
            "iterations": self.iterations,
        }


class JawCurve:
    """A dense curve along the template's jaw (embedded on its surface)."""

    def __init__(self, head: HeadModel, subdivisions: int = 6):
        lm = head.landmark_positions()
        ctrl = np.array([lm[i] for i in JAW_CURVE if i in lm])
        if len(ctrl) < 5:
            raise ValueError("the template head has no jaw landmarks")
        pts = []
        for a, b in zip(ctrl[:-1], ctrl[1:]):
            for s in np.linspace(0, 1, subdivisions, endpoint=False):
                pts.append(a + (b - a) * s)
        pts.append(ctrl[-1])
        self.tri, self.bary = embed_points(head.vertices, head.faces, np.asarray(pts))

    def positions(self, V: np.ndarray, F: np.ndarray) -> np.ndarray:
        return barycentric_points(V, F, self.tri, self.bary)


def _bary_matrix(n_vertices: int, faces: np.ndarray, tri: np.ndarray, bary: np.ndarray) -> sp.csr_matrix:
    f = faces[tri]
    rows = np.repeat(np.arange(len(tri)), 3)
    return sp.csr_matrix((bary.ravel(), (rows, f.ravel())), shape=(len(tri), n_vertices))


def landmark_targets_for_fit(head: HeadModel, targets: dict[int, np.ndarray],
                             regions: dict[str, np.ndarray]) -> dict[int, float]:
    """Choose which landmarks to trust and how much (beard-covered ones are dropped)."""
    beard = regions.get("beard")
    lips = set(L.LIPS_OUTER) | {13, 14, 0, 17, 78, 308, 81, 311, 178, 402}
    stable = set(L.STABLE)
    out = {}
    lm_ids, tri, bary = head.landmark_rows(sorted(targets))
    for i, t, b in zip(lm_ids, tri, bary):
        i = int(i)
        if i in L.LOWER_OVAL:
            continue
        if beard is not None:
            wb = float(beard[head.faces[t]] @ b)
            if wb > 0.5 and i not in lips:
                continue
        out[i] = 2.0 if i in stable else 1.0
    return out


def fit_head(head: HeadModel, landmark_targets: dict[int, np.ndarray], surfaces: list[TargetSurface],
             line_views: list[LineView], settings: FitSettings | None = None,
             start: np.ndarray | None = None, regions: dict[str, np.ndarray] | None = None
             ) -> tuple[np.ndarray, FitReport]:
    """Deform the template head. Returns the new vertex positions and a report."""
    s = settings or FitSettings()
    V0 = head.vertices
    F = head.faces
    n = len(V0)
    X = (start if start is not None else V0).copy()
    if regions is None:
        regions = compute_regions(V0, F, head.landmark_positions())
    Lap = uniform_laplacian(n, F)
    L3 = sp.block_diag([Lap, Lap, Lap]).tocsr()
    I3 = sp.identity(3 * n, format="csr")

    lm_w = landmark_targets_for_fit(head, landmark_targets, regions)
    lm_ids, lm_tri, lm_bary = head.landmark_rows(sorted(lm_w))
    B_lm = _bary_matrix(n, F, lm_tri, lm_bary)
    lm_target = np.array([landmark_targets[int(i)] for i in lm_ids]) if len(lm_ids) else np.zeros((0, 3))
    lm_weight = np.sqrt(s.landmark_weight * np.array([lm_w[int(i)] for i in lm_ids])) if len(lm_ids) else np.zeros(0)

    queries = [(surf, SurfaceQuery(surf.vertices, surf.faces)) for surf in surfaces]
    jaw = JawCurve(head) if line_views else None
    B_jaw = _bary_matrix(n, F, jaw.tri, jaw.bary) if jaw is not None else None

    report = FitReport()
    stiff = np.geomspace(s.stiffness_start, s.stiffness_end, max(s.iterations, 1))
    for it in range(s.iterations):
        rows_A = []
        rows_b = []
        # smoothness of the displacement and a weak pull to the rest shape
        w_l = np.sqrt(stiff[it])
        rows_A.append(w_l * L3)
        rows_b.append(w_l * (L3 @ V0.T.ravel()))
        w_r = np.sqrt(s.rest_weight)
        rows_A.append(w_r * I3)
        rows_b.append(w_r * V0.T.ravel())
        # landmarks
        if len(lm_ids):
            W = sp.diags(lm_weight)
            A_lm = sp.block_diag([W @ B_lm, W @ B_lm, W @ B_lm])
            rows_A.append(A_lm)
            rows_b.append((lm_weight[:, None] * lm_target).T.ravel())
        # surfaces: point-to-plane
        vn = vertex_normals(X, F)
        for surf, q in queries:
            w = surf.weights
            active = np.nonzero(w > 0.05)[0]
            if len(active) == 0:
                continue
            cp, d, tri, _ = q.closest(X[active])
            tn = q.normals[tri]
            agree = np.einsum("ij,ij->i", tn, vn[active])
            ok = (d < s.max_surface_distance) & (np.abs(agree) > s.min_normal_agreement)
            tn = tn * np.sign(agree)[:, None]
            idx = active[ok]
            if len(idx) == 0:
                continue
            ww = np.sqrt(s.surface_weight * w[idx])
            nrm = tn[ok]
            rr = np.arange(len(idx))
            data = (ww[:, None] * nrm).T.ravel()
            rows = np.tile(rr, 3)
            cols = np.concatenate([idx, idx + n, idx + 2 * n])
            rows_A.append(sp.csr_matrix((data, (rows, cols)), shape=(len(idx), 3 * n)))
            rows_b.append(ww * np.einsum("ij,ij->i", nrm, cp[ok]))
            report.surface_rms_mm[surf.name] = float(np.sqrt(np.mean(d[ok] ** 2)) * 1000)
        # jaw lines: move the jaw curve, in each image, toward the drawn line
        if jaw is not None:
            P = jaw.positions(X, F)
            Pn = normalize(B_jaw @ vn)
            for lv in line_views:
                A_rows, b_rows, err = _line_rows(lv, P, Pn, B_jaw, n, s)
                if A_rows is not None:
                    rows_A.append(A_rows)
                    rows_b.append(b_rows)
                    report.line_rms_px[lv.photo] = err
        A = sp.vstack(rows_A).tocsr()
        b = np.concatenate(rows_b)
        AtA = (A.T @ A).tocsc()
        Atb = A.T @ b
        x = spla.spsolve(AtA, Atb)
        X = x.reshape(3, n).T
        report.iterations = it + 1
    if len(lm_ids):
        P = B_lm @ X
        report.landmark_rms_mm = float(np.sqrt(np.mean(np.sum((P - lm_target) ** 2, axis=1))) * 1000)
    return X, report


def _line_rows(lv: LineView, P: np.ndarray, Pn: np.ndarray, B: sp.csr_matrix, n: int,
               s: FitSettings):
    cam = lv.camera
    px, z = cam.project(P)
    to_cam = normalize(cam.center[None] - P)
    facing = np.einsum("ij,ij->i", Pn, to_cam)
    usable = (z > 0) & (facing > -0.3)
    max_d = s.max_line_distance * max(cam.width, cam.height)
    rows_A, rows_b, errs = [], [], []
    for poly in lv.polylines:
        poly = np.asarray(poly, dtype=np.float64)
        if len(poly) < 2:
            continue
        c, d = closest_on_polyline(px, poly)
        # ignore curve points beyond the ends of the drawn line
        at_end = (np.linalg.norm(c - poly[0], axis=1) < 1e-6) | (np.linalg.norm(c - poly[-1], axis=1) < 1e-6)
        sel = np.nonzero(usable & (d < max_d) & ~(at_end & (d > 0.01 * max(cam.width, cam.height))))[0]
        if len(sel) == 0:
            continue
        off = c[sel] - px[sel]  # pixels
        dist = np.linalg.norm(off, axis=1)
        dir2 = off / np.maximum(dist, 1e-9)[:, None]
        # world-space direction of the image-plane offset
        d_cam = np.stack([dir2[:, 0], dir2[:, 1], np.zeros(len(sel))], axis=1)
        e = d_cam @ cam.R  # rows: R^T d
        metres = dist * z[sel] / cam.focal
        target = np.einsum("ij,ij->i", e, P[sel]) + metres
        w = np.sqrt(s.line_weight)
        Bs = B[sel]
        A = sp.hstack([sp.diags(w * e[:, 0]) @ Bs, sp.diags(w * e[:, 1]) @ Bs, sp.diags(w * e[:, 2]) @ Bs])
        rows_A.append(A)
        rows_b.append(w * target)
        errs.append(dist)
    if not rows_A:
        return None, None, 0.0
    allerr = np.concatenate(errs)
    return sp.vstack(rows_A), np.concatenate(rows_b), float(np.sqrt(np.mean(allerr ** 2)))


# ---------------------------------------------------------------------------
# Region adjustments requested by Claude after comparing renders with photos


def adjust_region(head: HeadModel, X: np.ndarray, regions: dict[str, np.ndarray], region: str,
                  translate_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
                  scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
                  strength: float = 1.0, smooth_iterations: int = 4) -> np.ndarray:
    """Move or scale one named region with a soft falloff.

    ``translate_mm`` is in head axes (x: toward the character's left, y: up,
    z: forward). ``scale`` is applied around the region's weighted centre.
    """
    if region not in regions:
        raise KeyError(f"unknown region '{region}'. Known: {sorted(regions)}")
    w = np.clip(regions[region] * strength, 0, 1)
    if w.max() <= 0:
        return X.copy()
    centre = (w[:, None] * X).sum(axis=0) / w.sum()
    disp = np.asarray(translate_mm, float) / 1000.0 + (X - centre) * (np.asarray(scale, float) - 1.0)
    D = w[:, None] * disp
    from .mesh import smooth_vertex_values

    D = smooth_vertex_values(D, head.faces, iterations=smooth_iterations, alpha=0.5)
    return X + D


def region_report(X: np.ndarray, X0: np.ndarray, regions: dict[str, np.ndarray]) -> dict[str, dict]:
    """How much each region moved compared to the template (mm)."""
    out = {}
    D = (X - X0) * 1000
    for name, w in regions.items():
        if w.sum() < 1e-6:
            continue
        m = (w[:, None] * D).sum(axis=0) / w.sum()
        out[name] = {"mean_shift_mm": [round(float(v), 2) for v in m],
                     "max_move_mm": round(float(np.max(np.linalg.norm(D[w > 0.5], axis=1))) if (w > 0.5).any() else 0.0, 2)}
    return out


def face_normals_of(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    return face_normals(X, F)
