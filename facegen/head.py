"""The template head the person's face is built from.

A :class:`HeadModel` is the welded skin surface of the PES template face
(``face_high.fmdl``) plus where each MediaPipe landmark sits on it. Fitting
moves its vertices; the PES exporter writes them back into the template files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import landmarks as L
from .geometry import Similarity, look_at_camera, umeyama
from .mesh import SurfaceQuery, barycentric_points, vertex_normals


@dataclass
class HeadModel:
    vertices: np.ndarray  # (N, 3) welded skin vertices, head space (metres)
    faces: np.ndarray  # (M, 3)
    lm_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    lm_tri: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    lm_bary: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    @property
    def has_landmarks(self) -> bool:
        return len(self.lm_ids) > 0

    def landmark_positions(self, vertices: np.ndarray | None = None) -> dict[int, np.ndarray]:
        V = self.vertices if vertices is None else vertices
        if not self.has_landmarks:
            return {}
        P = barycentric_points(V, self.faces, self.lm_tri, self.lm_bary)
        return {int(i): P[j] for j, i in enumerate(self.lm_ids)}

    def landmark_rows(self, ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(ids present, triangle ids, barycentrics) restricted to ``ids``."""
        pos = {int(i): j for j, i in enumerate(self.lm_ids)}
        keep = [i for i in ids if i in pos]
        idx = np.array([pos[i] for i in keep], dtype=np.int64)
        return np.array(keep, dtype=np.int64), self.lm_tri[idx], self.lm_bary[idx]

    def save(self, path: Path) -> None:
        np.savez_compressed(path, vertices=self.vertices, faces=self.faces, lm_ids=self.lm_ids,
                            lm_tri=self.lm_tri, lm_bary=self.lm_bary)

    @classmethod
    def load(cls, path: Path) -> "HeadModel":
        d = np.load(path)
        return cls(d["vertices"], d["faces"], d["lm_ids"], d["lm_tri"], d["lm_bary"])


# ---------------------------------------------------------------------------
# Landmark annotation


def embed_points(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points onto the surface; returns (triangle ids, barycentrics)."""
    q = SurfaceQuery(vertices, faces)
    _, _, tri, bary = q.closest(points)
    return tri, bary


def annotate_from_canonical(head: HeadModel, iterations: int = 80) -> tuple[HeadModel, float]:
    """Place MediaPipe landmarks on the head by fitting the canonical face to it.

    Works without the MediaPipe model. Uses a similarity ICP of the canonical
    face mesh onto the front of the head, followed by a light non-rigid
    adjustment. Returns the annotated head and the mean residual in metres.
    """
    cv, cf = L.canonical_face()
    V, F = head.vertices, head.faces
    # initial guess: nose tip = the most forward point near the middle of the head,
    # scale from the head's width at nose height
    vmin, vmax = V.min(axis=0), V.max(axis=0)
    xmid = 0.5 * (vmin[0] + vmax[0])
    height = vmax[1] - vmin[1]
    central = np.abs(V[:, 0] - xmid) < 0.12 * (vmax[0] - vmin[0])
    upper = V[:, 1] > vmin[1] + 0.25 * height
    cand = np.nonzero(central & upper)[0]
    nose = V[cand[np.argmax(V[cand, 2])]] if len(cand) else V[np.argmax(V[:, 2])]
    band = np.abs(V[:, 1] - nose[1]) < 0.02 * max(height, 1e-6) + 0.005
    width_head = np.ptp(V[band, 0]) if band.sum() > 3 else (vmax[0] - vmin[0])
    width_can = np.ptp(cv[np.abs(cv[:, 1] - cv[L.NOSE_TIP, 1]) < 0.01, 0])
    s = 0.95 * width_head / max(width_can, 1e-6)
    centre, c_centre = nose, cv[L.NOSE_TIP]
    sim = Similarity(s, np.eye(3), centre - s * c_centre)
    q = SurfaceQuery(V, F)
    normals = vertex_normals(cv, cf)
    vn = vertex_normals(V, F)
    stable = np.array(sorted(set(range(468)) - set(L.LOWER_OVAL)))
    for _ in range(iterations):
        P = sim.apply(cv)
        # forward: canonical -> head
        cp, d, _, _ = q.closest(P[stable])
        n = sim.apply_dirs(normals[stable])
        w = np.exp(-(d / (0.01 + np.median(d))) ** 2) * (n[:, 2] > -0.2)
        if w.sum() < 10:
            break
        # backward: front of the head under the canonical face -> canonical surface.
        # Using both directions stops scale-ICP from shrinking the face.
        lo, hi = P.min(axis=0), P.max(axis=0)
        pad = 0.05 * (hi - lo)
        sel = np.nonzero((V[:, 0] > lo[0] + pad[0]) & (V[:, 0] < hi[0] - pad[0]) &
                         (V[:, 1] > lo[1] + pad[1]) & (V[:, 1] < hi[1] - pad[1]) &
                         (V[:, 2] > lo[2]) & (vn[:, 2] > 0.3))[0]
        src, dst, ww = [P[stable]], [cp], [w]
        if len(sel) > 20:
            cq = SurfaceQuery(P, cf)
            bp, bd, _, _ = cq.closest(V[sel])
            src.append(bp)
            dst.append(V[sel])
            ww.append(np.exp(-(bd / (0.01 + np.median(bd))) ** 2) * len(stable) / len(sel))
        step = umeyama(np.vstack(src), np.vstack(dst), np.concatenate(ww))
        sim = step.compose(sim)
        moved = np.abs(step.apply(P[stable]) - P[stable]).max()
        if moved < 1e-6:
            break
    P = sim.apply(cv)
    cp, d, tri, bary = q.closest(P)
    annotated = HeadModel(V, F, np.arange(len(P), dtype=np.int64), tri, bary)
    return annotated, float(np.mean(d[stable]))


def head_view_cameras(V: np.ndarray, yaws=(0.0,), resolution: int = 1024, pitch: float = 0.0):
    """Portrait cameras around a head (head space), for previews and annotation."""
    centre = V.mean(axis=0)
    radius = np.linalg.norm(V - centre, axis=1).max()
    lo, hi = V.min(axis=0), V.max(axis=0)
    target = np.array([centre[0], lo[1] + 0.6 * (hi[1] - lo[1]), centre[2]])
    cams = []
    for yaw in yaws:
        a, p = np.radians(yaw), np.radians(pitch)
        eye = target + radius * 4.0 * np.array([np.sin(a) * np.cos(p), np.sin(p), np.cos(a) * np.cos(p)])
        cams.append(look_at_camera(eye, target, [0, 1, 0], resolution, resolution, resolution * 1.7))
    return cams


def annotate_with_detector(head: HeadModel, render_mesh: tuple[np.ndarray, np.ndarray, np.ndarray | None],
                           texture: np.ndarray | None, detector, resolution: int = 1024,
                           yaws: tuple[float, ...] = (0.0, -25.0, 25.0)) -> HeadModel:
    """Place landmarks by rendering the head and running MediaPipe on the renders.

    ``render_mesh`` is (vertices, faces, uvs) used for rendering: the unwelded skin,
    so the texture maps correctly. Hits are embedded on the (welded) ``head``.
    Each landmark keeps the view where its surface faces the camera most.
    """
    from .raster import render

    RV, RF, RUV = render_mesh
    vn = vertex_normals(RV, RF)
    best: dict[int, tuple[float, np.ndarray]] = {}
    for cam in head_view_cameras(RV, yaws, resolution):
        img, frag = render(cam, RV, RF, RUV, texture)
        pts = detector.detect_array((img * 255).astype(np.uint8))
        if pts is None:
            continue
        for i, (x, y) in enumerate(pts[:468]):
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < cam.width and 0 <= yi < cam.height):
                continue
            t = frag.tri[yi, xi]
            if t < 0:
                continue
            b = frag.bary[yi, xi]
            f = RF[t]
            point = RV[f].T @ b
            facing = float((vn[f].T @ b) @ (-cam.forward))
            if i not in best or facing > best[i][0]:
                best[i] = (facing, point)
    if len(best) < 100:
        raise RuntimeError("MediaPipe could not find the face on the template renders")
    ids = np.array(sorted(best), dtype=np.int64)
    tri, bary = embed_points(head.vertices, head.faces, np.array([best[i][1] for i in ids]))
    return HeadModel(head.vertices, head.faces, ids, tri, bary)
