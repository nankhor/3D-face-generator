"""FLAME head model: loading and a linear shape fit.

FLAME (https://flame.is.tue.mpg.de) is free for non-commercial use but needs a
registration, so the installer cannot download it. Put ``generic_model.pkl``
(FLAME 2020/2023) or a converted ``.npz`` in the facegen data folder.

The fit solves FLAME shape coefficients (expression and pose stay neutral) and
a similarity transform so FLAME matches the person's landmarks, scan and jaw
lines. FLAME's statistical prior gives a plausible skull and jaw where the
photos cannot see them: under the beard and behind the hair.
"""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import landmarks as L
from .geometry import Similarity, closest_on_polyline, normalize, umeyama
from .head import HeadModel, annotate_from_canonical
from .mesh import SurfaceQuery, barycentric_points, vertex_normals
from .regions import compute_regions


class _ChumpyStub:
    """Stands in for chumpy objects inside the FLAME pickle."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


class _FlameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _ChumpyStub
        return super().find_class(module, name)


def _to_array(obj):
    if isinstance(obj, _ChumpyStub):
        for key in ("x", "_x", "a", "r"):
            if key in obj.__dict__:
                return _to_array(obj.__dict__[key])
        raise ValueError("Could not read a chumpy array in the FLAME file; convert it with "
                         "`facegen flame-convert` on a machine with chumpy installed")
    if hasattr(obj, "toarray"):
        return obj.toarray()
    return np.asarray(obj)


@dataclass
class FlameModel:
    v_template: np.ndarray  # (5023, 3)
    shapedirs: np.ndarray  # (5023, 3, K)
    faces: np.ndarray  # (9976, 3)

    @classmethod
    def load(cls, path: str | Path, n_shape: int = 300) -> "FlameModel":
        path = Path(path)
        if path.suffix == ".npz":
            d = np.load(path)
            v, s, f = d["v_template"], d["shapedirs"], d["f"]
        else:
            raw = path.read_bytes()
            data = _FlameUnpickler(io.BytesIO(raw), encoding="latin1").load()
            v = _to_array(data["v_template"])
            s = _to_array(data["shapedirs"])
            f = _to_array(data["f"])
        v = np.asarray(v, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)[:, :, :n_shape]
        f = np.asarray(f, dtype=np.int64)
        return cls(v, s, f)

    def save_npz(self, path: Path) -> None:
        np.savez_compressed(path, v_template=self.v_template, shapedirs=self.shapedirs, f=self.faces)

    def vertices(self, beta: np.ndarray) -> np.ndarray:
        k = len(beta)
        return self.v_template + self.shapedirs[:, :, :k] @ beta

    def landmark_head(self, embedding: Path | None = None) -> HeadModel:
        """The FLAME template annotated with MediaPipe landmarks."""
        head = HeadModel(self.v_template, self.faces)
        if embedding is not None and Path(embedding).exists():
            d = np.load(embedding)
            ids = d["landmark_indices"].astype(np.int64)
            tri = d["lmk_face_idx"].astype(np.int64)
            bary = d["lmk_b_coords"].astype(np.float64)
            return HeadModel(self.v_template, self.faces, ids, tri, bary)
        annotated, _ = annotate_from_canonical(head)
        return annotated


@dataclass
class FlameFit:
    vertices: np.ndarray  # head space
    beta: np.ndarray
    similarity: Similarity
    landmark_rms_mm: float


def fit_flame(model: FlameModel, lm_head: HeadModel, targets: dict[int, np.ndarray],
              scan: tuple[np.ndarray, np.ndarray] | None = None, line_views: list | None = None,
              n_shape: int = 100, iterations: int = 20, prior: float = 2.0,
              line_weight: float = 2.0, scan_weight: float = 1.0) -> FlameFit:
    """Fit FLAME shape coefficients and pose to the person's data.

    ``targets`` are 3D landmarks in head space. ``scan`` is (vertices, faces) in head
    space. ``line_views`` are :class:`facegen.fitting.LineView` jaw lines.
    """
    n_shape = min(n_shape, model.shapedirs.shape[2])
    T = model.v_template
    S = model.shapedirs[:, :, :n_shape]
    F = model.faces
    lm_pos = lm_head.landmark_positions()
    regions = compute_regions(T, F, lm_pos)
    ids, tri, bary = lm_head.landmark_rows(sorted(targets))
    # drop beard-covered outline points; keep the stable ones heavier
    keep = np.array([int(i) not in L.LOWER_OVAL for i in ids])
    ids, tri, bary = ids[keep], tri[keep], bary[keep]
    w_lm = np.array([2.0 if int(i) in L.STABLE else 1.0 for i in ids])
    tgt = np.array([targets[int(i)] for i in ids])
    f = F[tri]
    p0 = (T[f] * bary[:, :, None]).sum(axis=1)  # (m, 3)
    B = (S[f] * bary[:, :, None, None]).sum(axis=1)  # (m, 3, K)

    stable_mask = np.array([int(i) in L.STABLE for i in ids])
    sim = umeyama(p0[stable_mask] if stable_mask.sum() >= 6 else p0, tgt[stable_mask] if stable_mask.sum() >= 6 else tgt)
    beta = np.zeros(n_shape)
    scan_q = SurfaceQuery(*scan) if scan is not None else None
    scan_w = regions["face"] * (1 - regions["beard"]) * (1 - regions["eyes"])
    scan_idx = np.nonzero(scan_w > 0.3)[0]

    jaw_tri = jaw_bary = None
    if line_views:
        from .fitting import JawCurve

        jc = JawCurve(lm_head)
        jaw_tri, jaw_bary = jc.tri, jc.bary

    for _ in range(iterations):
        sR = sim.scale * sim.rotation
        rows = []
        rhs = []
        # landmarks: sR (p0 + B beta) + t = tgt
        A_lm = np.einsum("ij,mjk->mik", sR, B)  # (m, 3, K)
        r_lm = tgt - (p0 @ sR.T + sim.translation)
        W = np.sqrt(w_lm)[:, None, None]
        rows.append((W * A_lm).reshape(-1, n_shape))
        rhs.append((np.sqrt(w_lm)[:, None] * r_lm).ravel())
        V = model.vertices(beta) @ sR.T + sim.translation
        # scan: point-to-plane on the open face
        if scan_q is not None and len(scan_idx):
            cp, d, t_id, _ = scan_q.closest(V[scan_idx])
            n = scan_q.normals[t_id]
            ok = d < 0.015
            if ok.any():
                ii = scan_idx[ok]
                nn = n[ok]
                A_s = np.einsum("ij,njk->nik", sR, S[ii])  # (q, 3, K)
                A_s = np.einsum("nj,njk->nk", nn, A_s)
                r_s = np.einsum("nj,nj->n", nn, cp[ok] - (T[ii] @ sR.T + sim.translation))
                ws = np.sqrt(scan_weight * scan_w[ii])
                rows.append(ws[:, None] * A_s)
                rhs.append(ws * r_s)
        # jaw lines
        if line_views and jaw_tri is not None:
            fj = F[jaw_tri]
            pj0 = (T[fj] * jaw_bary[:, :, None]).sum(axis=1)
            Bj = (S[fj] * jaw_bary[:, :, None, None]).sum(axis=1)
            Pj = (pj0 + np.einsum("mjk,k->mj", Bj, beta)) @ sR.T + sim.translation
            vn = vertex_normals(V, F)
            nj = normalize((vn[fj] * jaw_bary[:, :, None]).sum(axis=1))
            for lv in line_views:
                cam = lv.camera
                px, z = cam.project(Pj)
                facing = np.einsum("ij,ij->i", nj, normalize(cam.center[None] - Pj))
                usable = (z > 0) & (facing > -0.3)
                max_d = 0.06 * max(cam.width, cam.height)
                for poly in lv.polylines:
                    poly = np.asarray(poly, float)
                    if len(poly) < 2:
                        continue
                    c, d = closest_on_polyline(px, poly)
                    sel = np.nonzero(usable & (d < max_d))[0]
                    if len(sel) == 0:
                        continue
                    off = c[sel] - px[sel]
                    dist = np.linalg.norm(off, axis=1)
                    dir2 = off / np.maximum(dist, 1e-9)[:, None]
                    e = np.stack([dir2[:, 0], dir2[:, 1], np.zeros(len(sel))], axis=1) @ cam.R
                    metres = dist * z[sel] / cam.focal
                    # e . sR Bj beta_new = e . sR Bj beta_cur + metres
                    A_j = np.einsum("nj,njk->nk", e, np.einsum("ij,njk->nik", sR, Bj[sel]))
                    r_j = metres + A_j @ beta
                    wl = np.sqrt(line_weight)
                    rows.append(wl * A_j)
                    rhs.append(wl * r_j)
        # data rows are in metres; express them in millimetres so the prior (in units
        # of FLAME standard deviations) has a sensible weight
        A = np.vstack(rows) * 1000.0
        b = np.concatenate(rhs) * 1000.0
        A = np.vstack([A, np.sqrt(prior) * np.eye(n_shape)])
        b = np.concatenate([b, np.zeros(n_shape)])
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
        # pose step on the stable landmarks
        P_model = p0 + np.einsum("mjk,k->mj", B, beta)
        sel = stable_mask if stable_mask.sum() >= 6 else np.ones(len(ids), bool)
        sim = umeyama(P_model[sel], tgt[sel])
    V = model.vertices(beta) @ (sim.scale * sim.rotation).T + sim.translation
    P_model = (p0 + np.einsum("mjk,k->mj", B, beta)) @ (sim.scale * sim.rotation).T + sim.translation
    rms = float(np.sqrt(np.mean(np.sum((P_model - tgt) ** 2, axis=1))) * 1000)
    return FlameFit(V, beta, sim, rms)


def flame_landmark_positions(lm_head: HeadModel, vertices: np.ndarray) -> dict[int, np.ndarray]:
    P = barycentric_points(vertices, lm_head.faces, lm_head.lm_tri, lm_head.lm_bary)
    return {int(i): P[j] for j, i in enumerate(lm_head.lm_ids)}
