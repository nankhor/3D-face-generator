"""Small geometry toolkit: cameras, projection, rigid/similarity alignment, triangulation.

Conventions used everywhere in facegen:

* Head space is the coordinate system of the PES template head: metres, +Y up,
  +Z forward (the face looks toward +Z), +X toward the character's left.
* Cameras follow OpenCV: ``x_cam = R @ X + t``, pixel = ``K @ x_cam / z``,
  image +x right, +y down, camera looks along +z.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Rotations


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Axis-angle vector -> 3x3 rotation matrix."""
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)


def rotation_to_rvec(rot: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> axis-angle vector."""
    rot = np.asarray(rot, dtype=np.float64)
    cos = np.clip((np.trace(rot) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-9:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        # 180 degrees: axis from the symmetric part
        m = (rot + np.eye(3)) / 2
        axis = np.sqrt(np.clip(np.diag(m), 0, None))
        i = int(np.argmax(axis))
        axis[i] = np.sqrt(m[i, i])
        for j in range(3):
            if j != i:
                axis[j] = m[i, j] / axis[i]
        return axis / np.linalg.norm(axis) * theta
    v = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    return v / (2 * np.sin(theta)) * theta


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row rotation matrices taking unit vectors ``a`` onto ``b`` (N x 3 -> N x 3 x 3)."""
    a = normalize(np.atleast_2d(a))
    b = normalize(np.atleast_2d(b))
    axis = np.cross(a, b)
    sin = np.linalg.norm(axis, axis=1)
    cos = np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)
    out = np.repeat(np.eye(3)[None], len(a), axis=0)
    ok = sin > 1e-12
    k = np.zeros_like(axis)
    k[ok] = axis[ok] / sin[ok, None]
    kx = np.zeros((len(a), 3, 3))
    kx[:, 0, 1] = -k[:, 2]
    kx[:, 0, 2] = k[:, 1]
    kx[:, 1, 0] = k[:, 2]
    kx[:, 1, 2] = -k[:, 0]
    kx[:, 2, 0] = -k[:, 1]
    kx[:, 2, 1] = k[:, 0]
    kk = kx @ kx
    out = out + sin[:, None, None] * kx + (1 - cos)[:, None, None] * kk
    return out


def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-12)


# ---------------------------------------------------------------------------
# Similarity transforms


@dataclass
class Similarity:
    """x' = s * R @ x + t"""

    scale: float = 1.0
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def apply(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        return self.scale * pts @ self.rotation.T + self.translation

    def apply_dirs(self, dirs: np.ndarray) -> np.ndarray:
        return np.asarray(dirs, dtype=np.float64) @ self.rotation.T

    def inverse(self) -> "Similarity":
        rinv = self.rotation.T
        return Similarity(1.0 / self.scale, rinv, -(rinv @ self.translation) / self.scale)

    def compose(self, other: "Similarity") -> "Similarity":
        """Return self ∘ other (apply other first)."""
        return Similarity(
            self.scale * other.scale,
            self.rotation @ other.rotation,
            self.scale * self.rotation @ other.translation + self.translation,
        )

    def to_dict(self) -> dict:
        return {
            "scale": float(self.scale),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Similarity":
        return cls(float(d["scale"]), np.array(d["rotation"], float), np.array(d["translation"], float))


def umeyama(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None,
            with_scale: bool = True) -> Similarity:
    """Least-squares similarity taking ``src`` points onto ``dst`` points."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("umeyama needs at least 3 matching points")
    w = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    mu_s = w @ src
    mu_d = w @ dst
    xs = src - mu_s
    xd = dst - mu_d
    cov = (xd * w[:, None]).T @ xs
    u, d, vt = np.linalg.svd(cov)
    s_fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_fix[2, 2] = -1
    rot = u @ s_fix @ vt
    if with_scale:
        var_s = float(w @ np.sum(xs**2, axis=1))
        scale = float(np.trace(np.diag(d) @ s_fix) / max(var_s, 1e-18))
    else:
        scale = 1.0
    trans = mu_d - scale * rot @ mu_s
    return Similarity(scale, rot, trans)


def robust_umeyama(src: np.ndarray, dst: np.ndarray, iterations: int = 5,
                   with_scale: bool = True) -> Similarity:
    """Umeyama with iterative down-weighting of outliers (Huber-like)."""
    w = np.ones(len(src))
    sim = umeyama(src, dst, w, with_scale)
    for _ in range(iterations):
        res = np.linalg.norm(sim.apply(src) - dst, axis=1)
        sigma = 1.4826 * np.median(res) + 1e-9
        w = np.where(res <= 2 * sigma, 1.0, 2 * sigma / np.maximum(res, 1e-12))
        sim = umeyama(src, dst, w, with_scale)
    return sim


# ---------------------------------------------------------------------------
# Cameras


@dataclass
class Camera:
    """Pinhole camera, OpenCV convention."""

    width: int
    height: int
    K: np.ndarray
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @classmethod
    def from_focal(cls, width: int, height: int, focal_px: float,
                   R: np.ndarray | None = None, t: np.ndarray | None = None) -> "Camera":
        K = np.array([[focal_px, 0, width / 2.0], [0, focal_px, height / 2.0], [0, 0, 1.0]])
        return cls(int(width), int(height), K,
                   np.eye(3) if R is None else np.asarray(R, float),
                   np.zeros(3) if t is None else np.asarray(t, float))

    @property
    def focal(self) -> float:
        return float(self.K[0, 0])

    @property
    def center(self) -> np.ndarray:
        """Camera position in world space."""
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:
        """Viewing direction in world space."""
        return self.R.T @ np.array([0.0, 0.0, 1.0])

    def to_camera(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, dtype=np.float64) @ self.R.T + self.t

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points; returns (pixels Nx2, depth N)."""
        pc = self.to_camera(pts)
        z = pc[:, 2]
        zs = np.where(np.abs(z) < 1e-12, 1e-12, z)
        uv = (pc[:, :2] / zs[:, None]) * np.array([self.K[0, 0], self.K[1, 1]]) + self.K[:2, 2]
        return uv, z

    def pixel_rays(self, pix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World-space ray origin (shared) and unit directions for pixels."""
        pix = np.atleast_2d(np.asarray(pix, dtype=np.float64))
        x = (pix[:, 0] - self.K[0, 2]) / self.K[0, 0]
        y = (pix[:, 1] - self.K[1, 2]) / self.K[1, 1]
        d_cam = np.stack([x, y, np.ones_like(x)], axis=1)
        d = normalize(d_cam @ self.R)
        return self.center, d

    def projection_matrix(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t[:, None]])

    def transformed(self, sim: Similarity) -> "Camera":
        """Return the same physical camera expressed after the world is mapped by ``sim``.

        If world points X map to X' = sim(X), the camera that sees X' exactly like
        this camera sees X is returned.
        """
        inv = sim.inverse()
        # x_cam = R (inv(X')) + t = R (s' R' X' + t') + t
        # x_cam is scaled by inv.scale; projection is invariant to that, so divide it out
        R_new = self.R @ inv.rotation
        t_new = (self.R @ inv.translation + self.t) / inv.scale
        return Camera(self.width, self.height, self.K.copy(), R_new, t_new)

    def scaled(self, factor: float) -> "Camera":
        """Camera for the same view at a different image resolution."""
        K = self.K.copy()
        K[:2] *= factor
        return Camera(int(round(self.width * factor)), int(round(self.height * factor)), K,
                      self.R.copy(), self.t.copy())

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "K": self.K.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        return cls(int(d["width"]), int(d["height"]), np.array(d["K"], float),
                   np.array(d["R"], float), np.array(d["t"], float))


def look_at_camera(eye: np.ndarray, target: np.ndarray, up: np.ndarray,
                   width: int, height: int, focal_px: float) -> Camera:
    """Build an OpenCV camera at ``eye`` looking at ``target``."""
    eye = np.asarray(eye, float)
    fwd = normalize(np.asarray(target, float) - eye)
    right = normalize(np.cross(fwd, np.asarray(up, float)))
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])  # rows: camera axes in world
    t = -R @ eye
    return Camera.from_focal(width, height, focal_px, R, t)


def triangulate(cameras: list[Camera], pixels: list[np.ndarray]) -> np.ndarray:
    """Linear (DLT) triangulation of one point seen in several views."""
    rows = []
    for cam, px in zip(cameras, pixels):
        P = cam.projection_matrix()
        u, v = float(px[0]), float(px[1])
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.asarray(rows)
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    return X[:3] / X[3]


def triangulate_many(cameras: list[Camera], observations: dict[int, dict[int, np.ndarray]],
                     min_views: int = 2) -> dict[int, np.ndarray]:
    """Triangulate several points.

    ``observations[point_id][camera_index] = pixel``. Returns point_id -> XYZ for
    points seen in at least ``min_views`` cameras.
    """
    out = {}
    for pid, obs in observations.items():
        if len(obs) < min_views:
            continue
        cams = [cameras[i] for i in obs]
        pix = [obs[i] for i in obs]
        X = triangulate(cams, pix)
        # reject points behind any camera
        if all(c.to_camera(X[None])[0, 2] > 0 for c in cams):
            out[pid] = X
    return out


def focal_from_35mm(f35: float, width: int, height: int) -> float:
    """Convert a 35 mm-equivalent focal length to pixels (diagonal-based)."""
    diag_px = np.hypot(width, height)
    diag_35 = np.hypot(36.0, 24.0)
    return float(f35 * diag_px / diag_35)


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest points from points ``p`` (N x 2) to a segment set (M x 2 each).

    Returns (closest points N x M x 2, distances N x M).
    """
    ab = b - a
    denom = np.maximum(np.sum(ab * ab, axis=1), 1e-18)
    t = np.clip(np.einsum("nmk,mk->nm", p[:, None, :] - a[None], ab) / denom, 0, 1)
    c = a[None] + t[..., None] * ab[None]
    d = np.linalg.norm(p[:, None, :] - c, axis=2)
    return c, d


def closest_on_polyline(p: np.ndarray, poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest point on a 2D polyline for each point in ``p``. Returns (points, distances)."""
    poly = np.asarray(poly, dtype=np.float64)
    if len(poly) == 1:
        d = np.linalg.norm(p - poly[0], axis=1)
        return np.repeat(poly[:1], len(p), axis=0), d
    c, d = point_segment_distance(np.asarray(p, float), poly[:-1], poly[1:])
    idx = np.argmin(d, axis=1)
    rows = np.arange(len(p))
    return c[rows, idx], d[rows, idx]
