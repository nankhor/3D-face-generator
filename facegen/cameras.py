"""Camera estimation for the photo set.

Two sources:

* Meshroom's structure-from-motion output (``cameras.sfm``), when the scan worked.
* Our own solver: PnP of each photo against a 3D face model (template head
  landmarks, or MediaPipe's canonical face), then triangulation and bundle
  adjustment over all photos. Profile photos where MediaPipe finds no face are
  solved from anchors the person clicks in the line editor (nose tip, eye
  corner, mouth corner, ear).

Point keys are strings: ``"lm:33"`` for MediaPipe landmark 33, ``"anchor:ear_left"``
for a clicked anchor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from . import landmarks as L
from .geometry import Camera, Similarity, rodrigues, rotation_to_rvec, triangulate, umeyama
from .mesh import vertex_normals

Observations = dict[str, dict[str, np.ndarray]]  # view -> point key -> pixel


def lm_key(i: int) -> str:
    return f"lm:{i}"


def anchor_key(name: str) -> str:
    return f"anchor:{name}"


def anchor_model_key(name: str) -> str | None:
    idx = L.ANCHORS.get(name)
    return None if idx is None else lm_key(idx)


@dataclass
class CameraSolution:
    cameras: dict[str, Camera]
    points: dict[str, np.ndarray]
    errors: dict[str, float] = field(default_factory=dict)  # mean reprojection error (px)
    failed: list[str] = field(default_factory=list)
    source: str = "landmarks"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "cameras": {k: c.to_dict() for k, c in self.cameras.items()},
            "points": {k: np.asarray(v).tolist() for k, v in self.points.items()},
            "errors": self.errors,
            "failed": self.failed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraSolution":
        return cls(
            {k: Camera.from_dict(c) for k, c in d["cameras"].items()},
            {k: np.asarray(v, float) for k, v in d.get("points", {}).items()},
            {k: float(v) for k, v in d.get("errors", {}).items()},
            list(d.get("failed", [])),
            d.get("source", "landmarks"),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: Path) -> "CameraSolution":
        return cls.from_dict(json.loads(path.read_text()))

    def transformed(self, sim: Similarity) -> "CameraSolution":
        return CameraSolution(
            {k: c.transformed(sim) for k, c in self.cameras.items()},
            {k: sim.apply(v[None])[0] for k, v in self.points.items()},
            dict(self.errors), list(self.failed), self.source)


def build_observations(photo_landmarks: list[L.PhotoLandmarks], anchors: dict[str, dict[str, list]] | None,
                       landmark_ids: list[int] | None = None) -> Observations:
    """Merge automatic landmarks and user anchors into one observation table."""
    ids = landmark_ids if landmark_ids is not None else list(range(468))
    obs: Observations = {}
    for pl in photo_landmarks:
        if pl.points is None:
            continue
        obs[pl.photo] = {lm_key(i): pl.points[i] for i in ids if i < len(pl.points)}
    for photo, named in (anchors or {}).items():
        view = obs.setdefault(photo, {})
        for name, xy in named.items():
            view[anchor_key(name)] = np.asarray(xy, dtype=np.float64)
            mk = anchor_model_key(name)
            if mk is not None:
                # a clicked anchor overrides an automatic landmark (it is more reliable)
                view[mk] = np.asarray(xy, dtype=np.float64)
    return obs


def model_points_from_canonical() -> dict[str, np.ndarray]:
    verts, _ = L.canonical_face()
    pts = {lm_key(i): verts[i] for i in range(len(verts))}
    for name, idx in L.ANCHORS.items():
        if idx is not None:
            pts[anchor_key(name)] = verts[idx]
    return pts


def _canonical_normals() -> np.ndarray:
    v, f = L.canonical_face()
    return vertex_normals(v, f)


def _pnp(points3d: np.ndarray, points2d: np.ndarray, K: np.ndarray,
         guess: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    import cv2

    obj = np.ascontiguousarray(points3d, dtype=np.float64).reshape(-1, 1, 3)
    img = np.ascontiguousarray(points2d, dtype=np.float64).reshape(-1, 1, 2)
    dist = np.zeros(5)
    ok = False
    rvec = tvec = None
    if guess is not None:
        rvec = rotation_to_rvec(guess[0]).reshape(3, 1)
        tvec = np.asarray(guess[1], float).reshape(3, 1)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, rvec, tvec, useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        if len(points3d) >= 4:
            flags = cv2.SOLVEPNP_SQPNP
        else:
            return None
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=flags)
        if not ok:
            return None
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, rvec, tvec, useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R = rodrigues(rvec.ravel())
    t = tvec.ravel()
    # the face must be in front of the camera
    if np.median((points3d @ R.T + t)[:, 2]) <= 0:
        return None
    return R, t


def _visible_landmark_keys(cam: Camera, keys: list[str], points: dict[str, np.ndarray],
                           normals: np.ndarray, threshold: float = 0.15) -> list[str]:
    """Drop landmarks on the far side of the face (MediaPipe still predicts them)."""
    out = []
    centre = cam.center
    for k in keys:
        if not k.startswith("lm:"):
            out.append(k)
            continue
        i = int(k[3:])
        if i >= len(normals):
            out.append(k)
            continue
        X = points.get(k)
        if X is None:
            continue
        to_cam = centre - X
        to_cam /= max(np.linalg.norm(to_cam), 1e-12)
        if float(normals[i] @ to_cam) > threshold:
            out.append(k)
    return out


def solve_cameras(observations: Observations, sizes: dict[str, tuple[int, int]],
                  focals: dict[str, float], model_points: dict[str, np.ndarray] | None = None,
                  model_normals_rotation: np.ndarray | None = None, rounds: int = 3,
                  bundle_adjust: bool = True, refine_focal: bool = True,
                  prior_weight: float = 0.02, initial: dict[str, Camera] | None = None) -> CameraSolution:
    """Solve all cameras and the 3D landmark positions.

    ``model_points`` gives the 3D positions (head space, metres) used to solve each
    view by PnP; it also anchors the gauge (position, rotation and scale) of the result.
    ``initial`` cameras (e.g. Meshroom's, already in head space) seed the solve.
    """
    model = dict(model_points or model_points_from_canonical())
    stable_keys = {lm_key(i) for i in L.STABLE} | {anchor_key(n) for n in L.ANCHORS}
    cnormals = _canonical_normals()
    if model_normals_rotation is not None:
        cnormals = cnormals @ np.asarray(model_normals_rotation).T

    cameras: dict[str, Camera] = {k: v for k, v in (initial or {}).items() if k in observations}
    points = dict(model)
    failed: list[str] = []

    for _round in range(rounds):
        # current structure normals: rotate canonical normals with the best fit of
        # current points onto canonical (cheap and good enough for visibility)
        normals = cnormals
        for view, obs in observations.items():
            w, h = sizes[view]
            K = Camera.from_focal(w, h, focals[view]).K
            keys = [k for k in obs if k in points and (k in stable_keys or _round > 0)]
            if view in cameras:
                keys = _visible_landmark_keys(cameras[view], keys, points, normals)
            if len(keys) < 4:
                continue
            P3 = np.array([points[k] for k in keys])
            P2 = np.array([obs[k] for k in keys])
            guess = (cameras[view].R, cameras[view].t) if view in cameras else None
            res = _pnp(P3, P2, K, guess)
            if res is None:
                continue
            cameras[view] = Camera(w, h, K, res[0], res[1])
        if len(cameras) < 2:
            break
        # triangulate every key seen by >= 2 solved views (visible side only)
        per_key: dict[str, dict[str, np.ndarray]] = {}
        for view, cam in cameras.items():
            obs = observations[view]
            vis = _visible_landmark_keys(cam, list(obs), points, normals)
            for k in vis:
                per_key.setdefault(k, {})[view] = obs[k]
        new_points = dict(points)
        for k, views in per_key.items():
            if len(views) < 2:
                continue
            cams = [cameras[v] for v in views]
            X = triangulate(cams, [views[v] for v in views])
            if all(c.to_camera(X[None])[0, 2] > 0 for c in cams):
                new_points[k] = X
        # keep the gauge on the model: similarity of triangulated stable points onto model
        common = [k for k in stable_keys if k in new_points and k in model and k in per_key]
        if len(common) >= 6:
            sim = umeyama(np.array([new_points[k] for k in common]), np.array([model[k] for k in common]))
            new_points = {k: sim.apply(v[None])[0] for k, v in new_points.items()}
            cameras = {k: c.transformed(sim) for k, c in cameras.items()}
        points = new_points

    if bundle_adjust and len(cameras) >= 2:
        cameras, points = _bundle_adjust(cameras, points, observations, model, cnormals,
                                         refine_focal=refine_focal, prior_weight=prior_weight)

    errors = {}
    for view, cam in cameras.items():
        obs = observations[view]
        keys = _visible_landmark_keys(cam, [k for k in obs if k in points], points, cnormals)
        if not keys:
            continue
        px, _ = cam.project(np.array([points[k] for k in keys]))
        errors[view] = float(np.mean(np.linalg.norm(px - np.array([obs[k] for k in keys]), axis=1)))
    failed = sorted(set(observations) - set(cameras))
    return CameraSolution(cameras, points, errors, failed, "landmarks")


def _bundle_adjust(cameras: dict[str, Camera], points: dict[str, np.ndarray], observations: Observations,
                   model: dict[str, np.ndarray], normals: np.ndarray, refine_focal: bool,
                   prior_weight: float) -> tuple[dict[str, Camera], dict[str, np.ndarray]]:
    views = list(cameras)
    # observations to use: visible, and the point appears in the point table
    obs_list = []  # (view index, point index, x, y)
    keys: list[str] = []
    key_index: dict[str, int] = {}
    for vi, view in enumerate(views):
        cam = cameras[view]
        vis = _visible_landmark_keys(cam, [k for k in observations[view] if k in points], points, normals)
        for k in vis:
            if k not in key_index:
                key_index[k] = len(keys)
                keys.append(k)
            obs_list.append((vi, key_index[k], *observations[view][k]))
    if not obs_list:
        return cameras, points
    obs_arr = np.array(obs_list)
    vi_arr = obs_arr[:, 0].astype(int)
    pi_arr = obs_arr[:, 1].astype(int)
    uv = obs_arr[:, 2:4]
    nV, nP = len(views), len(keys)
    # pixel scale per view so residuals are comparable across resolutions
    scale = np.array([max(cameras[v].width, cameras[v].height) / 1000.0 for v in views])

    # one focal multiplier per camera group (same image size and focal guess = same phone
    # camera); views with few points cannot pin their own focal length
    groups: dict[tuple, int] = {}
    view_group = np.zeros(nV, dtype=int)
    for i, v in enumerate(views):
        c = cameras[v]
        key = (c.width, c.height, round(float(c.K[0, 0]) / max(c.width, c.height), 2))
        view_group[i] = groups.setdefault(key, len(groups))
    nG = len(groups) if refine_focal else 0
    cam_params = np.zeros((nV, 6))
    for i, v in enumerate(views):
        c = cameras[v]
        cam_params[i, :3] = rotation_to_rvec(c.R)
        cam_params[i, 3:6] = c.t
    pts = np.array([points[k] for k in keys])
    prior_idx = np.array([i for i, k in enumerate(keys) if k in model], dtype=int)
    prior_pts = np.array([model[keys[i]] for i in prior_idx]) if len(prior_idx) else np.zeros((0, 3))
    head_size = 0.15  # metres; normalises the prior residual
    Ks = [cameras[v].K for v in views]
    fx_all = np.array([K[0, 0] for K in Ks])
    fy_all = np.array([K[1, 1] for K in Ks])
    cx_all = np.array([K[0, 2] for K in Ks])
    cy_all = np.array([K[1, 2] for K in Ks])

    def unpack(x):
        cp = x[: nV * 6].reshape(nV, 6)
        g = x[nV * 6: nV * 6 + nG]
        P = x[nV * 6 + nG:].reshape(nP, 3)
        return cp, g, P

    def residuals(x):
        cp, g, P = unpack(x)
        out = np.empty(len(uv) * 2 + len(prior_idx) * 3)
        Rs = np.array([rodrigues(cp[i, :3]) for i in range(nV)])
        Xc = np.einsum("nij,nj->ni", Rs[vi_arr], P[pi_arr]) + cp[vi_arr, 3:6]
        f_mul = np.exp(g[view_group[vi_arr]]) if nG else np.ones(len(vi_arr))
        z = np.where(np.abs(Xc[:, 2]) < 1e-9, 1e-9, Xc[:, 2])
        px = fx_all[vi_arr] * f_mul * Xc[:, 0] / z + cx_all[vi_arr]
        py = fy_all[vi_arr] * f_mul * Xc[:, 1] / z + cy_all[vi_arr]
        s = scale[vi_arr]
        out[0:len(uv) * 2:2] = (px - uv[:, 0]) / s
        out[1:len(uv) * 2:2] = (py - uv[:, 1]) / s
        if len(prior_idx):
            out[len(uv) * 2:] = (prior_weight * (P[prior_idx] - prior_pts) / head_size * 1000.0).ravel()
        return out

    # sparsity pattern
    m = len(uv) * 2 + len(prior_idx) * 3
    n = nV * 6 + nG + nP * 3
    A = lil_matrix((m, n), dtype=int)
    rows = np.arange(len(uv))
    for k in range(6):
        A[2 * rows, vi_arr * 6 + k] = 1
        A[2 * rows + 1, vi_arr * 6 + k] = 1
    if nG:
        A[2 * rows, nV * 6 + view_group[vi_arr]] = 1
        A[2 * rows + 1, nV * 6 + view_group[vi_arr]] = 1
    for k in range(3):
        A[2 * rows, nV * 6 + nG + pi_arr * 3 + k] = 1
        A[2 * rows + 1, nV * 6 + nG + pi_arr * 3 + k] = 1
    for j, pi in enumerate(prior_idx):
        for k in range(3):
            A[len(uv) * 2 + 3 * j + k, nV * 6 + nG + pi * 3 + k] = 1

    x0 = np.concatenate([cam_params.ravel(), np.zeros(nG), pts.ravel()])
    res = least_squares(residuals, x0, jac_sparsity=A, loss="soft_l1", f_scale=3.0, x_scale="jac",
                        max_nfev=200, method="trf")
    cp, g, P = unpack(res.x)
    new_cams = {}
    for i, v in enumerate(views):
        K = Ks[i].copy()
        if nG:
            K[0, 0] *= np.exp(g[view_group[i]])
            K[1, 1] *= np.exp(g[view_group[i]])
        new_cams[v] = Camera(cameras[v].width, cameras[v].height, K, rodrigues(cp[i, :3]), cp[i, 3:6].copy())
    new_points = dict(points)
    for i, k in enumerate(keys):
        new_points[k] = P[i]
    return new_cams, new_points


# ---------------------------------------------------------------------------
# Meshroom


def _meshroom_raw(path: str | Path) -> list[tuple[str, int, int, np.ndarray, np.ndarray, np.ndarray]]:
    data = json.loads(Path(path).read_text())
    intrinsics = {str(i["intrinsicId"]): i for i in data.get("intrinsics", [])}
    poses = {str(p["poseId"]): p["pose"]["transform"] for p in data.get("poses", [])}
    out = []
    for view in data.get("views", []):
        pose = poses.get(str(view.get("poseId")))
        intr = intrinsics.get(str(view.get("intrinsicId")))
        if pose is None or intr is None:
            continue
        w = int(view.get("width") or intr.get("width"))
        h = int(view.get("height") or intr.get("height"))
        if "focalLength" in intr and "sensorWidth" in intr:
            f_px = float(intr["focalLength"]) / float(intr["sensorWidth"]) * max(w, h)
        else:
            f_px = float(intr.get("pxFocalLength", max(w, h)))
        pp = intr.get("principalPoint", [0.0, 0.0])
        pp = [float(pp[0]), float(pp[1])]
        # Newer AliceVision versions store the principal point as an offset from the
        # image centre, older ones as absolute pixels.
        if abs(pp[0]) < w * 0.25 and abs(pp[1]) < h * 0.25:
            pp = [w / 2.0 + pp[0], h / 2.0 + pp[1]]
        K = np.array([[f_px, 0, pp[0]], [0, f_px, pp[1]], [0, 0, 1.0]])
        rot = np.array([float(x) for x in pose["rotation"]])
        C = np.array([float(x) for x in pose["center"]])
        out.append((Path(view["path"]).name, w, h, K, rot, C))
    return out


# How the 9 rotation numbers map to a world->camera rotation. AliceVision's layout has
# changed over versions, so the variant that best explains the landmarks is chosen.
_ROT_VARIANTS = {
    "camera_to_world_rows": lambda r: r.reshape(3, 3).T,
    "world_to_camera_rows": lambda r: r.reshape(3, 3),
    "camera_to_world_cols": lambda r: r.reshape(3, 3, order="F").T,
    "world_to_camera_cols": lambda r: r.reshape(3, 3, order="F"),
}


def load_meshroom_sfm(path: str | Path, variant: str = "camera_to_world_rows") -> dict[str, Camera]:
    """Read cameras from a Meshroom/AliceVision ``cameras.sfm`` (JSON) file.

    Returns photo file name -> Camera (OpenCV convention, Meshroom's world).
    """
    out = {}
    for name, w, h, K, rot, C in _meshroom_raw(path):
        R = _ROT_VARIANTS[variant](rot)
        out[name] = Camera(w, h, K, R, -R @ C)
    return out


def load_meshroom_sfm_checked(path: str | Path, observations: Observations) -> tuple[dict[str, Camera], str, float]:
    """Like :func:`load_meshroom_sfm`, choosing the rotation layout whose cameras best
    reproject the photos' landmarks. Returns (cameras, variant, mean error in px)."""
    best = None
    for variant in _ROT_VARIANTS:
        cams = {k: v for k, v in load_meshroom_sfm(path, variant).items() if k in observations}
        if len(cams) < 2:
            continue
        per_key: dict[str, dict[str, np.ndarray]] = {}
        for view, cam in cams.items():
            for k in L.STABLE:
                px = observations[view].get(lm_key(k))
                if px is not None:
                    per_key.setdefault(lm_key(k), {})[view] = px
        errs = []
        for k, views in per_key.items():
            if len(views) < 2:
                continue
            vs = list(views)
            X = triangulate([cams[v] for v in vs], [views[v] for v in vs])
            for v in vs:
                px, z = cams[v].project(X[None])
                errs.append(np.linalg.norm(px[0] - views[v]) if z[0] > 0 else 1e6)
        if not errs:
            continue
        err = float(np.median(errs))
        if best is None or err < best[2]:
            best = (cams, variant, err)
    if best is None:
        raise ValueError("Meshroom cameras could not be matched to the photos")
    return best


def align_cameras_to_model(cameras: dict[str, Camera], observations: Observations,
                           model: dict[str, np.ndarray]) -> tuple[Similarity, dict[str, np.ndarray]]:
    """Find the similarity taking an arbitrary reconstruction (e.g. Meshroom's) into head space
    by triangulating stable landmarks and aligning them with the model points."""
    per_key: dict[str, dict[str, np.ndarray]] = {}
    for view, cam in cameras.items():
        for k, px in observations.get(view, {}).items():
            per_key.setdefault(k, {})[view] = px
    tri = {}
    for k, views in per_key.items():
        if len(views) >= 2:
            cams = [cameras[v] for v in views]
            tri[k] = triangulate(cams, [views[v] for v in views])
    stable = [lm_key(i) for i in L.STABLE]
    common = [k for k in stable if k in tri and k in model]
    if len(common) < 6:
        raise ValueError("Not enough landmarks seen in two or more photos to align the scan")
    from .geometry import robust_umeyama

    sim = robust_umeyama(np.array([tri[k] for k in common]), np.array([model[k] for k in common]))
    return sim, {k: sim.apply(v[None])[0] for k, v in tri.items()}
