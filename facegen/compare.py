"""Render the fitted head from each photo's camera and compare it with the photo."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from . import landmarks as L
from .geometry import Camera
from .raster import render, visibility_depth_test


@dataclass
class ViewComparison:
    photo: str
    landmark_error_pct: float | None = None  # mean error / interocular distance, in %
    worst_landmarks: list[tuple[str, float]] = field(default_factory=list)
    jaw_line_error_px: float | None = None
    image_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "photo": self.photo,
            "landmark_error_pct": None if self.landmark_error_pct is None else round(self.landmark_error_pct, 2),
            "worst_landmarks": [(n, round(e, 2)) for n, e in self.worst_landmarks],
            "jaw_line_error_px": None if self.jaw_line_error_px is None else round(self.jaw_line_error_px, 1),
            "image": self.image_path,
        }


LANDMARK_NAMES = {
    L.NOSE_TIP: "nose tip", 168: "nose bridge", 2: "under nose", L.EYE_R_OUTER: "right eye outer",
    L.EYE_L_OUTER: "left eye outer", L.EYE_R_INNER: "right eye inner", L.EYE_L_INNER: "left eye inner",
    L.MOUTH_R: "right mouth corner", L.MOUTH_L: "left mouth corner", 13: "upper lip", 14: "lower lip",
    105: "right brow", 334: "left brow", 10: "forehead top", 234: "right cheek edge",
    454: "left cheek edge", 98: "right nostril", 327: "left nostril",
}


def render_views(cameras: dict[str, Camera], vertices: np.ndarray, faces: np.ndarray,
                 uvs: np.ndarray | None = None, texture: np.ndarray | None = None,
                 max_side: int = 800) -> dict[str, tuple[np.ndarray, Camera]]:
    out = {}
    for name, cam in cameras.items():
        c = cam.scaled(min(1.0, max_side / max(cam.width, cam.height)))
        img, _ = render(c, vertices, faces, uvs, texture)
        out[name] = (img, c)
    return out


def compare_view(photo_name: str, photo: np.ndarray, cam: Camera, vertices: np.ndarray, faces: np.ndarray,
                 photo_landmarks: np.ndarray | None, head_landmarks: dict[int, np.ndarray],
                 jaw_points: np.ndarray | None, user_lines: list[np.ndarray], out_path: str,
                 uvs: np.ndarray | None = None, texture: np.ndarray | None = None,
                 max_side: int = 700) -> ViewComparison:
    """Side-by-side image (photo | render | overlay) plus numbers.

    Landmark error compares where the head's landmarks project with the landmarks
    found on the photo (beard-covered outline points excluded).
    """
    f = min(1.0, max_side / max(cam.width, cam.height))
    c = cam.scaled(f)
    img_r, frag = render(c, vertices, faces, uvs, texture, background=1.0)
    ph = np.asarray(Image.fromarray(photo).resize((c.width, c.height), Image.LANCZOS)).astype(float) / 255.0
    overlay = ph.copy()
    m = frag.mask
    overlay[m] = 0.45 * ph[m] + 0.55 * img_r[m]
    canvas = Image.fromarray((np.concatenate([ph, img_r, overlay], axis=1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    result = ViewComparison(photo_name)
    W = c.width

    if photo_landmarks is not None and head_landmarks:
        ids = [i for i in head_landmarks if i not in L.LOWER_OVAL and i < len(photo_landmarks)]
        P = np.array([head_landmarks[i] for i in ids])
        # only landmarks the camera can see (MediaPipe guesses hidden ones)
        scale = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
        vis, _ = visibility_depth_test(c, frag, P, 0.02 * scale)
        ids = [i for i, v in zip(ids, vis) if v]
        P = P[vis]
        px, z = c.project(P)
        obs = photo_landmarks[ids] * f
        err = np.linalg.norm(px - obs, axis=1)
        io = L.interocular(photo_landmarks * f)
        if io > 1:
            result.landmark_error_pct = float(np.mean(err) / io * 100)
            named = [(LANDMARK_NAMES[i], float(err[k] / io * 100)) for k, i in enumerate(ids) if i in LANDMARK_NAMES]
            result.worst_landmarks = sorted(named, key=lambda t: -t[1])[:5]
        for (x, y), (ox, oy) in zip(px, obs):
            for off in (0, 2 * W):
                draw.line([(ox + off, oy), (x + off, y)], fill=(255, 200, 0), width=1)
                draw.ellipse([x + off - 1.5, y - 1.5, x + off + 1.5, y + 1.5], fill=(255, 60, 60))

    if jaw_points is not None:
        jp, _ = c.project(jaw_points)
        for off in (W, 2 * W):
            draw.line([(x + off, y) for x, y in jp], fill=(255, 40, 40), width=2)
        errs = []
        for line in user_lines:
            line = np.asarray(line, float) * f
            for off in (0, 2 * W):
                draw.line([(x + off, y) for x, y in line], fill=(40, 220, 80), width=2)
            from .geometry import closest_on_polyline

            _, d = closest_on_polyline(line, jp)
            errs.append(d / f)
        if errs:
            result.jaw_line_error_px = float(np.mean(np.concatenate(errs)))
    canvas.save(out_path)
    result.image_path = out_path
    return result
