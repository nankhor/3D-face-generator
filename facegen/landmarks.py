"""Face landmarks (MediaPipe Face Landmarker, 478 points) and landmark groups.

MediaPipe needs a model file (``face_landmarker.task``) that the installer
downloads. Everything else in this module is plain numpy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"

# Ordered silhouette of the face (forehead -> subject's left -> chin -> right).
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54,
             103, 67, 109]
# Lower part of the oval: hidden or distorted by a beard.
LOWER_OVAL = [361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132]

NOSE_TIP = 4
NOSE_BRIDGE = [168, 6, 197, 195, 5]
NOSE_BASE = [2, 98, 327, 94]
EYE_R_OUTER, EYE_R_INNER, EYE_L_INNER, EYE_L_OUTER = 33, 133, 362, 263
MOUTH_R, MOUTH_L = 61, 291
UPPER_LIP, LOWER_LIP = 13, 14
CHIN = 152
FOREHEAD = 10
BROW_R = [70, 63, 105, 66, 107]
BROW_L = [300, 293, 334, 296, 336]
CHEEKBONE = [116, 123, 50, 345, 352, 280]
TEMPLE = [127, 162, 356, 389]
EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39,
              40, 185]

# Landmarks that do not move with expression and are not covered by a beard.
# Used for camera solving and rigid alignment.
STABLE = sorted(set([NOSE_TIP, *NOSE_BRIDGE, *NOSE_BASE, EYE_R_OUTER, EYE_R_INNER, EYE_L_INNER,
                     EYE_L_OUTER, *BROW_R, *BROW_L, *CHEEKBONE, *TEMPLE, FOREHEAD, 151, 9, 8,
                     MOUTH_R, MOUTH_L, 0]))

# Named anchors a person can click in the line editor (useful on profile photos where
# automatic detection fails). Anchors with a landmark index are tied to MediaPipe points;
# the others (ear) are triangulated from the views where they are clicked.
ANCHORS = {
    "nose_tip": NOSE_TIP,
    "eye_outer_right": EYE_R_OUTER,
    "eye_outer_left": EYE_L_OUTER,
    "mouth_corner_right": MOUTH_R,
    "mouth_corner_left": MOUTH_L,
    "ear_right": None,
    "ear_left": None,
}

# Landmark groups used to define regions on the head and for region adjustments.
GROUPS = {
    "nose": [NOSE_TIP, *NOSE_BRIDGE, *NOSE_BASE, 48, 278, 64, 294, 129, 358],
    "eyes": EYE_R + EYE_L,
    "brows": BROW_R + BROW_L,
    "mouth": LIPS_OUTER,
    "cheeks": CHEEKBONE + [205, 425, 187, 411],
    "jaw": LOWER_OVAL,
    "chin": [152, 148, 377, 176, 400, 199, 175],
    "forehead": [10, 151, 9, 67, 297, 109, 338, 108, 337],
}


@lru_cache(maxsize=1)
def canonical_face() -> tuple[np.ndarray, np.ndarray]:
    """MediaPipe canonical face model (468 vertices) in metres, head space axes.

    Source: mediapipe/modules/face_geometry/data/canonical_face_model.obj (Apache-2.0).
    """
    verts, faces = [], []
    for line in (DATA_DIR / "canonical_face_model.obj").read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(p.split("/")[0]) - 1 for p in line.split()[1:4]])
    return np.asarray(verts) / 100.0, np.asarray(faces, dtype=np.int64)


@dataclass
class PhotoLandmarks:
    """Landmarks detected on one photo, in pixel coordinates."""

    photo: str
    width: int
    height: int
    points: np.ndarray | None  # (478, 2) or None when no face was found
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "photo": self.photo,
            "width": self.width,
            "height": self.height,
            "points": None if self.points is None else np.round(self.points, 3).tolist(),
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PhotoLandmarks":
        pts = d.get("points")
        return cls(d["photo"], int(d["width"]), int(d["height"]),
                   None if pts is None else np.asarray(pts, dtype=np.float64), float(d.get("score", 0)))


def save_landmarks(path: Path, items: list[PhotoLandmarks]) -> None:
    path.write_text(json.dumps([i.to_dict() for i in items]))


def load_landmarks(path: Path) -> list[PhotoLandmarks]:
    if not path.exists():
        return []
    return [PhotoLandmarks.from_dict(d) for d in json.loads(path.read_text())]


class LandmarkDetector:
    """Thin wrapper over the MediaPipe Tasks Face Landmarker."""

    def __init__(self, model_path: str | Path, min_confidence: float = 0.3):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {model_path}. Run the installer, or download "
                "face_landmarker.task from https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        import mediapipe as mp  # imported lazily: heavy

        base = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            output_face_blendshapes=False,
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def detect_array(self, rgb: np.ndarray) -> np.ndarray | None:
        """Detect on an RGB uint8 array. Returns (478, 2) pixel coords or None."""
        rgb = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        h, w = rgb.shape[:2]
        lm = result.face_landmarks[0]
        return np.array([[p.x * w - 0.5, p.y * h - 0.5] for p in lm], dtype=np.float64)

    def detect_photo(self, path: str | Path, max_side: int = 1600) -> PhotoLandmarks:
        """Detect on a photo file (downscaled for speed, coordinates in full resolution)."""
        from .photos import load_rgb

        img = load_rgb(path)
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        small = img
        if scale < 1.0:
            from PIL import Image

            small = np.asarray(Image.fromarray(img).resize((round(w * scale), round(h * scale)),
                                                            Image.LANCZOS))
        pts = self.detect_array(small)
        if pts is not None:
            pts = (pts + 0.5) / scale - 0.5
        return PhotoLandmarks(Path(path).name, w, h, pts, 1.0 if pts is not None else 0.0)


def interocular(points: np.ndarray) -> float:
    return float(np.linalg.norm(points[EYE_R_OUTER] - points[EYE_L_OUTER]))
