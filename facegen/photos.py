"""Photo loading with EXIF orientation and focal-length handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp", ".tif", ".tiff"}


def list_photos(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_EXTENSIONS and p.is_file())


def _open(path: str | Path) -> Image.Image:
    path = Path(path)
    if path.suffix.lower() == ".heic":
        try:
            import pillow_heif  # optional

            pillow_heif.register_heif_opener()
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("HEIC photos need the optional 'pillow-heif' package, or export "
                               "the photos as JPG") from exc
    img = Image.open(path)
    return ImageOps.exif_transpose(img)


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(_open(path).convert("RGB"))


def photo_size(path: str | Path) -> tuple[int, int]:
    img = _open(path)
    return img.size  # (w, h) after EXIF rotation


def focal_35mm(path: str | Path) -> float | None:
    """35 mm-equivalent focal length from EXIF, if the phone wrote it."""
    try:
        img = Image.open(path)
        exif = img.getexif()
        sub = exif.get_ifd(0x8769)  # Exif IFD
        f35 = sub.get(0xA405) or exif.get(0xA405)  # FocalLengthIn35mmFilm
        if f35:
            return float(f35)
    except Exception:
        return None
    return None


def default_focal_px(width: int, height: int, path: str | Path | None = None) -> float:
    """Best focal-length guess in pixels: EXIF, else a typical phone main camera (26 mm eq.)."""
    from .geometry import focal_from_35mm

    f35 = focal_35mm(path) if path is not None else None
    return focal_from_35mm(f35 or 26.0, width, height)


def save_rgb(path: str | Path, img: np.ndarray) -> None:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.5 else 1.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def thumbnail(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s >= 1.0:
        return img
    return np.asarray(Image.fromarray(img).resize((round(w * s), round(h * s)), Image.LANCZOS))
