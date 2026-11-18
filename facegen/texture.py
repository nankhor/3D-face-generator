"""Bake the person's skin (and beard) from the photos into the template's UV layout."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .geometry import Camera, normalize
from .raster import interpolate, rasterize, rasterize_uv, sample_bilinear


@dataclass
class BakeSettings:
    max_photo_side: int = 2400
    depth_resolution: int = 900
    facing_power: float = 3.0
    blend_power: float = 2.0
    min_facing: float = 0.15
    depth_tolerance: float = 0.004  # metres
    border_fraction: float = 0.06
    exposure_match: bool = True
    shadow_flatten: float = 0.0  # 0 = keep photo shading, 1 = flatten low-frequency light
    feather_texels: int = 24


@dataclass
class BakeResult:
    texture: np.ndarray  # (H, W, 4) uint8
    coverage: np.ndarray  # (H, W) float, photo coverage weight
    per_photo_coverage: dict[str, float] = field(default_factory=dict)
    gains: dict[str, list[float]] = field(default_factory=dict)


def bake_texture(uv_vertices: np.ndarray, uv_faces: np.ndarray, uvs: np.ndarray,
                 normals: np.ndarray, photos: dict[str, np.ndarray], cameras: dict[str, Camera],
                 template: np.ndarray, settings: BakeSettings | None = None,
                 occluder: tuple[np.ndarray, np.ndarray] | None = None) -> BakeResult:
    """Project the photos onto the head and bake them into UV space.

    ``uv_vertices``/``uv_faces``/``uvs``: the skin mesh with per-vertex UVs (positions
    already fitted). ``normals``: per-vertex normals. ``template``: the template's base
    colour texture (H, W, 3|4), used for size, alpha and to fill unseen areas.
    ``occluder``: (vertices, faces) used for the visibility test (defaults to the skin).
    """
    s = settings or BakeSettings()
    H, W = template.shape[:2]
    frag = rasterize_uv(uvs, uv_faces, W, H)
    inside = frag.mask
    pos = interpolate(frag, uv_faces, uv_vertices)
    nrm = normalize(interpolate(frag, uv_faces, normals))
    texel_pos = pos[inside]
    texel_nrm = nrm[inside]
    occ_v, occ_f = occluder if occluder is not None else (uv_vertices, uv_faces)

    acc = np.zeros((len(texel_pos), 3))
    wsum = np.zeros(len(texel_pos))
    samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    densities: dict[str, np.ndarray] = {}
    for name, photo in photos.items():
        cam = cameras.get(name)
        if cam is None:
            continue
        img = photo
        f = min(1.0, s.max_photo_side / max(img.shape[:2]))
        cam_img = cam.scaled(img.shape[1] / cam.width * f) if f < 1.0 or img.shape[1] != cam.width else cam
        if f < 1.0:
            from PIL import Image

            img = np.asarray(Image.fromarray(img).resize((cam_img.width, cam_img.height), Image.LANCZOS))
        dcam = cam_img.scaled(min(1.0, s.depth_resolution / max(cam_img.width, cam_img.height)))
        dfrag = rasterize(dcam, occ_v, occ_f)
        px, z = cam_img.project(texel_pos)
        dpx, _ = dcam.project(texel_pos)
        xi = np.round(dpx[:, 0]).astype(np.int64)
        yi = np.round(dpx[:, 1]).astype(np.int64)
        inb = (xi >= 0) & (yi >= 0) & (xi < dcam.width) & (yi < dcam.height) & (z > 0)
        depth = np.full(len(texel_pos), np.inf)
        depth[inb] = dfrag.depth[yi[inb], xi[inb]]
        visible = inb & (z <= depth + s.depth_tolerance)
        view = normalize(cam.center[None] - texel_pos)
        facing = np.einsum("ij,ij->i", texel_nrm, view)
        visible &= facing > s.min_facing
        # fade near the image border
        bx = np.minimum(px[:, 0], cam_img.width - 1 - px[:, 0]) / (s.border_fraction * cam_img.width)
        by = np.minimum(px[:, 1], cam_img.height - 1 - px[:, 1]) / (s.border_fraction * cam_img.height)
        border = np.clip(np.minimum(bx, by), 0, 1)
        # pixels per metre on the skin (photos that see more detail are preferred later)
        density = cam_img.focal / np.maximum(z, 1e-6)
        w = np.where(visible, np.clip(facing, 0, 1) ** s.facing_power * border, 0.0)
        if not (w > 0).any():
            continue
        col = np.zeros((len(texel_pos), 3))
        col[w > 0] = sample_bilinear(img, px[w > 0, 0], px[w > 0, 1])[:, :3]
        samples[name] = (col, w)
        densities[name] = density

    if densities:
        ref_density = np.median(np.concatenate([d[samples[k][1] > 0] for k, d in densities.items()]))
        for k in samples:
            col, w = samples[k]
            samples[k] = (col, w * np.clip(densities[k] / max(ref_density, 1e-9), 0.25, 2.0))

    gains: dict[str, list[float]] = {}
    if s.exposure_match and len(samples) > 1:
        ref = max(samples, key=lambda k: float(samples[k][1].sum()))
        rc, rw = samples[ref]
        for name, (col, w) in samples.items():
            if name == ref:
                gains[name] = [1.0, 1.0, 1.0]
                continue
            both = (w > 0.2) & (rw > 0.2)
            if both.sum() < 200:
                gains[name] = [1.0, 1.0, 1.0]
                continue
            g = np.median(rc[both] + 1.0, axis=0) / np.median(col[both] + 1.0, axis=0)
            g = np.clip(g, 0.6, 1.6)
            gains[name] = [round(float(x), 3) for x in g]
            samples[name] = (np.clip(col * g, 0, 255), w)

    per_cov = {}
    for name, (col, w) in samples.items():
        wp = w ** s.blend_power
        acc += col * wp[:, None]
        wsum += wp
        per_cov[name] = round(float((w > 0.1).mean()), 3)

    tex = template[..., :3].astype(np.float64)
    if tex.max() <= 1.5:
        tex = tex * 255.0
    out = tex.copy()
    coverage = np.zeros((H, W))
    have = wsum > 1e-9
    baked = np.zeros((len(texel_pos), 3))
    baked[have] = acc[have] / wsum[have, None]
    cov_vals = np.clip(wsum / (np.percentile(wsum[have], 50) if have.any() else 1.0), 0, 1)
    coverage[inside] = np.where(have, cov_vals, 0)

    baked_img = np.zeros((H, W, 3))
    baked_img[inside] = baked
    if have.any() and s.shadow_flatten > 0:
        baked_img = _flatten_shading(baked_img, coverage > 0.05, s.shadow_flatten)

    # recolour the template to the person's skin tone where the photos did not reach
    covered = coverage > 0.3
    if covered.sum() > 100:
        src = tex[inside & covered]
        dst = baked_img[inside & covered]
        mu_s, sd_s = src.mean(axis=0), src.std(axis=0) + 1e-6
        mu_d, sd_d = dst.mean(axis=0), dst.std(axis=0) + 1e-6
        recolored = np.clip((tex - mu_s) / sd_s * np.minimum(sd_d, sd_s * 1.5) + mu_d, 0, 255)
    else:
        recolored = tex
    # feathered blend between baked photo colour and recoloured template
    dist_in = ndimage.distance_transform_edt(coverage > 0.05)
    alpha = np.clip(dist_in / max(s.feather_texels, 1), 0, 1) * np.clip(coverage * 2, 0, 1)
    out = recolored * (1 - alpha[..., None]) + baked_img * alpha[..., None]
    # pad outside the UV islands so mip-maps do not bleed in black
    _, (iy, ix) = ndimage.distance_transform_edt(~inside, return_indices=True)
    out = out[iy, ix]
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(out, 0, 255).astype(np.uint8)
    if template.shape[2] == 4:
        a = template[..., 3]
        rgba[..., 3] = a if a.dtype == np.uint8 else np.clip(a * (255 if a.max() <= 1.5 else 1), 0, 255).astype(np.uint8)
    else:
        rgba[..., 3] = 255
    return BakeResult(rgba, coverage, per_cov, gains)


def _flatten_shading(img: np.ndarray, mask: np.ndarray, strength: float, sigma: float = 0.03) -> np.ndarray:
    """Reduce baked-in lighting: divide out low-frequency luminance variation."""
    H, W = mask.shape
    s = sigma * max(H, W)
    lum = img.mean(axis=2)
    m = mask.astype(float)
    num = ndimage.gaussian_filter(lum * m, s)
    den = ndimage.gaussian_filter(m, s)
    low = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0)
    target = np.median(lum[mask]) if mask.any() else 1.0
    ratio = np.where(mask & (low > 1), target / np.maximum(low, 1), 1.0)
    ratio = 1.0 + strength * (np.clip(ratio, 0.6, 1.6) - 1.0)
    return np.clip(img * ratio[..., None], 0, 255)
