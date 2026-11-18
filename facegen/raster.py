"""A small z-buffer triangle rasteriser in numpy.

Used for visibility tests, texture baking, landmark annotation of meshes and the
quick preview renders that Claude compares against the photos. Blender is used
for nicer renders, but nothing in the core pipeline depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Camera, normalize
from .mesh import vertex_normals


@dataclass
class Fragments:
    depth: np.ndarray  # (H, W) float, inf where empty
    tri: np.ndarray  # (H, W) int, -1 where empty
    bary: np.ndarray  # (H, W, 3)

    @property
    def mask(self) -> np.ndarray:
        return self.tri >= 0


def _raster_triangles(xy: np.ndarray, z: np.ndarray | None, faces: np.ndarray, width: int,
                      height: int, pixel_offset: float) -> Fragments:
    depth = np.full((height, width), np.inf)
    tri_map = np.full((height, width), -1, dtype=np.int64)
    bary_map = np.zeros((height, width, 3))
    p = xy[faces]  # (M, 3, 2)
    zz = None if z is None else z[faces]
    xmin = np.floor(p[:, :, 0].min(axis=1) - pixel_offset).astype(np.int64)
    xmax = np.ceil(p[:, :, 0].max(axis=1) - pixel_offset).astype(np.int64)
    ymin = np.floor(p[:, :, 1].min(axis=1) - pixel_offset).astype(np.int64)
    ymax = np.ceil(p[:, :, 1].max(axis=1) - pixel_offset).astype(np.int64)
    xmin = np.clip(xmin, 0, width - 1)
    xmax = np.clip(xmax, 0, width - 1)
    ymin = np.clip(ymin, 0, height - 1)
    ymax = np.clip(ymax, 0, height - 1)
    valid = (xmax >= xmin) & (ymax >= ymin)
    if zz is not None:
        valid &= np.all(zz > 1e-6, axis=1)
    for t in np.nonzero(valid)[0]:
        (x0, y0), (x1, y1), (x2, y2) = p[t]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-14:
            continue
        xs = np.arange(xmin[t], xmax[t] + 1) + pixel_offset
        ys = np.arange(ymin[t], ymax[t] + 1) + pixel_offset
        gx, gy = np.meshgrid(xs, ys)
        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        sl = (slice(ymin[t], ymax[t] + 1), slice(xmin[t], xmax[t] + 1))
        if zz is not None:
            # perspective-correct depth: interpolate 1/z
            inv = w0 / zz[t, 0] + w1 / zz[t, 1] + w2 / zz[t, 2]
            dep = 1.0 / np.maximum(inv, 1e-12)
            # perspective-correct barycentrics
            b0 = w0 / zz[t, 0] * dep
            b1 = w1 / zz[t, 1] * dep
            b2 = w2 / zz[t, 2] * dep
            closer = inside & (dep < depth[sl])
        else:
            dep = np.zeros_like(w0)
            b0, b1, b2 = w0, w1, w2
            closer = inside & (tri_map[sl] < 0)
        if not closer.any():
            continue
        depth[sl][closer] = dep[closer]
        tri_map[sl][closer] = t
        bb = bary_map[sl]
        bb[..., 0][closer] = b0[closer]
        bb[..., 1][closer] = b1[closer]
        bb[..., 2][closer] = b2[closer]
    return Fragments(depth, tri_map, bary_map)


def rasterize(cam: Camera, vertices: np.ndarray, faces: np.ndarray) -> Fragments:
    """Rasterise a mesh seen by ``cam``. Pixel centres are at integer coordinates."""
    uv, z = cam.project(vertices)
    return _raster_triangles(uv, z, np.asarray(faces, np.int64), cam.width, cam.height, 0.0)


def rasterize_uv(uvs: np.ndarray, faces: np.ndarray, width: int, height: int) -> Fragments:
    """Rasterise UV space. Texel (row i, col j) has its centre at ((j+.5)/W, (i+.5)/H)."""
    xy = np.asarray(uvs, dtype=np.float64) * np.array([width, height])
    return _raster_triangles(xy, None, np.asarray(faces, np.int64), width, height, 0.5)


def interpolate(frag: Fragments, faces: np.ndarray, attr: np.ndarray) -> np.ndarray:
    """Interpolate a per-vertex attribute over covered pixels (zeros elsewhere)."""
    attr = np.asarray(attr, dtype=np.float64)
    out_shape = frag.tri.shape + attr.shape[1:]
    out = np.zeros(out_shape)
    m = frag.mask
    if not m.any():
        return out
    f = faces[frag.tri[m]]
    b = frag.bary[m]
    vals = sum(attr[f[:, k]] * (b[:, k:k + 1] if attr.ndim > 1 else b[:, k]) for k in range(3))
    out[m] = vals
    return out


def sample_bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample of ``img`` (H, W[, C]) at continuous pixel coords (centres at ints)."""
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = (x - x0)
    fy = (y - y0)
    if img.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    im = img.astype(np.float64)
    top = im[y0, x0] * (1 - fx) + im[y0, x1] * fx
    bot = im[y1, x0] * (1 - fx) + im[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def sample_texture_uv(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample a texture at UV coordinates (v=0 is the top row)."""
    h, w = tex.shape[:2]
    return sample_bilinear(tex, uv[..., 0] * w - 0.5, uv[..., 1] * h - 0.5)


def render(cam: Camera, vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray | None = None,
           texture: np.ndarray | None = None, background: float = 0.0,
           light_dir: np.ndarray | None = None) -> tuple[np.ndarray, Fragments]:
    """Quick preview render. Textured and unlit when a texture is given, else clay-shaded.

    Returns an (H, W, 3) float image in [0, 1] and the fragments.
    """
    frag = rasterize(cam, vertices, faces)
    img = np.full((cam.height, cam.width, 3), float(background))
    m = frag.mask
    if not m.any():
        return img, frag
    if texture is not None and uvs is not None:
        uv = interpolate(frag, faces, uvs)
        tex = texture[..., :3].astype(np.float64)
        if tex.max() > 1.5:
            tex = tex / 255.0
        img[m] = sample_texture_uv(tex, uv[m])
    else:
        n = interpolate(frag, faces, vertex_normals(vertices, faces))
        n = normalize(n)
        if light_dir is None:
            # light from slightly above the camera
            light_dir = -cam.forward + 0.4 * (cam.R.T @ np.array([0, -1.0, 0]))
        L = normalize(np.asarray(light_dir, float))
        lam = np.clip(np.einsum("ij,j->i", n[m], L), 0, 1)
        shade = 0.15 + 0.85 * lam
        img[m] = np.stack([shade * 0.86, shade * 0.80, shade * 0.76], axis=1)
    return np.clip(img, 0, 1), frag


def visibility_depth_test(cam: Camera, frag: Fragments, points: np.ndarray,
                          tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Which world points are visible in ``frag`` (rendered from ``cam``).

    Returns (visible mask, pixel coordinates).
    """
    px, z = cam.project(points)
    xi = np.round(px[:, 0]).astype(np.int64)
    yi = np.round(px[:, 1]).astype(np.int64)
    inside = (xi >= 0) & (yi >= 0) & (xi < cam.width) & (yi < cam.height) & (z > 0)
    vis = np.zeros(len(points), dtype=bool)
    d = np.full(len(points), np.inf)
    d[inside] = frag.depth[yi[inside], xi[inside]]
    vis[inside] = z[inside] <= d[inside] + tolerance
    return vis, px
