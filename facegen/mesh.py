"""Triangle mesh helpers built on numpy/scipy (no Blender needed)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .geometry import normalize


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) int64
    uvs: np.ndarray | None = None  # (N, 2) per-vertex UVs, DirectX convention (v=0 top)

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces.copy(), None if self.uvs is None else self.uvs.copy())

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    def face_normals(self) -> np.ndarray:
        return face_normals(self.vertices, self.faces)

    def vertex_normals(self) -> np.ndarray:
        return vertex_normals(self.vertices, self.faces)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


# ---------------------------------------------------------------------------
# Basic quantities


def face_normals(vertices: np.ndarray, faces: np.ndarray, unit: bool = True) -> np.ndarray:
    a = vertices[faces[:, 0]]
    n = np.cross(vertices[faces[:, 1]] - a, vertices[faces[:, 2]] - a)
    return normalize(n) if unit else n


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(face_normals(vertices, faces, unit=False), axis=1)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals."""
    fn = face_normals(vertices, faces, unit=False)
    vn = np.zeros_like(vertices, dtype=np.float64)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return normalize(vn)


def edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def adjacency(n: int, faces: np.ndarray) -> sp.csr_matrix:
    e = edges(faces)
    data = np.ones(len(e) * 2)
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def uniform_laplacian(n: int, faces: np.ndarray) -> sp.csr_matrix:
    """L = I - D^-1 A (rows sum to zero). Isolated vertices get a zero row."""
    A = adjacency(n, faces)
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0)
    eye = sp.diags((deg > 0).astype(float))
    return (eye - sp.diags(inv) @ A).tocsr()


def connected_components(n: int, faces: np.ndarray) -> np.ndarray:
    from scipy.sparse.csgraph import connected_components as cc

    _, labels = cc(adjacency(n, faces), directed=False)
    return labels


def smooth_vertex_values(values: np.ndarray, faces: np.ndarray, iterations: int = 5,
                         alpha: float = 0.5) -> np.ndarray:
    """Laplacian smoothing of any per-vertex quantity."""
    n = len(values)
    A = adjacency(n, faces)
    deg = np.asarray(A.sum(axis=1)).ravel()
    v = np.asarray(values, dtype=np.float64).copy()
    for _ in range(iterations):
        avg = (A @ v) / np.maximum(deg, 1)[:, None] if v.ndim > 1 else (A @ v) / np.maximum(deg, 1)
        mask = deg > 0
        v[mask] = (1 - alpha) * v[mask] + alpha * avg[mask]
    return v


# ---------------------------------------------------------------------------
# Welding (PES meshes duplicate vertices at UV seams and between sub-meshes)


def weld(vertices: np.ndarray, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices closer than ``tol``.

    Returns (welded_vertices, mapping) where ``welded_vertices[mapping[i]]`` is the
    representative of original vertex ``i``.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    tree = cKDTree(vertices)
    pairs = tree.query_pairs(tol, output_type="ndarray")
    parent = np.arange(len(vertices))

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(len(vertices))])
    uniq, mapping = np.unique(roots, return_inverse=True)
    return vertices[uniq], mapping


# ---------------------------------------------------------------------------
# Barycentric points and closest-point queries


def barycentric_points(vertices: np.ndarray, faces: np.ndarray, tri: np.ndarray,
                       bary: np.ndarray) -> np.ndarray:
    f = faces[np.asarray(tri, dtype=np.int64)]
    b = np.asarray(bary, dtype=np.float64)
    return (vertices[f[:, 0]] * b[:, :1] + vertices[f[:, 1]] * b[:, 1:2] + vertices[f[:, 2]] * b[:, 2:3])


def closest_point_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised closest point on triangle (Ericson). Returns (points, barycentrics)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    n = len(p)
    bary = np.zeros((n, 3))
    done = np.zeros(n, dtype=bool)

    def assign(mask, w0, w1, w2):
        m = mask & ~done
        bary[m, 0] = w0[m] if np.ndim(w0) else w0
        bary[m, 1] = w1[m] if np.ndim(w1) else w1
        bary[m, 2] = w2[m] if np.ndim(w2) else w2
        done[m] = True

    one = np.ones(n)
    zero = np.zeros(n)
    assign((d1 <= 0) & (d2 <= 0), one, zero, zero)
    assign((d3 >= 0) & (d4 <= d3), zero, one, zero)
    vc = d1 * d4 - d3 * d2
    v_ab = d1 / np.where(np.abs(d1 - d3) < 1e-30, 1e-30, d1 - d3)
    assign((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1 - v_ab, v_ab, zero)
    assign((d6 >= 0) & (d5 <= d6), zero, zero, one)
    vb = d5 * d2 - d1 * d6
    w_ac = d2 / np.where(np.abs(d2 - d6) < 1e-30, 1e-30, d2 - d6)
    assign((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1 - w_ac, zero, w_ac)
    va = d3 * d6 - d5 * d4
    denom_bc = (d4 - d3) + (d5 - d6)
    w_bc = (d4 - d3) / np.where(np.abs(denom_bc) < 1e-30, 1e-30, denom_bc)
    assign((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), zero, 1 - w_bc, w_bc)
    denom = va + vb + vc
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    v = vb / denom
    w = vc / denom
    assign(np.ones(n, dtype=bool), 1 - v - w, v, w)
    pts = a * bary[:, :1] + b * bary[:, 1:2] + c * bary[:, 2:3]
    return pts, bary


class SurfaceQuery:
    """Closest-point queries against a fixed triangle mesh."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, candidates: int = 8):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.candidates = min(candidates, len(self.faces))
        tri = self.vertices[self.faces]
        self.centroids = tri.mean(axis=1)
        # radius of each triangle around its centroid (for conservative candidate search)
        self.radius = np.max(np.linalg.norm(tri - self.centroids[:, None], axis=2), axis=1)
        self.tree = cKDTree(self.centroids)
        self.normals = face_normals(self.vertices, self.faces)

    def closest(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (closest points, distances, triangle ids, barycentrics)."""
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        k = self.candidates
        _, idx = self.tree.query(pts, k=k)
        if k == 1:
            idx = idx[:, None]
        best_d = np.full(len(pts), np.inf)
        best_p = np.zeros_like(pts)
        best_t = np.zeros(len(pts), dtype=np.int64)
        best_b = np.zeros((len(pts), 3))
        for j in range(k):
            t = idx[:, j]
            f = self.faces[t]
            cp, bary = closest_point_on_triangles(pts, self.vertices[f[:, 0]], self.vertices[f[:, 1]],
                                                  self.vertices[f[:, 2]])
            d = np.linalg.norm(cp - pts, axis=1)
            better = d < best_d
            best_d[better] = d[better]
            best_p[better] = cp[better]
            best_t[better] = t[better]
            best_b[better] = bary[better]
        return best_p, best_d, best_t, best_b


def sample_surface(vertices: np.ndarray, faces: np.ndarray, count: int,
                   rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-weighted random surface samples. Returns (points, triangle ids, barycentrics)."""
    rng = rng or np.random.default_rng(0)
    areas = face_areas(vertices, faces)
    prob = areas / max(areas.sum(), 1e-18)
    tri = rng.choice(len(faces), size=count, p=prob)
    r1 = np.sqrt(rng.random(count))
    r2 = rng.random(count)
    bary = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], axis=1)
    return barycentric_points(vertices, faces, tri, bary), tri, bary


# ---------------------------------------------------------------------------
# Displacement-field transfer (moves cards, hair, eyes along with the skin)


def interpolate_displacement(src_points: np.ndarray, src_disp: np.ndarray, query: np.ndarray,
                             k: int = 8, power: float = 2.0) -> np.ndarray:
    """Inverse-distance-weighted displacement at ``query`` points."""
    tree = cKDTree(src_points)
    k = min(k, len(src_points))
    d, idx = tree.query(query, k=k)
    if k == 1:
        d = d[:, None]
        idx = idx[:, None]
    w = 1.0 / np.maximum(d, 1e-9) ** power
    exact = d[:, 0] < 1e-9
    out = np.einsum("nk,nkd->nd", w, src_disp[idx]) / w.sum(axis=1, keepdims=True)
    out[exact] = src_disp[idx[exact, 0]]
    return out


# ---------------------------------------------------------------------------
# OBJ I/O (used for scans, previews and Blender hand-off)


def write_obj(path: str | Path, mesh: Mesh, mtl_texture: str | None = None,
              flip_v: bool = True) -> None:
    """Write an OBJ. UVs are written with v flipped to OBJ's bottom-left origin."""
    path = Path(path)
    lines = []
    if mtl_texture:
        mtl = path.with_suffix(".mtl")
        mtl.write_text(f"newmtl skin\nKd 1 1 1\nmap_Kd {mtl_texture}\n")
        lines.append(f"mtllib {mtl.name}")
    lines.extend(f"v {x:.7f} {y:.7f} {z:.7f}" for x, y, z in mesh.vertices)
    if mesh.uvs is not None:
        for u, v in mesh.uvs:
            lines.append(f"vt {u:.7f} {1.0 - v if flip_v else v:.7f}")
    if mtl_texture:
        lines.append("usemtl skin")
    f1 = mesh.faces + 1
    if mesh.uvs is not None:
        lines.extend(f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in f1)
    else:
        lines.extend(f"f {a} {b} {c}" for a, b, c in f1)
    path.write_text("\n".join(lines) + "\n")


def read_obj(path: str | Path, flip_v: bool = True) -> Mesh:
    """Read an OBJ (triangulating polygons). Per-corner UVs are collapsed by splitting
    vertices so the result has per-vertex UVs."""
    verts, uvs, faces_v, faces_t = [], [], [], []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                parts = line.split()[1:3]
                uvs.append([float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0])
            elif line.startswith("f "):
                items = line.split()[1:]
                vi, ti = [], []
                for it in items:
                    s = it.split("/")
                    vi.append(int(s[0]))
                    ti.append(int(s[1]) if len(s) > 1 and s[1] else 0)
                for j in range(1, len(vi) - 1):
                    faces_v.append([vi[0], vi[j], vi[j + 1]])
                    faces_t.append([ti[0], ti[j], ti[j + 1]])
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces_v, dtype=np.int64)
    F = np.where(F < 0, F + len(V) + 1, F) - 1
    if not uvs or not faces_t or np.all(np.asarray(faces_t) == 0):
        return Mesh(V, F, None)
    T = np.asarray(faces_t, dtype=np.int64)
    UV = np.asarray(uvs, dtype=np.float64)
    T = np.where(T < 0, T + len(UV) + 1, T) - 1
    if flip_v:
        UV = UV.copy()
        UV[:, 1] = 1.0 - UV[:, 1]
    pairs = np.stack([F.ravel(), T.ravel()], axis=1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    inv = inv.ravel()
    return Mesh(V[uniq[:, 0]], inv.reshape(-1, 3), UV[np.clip(uniq[:, 1], 0, len(UV) - 1)])


def largest_component(mesh: Mesh) -> Mesh:
    labels = connected_components(mesh.n_vertices, mesh.faces)
    counts = np.bincount(labels[mesh.faces[:, 0]])
    keep_label = int(np.argmax(counts))
    keep_faces = mesh.faces[labels[mesh.faces[:, 0]] == keep_label]
    used = np.unique(keep_faces)
    remap = -np.ones(mesh.n_vertices, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return Mesh(mesh.vertices[used], remap[keep_faces], None if mesh.uvs is None else mesh.uvs[used])
