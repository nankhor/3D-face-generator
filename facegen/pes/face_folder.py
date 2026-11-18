"""PES 2021 / Football Life face folders.

A face folder is named after the player ID::

    <ID>/
      #Win/face.fpk             face_high.fmdl, hair_high.fmdl, oral.fmdl, face_diff.bin
      sourceimages/#windx11/    face_bsm_alp.ftex, face_nrm.ftex, hair_parts_*.ftex, ...

The template face is morphed into the person's face: vertex positions (and
normals) change, the base colour texture is replaced, eye positions in
``face_diff.bin`` follow the eye sockets, and the ID is relinked. Everything
else is copied unchanged.
"""

from __future__ import annotations

import json
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..geometry import rotation_between
from ..head import HeadModel
from ..mesh import interpolate_displacement, vertex_normals, weld
from . import ftex
from .fmdl import Fmdl
from .fpk import FpkFile

EYE_R_OFFSET, EYE_L_OFFSET, MOUTH_OFFSET = 0x150, 0x160, 0x3C


def _find_ci(folder: Path, *parts: str) -> Path | None:
    """Case-insensitive path lookup."""
    cur = folder
    for part in parts:
        if not cur.is_dir():
            return None
        match = [p for p in cur.iterdir() if p.name.lower() == part.lower()]
        if not match:
            return None
        cur = match[0]
    return cur


def read_diff_eyes(buf: bytes) -> dict[str, np.ndarray]:
    """Eye (and mouth) positions from face_diff.bin, in head space (fox axes)."""
    if buf[:4] != b"FACE":
        return {}
    out = {}
    for name, off in (("eye_right", EYE_R_OFFSET), ("eye_left", EYE_L_OFFSET)):
        if len(buf) >= off + 12:
            fy, fz, fx = struct.unpack_from("<3f", buf, off)
            out[name] = np.array([fx, fy, fz])
    if len(buf) >= MOUTH_OFFSET + 12:
        fx, fy, fz = struct.unpack_from("<3f", buf, MOUTH_OFFSET)
        out["mouth"] = np.array([fx, fy, fz])
    return out


def write_diff_eyes(buf: bytes, points: dict[str, np.ndarray]) -> bytes:
    out = bytearray(buf)
    for name, off in (("eye_right", EYE_R_OFFSET), ("eye_left", EYE_L_OFFSET)):
        if name in points:
            x, y, z = points[name]
            struct.pack_into("<3f", out, off, y, z, x)
    if "mouth" in points:
        struct.pack_into("<3f", out, MOUTH_OFFSET, *points["mouth"])
    return bytes(out)


@dataclass
class SkinMap:
    """How the welded skin maps back to the fmdl meshes."""

    mesh_indices: list[int]
    offsets: list[int]  # start of each mesh in the concatenated vertex list
    mapping: np.ndarray  # concatenated vertex -> welded vertex


@dataclass
class TemplateFace:
    folder: Path
    face_id: str
    fpk_path: Path
    fpk: FpkFile
    face: Fmdl
    hair: Fmdl | None
    oral: Fmdl | None
    diff: bytes | None
    texture_dir: Path | None
    skin_meshes: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------
    @classmethod
    def open(cls, folder: str | Path) -> "TemplateFace":
        folder = Path(folder)
        fpk_path = _find_ci(folder, "#Win", "face.fpk")
        if fpk_path is None:
            raise FileNotFoundError(f"{folder} has no #Win/face.fpk. Pick a PES 2021 face folder "
                                    "(the one named after the player ID).")
        fpk = FpkFile.read(fpk_path)
        face_e = fpk.find("face_high.fmdl")
        if face_e is None:
            raise FileNotFoundError("face.fpk does not contain face_high.fmdl")
        face = Fmdl.parse(face_e.data)
        hair_e = fpk.find("hair_high.fmdl")
        oral_e = fpk.find("oral.fmdl")
        diff_e = fpk.find("face_diff.bin")
        tex_dir = _find_ci(folder, "sourceimages", "#windx11")
        tf = cls(folder, folder.name, fpk_path, fpk, face, Fmdl.parse(hair_e.data) if hair_e else None,
                 Fmdl.parse(oral_e.data) if oral_e else None, diff_e.data if diff_e else None, tex_dir)
        tf.skin_meshes = tf._find_skin_meshes()
        return tf

    def _find_skin_meshes(self) -> list[int]:
        skin = []
        for m in self.face.meshes:
            bt = m.base_texture()
            if bt and bt[0].lower().startswith("face_bsm") and m.uv is not None:
                skin.append(m.index)
        if skin:
            return skin
        # fallback: the mesh(es) with the largest area
        areas = []
        for m in self.face.meshes:
            v, f = m.positions, m.faces
            if len(f) == 0:
                areas.append(0.0)
                continue
            a = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum()
            areas.append(a)
        best = int(np.argmax(areas))
        return [i for i, a in enumerate(areas) if a >= 0.25 * areas[best] and self.face.meshes[i].uv is not None]

    # ------------------------------------------------------------------
    def base_texture_path(self) -> Path | None:
        if self.texture_dir is None:
            return None
        for name in ("face_bsm_alp.ftex", "face_bsm.ftex"):
            p = _find_ci(self.texture_dir, name)
            if p:
                return p
        for m in self.skin_meshes:
            bt = self.face.meshes[m].base_texture()
            if bt:
                p = _find_ci(self.texture_dir, Path(bt[0]).with_suffix(".ftex").name)
                if p:
                    return p
        return None

    def base_texture(self) -> np.ndarray | None:
        p = self.base_texture_path()
        if p is None:
            return None
        return ftex.decode(p.read_bytes())

    def skin(self, weld_tol: float = 1e-5) -> tuple[HeadModel, SkinMap, np.ndarray, np.ndarray, np.ndarray]:
        """Welded skin head + map, and the unwelded skin (vertices, faces, uvs) for texturing."""
        verts, faces, uvs, offsets = [], [], [], []
        n = 0
        for mi in self.skin_meshes:
            m = self.face.meshes[mi]
            offsets.append(n)
            verts.append(m.positions)
            faces.append(m.faces + n)
            uvs.append(m.uv)
            n += m.vertex_count
        V = np.concatenate(verts)
        F = np.concatenate(faces)
        UV = np.concatenate(uvs)
        W, mapping = weld(V, weld_tol)
        WF = mapping[F]
        WF = WF[(WF[:, 0] != WF[:, 1]) & (WF[:, 1] != WF[:, 2]) & (WF[:, 0] != WF[:, 2])]
        return HeadModel(W, WF), SkinMap(self.skin_meshes, offsets, mapping), V, F, UV

    def eyes(self) -> dict[str, np.ndarray]:
        return read_diff_eyes(self.diff) if self.diff else {}

    def summary(self) -> dict:
        def meshes(f: Fmdl | None):
            if f is None:
                return []
            return [{"index": m.index, "vertices": m.vertex_count, "triangles": len(m.faces),
                     "material": m.material, "base_texture": (m.base_texture() or ("", ""))[0]}
                    for m in f.meshes]

        return {
            "folder": str(self.folder), "id": self.face_id,
            "fpk_entries": [e.path for e in self.fpk.entries],
            "face_high": meshes(self.face), "hair_high": meshes(self.hair), "oral": meshes(self.oral),
            "skin_meshes": self.skin_meshes,
            "textures": sorted(p.name for p in self.texture_dir.iterdir()) if self.texture_dir else [],
            "eyes": {k: np.round(v, 4).tolist() for k, v in self.eyes().items()},
        }


# ---------------------------------------------------------------------------
# Export


@dataclass
class ExportResult:
    folder: Path
    face_id: str
    notes: list[str]


def _rotations_for(points_before: np.ndarray, src_points: np.ndarray, src_rot: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    _, idx = cKDTree(src_points).query(points_before, k=1)
    return src_rot[idx]


def _morph_fmdl(fmdl: Fmdl, skin_before: np.ndarray, skin_disp: np.ndarray, skin_rot: np.ndarray,
                skip: set[int] = frozenset()) -> float:
    """Move every mesh of ``fmdl`` (except ``skip``) with the skin's displacement field."""
    max_disp = 0.0
    for m in fmdl.meshes:
        if m.index in skip:
            continue
        d = interpolate_displacement(skin_before, skin_disp, m.positions)
        rot = _rotations_for(m.positions, skin_before, skin_rot)
        fmdl.set_positions(m.index, m.positions + d)
        if m.normals is not None:
            n = m.normals.copy()
            n[:, :3] = np.einsum("nij,nj->ni", rot, n[:, :3])
            fmdl.set_normals(m.index, n)
        if m.tangents is not None:
            t = m.tangents.copy()
            t[:, :3] = np.einsum("nij,nj->ni", rot, t[:, :3])
            fmdl.set_tangents(m.index, t)
        if len(d):
            max_disp = max(max_disp, float(np.linalg.norm(d, axis=1).max()))
    return max_disp


def export_face(template: TemplateFace, head: HeadModel, skin_map: SkinMap, fitted: np.ndarray,
                out_root: str | Path, new_id: str, texture_rgba: np.ndarray | None = None,
                hair_from: "TemplateFace | None" = None, hair_fitted_disp=None) -> ExportResult:
    """Write the morphed face folder ``<out_root>/<new_id>``.

    ``fitted`` are the new welded skin positions (same order as ``head.vertices``).
    ``hair_from``: take hair_high.fmdl and hair textures from another face; its hair is
    moved by ``hair_fitted_disp`` (a callable mapping points -> displacement) if given.
    """
    notes = []
    out = Path(out_root) / str(new_id)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(template.folder, out)
    old_id = template.face_id

    # fresh parse so the template object stays untouched
    face = Fmdl.parse(template.fpk.find("face_high.fmdl").data)
    before = head.vertices
    disp = fitted - before
    n_before = vertex_normals(before, head.faces)
    n_after = vertex_normals(fitted, head.faces)
    rot = rotation_between(n_before, n_after)

    # skin meshes: exact welded positions (keeps seams closed)
    for k, mi in enumerate(skin_map.mesh_indices):
        m = face.meshes[mi]
        start = skin_map.offsets[k]
        idx = skin_map.mapping[start: start + m.vertex_count]
        face.set_positions(mi, fitted[idx])
        if m.normals is not None:
            nn = m.normals.copy()
            nn[:, :3] = np.einsum("nij,nj->ni", rot[idx], nn[:, :3])
            face.set_normals(mi, nn)
        if m.tangents is not None:
            tt = m.tangents.copy()
            tt[:, :3] = np.einsum("nij,nj->ni", rot[idx], tt[:, :3])
            face.set_tangents(mi, tt)
    max_d = float(np.linalg.norm(disp, axis=1).max()) if len(disp) else 0.0
    max_d = max(max_d, _morph_fmdl(face, before, disp, rot, skip=set(skin_map.mesh_indices)))

    fpk = FpkFile.from_bytes(template.fpk.to_bytes())
    if hair_from is not None:
        hair = Fmdl.parse(hair_from.fpk.find("hair_high.fmdl").data)
        if hair_fitted_disp is not None:
            for m in hair.meshes:
                hair.set_positions(m.index, m.positions + hair_fitted_disp(m.positions))
        hair.relink(f"/{hair_from.face_id}/", f"/{new_id}/")
        notes.append(f"hair taken from face {hair_from.face_id}")
    else:
        hair = Fmdl.parse(template.fpk.find("hair_high.fmdl").data) if template.hair else None
        if hair is not None:
            max_d = max(max_d, _morph_fmdl(hair, before, disp, rot))
    oral = Fmdl.parse(template.fpk.find("oral.fmdl").data) if template.oral else None
    if oral is not None:
        max_d = max(max_d, _morph_fmdl(oral, before, disp, rot))

    for f in (face, hair, oral):
        if f is not None:
            f.grow_bounding_boxes(max_d + 0.002)
            f.relink(f"/{old_id}/", f"/{new_id}/")
    fpk.replace_data("face_high.fmdl", face.to_bytes())
    if hair is not None:
        if fpk.find("hair_high.fmdl") is None:
            notes.append("template had no hair_high.fmdl; hair not written")
        else:
            fpk.replace_data("hair_high.fmdl", hair.to_bytes())
    if oral is not None:
        fpk.replace_data("oral.fmdl", oral.to_bytes())
    if template.diff is not None:
        pts = read_diff_eyes(template.diff)
        if pts:
            names = list(pts)
            P = np.array([pts[k] for k in names])
            moved = P + interpolate_displacement(before, disp, P)
            fpk.replace_data("face_diff.bin", write_diff_eyes(template.diff, dict(zip(names, moved))))
            notes.append("eye positions in face_diff.bin moved with the eye sockets")
    fpk.relink(f"/{old_id}/", f"/{new_id}/")
    out_fpk = _find_ci(out, "#Win", "face.fpk")
    fpk.write(out_fpk)
    # a .fpkd next to the .fpk is a separate data package; leave it as copied

    tex_dir = _find_ci(out, "sourceimages", "#windx11")
    if texture_rgba is not None and tex_dir is not None:
        src = template.base_texture_path()
        header = src.read_bytes()[:64] if src else None
        name = src.name if src else "face_bsm_alp.ftex"
        (tex_dir / name).write_bytes(ftex.encode(texture_rgba, header))
        notes.append(f"{name} replaced with the baked photo texture")
    if hair_from is not None and hair_from.texture_dir is not None and tex_dir is not None:
        for p in hair_from.texture_dir.iterdir():
            if p.name.lower().startswith("hair"):
                shutil.copy2(p, tex_dir / p.name)
    (out / "facegen_export.json").write_text(json.dumps({
        "template_id": old_id, "new_id": str(new_id), "max_displacement_mm": round(max_d * 1000, 2),
        "notes": notes}, indent=2))
    return ExportResult(out, str(new_id), notes)
