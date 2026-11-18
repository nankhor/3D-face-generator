"""Build small but structurally valid PES face files for tests and the self-test.

These files exercise the same code paths as real faces (FPK container, FMDL
tables and buffers, FTEX textures) without needing any game assets.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from . import ftex
from .fmdl import F16x2, F16x4, F32x3, NORMAL, POSITION, TANGENT, UV0
from .fpk import new_fpk


def build_fmdl(meshes: list[dict]) -> bytes:
    """meshes: dicts with positions (N,3), faces (M,3), uv (N,2), normals (N,4), tangents (N,4),
    material (str), texture (filename), directory (str)."""
    strings: list[str] = [""]

    def sid(text: str) -> int:
        if text not in strings:
            strings.append(text)
        return strings.index(text)

    pos_buf = bytearray()
    dat_buf = bytearray()
    idx_buf = bytearray()
    mesh_entries, mat_entries, tex_entries, assign_entries, material_entries = [], [], [], [], []
    fa_entries, mf_entries, vf_entries, face_index, bboxes = [], [], [], [], []
    first_face_vertex = 0
    for mi, m in enumerate(meshes):
        V = np.asarray(m["positions"], "<f4")
        n = len(V)
        nrm = np.asarray(m.get("normals", np.tile([0, 0, 1, 1], (n, 1))), "<f2")
        tan = np.asarray(m.get("tangents", np.tile([1, 0, 0, 1], (n, 1))), "<f2")
        uv = np.asarray(m.get("uv", np.zeros((n, 2))), "<f2")
        F = np.asarray(m["faces"], "<u2")
        pos_off = len(pos_buf)
        pos_buf += V.tobytes()
        dat_off = len(dat_buf)
        rec = np.zeros(n, dtype=[("n", "<f2", 4), ("t", "<f2", 4), ("uv", "<f2", 2)])
        rec["n"], rec["t"], rec["uv"] = nrm, tan, uv
        dat_buf += rec.tobytes()
        idx_buf += F.tobytes()
        # material: one texture
        tex_id = len(tex_entries)
        tex_entries.append(struct.pack("<HH", sid(m.get("texture", "face_bsm_alp.ftex")),
                                       sid(m.get("directory", "/Assets/pes16/model/character/face/real/12345/sourceimages/"))))
        first_assign = len(assign_entries)
        assign_entries.append(struct.pack("<HH", sid("Base_Tex_SRGB"), tex_id))
        mat_id = len(material_entries)
        material_entries.append(struct.pack("<HH", sid("fox3DDF_Skin_Face"), sid("fox3DDF_Skin_Face")))
        mat_entries.append(struct.pack("<HHHBBHHI", sid(m.get("material", f"mat{mi}")), 0, mat_id, 1, 0,
                                       first_assign, 0, 0))
        first_mf = len(mf_entries)
        mf_entries.append(struct.pack("<BBBBI", 0, 1, 12, 0, pos_off))
        mf_entries.append(struct.pack("<BBBBI", 1, 3, 20, 1, dat_off))
        first_vf = len(vf_entries)
        vf_entries += [struct.pack("<BBH", POSITION, F32x3, 0), struct.pack("<BBH", NORMAL, F16x4, 0),
                       struct.pack("<BBH", TANGENT, F16x4, 8), struct.pack("<BBH", UV0, F16x2, 16)]
        fa_entries.append(struct.pack("<BBBBHH", 2, 4, 0, 1, first_mf, first_vf))
        face_index.append(struct.pack("<II", 0, F.size))
        mesh_entries.append(struct.pack("<BB2xHHHH4xIIQ16x", 0, 0, mi, 0, mi, n, first_face_vertex, F.size, mi))
        first_face_vertex += F.size
        mn, mx = V.min(axis=0), V.max(axis=0)
        bboxes.append(struct.pack("<8f", *mx, 1.0, *mn, 1.0))

    def pad16(b: bytearray) -> bytearray:
        while len(b) % 16:
            b += b"\x00"
        return b

    vbuf = bytearray()
    off0 = 0
    vbuf += pos_buf
    pad16(vbuf)
    off1 = len(vbuf)
    vbuf += dat_buf
    pad16(vbuf)
    off2 = len(vbuf)
    vbuf += idx_buf
    pad16(vbuf)
    buf_entries = [struct.pack("<IIII", 0, len(pos_buf), off0, 0), struct.pack("<IIII", 0, len(dat_buf), off1, 0),
                   struct.pack("<IIII", 1, len(idx_buf), off2, 0)]
    sbuf = bytearray()
    string_entries = []
    for s in strings:
        b = s.encode("utf-8")
        string_entries.append(struct.pack("<HHI", 3, len(b), len(sbuf)))
        sbuf += b + b"\x00"
    lod = [struct.pack("<I3f", 1, 1.0, 1.0, 1.0)]

    blocks0 = {3: mesh_entries, 4: mat_entries, 6: tex_entries, 7: assign_entries, 8: material_entries,
               9: fa_entries, 10: mf_entries, 11: vf_entries, 12: string_entries, 13: bboxes,
               14: buf_entries, 16: lod, 17: face_index}
    sec0 = bytearray()
    desc0 = []
    for bid, ents in blocks0.items():
        desc0.append(struct.pack("<HHI", bid, len(ents), len(sec0)))
        for e in ents:
            sec0 += e
        pad16(sec0)
    blocks1 = {0: bytearray(16), 2: vbuf, 3: sbuf}
    n0, n1 = len(desc0), len(blocks1)
    desc_off = 64
    s0_off = desc_off + 8 * n0 + 12 * n1
    s0_off += (-s0_off) % 16
    s1_off = s0_off + len(sec0)
    sec1 = bytearray()
    desc1 = []
    for bid, b in blocks1.items():
        desc1.append(struct.pack("<III", bid, len(sec1), len(b)))
        sec1 += b
        if bid != 3:
            pad16(sec1)
    header = struct.pack("<4sIQQQIIIIII", b"FMDL", 0x4001EB85, desc_off, 0, 0, n0, n1, s0_off, len(sec0),
                         s1_off, len(sec1))
    out = bytearray(header)
    out += b"\x00" * (desc_off - len(out))
    for d in desc0 + desc1:
        out += d
    out += b"\x00" * (s0_off - len(out))
    out += sec0
    out += sec1
    return bytes(out)


def sphere_head(radius: float = 0.1, rings: int = 24, segments: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A UV sphere with a seam (duplicated vertices), like real game meshes."""
    verts, uvs = [], []
    for i in range(rings + 1):
        th = np.pi * i / rings
        for j in range(segments + 1):
            ph = 2 * np.pi * j / segments
            verts.append([radius * np.sin(th) * np.sin(ph), radius * np.cos(th), radius * np.sin(th) * np.cos(ph)])
            uvs.append([j / segments, i / rings])
    faces = []
    for i in range(rings):
        for j in range(segments):
            a = i * (segments + 1) + j
            b = a + segments + 1
            if i > 0:
                faces.append([a, b, a + 1])
            if i < rings - 1:
                faces.append([a + 1, b, b + 1])
    return np.array(verts), np.array(faces), np.array(uvs)


def write_face_folder(root: Path, face_id: str, skin_meshes: list[dict], hair_meshes: list[dict] | None = None,
                      texture: np.ndarray | None = None) -> Path:
    folder = Path(root) / face_id
    (folder / "#Win").mkdir(parents=True, exist_ok=True)
    tex_dir = folder / "sourceimages" / "#windx11"
    tex_dir.mkdir(parents=True, exist_ok=True)
    directory = f"/Assets/pes16/model/character/face/real/{face_id}/sourceimages/"
    for m in skin_meshes + (hair_meshes or []):
        m.setdefault("directory", directory)
    for m in hair_meshes or []:
        m.setdefault("texture", "hair_parts_bsm_alp.ftex")
    files = {
        f"/Assets/pes16/model/character/face/real/{face_id}/face_high.fmdl": build_fmdl(skin_meshes),
        f"/Assets/pes16/model/character/face/real/{face_id}/hair_high.fmdl": build_fmdl(hair_meshes or skin_meshes[:1]),
        f"/Assets/pes16/model/character/face/real/{face_id}/face_diff.bin": _diff_bin(),
    }
    new_fpk(files).write(folder / "#Win" / "face.fpk")
    if texture is None:
        texture = np.zeros((256, 256, 4), np.uint8)
        texture[..., 0] = 180
        texture[..., 1] = 140
        texture[..., 2] = 120
        texture[..., 3] = 255
    (tex_dir / "face_bsm_alp.ftex").write_bytes(ftex.encode(texture))
    (tex_dir / "hair_parts_bsm_alp.ftex").write_bytes(ftex.encode(texture // 3))
    return folder


def _diff_bin() -> bytes:
    b = bytearray(0x200)
    b[:4] = b"FACE"
    struct.pack_into("<3f", b, 0x150, 0.0, 0.05, -0.03)  # (y, z, x) of the right eye
    struct.pack_into("<3f", b, 0x160, 0.0, 0.05, 0.03)
    return bytes(b)
