"""Fox Engine model (.fmdl) reader and in-place patcher.

Only what the face pipeline needs is decoded: meshes (positions, normals,
tangents, UVs, faces), material texture names, strings and bounding boxes.

Edits are applied *in place* on the original bytes: vertex positions (float32),
normals and tangents (float16) and bounding boxes are overwritten at their
exact offsets, so every other byte of the template stays identical. This is
deliberate: a full re-serialisation of ``face_high.fmdl`` is a known source of
in-game corruption.

Format summary: a 56-byte header points at block descriptors. Section 0 holds
fixed-size tables (meshes, formats, strings, bounding boxes...), section 1 holds
raw buffers (block 2: vertex and index data, block 3: string bytes).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

SECTION0_ENTRY_SIZES = {0: 48, 1: 8, 2: 32, 3: 48, 4: 16, 5: 68, 6: 4, 7: 4, 8: 4, 9: 8, 10: 8,
                        11: 4, 12: 8, 13: 32, 14: 16, 16: 16, 17: 8, 18: 8, 20: 128}

# vertex datum types
POSITION, BONE_WEIGHTS, NORMAL, COLOR, BONE_INDICES, UV0, UV1, UV2, UV3, TANGENT = 0, 1, 2, 3, 7, 8, 9, 10, 11, 14
# datum formats
F32x3, F32x2, F16x4, F16x2, U8x4N, U8x4 = 1, 2, 6, 7, 8, 9


class FmdlError(ValueError):
    pass


@dataclass
class FmdlMesh:
    index: int
    vertex_count: int
    material: str
    textures: list[tuple[str, str, str]]  # (role, filename, directory)
    faces: np.ndarray  # (M, 3)
    positions: np.ndarray  # (N, 3) float64
    uv: np.ndarray | None  # (N, 2), v=0 at the top (DirectX)
    normals: np.ndarray | None  # (N, 4)
    tangents: np.ndarray | None  # (N, 4)
    position_offsets: np.ndarray  # absolute file offsets of each vertex position
    normal_offsets: np.ndarray | None
    tangent_offsets: np.ndarray | None

    def base_texture(self) -> tuple[str, str] | None:
        for role, fn, d in self.textures:
            if "base" in role.lower() or "_bsm" in fn.lower():
                return fn, d
        return None


@dataclass
class Fmdl:
    data: bytearray
    strings: list[str]
    string_entries: list[tuple[int, int, int, int]]  # (entry file offset, s1 block, length, offset)
    meshes: list[FmdlMesh]
    bbox_offsets: list[int]
    section1: dict[int, tuple[int, int, int]]  # block id -> (abs offset, length, descriptor offset)
    header_section1_offset: int = 0
    descriptors_offset: int = 0
    n_section0: int = 0
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def parse(cls, buf: bytes) -> "Fmdl":
        data = bytearray(buf)
        if len(data) < 56 or data[:4] != b"FMDL":
            raise FmdlError("not an FMDL file")
        (_magic, _version, desc_off, _b0, _b1, n0, n1, s0_off, _s0_len, s1_off, _s1_len) = struct.unpack_from(
            "<4sIQQQIIIIII", data, 0)
        section0: dict[int, tuple[int, int, int]] = {}
        pos = desc_off
        for _ in range(n0):
            bid, count, boff = struct.unpack_from("<HHI", data, pos)
            pos += 8
            if bid in SECTION0_ENTRY_SIZES:
                section0[bid] = (s0_off + boff, count, SECTION0_ENTRY_SIZES[bid])
        section1: dict[int, tuple[int, int, int]] = {}
        for _ in range(n1):
            bid, boff, length = struct.unpack_from("<III", data, pos)
            section1[bid] = (s1_off + boff, length, pos)
            pos += 12

        def entries(bid):
            if bid not in section0:
                return []
            off, count, size = section0[bid]
            return [(off + i * size) for i in range(count)]

        # strings
        strings, string_entries = [], []
        for eo in entries(12):
            sb, ln, so = struct.unpack_from("<HHI", data, eo)
            if sb not in section1:
                raise FmdlError("string references a missing block")
            base = section1[sb][0]
            strings.append(bytes(data[base + so: base + so + ln]).decode("utf-8", "replace"))
            string_entries.append((eo, sb, ln, so))

        def s(i):
            return strings[i] if 0 <= i < len(strings) else ""

        # materials and textures
        textures = [struct.unpack_from("<HH", data, eo) for eo in entries(6)]
        assign = [struct.unpack_from("<HH", data, eo) for eo in entries(7)]
        mat_instances = []
        for eo in entries(4):
            name_id, _p, _mat, tex_count, _pc, first_tex, _fp, _p2 = struct.unpack_from("<HHHBBHHI", data, eo)
            texs = []
            for i in range(first_tex, first_tex + tex_count):
                if i < len(assign):
                    role_id, tex_id = assign[i]
                    if tex_id < len(textures):
                        fn_id, dir_id = textures[tex_id]
                        texs.append((s(role_id), s(fn_id), s(dir_id)))
            mat_instances.append((s(name_id), texs))

        # buffers and formats
        if 2 not in section1:
            raise FmdlError("vertex buffer block missing")
        vb_base = section1[2][0]
        buffer_offsets = [struct.unpack_from("<IIII", data, eo)[2] for eo in entries(14)]
        mesh_formats = [struct.unpack_from("<BBBBI", data, eo) for eo in entries(10)]
        vertex_formats = [struct.unpack_from("<BBH", data, eo) for eo in entries(11)]
        format_assign = []
        for eo in entries(9):
            mf_count, vf_count, _fu, _uc, first_mf, first_vf = struct.unpack_from("<BBBBHH", data, eo)
            bases = []
            for i in range(first_mf, first_mf + mf_count):
                buf_id, vf_n, stride, _type, boff = mesh_formats[i]
                for _ in range(vf_n):
                    bases.append((buffer_offsets[buf_id] + boff, stride))
            if len(bases) != vf_count:
                raise FmdlError("inconsistent vertex format table")
            fields = []
            for j in range(vf_count):
                dtype, dfmt, voff = vertex_formats[first_vf + j]
                base, stride = bases[j]
                fields.append((dtype, dfmt, base + voff, stride))
            format_assign.append(fields)
        face_index = [struct.unpack_from("<II", data, eo) for eo in entries(17)]

        meshes = []
        for mi, eo in enumerate(entries(3)):
            (_a, _s, mat_id, _bg, fmt_id, vcount, first_fv, _fv_count, first_fi) = struct.unpack_from(
                "<BB2xHHHH4xIIQ", data, eo)
            fields = format_assign[fmt_id]
            pos_off = nrm_off = tan_off = None
            positions = normals = tangents = uv = None
            idx = np.arange(vcount)
            for dtype, dfmt, off, stride in fields:
                offs = vb_base + off + idx * stride
                if dtype == POSITION:
                    if dfmt != F32x3:
                        raise FmdlError("unexpected position format")
                    pos_off = offs
                    positions = _gather(data, offs, "<f4", 3).astype(np.float64)
                elif dtype == NORMAL and dfmt == F16x4:
                    nrm_off = offs
                    normals = _gather(data, offs, "<f2", 4).astype(np.float64)
                elif dtype == TANGENT and dfmt == F16x4:
                    tan_off = offs
                    tangents = _gather(data, offs, "<f2", 4).astype(np.float64)
                elif dtype == UV0:
                    if dfmt == F16x2:
                        uv = _gather(data, offs, "<f2", 2).astype(np.float64)
                    elif dfmt == F32x2:
                        uv = _gather(data, offs, "<f4", 2).astype(np.float64)
            if positions is None:
                raise FmdlError(f"mesh {mi} has no positions")
            lod_first, lod_count = face_index[first_fi] if first_fi < len(face_index) else (0, 0)
            fstart = vb_base + buffer_offsets[2] + (first_fv + lod_first) * 2
            tri = np.frombuffer(bytes(data[fstart: fstart + lod_count * 2]), dtype="<u2").astype(np.int64)
            faces = tri.reshape(-1, 3)
            name, texs = mat_instances[mat_id] if mat_id < len(mat_instances) else ("", [])
            meshes.append(FmdlMesh(mi, vcount, name, texs, faces, positions, uv, normals, tangents,
                                   pos_off, nrm_off, tan_off))

        return cls(data, strings, string_entries, meshes, entries(13), section1, s1_off, desc_off, n0)

    # ------------------------------------------------------------------
    def set_positions(self, mesh_index: int, positions: np.ndarray) -> None:
        m = self.meshes[mesh_index]
        if positions.shape != (m.vertex_count, 3):
            raise FmdlError("position array has the wrong shape")
        _scatter(self.data, m.position_offsets, np.asarray(positions, "<f4"))
        m.positions = np.asarray(positions, np.float64)

    def set_normals(self, mesh_index: int, normals: np.ndarray) -> None:
        m = self.meshes[mesh_index]
        if m.normal_offsets is None:
            return
        _scatter(self.data, m.normal_offsets, np.asarray(normals, "<f2"))
        m.normals = np.asarray(normals, np.float64)

    def set_tangents(self, mesh_index: int, tangents: np.ndarray) -> None:
        m = self.meshes[mesh_index]
        if m.tangent_offsets is None:
            return
        _scatter(self.data, m.tangent_offsets, np.asarray(tangents, "<f2"))
        m.tangents = np.asarray(tangents, np.float64)

    def grow_bounding_boxes(self, margin: float) -> None:
        """Enlarge every bounding box by ``margin`` (metres) so culling stays safe."""
        for eo in self.bbox_offsets:
            vals = list(struct.unpack_from("<8f", self.data, eo))
            for k in range(3):
                vals[k] += margin
                vals[4 + k] -= margin
            struct.pack_into("<8f", self.data, eo, *vals)

    def relink(self, old: str, new: str) -> int:
        """Replace ``old`` with ``new`` in every string (texture paths carry the face ID).

        Same-length replacements are patched in place. Otherwise the new string is
        appended to the end of the string block, which must be the last block of the
        file (true for game and tool-written faces). Returns the number of strings changed.
        """
        changed = 0
        for k, (eo, sb, ln, so) in enumerate(self.string_entries):
            text = self.strings[k]
            if old not in text:
                continue
            new_text = text.replace(old, new)
            nb = new_text.encode("utf-8")
            base, length, desc_pos = self.section1[sb]
            if len(nb) == ln:
                self.data[base + so: base + so + ln] = nb
            else:
                if any(other_base > base for bid, (other_base, _l, _d) in self.section1.items() if bid != sb):
                    raise FmdlError("string block is not at the end of the file; cannot relink "
                                    "IDs of different lengths")
                # string data offset relative to the block, appended at the true end
                block_len = len(self.data) - base
                new_off = block_len
                self.data += nb + b"\x00"
                while len(self.data) % 16:
                    self.data += b"\x00"
                struct.pack_into("<HHI", self.data, eo, sb, len(nb), new_off)
                new_len = len(self.data) - base
                struct.pack_into("<I", self.data, desc_pos + 8, new_len)
                self.section1[sb] = (base, new_len, desc_pos)
                # header: section 1 length
                s1_len = len(self.data) - self.header_section1_offset
                struct.pack_into("<I", self.data, 52, s1_len)
                self.string_entries[k] = (eo, sb, len(nb), new_off)
            self.strings[k] = new_text
            changed += 1
        return changed

    def to_bytes(self) -> bytes:
        return bytes(self.data)


def _gather(data: bytearray, offsets: np.ndarray, dtype: str, count: int) -> np.ndarray:
    item = np.dtype(dtype).itemsize * count
    buf = np.frombuffer(bytes(data), dtype=np.uint8)
    idx = offsets[:, None] + np.arange(item)[None, :]
    raw = buf[idx].copy()
    return raw.view(dtype).reshape(len(offsets), count)


def _scatter(data: bytearray, offsets: np.ndarray, values: np.ndarray) -> None:
    raw = np.ascontiguousarray(values).view(np.uint8).reshape(len(offsets), -1)
    item = raw.shape[1]
    arr = np.frombuffer(data, dtype=np.uint8)
    idx = offsets[:, None] + np.arange(item)[None, :]
    arr[idx] = raw
