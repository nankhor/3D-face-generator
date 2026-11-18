"""Fox Engine textures (.ftex): decode to RGBA, encode from RGBA.

Header (64 bytes): ``FTEX``, version (float), pixel format, width, height, depth,
mip count, then fields whose meaning is not fully known (they are copied from the
template texture when writing), then 16-byte mip entries (offset, sizes, index,
chunk count). Mip data is a list of zlib chunks of at most 16 KiB.

New textures are written as BC3/DXT5 (pixel format 4), the format PES face
textures normally use; the 64-byte header is copied from the template texture so
the unknown fields stay valid.
"""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image

HEADER = struct.Struct("<4sfHHHHBBHIIIBB14x8s8s")
MIP = struct.Struct("<IIIBBH")
CHUNK = struct.Struct("<HHI")
CHUNK_SIZE = 0x4000

FOURCC = {2: b"DXT1", 3: b"DXT3", 4: b"DXT5"}
DXGI = {1: 61, 8: 80, 9: 83, 10: 95, 11: 98, 12: 10, 13: 2, 14: 24, 15: 26}


@dataclass
class FtexInfo:
    version: float
    pixel_format: int
    width: int
    height: int
    depth: int
    mip_count: int
    texture_type: int
    ftexs_count: int
    raw_header: bytes


def read_info(buf: bytes) -> FtexInfo:
    (magic, version, fmt, w, h, depth, mips, _nrt, _flags, _u1, _u2, ttype, ftexs, _u3, _h1, _h2) = HEADER.unpack_from(buf, 0)
    if magic != b"FTEX":
        raise ValueError("not an FTEX file")
    return FtexInfo(version, fmt, w, h, depth, mips, ttype, ftexs, bytes(buf[:64]))


def _read_mip(buf: bytes, offset: int, chunk_count: int, uncompressed: int, compressed: int) -> bytes:
    if chunk_count == 0:
        raw = buf[offset: offset + (compressed or uncompressed)]
        return zlib.decompress(raw) if compressed else bytes(raw)
    parts = []
    for i in range(chunk_count):
        c_size, _u_size, c_off = CHUNK.unpack_from(buf, offset + i * 8)
        stored = (c_off & 0x80000000) != 0
        c_off &= 0x7FFFFFFF
        piece = buf[offset + c_off: offset + c_off + c_size]
        parts.append(bytes(piece) if stored else zlib.decompress(piece))
    return b"".join(parts)


def read_mips(buf: bytes) -> tuple[FtexInfo, list[bytes]]:
    info = read_info(buf)
    if info.ftexs_count:
        raise ValueError("textures split across .ftexs files are not supported")
    images = 6 if info.texture_type & 4 else 1
    mips = []
    pos = 64
    for _ in range(images * max(info.mip_count, 1)):
        off, usize, csize, _idx, _n, chunks = MIP.unpack_from(buf, pos)
        pos += 16
        mips.append(_read_mip(buf, off, chunks, usize, csize))
    return info, mips


def to_dds(buf: bytes) -> bytes:
    """Convert the top-level 2D image of an FTEX to a single-mip DDS."""
    info, mips = read_mips(buf)
    data = mips[0]
    flags = 0x1 | 0x2 | 0x4 | 0x1000
    caps = 0x1000
    dx10 = None
    if info.pixel_format == 0:
        flags |= 0x8
        pf = struct.pack("<II4sIIIII", 32, 0x41, b"\0\0\0\0", 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
        pitch = 4 * info.width
    elif info.pixel_format in FOURCC:
        flags |= 0x80000
        pf = struct.pack("<II4sIIIII", 32, 0x4, FOURCC[info.pixel_format], 0, 0, 0, 0, 0)
        pitch = len(data)
    elif info.pixel_format in DXGI:
        flags |= 0x80000
        pf = struct.pack("<II4sIIIII", 32, 0x4, b"DX10", 0, 0, 0, 0, 0)
        pitch = len(data)
        dx10 = struct.pack("<5I", DXGI[info.pixel_format], 3, 0, 1, 0)
    else:
        raise ValueError(f"unsupported FTEX pixel format {info.pixel_format}")
    header = b"DDS " + struct.pack("<7I44x", 124, flags, info.height, info.width, pitch, 1, 0) + pf + \
        struct.pack("<II12x", caps, 0)
    return header + (dx10 or b"") + data


def decode(buf: bytes) -> np.ndarray:
    """FTEX bytes -> (H, W, 4) uint8 RGBA."""
    img = Image.open(io.BytesIO(to_dds(buf)))
    return np.asarray(img.convert("RGBA"))


def _dxt5_level(rgba: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(out, "DDS", pixel_format="DXT5")
    raw = out.getvalue()
    if raw[84:88] == b"DX10":
        return raw[148:]
    return raw[128:]


def _chunked(level: bytes) -> bytes:
    pieces = [level[i:i + CHUNK_SIZE] for i in range(0, len(level), CHUNK_SIZE)] or [b""]
    table = bytearray()
    blob = bytearray()
    first = 8 * len(pieces)
    for p in pieces:
        comp = zlib.compress(p, 9)
        if len(comp) < len(p):
            table += CHUNK.pack(len(comp), len(p), first + len(blob))
            blob += comp
        else:
            table += CHUNK.pack(len(p), len(p), (first + len(blob)) | 0x80000000)
            blob += p
    return bytes(table + blob)


def encode(rgba: np.ndarray, template_header: bytes | None = None, mip_count: int | None = None) -> bytes:
    """RGBA uint8 -> FTEX (BC3/DXT5 with mip-maps)."""
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w = rgba.shape[:2]
    full = int(np.floor(np.log2(max(w, h)))) + 1
    mips = min(mip_count or full, full)
    header = bytearray(template_header[:64] if template_header else _default_header())
    struct.pack_into("<H", header, 8, 4)  # pixel format: DXT5
    struct.pack_into("<HHH", header, 10, w, h, 1)
    header[16] = mips
    header[32] = 0  # no .ftexs files
    levels = []
    img = Image.fromarray(rgba, "RGBA")
    for i in range(mips):
        lw, lh = max(1, w >> i), max(1, h >> i)
        lvl = rgba if i == 0 else np.asarray(img.resize((lw, lh), Image.LANCZOS))
        levels.append(_dxt5_level(lvl))
    body = bytearray()
    entries = bytearray()
    data_start = 64 + 16 * mips
    for i, lvl in enumerate(levels):
        packed = _chunked(lvl)
        n_chunks = max(1, -(-len(lvl) // CHUNK_SIZE))
        entries += MIP.pack(data_start + len(body), len(lvl), len(packed), i, 0, n_chunks)
        body += packed
        while len(body) % 16:
            body += b"\x00"
    return bytes(header) + bytes(entries) + bytes(body)


def _default_header() -> bytes:
    return HEADER.pack(b"FTEX", 2.04, 4, 0, 0, 1, 1, 2, 0x11, 1, 0, 1, 0, 1, b"\0" * 8, b"\0" * 8)
