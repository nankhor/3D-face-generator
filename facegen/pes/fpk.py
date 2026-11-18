"""Fox Engine package (.fpk / .fpkd) reader and writer.

Layout (little endian), as used by PES 2021 ``face.fpk``:

* 48-byte header: ``foxf`` ``pk`` type ``w`` ``in``, file size, 18 zero bytes,
  ``2``, entry count, reference count, 4 zero bytes.
* 48-byte entries: data offset, data size, name string (offset, length) and the
  MD5 of the name.
* 16-byte references: name string (offset, length).
* Null-terminated name strings, padded to 16; then each entry's data padded to 16.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"foxfpk"


@dataclass
class FpkEntry:
    name: bytes  # file path as stored
    md5: bytes  # 16 bytes
    data: bytes

    @property
    def path(self) -> str:
        return self.name.decode("latin-1")

    @property
    def filename(self) -> str:
        return self.path.replace("\\", "/").rsplit("/", 1)[-1]

    def name_hash_matches(self) -> bool:
        return hashlib.md5(self.name).digest() == self.md5


@dataclass
class FpkFile:
    fpk_type: int = 0x20  # b' ' for .fpk, b'd' for .fpkd
    entries: list[FpkEntry] = field(default_factory=list)
    references: list[bytes] = field(default_factory=list)

    # -- reading ---------------------------------------------------------
    @classmethod
    def read(cls, path: str | Path) -> "FpkFile":
        return cls.from_bytes(Path(path).read_bytes())

    @classmethod
    def from_bytes(cls, buf: bytes) -> "FpkFile":
        if len(buf) < 48 or buf[0:4] != b"foxf" or buf[4:6] != b"pk":
            raise ValueError("not a Fox Engine package (missing 'foxfpk' magic)")
        fpk_type = buf[6]
        (version, n_entries, n_refs) = struct.unpack_from("<III", buf, 32)
        if version != 2:
            raise ValueError(f"unsupported fpk version {version}")
        entries = []
        pos = 48
        for _ in range(n_entries):
            off, _z1, size, _z2, s_off, _z3, s_len, _z4 = struct.unpack_from("<IIiIiIiI", buf, pos)
            md5 = buf[pos + 32: pos + 48]
            name = buf[s_off: s_off + s_len]
            data = buf[off: off + size]
            entries.append(FpkEntry(name, md5, data))
            pos += 48
        refs = []
        for _ in range(n_refs):
            s_off, _z1, s_len, _z2 = struct.unpack_from("<iIiI", buf, pos)
            refs.append(buf[s_off: s_off + s_len])
            pos += 16
        return cls(fpk_type, entries, refs)

    # -- access ----------------------------------------------------------
    def find(self, filename: str) -> FpkEntry | None:
        low = filename.lower()
        for e in self.entries:
            if e.filename.lower() == low:
                return e
        return None

    def replace_data(self, filename: str, data: bytes) -> None:
        e = self.find(filename)
        if e is None:
            raise KeyError(filename)
        e.data = data

    def relink(self, old: str, new: str) -> None:
        """Replace ``old`` with ``new`` inside entry and reference paths (e.g. player IDs)."""
        ob, nb = old.encode("latin-1"), new.encode("latin-1")
        for e in self.entries:
            if ob in e.name:
                matched = e.name_hash_matches()
                e.name = e.name.replace(ob, nb)
                if matched:
                    e.md5 = hashlib.md5(e.name).digest()
        self.references = [r.replace(ob, nb) for r in self.references]

    # -- writing ---------------------------------------------------------
    def to_bytes(self) -> bytes:
        n_e, n_r = len(self.entries), len(self.references)
        table_end = 48 + 48 * n_e + 16 * n_r
        out = bytearray(table_end)
        string_pos = []
        for name in [e.name for e in self.entries] + self.references:
            string_pos.append((len(out), len(name)))
            out += name + b"\x00"
        _pad(out)
        data_pos = []
        for e in self.entries:
            data_pos.append((len(out), len(e.data)))
            out += e.data
            _pad(out)
        struct.pack_into("<4s2sBB2sI", out, 0, b"foxf", b"pk", self.fpk_type, ord("w"), b"in", len(out))
        struct.pack_into("<III", out, 32, 2, n_e, n_r)
        pos = 48
        for i, e in enumerate(self.entries):
            d_off, d_len = data_pos[i]
            s_off, s_len = string_pos[i]
            struct.pack_into("<IIiIiIiI", out, pos, d_off, 0, d_len, 0, s_off, 0, s_len, 0)
            md5 = e.md5 if len(e.md5) == 16 else hashlib.md5(e.name).digest()
            out[pos + 32: pos + 48] = md5
            pos += 48
        for j in range(n_r):
            s_off, s_len = string_pos[n_e + j]
            struct.pack_into("<iIiI", out, pos, s_off, 0, s_len, 0)
            pos += 16
        return bytes(out)

    def write(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_bytes())


def _pad(buf: bytearray, align: int = 16) -> None:
    r = len(buf) % align
    if r:
        buf += b"\x00" * (align - r)


def new_fpk(files: dict[str, bytes], fpk_type: int = 0x20) -> FpkFile:
    return FpkFile(fpk_type, [FpkEntry(n.encode("latin-1"), hashlib.md5(n.encode("latin-1")).digest(), d)
                              for n, d in files.items()], [])
