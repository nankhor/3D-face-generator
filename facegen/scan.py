"""Photogrammetry scan with Meshroom (AliceVision), run as a background process."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .geometry import Similarity
from .mesh import Mesh, largest_component, read_obj


def find_meshroom(configured: str | None) -> Path | None:
    candidates = []
    if configured:
        candidates.append(Path(configured))
    for name in ("meshroom_batch", "meshroom_batch.exe"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    for c in candidates:
        if c.is_file():
            return c
        if c.is_dir():
            for name in ("meshroom_batch.exe", "meshroom_batch", "aliceVision/bin/meshroom_batch"):
                if (c / name).exists():
                    return c / name
    return None


def meshroom_command(meshroom: Path, photos: Path, output: Path, cache: Path) -> list[str]:
    return [str(meshroom), "--input", str(photos), "--output", str(output), "--cache", str(cache)]


def find_outputs(output: Path, cache: Path) -> tuple[Path | None, Path | None]:
    """Locate the textured mesh and the SfM cameras after a Meshroom run."""
    mesh = None
    for name in ("texturedMesh.obj", "mesh.obj"):
        found = sorted(output.rglob(name), key=lambda p: p.stat().st_mtime)
        if found:
            mesh = found[-1]
            break
    if mesh is None:
        found = sorted(cache.rglob("texturedMesh.obj"), key=lambda p: p.stat().st_mtime)
        mesh = found[-1] if found else None
    sfm = None
    for pattern in ("StructureFromMotion/*/cameras.sfm", "*/cameras.sfm", "cameras.sfm"):
        found = sorted(cache.rglob(pattern) if "/" in pattern else cache.rglob(pattern),
                       key=lambda p: p.stat().st_mtime)
        if found:
            sfm = found[-1]
            break
    return mesh, sfm


def load_scan(mesh_path: Path, to_head: Similarity, centre: np.ndarray, radius: float = 0.16) -> Mesh:
    """Load the Meshroom mesh, move it to head space and keep the head only."""
    mesh = read_obj(mesh_path)
    V = to_head.apply(mesh.vertices)
    keep_v = np.linalg.norm(V - centre, axis=1) < radius
    keep_f = keep_v[mesh.faces].all(axis=1)
    faces = mesh.faces[keep_f]
    used = np.unique(faces)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    cropped = Mesh(V[used], remap[faces], None if mesh.uvs is None else mesh.uvs[used])
    if len(cropped.faces) == 0:
        raise ValueError("The scan does not overlap the head after alignment")
    return largest_component(cropped)
