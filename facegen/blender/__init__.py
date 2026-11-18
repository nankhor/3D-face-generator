"""Blender hand-off: nicer renders and opening the result in the Blender UI.

Blender is optional. The pipeline's own renderer covers fitting and comparing;
Blender adds lit renders and lets you inspect the head with every photo camera.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np

JOB_SCRIPT = Path(__file__).parent / "blender_job.py"


def find_blender(configured: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    found = shutil.which("blender")
    if found:
        candidates.append(Path(found))
    system = platform.system()
    if system == "Windows":
        roots = [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Blender Foundation"]
        for root in roots:
            if root.exists():
                for d in sorted(root.iterdir(), reverse=True):
                    candidates.append(d / "blender.exe")
        steam = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Steam" / "steamapps" / "common" / "Blender" / "blender.exe"
        candidates.append(steam)
    elif system == "Darwin":
        candidates.append(Path("/Applications/Blender.app/Contents/MacOS/Blender"))
    for c in candidates:
        if c.is_file():
            return c
    return None


def blender_version(blender: Path) -> str | None:
    try:
        out = subprocess.run([str(blender), "-b", "--factory-startup", "--version"], capture_output=True,
                             text=True, timeout=120)
        for line in out.stdout.splitlines():
            if line.startswith("Blender"):
                return line.strip()
    except Exception:
        return None
    return None


def write_mesh_npz(path: Path, vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray | None = None) -> Path:
    data = {"vertices": np.asarray(vertices, np.float64), "faces": np.asarray(faces, np.int64)}
    if uvs is not None:
        data["uvs"] = np.asarray(uvs, np.float64)
    np.savez(path, **data)
    return path


def run_job(blender: Path, job: dict, work_dir: Path, timeout: int = 1800) -> list[str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    job_path = work_dir / "blender_job.json"
    job_path.write_text(json.dumps(job))
    cmd = [str(blender), "-b", "--factory-startup", "--python", str(JOB_SCRIPT), "--", str(job_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    for line in proc.stdout.splitlines():
        if line.startswith("FACEGEN_DONE "):
            return json.loads(line[len("FACEGEN_DONE "):])
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    raise RuntimeError(f"Blender job failed:\n{tail}")


def open_in_ui(blender: Path, blend_file: Path) -> None:
    """Open a .blend in the Blender UI without waiting for it to close."""
    kwargs = {}
    if platform.system() == "Windows":
        kwargs["creationflags"] = 0x00000008  # DETACHED_PROCESS
    subprocess.Popen([str(blender), str(blend_file)], **kwargs)
