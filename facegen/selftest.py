"""Environment report and a synthetic end-to-end run (no game files or photos needed)."""

from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import landmarks as L
from .config import Config


def environment_report(cfg: Config) -> dict:
    from .blender import blender_version, find_blender
    from .scan import find_meshroom

    pkgs = {}
    for name in ("numpy", "scipy", "cv2", "PIL", "mediapipe", "mcp"):
        try:
            mod = importlib.import_module(name)
            pkgs[name] = getattr(mod, "__version__", "installed")
        except Exception as exc:  # noqa: BLE001
            pkgs[name] = f"missing ({exc.__class__.__name__})"
    blender = find_blender(cfg.blender)
    meshroom = find_meshroom(cfg.meshroom)
    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001
            gpu = None
    model = cfg.mediapipe_model_path()
    flame = cfg.flame_model_path()
    todo = []
    if not model.exists():
        todo.append(f"Download the MediaPipe face model to {model} (the installer does this).")
    if flame is None:
        todo.append("Optional: FLAME face model. Register at https://flame.is.tue.mpg.de, download FLAME 2020 "
                    f"or 2023 and put generic_model.pkl in {cfg.models_dir}.")
    if blender is None:
        todo.append("Optional: install Blender 3.6 or newer for lit previews and opening results in Blender.")
    if meshroom is None:
        todo.append("Optional: install Meshroom (2023.3 or newer) for the photogrammetry scan.")
    if not cfg.faces_folder:
        todo.append("Set faces_folder to FL27's ...\\livecpk\\Faces\\Asset\\model\\character\\face\\real "
                    "to browse templates and install faces.")
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "facegen_home": cfg.home,
        "packages": pkgs,
        "mediapipe_model": str(model) if model.exists() else None,
        "blender": f"{blender} ({blender_version(blender)})" if blender else None,
        "meshroom": str(meshroom) if meshroom else None,
        "flame_model": str(flame) if flame else None,
        "faces_folder": cfg.faces_folder,
        "nvidia_gpu": gpu,
        "still_to_do": todo,
    }


# ---------------------------------------------------------------------------
# Synthetic end-to-end run


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _person_deform(P: np.ndarray) -> np.ndarray:
    """The 'person': wider jaw, longer nose, slightly longer face than the template."""
    Q = P.copy()
    low = _smooth((-0.02 - P[:, 1]) / 0.04)
    Q[:, 0] *= 1.0 + 0.10 * low
    Q[:, 1] -= 0.006 * low
    nose = np.exp(-np.sum((P - np.array([0.0, -0.005, 0.075])) ** 2, axis=1) / 0.018 ** 2)
    Q[:, 2] += 0.007 * nose
    return Q


def _skin_texture(V, F, UV, size, beard: bool) -> np.ndarray:
    from .raster import interpolate, rasterize_uv

    frag = rasterize_uv(UV, F, size, size)
    pos = interpolate(frag, F, V)
    rng = np.random.default_rng(3)
    base = np.array([196.0, 150.0, 128.0])
    tex = np.tile(base, (size, size, 1))
    # fine pattern so the texture carries detail the bake must reproduce
    tex += 10 * np.sin(pos[..., :1] * 900) * np.cos(pos[..., 1:2] * 700)
    if beard:
        b = _smooth((-0.035 - pos[..., 1]) / 0.015)[..., None]
        tex = tex * (1 - 0.75 * b) + np.array([45.0, 35.0, 30.0]) * 0.75 * b
    tex += rng.normal(0, 3, tex.shape)
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., :3] = np.clip(tex, 0, 255).astype(np.uint8)
    rgba[..., 3] = 255
    return rgba


def run_selftest(workdir: str | Path | None = None, log=print) -> dict:
    """Build a fake template face and fake photos of a known 'person', run the whole pipeline
    and check the result against the truth."""
    from . import pipeline as P
    from .fitting import JawCurve
    from .geometry import look_at_camera
    from .mesh import read_obj, vertex_normals
    from .pes.face_folder import TemplateFace
    from .pes.synthetic import write_face_folder
    from .photos import save_rgb
    from .raster import render

    t0 = time.time()
    root = Path(workdir or tempfile.mkdtemp(prefix="facegen_selftest_"))
    cfg = Config(home=str(root / "home"))
    cfg.mediapipe_model = str(root / "no_model.task")  # force the detector-free paths
    # template skin: MediaPipe's canonical face with its UV layout
    mesh = read_obj(L.DATA_DIR / "canonical_face_model.obj")
    V = mesh.vertices / 100.0
    F, UV = mesh.faces, mesh.uvs
    N = vertex_normals(V, F)
    normals = np.c_[N, np.ones(len(V))]
    template_tex = _skin_texture(V, F, UV, 256, beard=False)
    folder = write_face_folder(root / "faces", "12345",
                               [dict(positions=V, faces=F, uv=UV, normals=normals, material="face")],
                               hair_meshes=[dict(positions=V[V[:, 1] > 0.03] * 1.04,
                                                 faces=np.zeros((0, 3), int), uv=UV[V[:, 1] > 0.03])],
                               texture=template_tex)
    # the person
    PV = _person_deform(V)
    person_tex = _skin_texture(V, F, UV, 512, beard=True)
    cv, _ = L.canonical_face()
    person_lm = _person_deform(cv)
    photos_dir = root / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)
    cams = {}
    rng = np.random.default_rng(7)
    for yaw in (-60, -40, -20, 0, 20, 40, 60):
        for pitch in ((0, -12) if yaw in (-40, 0, 40) else (0,)):
            a, p = np.radians(yaw), np.radians(pitch)
            eye = np.array([0.5 * np.sin(a) * np.cos(p), 0.5 * np.sin(p), 0.5 * np.cos(a) * np.cos(p)])
            cam = look_at_camera(eye, [0, -0.005, 0.03], [0, 1, 0], 600, 800, 1100)
            img, _ = render(cam, PV, F, UV, person_tex, background=0.55)
            name = f"y{yaw:+d}_p{pitch:+d}.jpg"
            save_rgb(photos_dir / name, img)
            cams[name] = cam
    # create the project with the real pipeline
    info = P.create_project(cfg, "selftest", str(photos_dir), str(folder), face_id="654321")
    proj = P.open_project(cfg, "selftest")
    # landmarks: projections of the person's landmarks plus noise (stands in for MediaPipe)
    items = []
    for name, cam in cams.items():
        px, _ = cam.project(person_lm)
        pts = np.vstack([px + rng.normal(0, 0.7, px.shape), np.zeros((10, 2))])
        items.append(L.PhotoLandmarks(name, cam.width, cam.height, pts, 1.0))
    L.save_landmarks(proj.path("landmarks.json"), items)
    P.prepare_template(proj, cfg, log)
    # the truth for every welded template vertex
    from scipy.spatial import cKDTree

    head = proj.head()
    _, idx = cKDTree(V).query(head.vertices)
    truth = PV[idx]
    # jaw lines drawn on three photos: the person's true jaw, projected
    jaw_true = JawCurve(head).positions(truth, head.faces)
    lines = {}
    for name in ("y+0_p-12.jpg", "y+60_p+0.jpg", "y-60_p+0.jpg"):
        px, _ = cams[name].project(jaw_true)
        lines[name] = {"jaw": [px.round(1).tolist()]}
    proj.save_lines(lines)
    cam_info = P.solve_project_cameras(proj, cfg, log)
    fit_info = P.fit(proj, cfg, log)
    X = proj.fitted()
    # the head's absolute placement follows the template (it must sit on the game's neck),
    # so judge the shape after removing a similarity transform
    from .geometry import umeyama

    sim = umeyama(X, truth)
    err = np.linalg.norm(sim.apply(X) - truth, axis=1) * 1000
    t_sim = umeyama(head.vertices, truth)
    err_t = np.linalg.norm(t_sim.apply(head.vertices) - truth, axis=1) * 1000
    regions = proj.regions()
    jaw_mask = regions["jaw"] > 0.5
    bake_info = P.bake(proj, cfg, log)
    cmp_info = P.compare(proj, cfg, max_views=3)
    exp = P.export(proj, cfg)
    back = TemplateFace.open(exp["folder"])
    hb, _, _, _, _ = back.skin()
    roundtrip = float(np.abs(np.sort(hb.vertices, axis=0) - np.sort(X, axis=0)).max() * 1000)
    tex_back = back.base_texture()
    report = {
        "workdir": str(root),
        "seconds": round(time.time() - t0, 1),
        "photos": len(cams),
        "cameras": {k: cam_info[k] for k in ("solved", "of", "mean_error_px")},
        "fit_rms_error_mm": round(float(np.sqrt(np.mean(err ** 2))), 3),
        "jaw_rms_error_mm": round(float(np.sqrt(np.mean(err[jaw_mask] ** 2))), 3),
        "template_vs_person_rms_mm": round(float(np.sqrt(np.mean(err_t ** 2))), 3),
        "template_vs_person_jaw_rms_mm": round(float(np.sqrt(np.mean(err_t[jaw_mask] ** 2))), 3),
        "fit_report": {k: fit_info[k] for k in ("landmark_rms_mm", "line_rms_px")},
        "texture_covered": bake_info["covered_fraction_of_uv"],
        "compare_landmark_error_pct": cmp_info["mean_landmark_error_pct"],
        "export_folder": exp["folder"],
        "export_roundtrip_error_mm": round(roundtrip, 4),
        "exported_texture": list(tex_back.shape) if tex_back is not None else None,
    }
    ok = (report["fit_rms_error_mm"] < 0.5 * report["template_vs_person_rms_mm"] and report["export_roundtrip_error_mm"] < 0.01
          and report["cameras"]["solved"] == report["photos"])
    report["passed"] = bool(ok)
    (root / "selftest_report.json").write_text(json.dumps(report, indent=2))
    return report
