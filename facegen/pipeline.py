"""The face pipeline, step by step. Every MCP tool and CLI command calls into here.

A project folder holds everything for one person::

    <FaceGen home>/projects/<name>/
      project.json          name, template folder, face ID, step status
      photos/               the person's photos
      work/                 landmarks, cameras, lines, template head, scan, fits, texture
      renders/              comparison and preview images
      export/<ID>/          the finished PES face folder
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from . import landmarks as L
from .cameras import (CameraSolution, align_cameras_to_model, anchor_key, build_observations,
                      load_meshroom_sfm_checked, lm_key, solve_cameras)
from .config import Config
from .fitting import FitSettings, JawCurve, LineView, TargetSurface, adjust_region, fit_head, region_report
from .geometry import Camera
from .head import HeadModel, annotate_from_canonical, annotate_with_detector, head_view_cameras
from .mesh import Mesh, interpolate_displacement, vertex_normals, write_obj
from .photos import default_focal_px, list_photos, load_rgb, photo_size, save_rgb
from .regions import compute_regions

Log = Callable[[str], None]


def _nolog(_msg: str) -> None:
    pass


class PipelineError(RuntimeError):
    """A problem the person can fix (missing photos, step out of order...)."""


# ---------------------------------------------------------------------------
# Project


class Project:
    def __init__(self, root: Path):
        self.root = Path(root)
        if not (self.root / "project.json").exists():
            raise PipelineError(f"No project at {self.root}")
        self.state = json.loads((self.root / "project.json").read_text())

    # paths
    @property
    def name(self) -> str:
        return self.state["name"]

    @property
    def photos_dir(self) -> Path:
        return self.root / "photos"

    @property
    def work(self) -> Path:
        p = self.root / "work"
        p.mkdir(exist_ok=True)
        return p

    @property
    def renders(self) -> Path:
        p = self.root / "renders"
        p.mkdir(exist_ok=True)
        return p

    def path(self, name: str) -> Path:
        return self.work / name

    # state
    def save(self) -> None:
        (self.root / "project.json").write_text(json.dumps(self.state, indent=2))

    def mark(self, step: str, info: dict | None = None) -> None:
        self.state.setdefault("steps", {})[step] = {"at": time.strftime("%Y-%m-%d %H:%M"), **(info or {})}
        self.save()

    def done(self, step: str) -> bool:
        return step in self.state.get("steps", {})

    # data
    def photos(self) -> list[str]:
        return [p.name for p in list_photos(self.photos_dir)]

    def lines(self) -> dict:
        p = self.path("lines.json")
        return json.loads(p.read_text()) if p.exists() else {}

    def save_lines(self, data: dict) -> None:
        self.path("lines.json").write_text(json.dumps(data))

    def landmarks(self) -> list[L.PhotoLandmarks]:
        return L.load_landmarks(self.path("landmarks.json"))

    def cameras(self) -> CameraSolution:
        p = self.path("cameras.json")
        if not p.exists():
            raise PipelineError("Cameras are not solved yet: run solve_cameras first")
        return CameraSolution.load(p)

    def head(self) -> HeadModel:
        p = self.path("template_head.npz")
        if not p.exists():
            raise PipelineError("The template is not prepared yet: run prepare_template first")
        return HeadModel.load(p)

    def regions(self) -> dict[str, np.ndarray]:
        d = np.load(self.path("regions.npz"))
        return {k: d[k] for k in d.files}

    def fitted(self) -> np.ndarray:
        p = self.path("fit.npz")
        if not p.exists():
            raise PipelineError("No fit yet: run fit_head first")
        return np.load(p)["vertices"]

    def save_fitted(self, X: np.ndarray, note: str) -> None:
        hist = self.work / "fit_history"
        hist.mkdir(exist_ok=True)
        cur = self.path("fit.npz")
        if cur.exists():
            shutil.copy2(cur, hist / f"fit_{int(time.time() * 1000)}.npz")
        np.savez_compressed(cur, vertices=X, note=np.array(note))


def projects_dir(cfg: Config) -> Path:
    d = cfg.projects_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_projects(cfg: Config) -> list[dict]:
    out = []
    for p in sorted(projects_dir(cfg).iterdir()):
        if (p / "project.json").exists():
            st = json.loads((p / "project.json").read_text())
            out.append({"name": st["name"], "steps": sorted(st.get("steps", {})), "face_id": st.get("face_id")})
    return out


def open_project(cfg: Config, name: str) -> Project:
    return Project(projects_dir(cfg) / name)


def create_project(cfg: Config, name: str, photos_folder: str, template_folder: str,
                   face_id: str | None = None, log: Log = _nolog) -> dict:
    """Start a project: copy the photos and remember the template face folder."""
    from .pes.face_folder import TemplateFace

    root = projects_dir(cfg) / name
    if (root / "project.json").exists():
        raise PipelineError(f"A project called '{name}' already exists")
    src = Path(photos_folder).expanduser()
    photos = list_photos(src)
    if len(photos) < 3:
        raise PipelineError(f"Found {len(photos)} photos in {src}; at least 3 are needed (30-50 is best)")
    template = TemplateFace.open(Path(template_folder).expanduser())
    (root / "photos").mkdir(parents=True)
    for p in photos:
        shutil.copy2(p, root / "photos" / p.name)
    state = {"name": name, "template": str(Path(template_folder).expanduser()),
             "face_id": face_id or template.face_id, "created": time.strftime("%Y-%m-%d %H:%M"), "steps": {}}
    (root / "project.json").write_text(json.dumps(state, indent=2))
    log(f"project {name}: {len(photos)} photos, template {template.face_id}")
    return {"project": name, "photos": len(photos), "template": template.summary()}


# ---------------------------------------------------------------------------
# Landmarks and template


def detect_landmarks(project: Project, cfg: Config, log: Log = _nolog) -> dict:
    with L.LandmarkDetector(cfg.mediapipe_model_path()) as det:
        results = []
        for name in project.photos():
            r = det.detect_photo(project.photos_dir / name)
            log(f"{name}: {'face found' if r.points is not None else 'no face'}")
            results.append(r)
    L.save_landmarks(project.path("landmarks.json"), results)
    missing = [r.photo for r in results if r.points is None]
    project.mark("landmarks", {"found": len(results) - len(missing), "missing": missing})
    return {"photos": len(results), "faces_found": len(results) - len(missing), "no_face": missing,
            "hint": "Photos without a face (usually full profiles) need anchor clicks in the line editor."
            if missing else ""}


def prepare_template(project: Project, cfg: Config, log: Log = _nolog) -> dict:
    """Weld the template skin, place landmarks on it and compute regions."""
    from .pes.face_folder import TemplateFace
    from .raster import render

    t = TemplateFace.open(project.state["template"])
    head, smap, RV, RF, RUV = t.skin()
    tex = t.base_texture()
    method = "canonical"
    annotated = None
    model = cfg.mediapipe_model_path()
    if model.exists():
        try:
            with L.LandmarkDetector(model) as det:
                annotated = annotate_with_detector(head, (RV, RF, RUV), tex, det)
                method = "mediapipe"
        except Exception as exc:  # fall back below
            log(f"MediaPipe annotation failed ({exc}); using the canonical-face fit instead")
    if annotated is None:
        annotated, resid = annotate_from_canonical(head)
        log(f"canonical-face fit residual {resid * 1000:.1f} mm")
    annotated.save(project.path("template_head.npz"))
    regions = compute_regions(annotated.vertices, annotated.faces, annotated.landmark_positions())
    np.savez_compressed(project.path("regions.npz"), **regions)
    np.savez_compressed(project.path("skin_map.npz"), mapping=smap.mapping, offsets=np.array(smap.offsets),
                        meshes=np.array(smap.mesh_indices), faces=RF, uvs=RUV)
    if tex is not None:
        save_rgb(project.path("template_texture.png"), tex[..., :3])
    images = []
    for i, cam in enumerate(head_view_cameras(RV, (0.0, 40.0), 640)):
        img, _ = render(cam, RV, RF, RUV, tex, background=1.0)
        path = project.renders / f"template_{i}.png"
        save_rgb(path, img)
        images.append(str(path))
    info = {"template_id": t.face_id, "skin_vertices": len(annotated.vertices), "landmarks": len(annotated.lm_ids),
            "annotation": method, "meshes": t.summary()["face_high"]}
    project.mark("template", {"annotation": method})
    return {**info, "images": images}


def _template_skin(project: Project) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(project.path("skin_map.npz"))
    return d["mapping"], d["faces"], d["uvs"]


# ---------------------------------------------------------------------------
# Cameras


def _model_points(head: HeadModel) -> dict[str, np.ndarray]:
    lm = head.landmark_positions()
    pts = {lm_key(i): p for i, p in lm.items()}
    for name, idx in L.ANCHORS.items():
        if idx is not None and idx in lm:
            pts[anchor_key(name)] = lm[idx]
    return pts


def solve_project_cameras(project: Project, cfg: Config, log: Log = _nolog) -> dict:
    head = project.head()
    lms = project.landmarks()
    if not lms:
        raise PipelineError("Run detect_landmarks first")
    anchors = {photo: v.get("anchors", {}) for photo, v in project.lines().items() if v.get("anchors")}
    obs = build_observations(lms, anchors)
    sizes, focals = {}, {}
    for name in project.photos():
        w, h = photo_size(project.photos_dir / name)
        sizes[name] = (w, h)
        focals[name] = default_focal_px(w, h, project.photos_dir / name)
    model = _model_points(head)
    initial = None
    sfm = project.path("scan/cameras.sfm")
    if sfm.exists():
        mr, variant, err = load_meshroom_sfm_checked(sfm, obs)
        log(f"Meshroom cameras: {len(mr)} matched, layout {variant}, landmark error {err:.1f} px")
        if len(mr) >= 2:
            sim, tri_head = align_cameras_to_model(mr, obs, model)
            initial = {k: c.transformed(sim) for k, c in mr.items()}
            log(f"using {len(initial)} Meshroom cameras as a starting point")
    sol = solve_cameras(obs, sizes, focals, model, initial=initial)
    sol.source = "meshroom+landmarks" if initial else "landmarks"
    if initial:
        # map Meshroom's world onto the final solution (not just the template), so the scan
        # and the landmarks agree exactly
        from .geometry import robust_umeyama

        common = [k for k in tri_head if k in sol.points]
        src = np.array([sim.inverse().apply(tri_head[k][None])[0] for k in common])
        final = robust_umeyama(src, np.array([sol.points[k] for k in common]))
        np.save(project.path("meshroom_to_head.npy"), np.array([json.dumps(final.to_dict())]))
    sol.save(project.path("cameras.json"))
    worst = sorted(sol.errors.items(), key=lambda kv: -kv[1])[:5]
    project.mark("cameras", {"solved": len(sol.cameras), "failed": sol.failed})
    return {"solved": len(sol.cameras), "of": len(project.photos()), "failed": sol.failed,
            "mean_error_px": round(float(np.mean(list(sol.errors.values()))), 2) if sol.errors else None,
            "worst_photos": [(k, round(v, 1)) for k, v in worst], "source": sol.source,
            "hint": ("Photos listed under 'failed' have no camera yet. Open the line editor and click the "
                     "anchors (nose tip, eye corner, mouth corner, ear) on them, then solve again.")
            if sol.failed else ""}


def _person_landmarks(sol: CameraSolution) -> dict[int, np.ndarray]:
    return {int(k[3:]): v for k, v in sol.points.items() if k.startswith("lm:")}


def _line_views(project: Project, sol: CameraSolution) -> list[LineView]:
    out = []
    for photo, entry in project.lines().items():
        polys = [np.asarray(p, float) for p in entry.get("jaw", []) if len(p) >= 2]
        if polys and photo in sol.cameras:
            out.append(LineView(photo, sol.cameras[photo], polys))
    return out


# ---------------------------------------------------------------------------
# Scan (Meshroom)


def run_scan(project: Project, cfg: Config, log: Log = _nolog) -> dict:
    import subprocess

    from .scan import find_meshroom, find_outputs, meshroom_command

    meshroom = find_meshroom(cfg.meshroom)
    if meshroom is None:
        raise PipelineError("Meshroom was not found. Run the installer or set its path with set_config.")
    out = project.work / "scan" / "output"
    cache = project.work / "scan" / "cache"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    cmd = meshroom_command(meshroom, project.photos_dir, out, cache)
    log("running: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in proc.stdout:
        log(line.rstrip())
    code = proc.wait()
    mesh, sfm = find_outputs(out, cache)
    if sfm is not None:
        shutil.copy2(sfm, project.path("scan/cameras.sfm"))
    if code != 0 or mesh is None:
        raise PipelineError(f"Meshroom stopped (exit code {code}). Skin often lacks texture for "
                            "photogrammetry: more overlapping photos and even light help. The pipeline "
                            "still works without a scan (landmarks, face model and jaw lines).")
    project.state["scan_mesh"] = str(mesh)
    project.mark("scan", {"mesh": str(mesh)})
    return {"mesh": str(mesh), "cameras": str(sfm) if sfm else None,
            "next": "Run solve_cameras again (it uses Meshroom's cameras), then import_scan."}


def import_scan(project: Project, cfg: Config, log: Log = _nolog) -> dict:
    from .geometry import Similarity
    from .scan import load_scan

    mesh_path = project.state.get("scan_mesh")
    sim_path = project.path("meshroom_to_head.npy")
    if not mesh_path or not sim_path.exists():
        raise PipelineError("Run run_scan and then solve_cameras before import_scan")
    sim = Similarity.from_dict(json.loads(str(np.load(sim_path)[0])))
    head = project.head()
    lm = head.landmark_positions()
    centre = np.mean([lm[i] for i in (L.NOSE_TIP, L.EYE_R_OUTER, L.EYE_L_OUTER, L.CHIN) if i in lm], axis=0)
    centre = centre - np.array([0, 0, 0.06])
    scan = load_scan(Path(mesh_path), sim, centre)
    np.savez_compressed(project.path("scan_head.npz"), vertices=scan.vertices, faces=scan.faces)
    write_obj(project.path("scan_head.obj"), scan)
    project.mark("scan_import", {"vertices": len(scan.vertices)})
    return {"scan_vertices": len(scan.vertices), "obj": str(project.path("scan_head.obj"))}


def _scan(project: Project) -> tuple[np.ndarray, np.ndarray] | None:
    p = project.path("scan_head.npz")
    if not p.exists():
        return None
    d = np.load(p)
    return d["vertices"], d["faces"]


# ---------------------------------------------------------------------------
# Face model (FLAME)


def fit_face_model(project: Project, cfg: Config, log: Log = _nolog) -> dict:
    from .flame import FlameModel, fit_flame

    path = cfg.flame_model_path()
    if path is None:
        raise PipelineError("No FLAME model found. Register at https://flame.is.tue.mpg.de, download "
                            "FLAME 2020 or 2023, and put generic_model.pkl in "
                            f"{cfg.models_dir}. The pipeline works without it, with less help under the beard.")
    model = FlameModel.load(path)
    lm_head = model.landmark_head(cfg.flame_embedding_path())
    sol = project.cameras()
    targets = _person_landmarks(sol)
    fit = fit_flame(model, lm_head, targets, scan=_scan(project), line_views=_line_views(project, sol))
    np.savez_compressed(project.path("flame_fit.npz"), vertices=fit.vertices, faces=model.faces, beta=fit.beta)
    write_obj(project.path("flame_fit.obj"), Mesh(fit.vertices, model.faces))
    project.mark("face_model", {"landmark_rms_mm": round(fit.landmark_rms_mm, 2)})
    return {"landmark_rms_mm": round(fit.landmark_rms_mm, 2), "obj": str(project.path("flame_fit.obj"))}


def _flame(project: Project) -> tuple[np.ndarray, np.ndarray] | None:
    p = project.path("flame_fit.npz")
    if not p.exists():
        return None
    d = np.load(p)
    return d["vertices"], d["faces"]


# ---------------------------------------------------------------------------
# Head fit


def fit(project: Project, cfg: Config, log: Log = _nolog, stiffness: float | None = None,
        line_weight: float | None = None, landmark_weight: float | None = None,
        iterations: int | None = None) -> dict:
    head = project.head()
    regions = project.regions()
    sol = project.cameras()
    targets = _person_landmarks(sol)
    if len(targets) < 30:
        raise PipelineError("Too few 3D landmarks; solve the cameras with more photos first")
    surfaces = []
    scan = _scan(project)
    if scan is not None:
        w = regions["face"] * (1 - regions["beard"]) * (1 - 0.7 * regions["eyes"])
        w = np.maximum(w, 0.4 * regions["ears"])
        surfaces.append(TargetSurface("scan", scan[0], scan[1], w))
    flame = _flame(project)
    if flame is not None:
        w = np.maximum.reduce([0.8 * regions["beard"], 0.6 * regions["jaw"], 0.5 * regions["neck"],
                               0.5 * regions["scalp"], 0.15 * regions["face"]])
        surfaces.append(TargetSurface("face_model", flame[0], flame[1], w))
    settings = FitSettings()
    if stiffness is not None:
        settings.stiffness_end = float(stiffness)
        settings.stiffness_start = max(float(stiffness) * 10, settings.stiffness_start)
    if line_weight is not None:
        settings.line_weight = float(line_weight)
    if landmark_weight is not None:
        settings.landmark_weight = float(landmark_weight)
    if iterations is not None:
        settings.iterations = int(iterations)
    lines = _line_views(project, sol)
    X, report = fit_head(head, targets, surfaces, lines, settings, regions=regions)
    project.save_fitted(X, "fit_head")
    info = {**report.to_dict(), "surfaces": [s.name for s in surfaces], "jaw_line_photos": [lv.photo for lv in lines],
            "settings": asdict(settings)}
    project.mark("fit", {k: info[k] for k in ("landmark_rms_mm", "surfaces")})
    if not lines:
        info["hint"] = ("No jaw lines yet. With a beard, drawing the jaw on 2-4 photos in the line editor "
                        "makes the jaw much more accurate.")
    return info


def adjust(project: Project, region: str, translate_mm=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0),
           strength: float = 1.0) -> dict:
    head = project.head()
    X = project.fitted()
    regions = project.regions()
    X2 = adjust_region(head, X, regions, region, tuple(translate_mm), tuple(scale), strength)
    project.save_fitted(X2, f"adjust {region}")
    moved = float(np.linalg.norm(X2 - X, axis=1).max() * 1000)
    return {"region": region, "max_move_mm": round(moved, 2)}


def undo(project: Project) -> dict:
    hist = sorted((project.work / "fit_history").glob("fit_*.npz"))
    if not hist:
        raise PipelineError("Nothing to undo")
    shutil.copy2(hist[-1], project.path("fit.npz"))
    hist[-1].unlink()
    return {"restored": hist[-1].name, "remaining_history": len(hist) - 1}


def describe_fit(project: Project) -> dict:
    head = project.head()
    return region_report(project.fitted(), head.vertices, project.regions())


# ---------------------------------------------------------------------------
# Rendering and comparison


def _fitted_skin(project: Project) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unwelded skin with fitted positions: (vertices, faces, uvs, normals)."""
    mapping, RF, RUV = _template_skin(project)
    X = project.fitted() if project.path("fit.npz").exists() else project.head().vertices
    head = project.head()
    vn = vertex_normals(X, head.faces)
    return X[mapping], RF, RUV, vn[mapping]


def _texture_for_preview(project: Project) -> np.ndarray | None:
    for name in ("texture.png", "template_texture.png"):
        p = project.path(name)
        if p.exists():
            return load_rgb(p)
    return None


def compare(project: Project, cfg: Config, photos: list[str] | None = None, log: Log = _nolog,
            max_views: int = 6) -> dict:
    from .compare import compare_view

    sol = project.cameras()
    head = project.head()
    X = project.fitted() if project.path("fit.npz").exists() else head.vertices
    V, F, UV, _ = _fitted_skin(project)
    tex = _texture_for_preview(project)
    lm_photo = {p.photo: p.points for p in project.landmarks()}
    lines = project.lines()
    jaw = JawCurve(head).positions(X, head.faces)
    names = photos or _spread_views(sol, max_views)
    out_dir = project.renders / "compare"
    out_dir.mkdir(exist_ok=True)
    results = []
    for name in names:
        if name not in sol.cameras:
            continue
        img = load_rgb(project.photos_dir / name)
        user = [np.asarray(p, float) for p in lines.get(name, {}).get("jaw", [])]
        r = compare_view(name, img, sol.cameras[name], V, F, lm_photo.get(name), head.landmark_positions(X),
                         jaw, user, str(out_dir / (Path(name).stem + ".png")), UV, tex)
        results.append(r.to_dict())
        log(f"{name}: {r.landmark_error_pct}")
    errs = [r["landmark_error_pct"] for r in results if r["landmark_error_pct"] is not None]
    return {"views": results, "mean_landmark_error_pct": round(float(np.mean(errs)), 2) if errs else None,
            "how_to_read": "Each image: photo | render from the photo's camera | overlay. Red dots: head "
                           "landmarks, yellow lines: offset to the photo's landmarks, red curve: head jaw, "
                           "green: jaw lines drawn by the person. Error is % of the eye-to-eye distance."}


def _spread_views(sol: CameraSolution, k: int) -> list[str]:
    """Pick up to k photos spread around the head (by camera yaw)."""
    items = []
    for name, cam in sol.cameras.items():
        c = cam.center
        items.append((float(np.degrees(np.arctan2(c[0], c[2]))), name))
    items.sort()
    if len(items) <= k:
        return [n for _, n in items]
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i][1] for i in idx]


def render_preview(project: Project, cfg: Config, style: str = "textured", photos: list[str] | None = None,
                   log: Log = _nolog) -> dict:
    """Preview renders from photo cameras (Blender if installed, else the built-in renderer)."""
    from .blender import find_blender, run_job, write_mesh_npz
    from .raster import render

    sol = project.cameras()
    V, F, UV, _ = _fitted_skin(project)
    names = photos or _spread_views(sol, 4)
    out = project.renders / "preview"
    out.mkdir(exist_ok=True)
    blender = find_blender(cfg.blender)
    tex_path = project.path("texture.png") if project.path("texture.png").exists() else project.path("template_texture.png")
    if blender is not None:
        npz = write_mesh_npz(project.path("preview_head.npz"), V, F, UV)
        job = {"meshes": [{"name": "head", "npz": str(npz), "texture": str(tex_path) if tex_path.exists() else None}],
               "cameras": [{"name": Path(n).stem, **sol.cameras[n].to_dict()} for n in names if n in sol.cameras],
               "style": style, "renders_dir": str(out), "samples": 8, "max_side": 900}
        files = run_job(blender, job, project.work / "blender")
        return {"renderer": "blender", "images": files}
    tex = load_rgb(tex_path) if (style == "textured" and tex_path.exists()) else None
    files = []
    for n in names:
        cam = sol.cameras[n].scaled(min(1.0, 900 / max(sol.cameras[n].width, sol.cameras[n].height)))
        img, _ = render(cam, V, F, UV, tex, background=1.0)
        p = out / (Path(n).stem + ".png")
        save_rgb(p, img)
        files.append(str(p))
    return {"renderer": "built-in", "images": files}


def open_in_blender(project: Project, cfg: Config, log: Log = _nolog) -> dict:
    from .blender import find_blender, open_in_ui, run_job, write_mesh_npz

    blender = find_blender(cfg.blender)
    if blender is None:
        raise PipelineError("Blender was not found. Install it or set its path with set_config.")
    sol = project.cameras()
    V, F, UV, _ = _fitted_skin(project)
    npz = write_mesh_npz(project.path("preview_head.npz"), V, F, UV)
    tex_path = project.path("texture.png") if project.path("texture.png").exists() else project.path("template_texture.png")
    meshes = [{"name": "head", "npz": str(npz), "texture": str(tex_path) if tex_path.exists() else None}]
    if project.path("scan_head.npz").exists():
        meshes.append({"name": "scan", "npz": str(project.path("scan_head.npz")), "texture": None})
    blend = project.work / "facegen_scene.blend"
    job = {"meshes": meshes, "style": "textured",
           "cameras": [{"name": Path(n).stem, **c.to_dict(), "background": str(project.photos_dir / n)}
                       for n, c in sol.cameras.items()],
           "save_blend": str(blend)}
    run_job(blender, job, project.work / "blender")
    open_in_ui(blender, blend)
    return {"blend": str(blend), "tip": "Look through a photo camera (Numpad 0) to see the head over the photo."}


# ---------------------------------------------------------------------------
# Texture


def bake(project: Project, cfg: Config, log: Log = _nolog, shadow_flatten: float = 0.0) -> dict:
    from .texture import BakeSettings, bake_texture

    sol = project.cameras()
    V, F, UV, N = _fitted_skin(project)
    tpl = project.path("template_texture.png")
    if not tpl.exists():
        raise PipelineError("The template has no readable face_bsm_alp texture")
    from .pes.face_folder import TemplateFace

    template_rgba = TemplateFace.open(project.state["template"]).base_texture()
    photos = {}
    for name in project.photos():
        if name in sol.cameras:
            photos[name] = load_rgb(project.photos_dir / name)
    log(f"baking from {len(photos)} photos")
    res = bake_texture(V, F, UV, N, photos, sol.cameras, template_rgba, BakeSettings(shadow_flatten=shadow_flatten))
    from PIL import Image

    Image.fromarray(res.texture, "RGBA").save(project.path("texture_rgba.png"))
    save_rgb(project.path("texture.png"), res.texture[..., :3])
    save_rgb(project.renders / "texture_coverage.png", np.repeat(res.coverage[..., None], 3, axis=2))
    covered = float((res.coverage > 0.3).mean())
    project.mark("texture", {"covered": round(covered, 3)})
    return {"texture": str(project.path("texture.png")), "coverage_image": str(project.renders / "texture_coverage.png"),
            "covered_fraction_of_uv": round(covered, 3), "per_photo_coverage": res.per_photo_coverage,
            "exposure_gains": res.gains}


# ---------------------------------------------------------------------------
# Hair from another face


def _hair_transfer(project: Project, other_folder: str, log: Log = _nolog):
    """Return (TemplateFace, displacement function) moving another face's hair onto ours."""
    from .pes.face_folder import TemplateFace

    other = TemplateFace.open(other_folder)
    ohead, _, _, _, _ = other.skin()
    ohead, _ = annotate_from_canonical(ohead)
    head = project.head()
    X = project.fitted()
    ours = head.landmark_positions(X)
    targets = {i: p for i, p in ours.items()}
    regions = compute_regions(ohead.vertices, ohead.faces, ohead.landmark_positions())
    surf = TargetSurface("ours", X, head.faces, np.clip(regions["scalp"] + regions["face"] + regions["neck"]
                                                        + regions["beard"] + regions["ears"], 0, 1))
    OX, _ = fit_head(ohead, targets, [surf], [], FitSettings(iterations=8), regions=regions)
    disp = OX - ohead.vertices
    base = ohead.vertices
    log(f"hair from {other.face_id}: max move {np.linalg.norm(disp, axis=1).max() * 1000:.1f} mm")
    return other, (lambda pts: interpolate_displacement(base, disp, pts))


def preview_face(folder: str, out_path: str, yaw: float = 25.0) -> str:
    """Render a face folder (skin + hair) so Claude and the person can judge hair and beard styles."""
    from .pes import ftex
    from .pes.face_folder import TemplateFace, _find_ci
    from .raster import rasterize, interpolate, sample_texture_uv

    t = TemplateFace.open(folder)
    _, _, RV, RF, RUV = t.skin()
    cam = head_view_cameras(RV, (yaw,), 640)[0]
    from .raster import render

    img, frag = render(cam, RV, RF, RUV, t.base_texture(), background=1.0)
    if t.hair is not None and t.texture_dir is not None:
        for m in t.hair.meshes:
            if m.uv is None or len(m.faces) == 0:
                continue
            bt = m.base_texture()
            tex = None
            if bt:
                p = _find_ci(t.texture_dir, Path(bt[0]).with_suffix(".ftex").name)
                if p:
                    tex = ftex.decode(p.read_bytes()).astype(float) / 255.0
            hf = rasterize(cam, m.positions, m.faces)
            cover = hf.mask & (hf.depth < frag.depth)
            if not cover.any():
                continue
            uv = interpolate(hf, m.faces, m.uv)
            if tex is not None:
                col = sample_texture_uv(tex, uv[cover])
                keep = col[:, 3] > 0.45
                ys, xs = np.nonzero(cover)
                img[ys[keep], xs[keep]] = col[keep, :3]
            else:
                img[cover] = 0.15
    save_rgb(out_path, img)
    return out_path


def find_faces(folder: str, limit: int = 200) -> list[str]:
    root = Path(folder).expanduser()
    out = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and any(c.name.lower() == "#win" for c in p.iterdir() if c.is_dir()):
            out.append(p.name)
            if len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------------------
# Export


def export(project: Project, cfg: Config, face_id: str | None = None, hair_from: str | None = None,
           install: bool = False, log: Log = _nolog) -> dict:
    from PIL import Image

    from .pes.face_folder import TemplateFace, export_face

    t = TemplateFace.open(project.state["template"])
    head_raw, smap, _, _, _ = t.skin()
    head = project.head()
    if len(head_raw.vertices) != len(head.vertices):
        raise PipelineError("The template changed since prepare_template; run it again")
    X = project.fitted()
    tex = None
    if project.path("texture_rgba.png").exists():
        tex = np.asarray(Image.open(project.path("texture_rgba.png")).convert("RGBA"))
    fid = str(face_id or project.state.get("face_id") or t.face_id)
    other, disp_fn = (None, None)
    if hair_from:
        other, disp_fn = _hair_transfer(project, hair_from, log)
    res = export_face(t, head_raw, smap, X, project.root / "export", fid, tex, other, disp_fn)
    info = {"folder": str(res.folder), "face_id": fid, "notes": res.notes}
    if install:
        if not cfg.faces_folder:
            raise PipelineError("Set faces_folder first (your ...\\livecpk\\Faces\\Asset\\model\\character\\face\\real)")
        dest = Path(cfg.faces_folder) / fid
        if dest.exists():
            backup = project.root / "backup" / f"{fid}_{int(time.time())}"
            backup.parent.mkdir(exist_ok=True)
            shutil.move(str(dest), str(backup))
            info["backup_of_previous"] = str(backup)
        shutil.copytree(res.folder, dest)
        (dest / "facegen_export.json").unlink(missing_ok=True)
        info["installed_to"] = str(dest)
    project.mark("export", {"face_id": fid, "installed": install})
    info["next"] = ("Copy the folder into ...\\livecpk\\Faces\\Asset\\model\\character\\face\\real (Sider), "
                    "start FL27 and check the face in Edit mode.") if not install else "Start FL27 and check the face in Edit mode."
    return info


def export_test_face(cfg: Config, template_folder: str, face_id: str, nose_mm: float = 0.0,
                     log: Log = _nolog) -> dict:
    """Write a copy of a face through FaceGen's PES writer, optionally with the nose pushed forward.

    Use it once before the first real face: if this copy looks right in FL27, the file
    round trip works on your install. With ``nose_mm`` > 0 the change is easy to spot.
    """
    from .pes.face_folder import TemplateFace, export_face

    t = TemplateFace.open(template_folder)
    head, smap, _, _, _ = t.skin()
    X = head.vertices.copy()
    if nose_mm:
        annotated, _ = annotate_from_canonical(head)
        nose = annotated.landmark_positions()[L.NOSE_TIP]
        w = np.exp(-np.sum((X - nose) ** 2, axis=1) / 0.015 ** 2)
        X[:, 2] += w * nose_mm / 1000.0
    out = cfg.home_path / "test_exports"
    res = export_face(t, head, smap, X, out, face_id)
    return {"folder": str(res.folder), "face_id": face_id, "nose_moved_mm": nose_mm,
            "next": "Copy it into your faces folder, assign the ID to a player and check it in FL27. "
                    "If it loads and looks like the original (with the nose change), exporting works."}


# ---------------------------------------------------------------------------
# Status


def status(project: Project, cfg: Config) -> dict:
    steps = project.state.get("steps", {})
    lines = project.lines()
    order = ["landmarks", "template", "cameras", "scan", "scan_import", "face_model", "fit", "texture", "export"]
    nxt = next((s for s in order if s not in steps and s not in ("scan", "scan_import", "face_model")), None)
    return {
        "project": project.name,
        "photos": len(project.photos()),
        "template": project.state.get("template"),
        "face_id": project.state.get("face_id"),
        "steps_done": {k: steps[k] for k in order if k in steps},
        "jaw_lines_on": [p for p, v in lines.items() if v.get("jaw")],
        "anchors_on": [p for p, v in lines.items() if v.get("anchors")],
        "optional": {"scan (Meshroom)": "scan_import" in steps, "face model (FLAME)": "face_model" in steps,
                     "flame_installed": cfg.flame_model_path() is not None},
        "next_step": nxt,
    }
