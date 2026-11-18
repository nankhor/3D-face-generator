"""MCP server: exposes the face pipeline to Claude.

Start with ``facegen-mcp`` (the installer registers it with Claude Desktop and
Claude Code). Long steps run as background jobs; poll them with ``job_status``.
"""

from __future__ import annotations

import functools
import io
import json
import traceback
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from . import pipeline as P
from .config import Config
from .jobs import JOBS

try:  # mcp >= 2
    from mcp.server.mcpserver import Image, MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server, Image  # type: ignore

INSTRUCTIONS = """FaceGen builds a PES 2021 / Football Life face from photos of a real person.

Typical order (each step's result says what to do next):
1. check_setup. Before the first real face, export_test_face once and have the person check it in FL27.
   Then create_project (photos folder + a template face folder from the game).
2. detect_landmarks, prepare_template, then open_line_editor so the person draws their jaw
   (essential with a beard) and clicks anchors on side photos.
3. solve_cameras. Optional: run_scan (Meshroom, slow) then solve_cameras again and import_scan.
   Optional: fit_face_model (needs the FLAME model).
4. fit_head, then compare_to_photos. Look at the images: fix what you see with adjust_region
   (millimetres, head axes: x = the character's left, y = up, z = forward), compare again. undo_fit reverts.
5. bake_texture, compare again, pick hair with find_face_folders/preview_face_folder, export_face.
Long steps return a job id: call job_status until it is done. Never claim likeness from numbers
alone: show the comparison images to the person and let them judge."""

mcp = _Server("facegen", instructions=INSTRUCTIONS)
_editors: dict[str, Any] = {}


def _cfg() -> Config:
    return Config.load()


def _project(name: str) -> P.Project:
    return P.open_project(_cfg(), name)


def _image(path: str | Path, max_side: int = 1400) -> Image:
    img = PILImage.open(path)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=85)
    return Image(data=buf.getvalue(), format="jpeg")


def _text(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def _safe(fn):
    """Return errors as readable text instead of crashing the tool call."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except P.PipelineError as exc:
            return f"Cannot do that yet: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"Error: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"

    return wrapper


def _start(name: str, project: str, fn, **kwargs) -> str:
    running = JOBS.running(name)
    if any(j for j in running if j.name == name):
        return f"A '{name}' job is already running ({running[0].id}). Check it with job_status."
    cfg = _cfg()
    proj = _project(project)
    job = JOBS.start(name, fn, proj, cfg, **kwargs)
    return f"Started {name} as job {job.id}. Check it with job_status('{job.id}')."


# ---------------------------------------------------------------- setup


@mcp.tool()
@_safe
def check_setup() -> str:
    """Report which parts of the toolchain are installed (MediaPipe model, Blender, Meshroom, FLAME)."""
    from .selftest import environment_report

    return _text(environment_report(_cfg()))


@mcp.tool()
@_safe
def set_config(blender: str = "", meshroom: str = "", flame_model: str = "", faces_folder: str = "") -> str:
    """Save tool paths. faces_folder is FL27's ...\\livecpk\\Faces\\Asset\\model\\character\\face\\real
    (used by export_face(install=True) and find_face_folders). Empty values are left unchanged."""
    cfg = _cfg()
    for key, val in (("blender", blender), ("meshroom", meshroom), ("flame_model", flame_model),
                     ("faces_folder", faces_folder)):
        if val:
            setattr(cfg, key, val)
    cfg.save()
    return _text({k: getattr(cfg, k) for k in ("home", "blender", "meshroom", "flame_model", "faces_folder")})


@mcp.tool()
def photo_guide() -> str:
    """How to take the photos (share this with the person before they shoot)."""
    return (Path(__file__).parent / "data" / "photo_guide.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- projects


@mcp.tool()
@_safe
def list_projects() -> str:
    """List face projects."""
    return _text(P.list_projects(_cfg()))


@mcp.tool()
@_safe
def create_project(name: str, photos_folder: str, template_face_folder: str, face_id: str = "") -> str:
    """Create a project. photos_folder: the person's photos. template_face_folder: an existing PES 2021
    face folder (named after a player ID, containing #Win/face.fpk) to morph; pick one with a similar
    head shape and skin tone. face_id: the player ID the new face should use (default: template's)."""
    return _text(P.create_project(_cfg(), name, photos_folder, template_face_folder, face_id or None))


@mcp.tool()
@_safe
def project_status(project: str) -> str:
    """What is done for a project and what to do next."""
    return _text(P.status(_project(project), _cfg()))


@mcp.tool()
@_safe
def find_face_folders(folder: str = "", limit: int = 200) -> str:
    """List face folders (player IDs) in a faces folder; defaults to the configured faces_folder."""
    folder = folder or _cfg().faces_folder or ""
    if not folder:
        return "Give a folder, or set faces_folder with set_config."
    return _text(P.find_faces(folder, limit))


@mcp.tool()
@_safe
def preview_face_folder(folder: str, yaw: float = 25.0) -> list:
    """Render an existing face folder (skin and hair) to judge it as a template or a hair/beard source."""
    out = Path(_cfg().home) / "previews"
    out.mkdir(parents=True, exist_ok=True)
    path = P.preview_face(folder, str(out / f"{Path(folder).name}_{int(yaw)}.png"), yaw)
    from .pes.face_folder import TemplateFace

    return [_text(TemplateFace.open(folder).summary()), _image(path)]


# ---------------------------------------------------------------- steps


@mcp.tool()
@_safe
def detect_landmarks(project: str) -> str:
    """Find face landmarks on every photo (background job)."""
    return _start("detect_landmarks", project, lambda p, c, log: P.detect_landmarks(p, c, log))


@mcp.tool()
@_safe
def prepare_template(project: str) -> str:
    """Read the template face, place landmarks on it and compute head regions (background job)."""
    return _start("prepare_template", project, lambda p, c, log: P.prepare_template(p, c, log))


@mcp.tool()
@_safe
def open_line_editor(project: str) -> str:
    """Open the jaw-line editor in the person's browser. They draw where their jawbone is on 2-4 photos
    (side and front) and click anchors on photos where no face was detected. Lines save automatically."""
    from .line_editor.server import LineEditor

    if project in _editors:
        return f"The editor is already open at {_editors[project].url}"
    ed = LineEditor(_project(project))
    url = ed.start(open_browser=True)
    _editors[project] = ed
    return (f"Line editor opened at {url}. Ask the person to draw their jaw (bone, not beard edge) on a "
            "side photo and a front photo, then tell you when they are done.")


@mcp.tool()
@_safe
def solve_cameras(project: str) -> str:
    """Estimate where each photo was taken from (background job). Uses landmarks, anchors and, if a scan
    exists, Meshroom's cameras."""
    return _start("solve_cameras", project, lambda p, c, log: P.solve_project_cameras(p, c, log))


@mcp.tool()
@_safe
def run_scan(project: str) -> str:
    """Optional photogrammetry scan with Meshroom (background job, 20-60 minutes on a GPU)."""
    return _start("run_scan", project, lambda p, c, log: P.run_scan(p, c, log))


@mcp.tool()
@_safe
def import_scan(project: str) -> str:
    """Bring the Meshroom scan into head space (after run_scan and solve_cameras)."""
    return _text(P.import_scan(_project(project), _cfg()))


@mcp.tool()
@_safe
def fit_face_model(project: str) -> str:
    """Optional: fit the FLAME face model for a plausible jaw under the beard (background job)."""
    return _start("fit_face_model", project, lambda p, c, log: P.fit_face_model(p, c, log))


@mcp.tool()
@_safe
def fit_head(project: str, stiffness: float = 0.0, line_weight: float = 0.0, landmark_weight: float = 0.0,
             iterations: int = 0) -> str:
    """Deform the template head to the person (background job). Leave values at 0 for defaults.
    Lower stiffness (default 3) follows the data more closely; higher keeps the template smoother.
    line_weight (default 4) is how strongly the jaw follows the drawn lines."""
    kw = {k: v for k, v in (("stiffness", stiffness), ("line_weight", line_weight),
                            ("landmark_weight", landmark_weight), ("iterations", iterations)) if v}
    return _start("fit_head", project, lambda p, c, log, **k: P.fit(p, c, log, **k), **kw)


@mcp.tool()
@_safe
def compare_to_photos(project: str, photos: list[str] | None = None, max_views: int = 4) -> list:
    """Render the current head from photo cameras next to the photos. Returns numbers and images
    (photo | render | overlay). Use them to decide adjustments; show them to the person."""
    res = P.compare(_project(project), _cfg(), photos, max_views=max_views)
    out: list = [_text(res)]
    for v in res["views"][:max_views]:
        if v.get("image"):
            out.append(_image(v["image"], 1800))
    return out


@mcp.tool()
@_safe
def adjust_region(project: str, region: str, translate_mm: list[float] | None = None,
                  scale: list[float] | None = None, strength: float = 1.0) -> str:
    """Move or scale a region of the fitted head. Regions: face, beard, jaw, chin, neck, scalp, ears, nose,
    eyes, brows, mouth, cheeks, forehead. translate_mm = [x, y, z] in millimetres (x toward the character's
    left, y up, z forward). scale = [sx, sy, sz] around the region centre (e.g. [0.97, 1, 1] narrows)."""
    return _text(P.adjust(_project(project), region, translate_mm or [0, 0, 0], scale or [1, 1, 1], strength))


@mcp.tool()
@_safe
def undo_fit(project: str) -> str:
    """Revert the last fit or adjustment."""
    return _text(P.undo(_project(project)))


@mcp.tool()
@_safe
def describe_fit(project: str) -> str:
    """How far each region moved from the template, in millimetres."""
    return _text(P.describe_fit(_project(project)))


@mcp.tool()
@_safe
def bake_texture(project: str, shadow_flatten: float = 0.0) -> str:
    """Project the photos onto the head to make the face texture, beard included (background job).
    shadow_flatten 0..1 evens out baked-in lighting. Keep 0 for bearded faces: flattening also
    lightens a dark beard."""
    return _start("bake_texture", project, lambda p, c, log, **k: P.bake(p, c, log, **k),
                  shadow_flatten=shadow_flatten)


@mcp.tool()
@_safe
def render_preview(project: str, style: str = "textured", photos: list[str] | None = None) -> str:
    """Preview renders from photo cameras, with Blender if installed (background job).
    style: textured or clay."""
    return _start("render_preview", project, lambda p, c, log, **k: P.render_preview(p, c, log=log, **k),
                  style=style, photos=photos)


@mcp.tool()
@_safe
def open_in_blender(project: str) -> str:
    """Open the head with every photo as a camera background in the Blender UI."""
    return _text(P.open_in_blender(_project(project), _cfg()))


@mcp.tool()
@_safe
def export_face(project: str, face_id: str = "", hair_from_folder: str = "", install: bool = False) -> str:
    """Write the finished PES face folder (background job). face_id: player ID to use.
    hair_from_folder: take hair and beard meshes from another face folder (fitted onto this head).
    install: also copy it into the configured faces_folder (the previous face is backed up)."""
    return _start("export_face", project, lambda p, c, log, **k: P.export(p, c, log=log, **k),
                  face_id=face_id or None, hair_from=hair_from_folder or None, install=install)


@mcp.tool()
@_safe
def export_test_face(template_face_folder: str, face_id: str, nose_mm: float = 4.0) -> str:
    """Do this once first: copy a game face through FaceGen's writer (nose pushed forward by nose_mm so the
    change is visible). If it works in FL27, real exports will too."""
    return _text(P.export_test_face(_cfg(), template_face_folder, face_id, nose_mm))


@mcp.tool()
@_safe
def run_basic_pipeline(project: str) -> str:
    """Run landmarks, template, cameras, face model (if installed), fit, texture and comparison in one
    background job. Good for a first pass; jaw lines should be drawn before for bearded faces."""

    def run(p, c, log):
        out = {}
        out["landmarks"] = P.detect_landmarks(p, c, log)
        out["template"] = P.prepare_template(p, c, log)
        out["cameras"] = P.solve_project_cameras(p, c, log)
        if c.flame_model_path() is not None:
            try:
                out["face_model"] = P.fit_face_model(p, c, log)
            except Exception as exc:  # noqa: BLE001
                out["face_model"] = f"skipped: {exc}"
        out["fit"] = P.fit(p, c, log)
        out["texture"] = P.bake(p, c, log)
        out["compare"] = P.compare(p, c, log=log)
        return out

    return _start("run_basic_pipeline", project, run)


@mcp.tool()
@_safe
def job_status(job_id: str = "") -> list:
    """Status of a background job (or all jobs when job_id is empty). Returns images the job produced."""
    if not job_id:
        return [_text([j.to_dict(tail=3) for j in JOBS.all()])]
    job = JOBS.get(job_id)
    if job is None:
        return [f"No job {job_id}. Jobs are lost when Claude restarts the server."]
    out: list = [_text(job.to_dict())]
    if job.status == "done":
        for path in _image_paths(job.result)[:4]:
            out.append(_image(path))
    return out


def _image_paths(obj) -> list[str]:
    found = []
    if isinstance(obj, dict):
        for v in obj.values():
            found += _image_paths(v)
    elif isinstance(obj, list):
        for v in obj:
            found += _image_paths(v)
    elif isinstance(obj, str) and obj.lower().endswith((".png", ".jpg", ".jpeg")) and Path(obj).exists():
        found.append(obj)
    return found


@mcp.tool()
@_safe
def show_image(path: str) -> list:
    """Show any image the pipeline wrote (renders, texture, coverage map)."""
    return [_image(path, 1800)]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
