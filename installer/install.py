"""FaceGen installer, second stage (runs inside the project's virtual environment).

install.ps1 / install.sh create the environment and then call this script. It:

1. creates the FaceGen home folder (~/FaceGen),
2. downloads the MediaPipe face model,
3. finds Blender (offers winget on Windows) and Meshroom (offers to download it),
4. stops for the FLAME model, which needs your own registration (optional),
5. asks for FL27's faces folder (optional),
6. registers the MCP server with Claude Desktop and Claude Code,
7. runs the self-test.

Every download and change is announced first. Re-running is safe.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from facegen.blender import blender_version, find_blender  # noqa: E402
from facegen.config import Config  # noqa: E402
from facegen.scan import find_meshroom  # noqa: E402

MEDIAPIPE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
                 "face_landmarker.task")
FLAME_URL = "https://flame.is.tue.mpg.de/download.php"
IS_WINDOWS = platform.system() == "Windows"


class Installer:
    def __init__(self, yes: bool, skip_meshroom: bool):
        self.yes = yes
        self.skip_meshroom = skip_meshroom
        self.cfg = Config.load()
        self.manual: list[str] = []

    # -- helpers ---------------------------------------------------------
    def say(self, msg: str = "") -> None:
        print(msg, flush=True)

    def step(self, title: str) -> None:
        self.say()
        self.say(f"== {title}")

    def ask(self, question: str, default: bool = True) -> bool:
        if self.yes:
            return default
        hint = "Y/n" if default else "y/N"
        try:
            ans = input(f"{question} [{hint}] ").strip().lower()
        except EOFError:
            return default
        return default if not ans else ans.startswith("y")

    def prompt(self, question: str) -> str:
        if self.yes:
            return ""
        try:
            return input(question).strip().strip('"')
        except EOFError:
            return ""

    def download(self, url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        self.say(f"   downloading {url}")
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            last = time.time()
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total and time.time() - last > 2:
                    self.say(f"   {done * 100 // total}% of {total // (1 << 20)} MB")
                    last = time.time()
        tmp.replace(dest)
        return dest

    # -- steps -----------------------------------------------------------
    def home(self) -> None:
        self.step("FaceGen folder")
        self.cfg.models_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.projects_dir.mkdir(parents=True, exist_ok=True)
        self.say(f"   {self.cfg.home}")

    def mediapipe(self) -> None:
        self.step("MediaPipe face model")
        dest = self.cfg.mediapipe_model_path()
        if dest.exists() and dest.stat().st_size > 1_000_000:
            self.say(f"   already there: {dest}")
            return
        try:
            self.download(MEDIAPIPE_URL, dest)
            self.say(f"   saved to {dest}")
        except Exception as exc:  # noqa: BLE001
            self.manual.append(f"Download {MEDIAPIPE_URL} and save it as {dest}")
            self.say(f"   could not download ({exc}); added to the manual list")

    def blender(self) -> None:
        self.step("Blender (optional: lit previews and opening results)")
        found = find_blender(self.cfg.blender)
        if found:
            self.cfg.blender = str(found)
            self.say(f"   found {found} ({blender_version(found) or 'version unknown'})")
            return
        if IS_WINDOWS and shutil.which("winget") and self.ask("   Blender not found. Install it with winget?"):
            subprocess.run(["winget", "install", "--id", "BlenderFoundation.Blender", "-e",
                            "--accept-package-agreements", "--accept-source-agreements"])
            found = find_blender(None)
            if found:
                self.cfg.blender = str(found)
                self.say(f"   installed: {found}")
                return
        path = self.prompt("   Path to blender.exe (Enter to skip): ")
        if path and Path(path).is_file():
            self.cfg.blender = path
        else:
            self.say("   skipped. FaceGen works without Blender; install it later and run set_config.")

    def meshroom(self) -> None:
        self.step("Meshroom (optional: photogrammetry scan, about 1.5 GB)")
        found = find_meshroom(self.cfg.meshroom)
        tools = self.cfg.home_path / "tools"
        if not found and tools.exists():
            for p in tools.rglob("meshroom_batch*"):
                if p.is_file():
                    found = p
                    break
        if found:
            self.cfg.meshroom = str(found)
            self.say(f"   found {found}")
            return
        if self.skip_meshroom or not IS_WINDOWS:
            self.say("   skipped (download it from https://alicevision.org/#meshroom and run set_config)")
            return
        if not self.ask("   Download Meshroom now? It is large but the scan improves the open face.", False):
            self.say("   skipped. You can run the installer again later.")
            return
        try:
            with urllib.request.urlopen("https://api.github.com/repos/alicevision/Meshroom/releases/latest") as r:
                rel = json.loads(r.read())
            asset = next(a for a in rel["assets"] if "win" in a["name"].lower() and a["name"].endswith(".zip"))
            zpath = self.download(asset["browser_download_url"], tools / asset["name"])
            self.say("   extracting...")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(tools)
            zpath.unlink()
            for p in tools.rglob("meshroom_batch*"):
                if p.is_file():
                    self.cfg.meshroom = str(p)
                    self.say(f"   installed: {p}")
                    return
            self.manual.append("Meshroom was extracted but meshroom_batch was not found; set its path with set_config")
        except Exception as exc:  # noqa: BLE001
            self.manual.append(f"Install Meshroom manually from https://alicevision.org/#meshroom ({exc})")
            self.say(f"   download failed: {exc}")

    def flame(self) -> None:
        self.step("FLAME face model (optional: plausible jaw under the beard)")
        found = self.cfg.flame_model_path()
        if found:
            self.say(f"   found {found}")
            return
        self.say("   FLAME is free for non-commercial use but needs your own registration, so it cannot be")
        self.say("   downloaded automatically. Steps:")
        self.say("     1. Register at https://flame.is.tue.mpg.de and accept the license")
        self.say("     2. Download 'FLAME 2020' (or 2023) and unzip it")
        self.say(f"     3. Copy generic_model.pkl into {self.cfg.models_dir}")
        if self.yes:
            self.manual.append(f"Optional: FLAME model into {self.cfg.models_dir} (see step above)")
            return
        if self.ask("   Open the FLAME website now?", True):
            webbrowser.open("https://flame.is.tue.mpg.de")
        self.prompt("   Press Enter when the file is in place (or Enter to skip for now): ")
        found = self.cfg.flame_model_path()
        if found:
            self.say(f"   found {found}")
            try:
                from facegen.flame import FlameModel

                m = FlameModel.load(found)
                self.say(f"   loaded: {len(m.v_template)} vertices, {m.shapedirs.shape[2]} shape components")
            except Exception as exc:  # noqa: BLE001
                self.manual.append(f"FLAME file found but could not be read ({exc}). Try FLAME 2023, or convert "
                                   "it with `facegen flame-convert` on a machine with chumpy.")
        else:
            self.manual.append(f"Optional: FLAME model into {self.cfg.models_dir}")

    def faces_folder(self) -> None:
        self.step("FL27 faces folder (optional: browse templates, install faces)")
        if self.cfg.faces_folder and Path(self.cfg.faces_folder).exists():
            self.say(f"   {self.cfg.faces_folder}")
            return
        self.say("   This is the folder with one sub-folder per player ID, usually")
        self.say("   ...\\livecpk\\Faces\\Asset\\model\\character\\face\\real")
        path = self.prompt("   Paste its path (Enter to skip): ")
        if path and Path(path).is_dir():
            self.cfg.faces_folder = path
        elif path:
            self.say("   that folder does not exist; skipped")

    def register(self) -> None:
        self.step("Connect to Claude")
        python = Path(sys.executable)
        server = {"command": str(python), "args": ["-m", "facegen.server"],
                  "env": {"FACEGEN_HOME": str(self.cfg.home)}}
        if IS_WINDOWS:
            conf = Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
        elif platform.system() == "Darwin":
            conf = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        else:
            conf = Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
        if conf.parent.exists() or self.ask("   Claude Desktop config folder not found. Create it?", True):
            conf.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if conf.exists():
                shutil.copy2(conf, conf.with_suffix(".json.bak"))
                try:
                    data = json.loads(conf.read_text(encoding="utf-8") or "{}")
                except json.JSONDecodeError:
                    self.manual.append(f"{conf} is not valid JSON; add the facegen server by hand")
                    data = None
            if data is not None:
                data.setdefault("mcpServers", {})["facegen"] = server
                conf.write_text(json.dumps(data, indent=2), encoding="utf-8")
                self.say(f"   Claude Desktop: added 'facegen' to {conf} (restart Claude Desktop)")
        claude = shutil.which("claude")
        if claude:
            subprocess.run([claude, "mcp", "remove", "facegen", "-s", "user"], capture_output=True)
            r = subprocess.run([claude, "mcp", "add", "facegen", "-s", "user", "-e", f"FACEGEN_HOME={self.cfg.home}",
                                "--", str(python), "-m", "facegen.server"], capture_output=True, text=True)
            self.say("   Claude Code: registered" if r.returncode == 0 else f"   Claude Code: {r.stderr.strip()}")
        else:
            self.say(f"   Claude Code not found. To add it later: claude mcp add facegen -s user -- \"{python}\" -m facegen.server")

    def selftest(self) -> bool:
        self.step("Self-test")
        from facegen.selftest import run_selftest

        rep = run_selftest(log=lambda m: None)
        self.say(f"   synthetic face: shape error {rep['fit_rms_error_mm']} mm (template was "
                 f"{rep['template_vs_person_rms_mm']} mm), export round-trip {rep['export_roundtrip_error_mm']} mm")
        self.say("   PASSED" if rep["passed"] else f"   FAILED, details in {rep['workdir']}")
        return bool(rep["passed"])

    def run(self) -> int:
        self.say("FaceGen setup")
        self.home()
        self.mediapipe()
        self.blender()
        self.meshroom()
        self.flame()
        self.faces_folder()
        self.cfg.save()
        self.register()
        ok = self.selftest()
        self.step("Done")
        if self.manual:
            self.say("   Still to do by hand:")
            for m in self.manual:
                self.say(f"   - {m}")
        self.say("   Restart Claude, then ask: \"Check my FaceGen setup\".")
        return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="accept defaults, no questions")
    ap.add_argument("--skip-meshroom", action="store_true")
    a = ap.parse_args()
    return Installer(a.yes, a.skip_meshroom).run()


if __name__ == "__main__":
    sys.exit(main())
