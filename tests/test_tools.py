import ast
import json
import urllib.request
from pathlib import Path

import pytest

from facegen.config import Config
from facegen.selftest import run_selftest


@pytest.fixture(scope="module")
def project_home(tmp_path_factory):
    root = tmp_path_factory.mktemp("st")
    run_selftest(root, log=lambda m: None)
    return root / "home"


def test_blender_script_is_python39_compatible():
    src = (Path(__file__).resolve().parents[1] / "facegen" / "blender" / "blender_job.py").read_text()
    ast.parse(src, feature_version=(3, 9))


def test_line_editor_endpoints(project_home):
    from facegen import pipeline as P
    from facegen.line_editor.server import LineEditor

    proj = P.open_project(Config(home=str(project_home)), "selftest")
    ed = LineEditor(proj)
    url = ed.start(open_browser=False)
    try:
        state = json.loads(urllib.request.urlopen(url + "api/state").read())
        assert len(state["photos"]) == 10 and state["photos"][0]["camera"]
        first = state["photos"][0]["name"]
        ov = json.loads(urllib.request.urlopen(url + f"api/overlay/{first}").read())
        assert ov["landmarks"] and ov["jaw"]
        body = json.dumps({first: {"jaw": [[[1, 2], [3, 4]]], "anchors": {"nose_tip": [5, 6]}}}).encode()
        req = urllib.request.Request(url + "api/lines", data=body, headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req).read())["ok"]
        assert proj.lines()[first]["anchors"]["nose_tip"] == [5, 6]
        assert urllib.request.urlopen(url).status == 200
        assert urllib.request.urlopen(url + f"api/photo/{first}?max=300").headers["Content-Type"] == "image/jpeg"
    finally:
        ed.stop()


def test_mcp_tools_registered_and_callable(project_home, monkeypatch):
    monkeypatch.setenv("FACEGEN_HOME", str(project_home))
    import asyncio

    from facegen import server

    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    for expected in ("create_project", "detect_landmarks", "open_line_editor", "solve_cameras", "fit_head",
                     "compare_to_photos", "adjust_region", "bake_texture", "export_face", "job_status"):
        assert expected in names
    status = json.loads(server.project_status("selftest"))
    assert status["photos"] == 10
    moved = json.loads(server.adjust_region("selftest", "nose", [0, 0, 2], [1, 1, 1]))
    assert moved["max_move_mm"] > 1.0
    assert "restored" in json.loads(server.undo_fit("selftest"))
    out = server.compare_to_photos("selftest", None, 2)
    assert isinstance(out, list) and len(out) == 3
