"""Command line: the same steps as the MCP tools, for running without Claude or for debugging."""

from __future__ import annotations

import argparse
import json
import sys

from . import pipeline as P
from .config import Config


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="facegen", description="Build a PES 2021 / FL27 face from photos.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="show what is installed")
    st = sub.add_parser("selftest", help="run a synthetic end-to-end test")
    st.add_argument("--dir", default=None)
    sub.add_parser("mcp", help="start the MCP server (stdio)")
    c = sub.add_parser("config", help="set tool paths")
    for k in ("blender", "meshroom", "flame-model", "faces-folder"):
        c.add_argument(f"--{k}")
    n = sub.add_parser("new", help="create a project")
    n.add_argument("name")
    n.add_argument("photos")
    n.add_argument("template")
    n.add_argument("--face-id")
    for name, hlp in (("landmarks", "detect landmarks"), ("template", "prepare the template head"),
                      ("cameras", "solve cameras"), ("scan", "run Meshroom"), ("import-scan", "import the scan"),
                      ("face-model", "fit FLAME"), ("fit", "fit the head"), ("bake", "bake the texture"),
                      ("compare", "comparison images"), ("status", "project status"),
                      ("editor", "open the jaw-line editor"), ("preview", "preview renders"),
                      ("blender", "open in Blender"), ("all", "landmarks to compare in one go")):
        sp = sub.add_parser(name, help=hlp)
        sp.add_argument("project")
    ex = sub.add_parser("export", help="write the PES face folder")
    ex.add_argument("project")
    ex.add_argument("--face-id")
    ex.add_argument("--hair-from")
    ex.add_argument("--install", action="store_true")
    fc = sub.add_parser("flame-convert", help="convert a FLAME .pkl to .npz (needs chumpy)")
    fc.add_argument("pkl")
    fc.add_argument("out")
    args = ap.parse_args(argv)
    cfg = Config.load()
    log = print

    if args.cmd == "check":
        from .selftest import environment_report

        _print(environment_report(cfg))
    elif args.cmd == "selftest":
        from .selftest import run_selftest

        rep = run_selftest(args.dir, log=log)
        _print(rep)
        return 0 if rep["passed"] else 1
    elif args.cmd == "mcp":
        from .server import main as serve

        serve()
    elif args.cmd == "config":
        for k in ("blender", "meshroom", "flame_model", "faces_folder"):
            v = getattr(args, k)
            if v:
                setattr(cfg, k, v)
        cfg.save()
        _print({k: getattr(cfg, k) for k in ("home", "blender", "meshroom", "flame_model", "faces_folder")})
    elif args.cmd == "new":
        _print(P.create_project(cfg, args.name, args.photos, args.template, args.face_id, log))
    elif args.cmd == "flame-convert":
        from .flame import FlameModel

        FlameModel.load(args.pkl).save_npz(args.out)
        print(f"wrote {args.out}")
    else:
        proj = P.open_project(cfg, args.project)
        if args.cmd == "landmarks":
            _print(P.detect_landmarks(proj, cfg, log))
        elif args.cmd == "template":
            _print(P.prepare_template(proj, cfg, log))
        elif args.cmd == "cameras":
            _print(P.solve_project_cameras(proj, cfg, log))
        elif args.cmd == "scan":
            _print(P.run_scan(proj, cfg, log))
        elif args.cmd == "import-scan":
            _print(P.import_scan(proj, cfg, log))
        elif args.cmd == "face-model":
            _print(P.fit_face_model(proj, cfg, log))
        elif args.cmd == "fit":
            _print(P.fit(proj, cfg, log))
        elif args.cmd == "bake":
            _print(P.bake(proj, cfg, log))
        elif args.cmd == "compare":
            _print(P.compare(proj, cfg, log=log))
        elif args.cmd == "status":
            _print(P.status(proj, cfg))
        elif args.cmd == "preview":
            _print(P.render_preview(proj, cfg, log=log))
        elif args.cmd == "blender":
            _print(P.open_in_blender(proj, cfg, log))
        elif args.cmd == "export":
            _print(P.export(proj, cfg, args.face_id, args.hair_from, args.install, log))
        elif args.cmd == "editor":
            from .line_editor.server import LineEditor

            ed = LineEditor(proj)
            print(f"Line editor at {ed.start()} (Ctrl+C to stop)")
            try:
                ed.thread.join()
            except KeyboardInterrupt:
                ed.stop()
        elif args.cmd == "all":
            out = {"landmarks": P.detect_landmarks(proj, cfg, log), "template": P.prepare_template(proj, cfg, log),
                   "cameras": P.solve_project_cameras(proj, cfg, log)}
            if cfg.flame_model_path():
                out["face_model"] = P.fit_face_model(proj, cfg, log)
            out["fit"] = P.fit(proj, cfg, log)
            out["texture"] = P.bake(proj, cfg, log)
            out["compare"] = P.compare(proj, cfg, log=log)
            _print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
