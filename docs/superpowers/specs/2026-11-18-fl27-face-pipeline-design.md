# FL27 Face Pipeline: Design

Status: draft for approval. Nothing is built yet.

## Intent and why

Parwez wants to put his own real face, and a friend's, into FL27 (the PES 2021-based Football Life mod) at the best quality possible. The first attempt, a head sculpted from scratch in Blender by Claude, was good but not good enough. The fix is a pipeline built on real reconstruction tools, driven by Claude over MCP, so Claude can build the face, check it against the photos and correct it.

## Goals

- Photos in, an FL27-ready face (geometry, textures, hair and beard) out, with Claude orchestrating over MCP.
- Everything free to use (Blender, Meshroom, Pixel3DMM or another FLAME fitter, MediaPipe, community PES add-ons).
- Works with a big beard (the main user has one).
- One-time installer that sets up the dependencies and stops only for unavoidable manual steps. Parwez accepts one or two manual stops.
- Usable for a second person (the friend) with no code changes.
- Runs on Windows with an NVIDIA GPU (RTX 2070 Super, 8 GB VRAM).

## Non-goals

- True FC27 skin detail. The ceiling is PES 2021's engine.
- Commercial use or distribution of faces made from other people's assets.
- A graphical app. Claude is the interface.
- Full-body or kit generation.

## Anti-goals (failures even if it technically works)

- A head that loads in the game but looks like a generic face.
- A pipeline that needs hand-editing in Blender at every step.
- Dependencies that conflict with Blender's own Python.
- A face that breaks the rig, eyes or UVs in game.

## Constraints

- Windows PC, RTX 2070 Super (8 GB VRAM). Models must fit in 8 GB.
- FLAME model license: free for non-commercial use, but requires manual registration and acceptance. The installer cannot fetch it silently (manual stop 1).
- Blender: use the user's existing install. The PES face and hair add-ons work well only on certain Blender versions (community docs mention 2.8x to 2.9x for some PES add-ons), so version pinning is needed. This is unverified and must be checked first.
- Anything tied to the PES export format is unverified until tested in the game.

## Approaches considered

- **A. Pipeline MCP server plus a small Blender add-on (chosen).** Heavy ML runs outside Blender in its own Python environment. Blender-side work goes through a socket add-on. Isolated, debuggable, each stage can run alone.
- **B. Everything inside one Blender add-on.** Rejected: PyTorch, Meshroom and Pixel3DMM conflict with Blender's bundled Python.
- **C. Fully automatic command-line tool with no Claude in the loop.** Rejected: loses the render, compare and fix loop that is the main reason for using Claude.

Parwez picked A.

## Architecture

```
Photos folder
    |
Claude <-- MCP --> Face-pipeline server (Python, own venv)
    |                 |- Meshroom (photogrammetry scan)
    |                 |- Pixel3DMM / FLAME fit
    |                 |- MediaPipe landmarks
    |                 |- texture bake helpers
    |                 `- PES export wrapper
    |
    `-- MCP --> Blender add-on (socket) --> Blender
                  fit, sculpt, render from photo cameras, bake
```

## MCP tools (first list)

- `check_environment`: GPU, drivers, Blender version, add-ons, model files.
- `scan_photos(folder)`: runs Meshroom and returns the scan mesh and texture.
- `fit_face_model(photos)`: FLAME fit from the best photos.
- `detect_landmarks(photos)`: about 468 landmarks per photo.
- `blender_load_base_head(fl27_face_id)`: loads the PES base head.
- `blender_align_and_shrinkwrap(scan, base)`: landmark alignment, then Shrinkwrap and Surface Deform.
- `blender_render_from_photo_cameras()`: renders the head from each photo's estimated camera.
- `compare_render_to_photo(i)`: landmark distance, image difference, returned with both images for Claude to look at.
- `blender_sculpt_fix(region, delta)`: bounded local corrections Claude requests.
- `bake_texture()`: scan colour onto the PES UVs, seam blending.
- `list_beard_hair_assets()` and `blender_fit_beard_mesh(asset)`: test beard and hair meshes.
- `export_to_pes(face_id)`: writes the face folder for the game.

## Beard handling

- The jaw under a beard cannot be scanned. Use the FLAME fit's jaw plus ear, cheekbone and neck landmarks, and check against side-profile photos. A clean-shaven reference photo, if one exists, is much better. (Decided: Parwez has no clean-shaven photo, so the jaw is estimated and checked against side-profile and under-jaw photos. He does have photos showing his head hair, so hair can be matched from real references.)
- The beard is added two ways: baked into the face texture (density and colour from the photos) and, if the FL27 export accepts it, a separate beard or hair mesh fitted to the jaw. Mesh support is unverified and tested early.

## Line-marking tool (added at Parwez's request)

A small local web page, opened by the app, where the user draws lines on each photo:
- Side photos: the chin, the jaw angle, and the back edge of the jaw going up to the ear.
- Front photos: the chin and the jawline on both sides.
- Optional: beard outline, hairline, ear outline.

How it is used: each 2D line is converted to a 3D curve using the camera pose estimated for that photo, and lines from several photos are combined. The resulting jaw curve becomes a hard target for the fit and sculpt steps, and the compare step measures the head against it. Lines drawn by hand are treated as more reliable than landmarks detected under the beard. Tools: `open_line_editor(photos)` and `get_user_lines()`.

Because this adds a user interface, the earlier "no graphical app" non-goal now means: no GUI except this line-marking page.

## Photo protocol (shown to the user by the tool)

30 to 50 photos around the head, neutral expression, soft even light, hair tied back, ears and jawline visible, extra straight-on and side-profile shots with the chin lifted, one from under the jaw.

## Installer

First run creates an isolated Python venv, installs PyTorch with CUDA, MediaPipe and the other packages, downloads Meshroom and the Pixel3DMM code, installs the Blender add-on and registers the MCP server with Claude, then runs a self-test (fake scan, fit, render) and reports failures.

Planned manual stops: (1) FLAME registration and download, (2) Blender or NVIDIA driver version fixes if the check finds a problem, and (3) the PES add-ons if no stable download link exists.

## Build order

1. Blender add-on, photo-camera render and PES import and export test. This proves the riskiest part first, including whether a beard mesh exports.
2. MCP server skeleton and Claude connected to Blender and the server.
3. Scan and landmark tools.
4. FLAME fit and fitting onto the PES head.
5. Texture bake, beard texture, beard mesh, hair options.
6. Render, compare and fix loop, tuned on real photos.
7. Installer and self-test (built alongside steps 2 to 4 and finished last).

## Timeline (estimates)

- First end-to-end head in the game: about 3 to 7 days, limited by setup and test rounds on Parwez's PC.
- A face he is happy with, beard included: about 2 to 3 weeks with regular testing.

## Risks

- Meshroom may fail on skin with little texture. The FLAME fit is the fallback.
- PES export format or beard mesh support may differ from community tutorials.
- Windows installs of CUDA packages can fail in surprising ways.
- Claude's visual comparison is good at proportions, weaker at subtle likeness, so the final judgement stays with the user.

## Left to the builder

- Exact Blender version and add-on versions.
- Which FLAME fitter to use if Pixel3DMM does not run in 8 GB.
- Camera estimation method for the photo-angle renders.
- Package layout and MCP transport details.

## Calls I made (not decided by Parwez)

- Windows with an NVIDIA GPU as the target.
- Pixel3DMM as the first FLAME fitter, with DECA or SMIRK as backups.
- Reusing the user's existing Blender install instead of installing one.
- Testing beard mesh export before building the rest.

## Implementation notes (2026-11-18, after research into the game formats)

Parwez approved the design and asked for the whole program to be built. Research changed four things:

- **No PES Blender add-on in the loop.** PES 2021 faces are `<ID>/#Win/face.fpk` (FPK archive with
  `face_high.fmdl`, `hair_high.fmdl`, `oral.fmdl`, `face_diff.bin`) plus `.ftex` textures. FaceGen reads and
  writes these itself and patches only vertex positions, normals and bounding boxes in place. A public
  attempt that fully re-serialised `face_high.fmdl` showed corrupted faces in game, so in-place patching is
  the safer route. This removes the manual step for the PES add-ons and the Blender version constraint.
- **Blender runs headless, not through a live socket add-on.** The core loop (fit, render, compare) uses
  FaceGen's own numpy renderer, so it works without Blender. Blender is used for lit previews and for
  opening the result with every photo as a camera background (`open_in_blender`). Fewer moving parts,
  nothing has to stay open.
- **Jaw lines as image-space constraints.** Instead of lifting each drawn line to a 3D curve, a dense curve
  along the template's jaw is projected into each photo and pulled toward the drawn line in the image plane.
  This handles silhouettes and creases the same way and needs no triangulation of the lines.
- **Template-based export with a test step.** `export_test_face` writes a copy of a game face (nose pushed
  forward a few mm) through the same writer, so the file round trip can be checked in FL27 before the first
  real face. This is build-order step 1 from above, done by the person on their install.

FLAME is fitted with FaceGen's own linear shape fit (numpy), not Pixel3DMM, so it runs on Windows without
PyTorch. Meshroom's camera file layout is detected from the landmarks.

Verified here on synthetic faces (template = MediaPipe canonical face, person = known deformation, photos
rendered from known cameras): shape error drops from 2.1 mm to 0.4 mm, the jaw under the "beard" from
3.1 mm to 0.6 mm, and exported files read back exactly. Not verified: real game files, real photos,
MediaPipe on Windows, Meshroom output from a real run, a real FLAME file.
