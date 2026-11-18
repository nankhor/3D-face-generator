# FaceGen

Build your own face for **FL27 / PES 2021** from photos, with Claude doing the work in the background.

You take 30 to 50 photos, draw where your jaw is on two of them, and pick an existing face from the game
as a starting point. FaceGen reshapes that face to match you, paints your skin and beard onto it from the
photos, and writes a face folder you drop into your `livecpk` faces folder. Claude drives each step over
MCP, compares the result with your photos and fixes what doesn't match.

Everything is free and runs on your PC.

## Install (Windows)

1. Install [Claude Desktop](https://claude.ai/download) (or Claude Code).
2. Download or clone this repository.
3. Double-click **`install.bat`**.

The installer sets up Python, downloads the MediaPipe face model, finds Blender (or offers to install it
with winget), offers to download Meshroom, and connects FaceGen to Claude. It stops at most twice:

- **FLAME face model (optional):** free, but you have to register at <https://flame.is.tue.mpg.de> yourself
  and copy `generic_model.pkl` into `%USERPROFILE%\FaceGen\models`. It helps Claude estimate the jaw under a beard.
- **Your faces folder (optional):** paste the path to
  `...\livecpk\Faces\Asset\model\character\face\real` so Claude can browse faces and install the result.

At the end it runs a self-test on a synthetic face. Restart Claude Desktop afterwards.

macOS and Linux: `./installer/install.sh` (the photogrammetry download is Windows only).

## Make a face

Read the [photo guide](facegen/data/photo_guide.md) first: light and sharpness matter more than any setting.

Then, in Claude:

> Check my FaceGen setup. Then make a test export of face 12345 as ID 99999 so I can check it in the game.

Once the test face works in FL27:

> Make a new FaceGen project called "parwez" from the photos in `C:\Users\me\Pictures\face` using face
> 12345 from my faces folder as the template, ID 99999.

Claude finds the landmarks, prepares the template, then opens the **line editor** in your browser. Draw your
jawbone (press on your jaw to feel the bone, draw the bone and not the edge of the beard) on one side
photo and one front photo. Full-profile photos where no face is detected need four clicks (nose tip, eye
corner, mouth corner, ear opening). Tell Claude when you are done.

Claude then solves the cameras, fits the head, bakes the texture and shows you comparison images
(photo | render | overlay) for each angle. Tell it what looks off ("the nose is too short"); it adjusts the
region and compares again. For hair and beard meshes, ask it to find a face with a similar style in your
faces folder and use it as the hair source. Finally:

> Export the face as ID 99999 and install it.

## What happens under the hood

| Step | What it does | Tools |
|---|---|---|
| Landmarks | 478 face points on every photo | MediaPipe Face Landmarker |
| Template | Welds the template's skin, places the same points on it, splits it into regions (jaw, beard, nose...) | own code |
| Cameras | Works out where every photo was taken from (and the focal length) | OpenCV PnP + bundle adjustment |
| Scan (optional) | Photogrammetry of the open face | Meshroom |
| Face model (optional) | Plausible skull and jaw under the beard | FLAME |
| Fit | Reshapes the template to the landmarks, scan, face model and your jaw lines, keeping its topology | sparse least squares |
| Compare | Renders the head from each photo's camera next to the photo | own renderer, Blender for lit previews |
| Texture | Projects the photos onto the head's UV layout, beard included | own code |
| Export | Writes `#Win/face.fpk` and `sourceimages/#windx11/*.ftex`, relinks the ID | own FPK/FMDL/FTEX code |

The export patches only vertex positions, normals and bounding boxes inside the template's files and
leaves every other byte alone. Eye positions in `face_diff.bin` follow the eye sockets, and the template's
hair moves with the scalp. More detail in [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Command line

Everything Claude does is also available as `facegen <command>` inside the virtual environment
(`.venv\Scripts\facegen.exe`): `check`, `selftest`, `new`, `landmarks`, `template`, `editor`, `cameras`,
`scan`, `import-scan`, `face-model`, `fit`, `bake`, `compare`, `export`, `status`.

## Limits

- Tested on synthetic faces only so far. The game files are written from the documented formats; run the
  test export once and check it in FL27 before trusting a real face.
- The game engine sets the ceiling: expect a good modder-quality likeness, not EA-level skin detail.
- Claude's image comparison is good at proportions and weak at subtle likeness. You judge the final face.
- Hair and beard meshes come from existing faces; the beard's colour and density come from your photos.

## Development

```
uv venv --python 3.12 .venv && uv pip install --python .venv -e ".[dev]"
.venv/bin/python -m pytest
```

Third-party data: `facegen/data/canonical_face_model.obj` is from MediaPipe (Apache-2.0).
