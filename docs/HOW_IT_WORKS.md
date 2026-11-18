# How FaceGen works

## Coordinates

Everything happens in **head space**: the template face's own coordinates from `face_high.fmdl`, in metres,
with +Y up, +Z forward (where the face looks) and +X toward the character's left. Keeping the template's
frame means the finished head still sits on the game's neck.

Cameras use the OpenCV convention (`x_cam = R X + t`, image y down). Blender scenes are built by rotating
head space so +Y becomes Blender's +Z.

## Pipeline

1. **Landmarks.** MediaPipe Face Landmarker finds 478 points per photo. Hidden points on the far side of a
   turned face are guessed by MediaPipe, so they are dropped using the face's surface normals once a camera
   is known.
2. **Template.** The skin meshes of `face_high.fmdl` (those textured with `face_bsm_alp`) are welded into
   one surface. Landmarks are placed on it by rendering it with its own texture and running MediaPipe on
   the renders (same detector as on the photos, so its biases cancel). Without the MediaPipe model, the
   canonical MediaPipe face is fitted to the template instead. Regions (face, beard, jaw, chin, neck,
   scalp, ears, nose, eyes, brows, mouth, cheeks, forehead) are derived from the landmark positions.
3. **Cameras.** Each photo is solved by PnP against the template's landmark positions, the landmarks are
   triangulated, and a bundle adjustment refines all cameras, one focal length per phone camera and the
   person's 3D landmarks, with a weak prior toward the template that fixes position, rotation and scale.
   Photos with no detected face (full profiles) are solved from clicked anchors; the ear anchor is
   triangulated from the three-quarter photos where it is clicked too. With a Meshroom scan, Meshroom's
   cameras seed the solve; the layout of its rotation numbers is detected by checking which one explains
   the landmarks.
4. **Face model (optional).** FLAME shape coefficients and a similarity are fitted to the landmarks, the
   scan (open face only) and the jaw lines. The result is a plausible head where photos cannot see the
   skin: under the beard, behind the hair.
5. **Fit.** The welded template skin is deformed by minimising, in sparse linear least squares:
   - the Laplacian of the displacement (keeps the template's detail and topology), stiff at first and
     relaxed over the iterations,
   - landmark distances (beard-covered outline points excluded),
   - point-to-plane distances to the scan (open face) and to the FLAME fit (beard, jaw, neck, scalp),
   - jaw-line terms: a dense curve along the template's jaw is projected into every photo with lines; each
     point is pulled, in the image plane, toward the nearest point of the line drawn by the person.
   Closest points and line matches are refreshed each iteration.
6. **Adjust.** Claude can move or scale any region with a smooth falloff (`adjust_region`) after looking at
   the comparisons, and undo.
7. **Texture.** Every texel of the template's UV layout is projected into every photo. Colours are blended
   with weights from the viewing angle, distance to the image border, pixel density and visibility (a depth
   test), after matching each photo's exposure to the best one. Areas no photo sees are filled with the
   template's texture recoloured to the person's skin tone. The template's alpha channel is kept.
8. **Export.** The template folder is copied. In `face_high.fmdl`, skin vertices get their exact welded
   positions (seams stay closed); other meshes in it, `hair_high.fmdl` and `oral.fmdl` move with the skin's
   displacement field. Normals and tangents are rotated with the surface. Bounding boxes grow by the largest
   displacement. All of this is written in place, so nothing else in the files changes. Texture paths and
   FPK entry names are relinked to the new ID. `face_bsm_alp.ftex` is replaced by the baked texture (BC3 with
   mip-maps, header copied from the template's texture). Eye positions in `face_diff.bin` move with the eye
   sockets.

## PES 2021 face folder

```
<ID>/
  #Win/face.fpk             face_high.fmdl, hair_high.fmdl, oral.fmdl (optional), face_diff.bin
  sourceimages/#windx11/    face_bsm_alp.ftex, face_nrm.ftex, face_srm.ftex, face_trm.ftex,
                            eye_occlusion_alp.ftex, hair_parts_*.ftex
```

Format notes come from the community tools (GzsTool for FPK, pes-fmdl-blender and PES Facemod for FMDL,
FTEX and `face_diff.bin`). FaceGen reimplements what it needs and does not bundle them.

`face_diff.bin`: eye positions at 0x150 (right) and 0x160 (left) stored as (y, z, x) in head space; mouth
position at 0x3C as (x, y, z), used by the game only when there is no `oral.fmdl`.

## Files in a project

```
project.json              template, face ID, finished steps
photos/                   copies of the photos
work/landmarks.json       MediaPipe points per photo
work/lines.json           jaw lines, beard outline, hairline, anchors from the line editor
work/template_head.npz    welded template skin + landmark positions
work/regions.npz          per-vertex region weights
work/cameras.json         solved cameras and the person's 3D landmarks
work/scan/                Meshroom output and cache
work/scan_head.npz        scan in head space
work/flame_fit.npz        FLAME fit in head space
work/fit.npz              current head (fit_history/ keeps every earlier version for undo)
work/texture_rgba.png     baked texture
renders/compare/          photo | render | overlay images
export/<ID>/              the finished face folder
```
