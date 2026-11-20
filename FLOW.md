# How the current pipeline works (code walkthrough)

Last updated: 2026-11-20. This explains what runs, in which order, and which file does what, for the **Daz /
Genesis 9 character path** (the current main path). Older paths (Human Generator, MPFB, video-to-motion) are
listed at the end.

## Tools it relies on (all under `tools/`, not in git)

| Tool | Path | Used for |
|---|---|---|
| Blender 5.2.3 (portable) | `tools/blender-5.2.3-windows-x64/blender.exe` | face morphs, bakes, landmark renders (`-b -P script.py -- args`) |
| Blender 4.2.23 LTS (portable) | `tools/blender-4.2.23-windows-x64/blender.exe` | only KeenTools FaceBuilder (its core needs Python ≤ 3.11) |
| Python env (mediapipe, opencv, numpy) | `tools/pose-env/Scripts/python.exe -I` | landmarks, FLAME fit, texture work |
| FLAME 2023 Open (CC-BY-4.0, MPI-IS) | `tools/flame/flame2023_Open.pkl`, `tools/flame/flame_landmarks.json` | free head model and its MediaPipe landmark map |
| MediaPipe face model | `tools/face-model/face_landmarker.task` | 478 face points on photos and renders |
| KeenTools (owner's, trial) | `tools/keentools/` → installed into Blender 4.2 by `pipeline/keentools/install_fb.py` | FaceBuilder head from photos |
| Daz Studio 6 | `C:\Program Files\DAZ 3D\DAZStudio6\DAZStudio.exe` | builds and renders Genesis 9 characters, run by script in its own instance |
| Daz content library | `C:\Users\Public\Documents\My DAZ 3D Library` | Genesis 9, skins, hair, outfits; our morphs and textures are written in here |
| Browser profiles | `tools/browser-profiles/flow`, `.../grok` | signed-in Flow and Grok windows for the picture drivers |

## 1. Pictures: concept → portrait → turned views

- `engine/imagegen/imagegen.mjs` drives Google Flow or Grok Imagine in their own Edge windows (DevTools
  protocol). `state` / `open` / `image` / `edit` / `video`. The owner signs in; the script never does.
- Portrait from a concept: `node imagegen.mjs edit flow 3:4 <concept.png> <face_run>/front.png "<prompt>"`.
- Turned views: `node face_views.mjs flow <front.png> <face_run> left45,left75` (edits of the portrait, so the
  person stays the same); `right45/right75` are mirrored copies (`cv2.flip`).
- Result: `<char_dir>/face_run/{front,left45,left75,right45,right75}.png`.

## 2. One command per character: `pipeline/daz/make_daz_character.py <char_dir> <Name> [--from step] [--rounds 2]`

Inputs: `<char_dir>/face_run/` (the five views) and `<char_dir>/char.json`:

```json
{"sex": "female", "height_cm": 168, "facial_hair": false,
 "body_morphs": {"BaseFeminine_body_bs_Body": 1.0},
 "figure_presets": ["/People/Genesis 9/Materials/.../G9 Iris 05 Deep Brown.duf"],
 "wear": [{"wear": "/People/Genesis 9/Hair/.../G9 Base dForce Pixie Hair.duf",
           "presets": ["...Pixie Hair Black.duf", "...Style Jaunty.duf"], "tint": "#2B4F95"},
          {"wear": "/People/Genesis 9/Clothing/.../Worker Uniform Tshirt.duf"}]}
```

Steps (`--from` restarts at any of them):

| Step | Script | What it does | Output |
|---|---|---|---|
| `flame` | inline + `pipeline/flame/flame_photofit.py` | hair colour from the top of the front picture; FLAME fitted to all five views at once (one shape, per-view pose, lens, expression; damped Gauss-Newton on MediaPipe points); a texture blended from the views | `<char_dir>/flame/` (`heads.npz`, `texture.png`, `fit.json`) |
| `fb` | `pipeline/keentools/fb_head.py` (Blender 4.2) | FaceBuilder: adds a head, loads the views, auto-pins the face in each, solves, bakes a texture. Skipped (FLAME used) if not installed/licensed | `<char_dir>/fb/fb_head.blend`, `fb_texture.png` |
| `face` | `pipeline/flame/flame_head.py`, `pipeline/keentools/fb_textures.py`, `pipeline/keentools/fb_to_run.py`, `pipeline/daz/g9_face.py` | builds a "head run" (head.blend with eyes, `subject_fit.json` landmarks, textures, hair mask); then **g9_face**: reads Genesis 9's mesh from `Genesis9.dsf`, finds its landmarks (render + MediaPipe + ray cast, `pipeline/humgen/hg_landmarks.py`), lines our head up (similarity), moves Genesis 9's head onto ours (biharmonic interpolation of landmark moves, outlier filters, light surface pull, mesh smoothing), writes a **DSON morph** with **rig formulas** for 40 head/neck joints into the Daz library, and saves `g9_<Name>_fit.blend` with a texture layout, a `face` region and a `keep` region (Genesis 9's own eyes/nostrils/lips) | `.../Morphs/Ours/Faces/OURS_<Name>.dsf`; `build.json` (`head_run`, `face_from`) |
| `skin` | `pipeline/daz/g9_skin.py` (+ `pipeline/humgen/face_blend.py`) | bakes (Cycles) photo colour, coverage, hair, face and keep masks from our head onto Genesis 9's head tile; picks the closest Genesis 9 skin of that sex; blends the photo's **detail** onto that skin's own colour (no patch, no edge); keeps Genesis 9's eyes/nostrils/lips; filters hair (dark-grey and any coloured hair); one tone shift on all four tiles; facial-hair colour correction and a translucency mask | `.../Runtime/Textures/Ours/<Name>/` (`*_D_100N.jpg`, `*_TW_1001.png`, `skin.json`) |
| `daz` | `pipeline/daz/g9_calibrate.py` → `pipeline/daz/g9_build.dsa` | runs **Daz Studio 6 in its own instance** (`-instanceName # -noPrompt -scriptArg ...`): loads Genesis 9, dials our face morph, body morphs, exact height (measures and solves Proportion Height), skin preset + our colour maps + extra maps, eye preset, hair and clothes (+ tint), saves the scene, renders face / three-quarter / body (Iray, CPU-friendly: denoiser, 150 samples, texture compression, retries). Then measures skin and upper lip on the render vs the portrait and corrects the textures; repeats `--rounds` times; resumes from the last calibration | `tests/daz/<name>_g9.duf`, `<name>_g9_{face,face34,body}.png`, `<name>_calibration.json`, `build_<name>.txt` |

Every script prints one `..._OK` line on success (`MAKE_DAZ_CHARACTER_OK`, `G9_FACE_OK`, `G9_SKIN_OK`,
`G9_CALIBRATE_OK`, ...) and `..._FAILED at <step>` otherwise.

## 3. Supporting scripts

- `pipeline/daz/daz_library.py`: index of installed Daz content from `Runtime/Support/*.dsx` (kind, fits,
  thumbnails) → `tools/daz/library.json`; `--find "words"` searches it.
- `pipeline/flame/flame_fit.py`: FLAME plausibility check of any head's landmarks (how unusual, where it can't follow).
- `pipeline/flame/flame_model.py`: FLAME loader without chumpy (safe unpickler).
- `pipeline/humgen/hg_landmarks.py`: `surface_landmarks` (MediaPipe points on any mesh), `similarity`, `rbf_moves`.

## 4. Daz scripting facts (learned the hard way)

- Run Daz **one instance at a time**: two Iray renders on a 4 GB card crash.
- Daz Studio 4.24 shows an update dialog that blocks scripts; use **Daz Studio 6**.
- Daz's Iray dropped GeForce 10-series GPUs: it renders on the **CPU**. Big textures + strand hair can exhaust RAM
  ("Memory allocation failed") and hang Daz: keep texture compression on.
- "Not responding" during a scripted render is normal; check CPU use before calling it stuck.
- Actions with dialogs (e.g. Adjust Rigging to Shape) block scripts: put rig changes into the morph's formulas.
- Bone positions: use `getWSPos()` (world, after morphs and height), not `getOrigin()`.
- New wearables: select the newly added node before applying its presets; a hair item adds several nodes (cap +
  strands): tint all of them.

## 5. Older paths (still in the repo)

- **Human Generator** (`pipeline/humgen/`, needs `tools/Human`): HG characters, children ages, gown, Indian
  costumes, braid, long hair, glTF export for Character Studio. Kept until the studio runs on Daz.
- **Character Studio** (`pipeline/studio/`): web app (`node pipeline/studio/server.mjs 4280`): scenario → pictures
  → face → HG model → sliders and costumes. To be rewired to Daz (milestone M3).
- **MPFB / video-to-motion** (`pipeline/*.py`, `pipeline/mpfb/`): the original video-to-animation pipeline; game001's
  first cast used MPFB.
