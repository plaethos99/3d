# How the current pipeline works (code walkthrough)

Last updated: 2026-11-21. This explains what runs, in which order, and which file does what, for the **Daz /
Genesis 9 character path** (the main path). The flow is final for now: **face and skin are made in Daz, the head
hair is grown in Blender, Blender renders**; known problems are listed in PLAN.md ("Later fixes") and get fixed
during character design. Older paths (Human Generator, MPFB, video-to-motion) are listed at the end.

## Tools it relies on (all under `tools/`, not in git)

| Tool | Path | Used for |
|---|---|---|
| Blender 5.2.3 (portable) | `tools/blender-5.2.3-windows-x64/blender.exe` | face morphs, bakes, landmark renders, the Daz import, hair, renders (`-b -P script.py -- args`) |
| Diffeomorphic DAZ Importer 5.2.0 (GPL) | installed in that Blender as extension `import_daz` (source zip in `tools/downloads/diffeo/`) | reads Daz scenes into Blender (morphs, rig, materials) |
| Blender 4.2.23 LTS (portable) | `tools/blender-4.2.23-windows-x64/blender.exe` | only KeenTools FaceBuilder (its core needs Python ≤ 3.11) |
| Python env (mediapipe, opencv, numpy) | `tools/pose-env/Scripts/python.exe -I` | landmarks, FLAME fit, texture work |
| FLAME 2023 Open (CC-BY-4.0, MPI-IS) | `tools/flame/flame2023_Open.pkl`, `tools/flame/flame_landmarks.json` | free head model and its MediaPipe landmark map |
| MediaPipe face model | `tools/face-model/face_landmarker.task` | 478 face points on photos and renders |
| KeenTools (owner's, trial) | `tools/keentools/` → installed into Blender 4.2 by `pipeline/keentools/install_fb.py` | FaceBuilder head from photos |
| Daz Studio 6 | `C:\Program Files\DAZ 3D\DAZStudio6\DAZStudio.exe` | builds Genesis 9 characters (and calibrates their skin with small Iray renders), run by script in its own instance |
| Daz content library | `C:\Users\Public\Documents\My DAZ 3D Library` | Genesis 9, skins, hair, outfits; our morphs and textures are written in here |
| Browser profiles | `tools/browser-profiles/flow`, `.../grok` | signed-in Flow and Grok windows for the picture drivers |

## 1. Pictures: concept → portrait → turned views

- `engine/imagegen/imagegen.mjs` drives Google Flow or Grok Imagine in their own Edge windows (DevTools
  protocol). `state` / `open` / `image` / `edit` / `video`. The owner signs in; the script never does.
- Portrait from a concept: `node imagegen.mjs edit flow 3:4 <concept.png> <face_run>/front.png "<prompt>"`.
- Turned views: `node face_views.mjs flow <front.png> <face_run> left45,left75` (edits of the portrait, so the
  person stays the same); `right45/right75` are mirrored copies (`cv2.flip`).
- Result: `<char_dir>/face_run/{front,left45,left75,right45,right75}.png`.

## 2. One command per character: `pipeline/daz/make_daz_character.py <char_dir> <Name> [--from step]`

Inputs: `<char_dir>/face_run/` (the five views) and `<char_dir>/char.json`:

```json
{"sex": "male", "height_cm": 175, "facial_hair": false,
 "body_morphs": {"BaseMasculine_body_bs_Body": 1.0},
 "figure_presets": ["/People/Genesis 9/Materials/.../G9 Iris 05 Deep Brown.duf"],
 "wear": [{"wear": "/People/Genesis 9/Clothing/.../Worker Uniform Tshirt.duf"}],
 "face_shape": {"jaw_width": -0.35, "chin_length": 0.2},
 "blender_hair": {"length_cm": 11, "messy": 0.7, "cut_cm": {"front": 0.6, "side": -1.5, "back": -7}}}
```

`face_shape` and `blender_hair` are what Character Studio's sliders edit (section 3). Head hair listed in `wear` is
left out of Daz (grown in Blender instead); beards stay in Daz.

Steps (`--from` restarts at any of them):

| Step | Script | What it does | Output |
|---|---|---|---|
| `flame` | inline + `pipeline/flame/flame_photofit.py` | hair colour from the top of the front picture; FLAME fitted to all five views at once; a texture blended from the views | `<char_dir>/flame/` |
| `fb` | `pipeline/keentools/fb_head.py` (Blender 4.2) | FaceBuilder head from the five views (skipped, FLAME used, if not installed/licensed) | `<char_dir>/fb/` |
| `face` | `flame_head.py`, `fb_textures.py`, `fb_to_run.py`, **`pipeline/daz/g9_face.py`** | a "head run" (head.blend, landmarks, textures); then g9_face: Genesis 9's mesh from `Genesis9.dsf`, the same 478 MediaPipe points on both heads, our head lined up (scale/position only), Genesis 9's head moved onto ours (smooth interpolation of the point moves, light surface pull, smoothing); then the **shape controls** from `face_shape` (strength, jaw/chin width, chin length/forward, mouth width, lip fullness, nose width: smooth regions round landmarks; the jaw control spares the lips); writes a **DSON morph** with **rig formulas** for 40 head/neck joints; saves `g9_<Name>_fit.blend` (our head reshaped the same way, so the skin bake follows) | `.../Morphs/Ours/Faces/OURS_<Name>.dsf` |
| `skin` | `pipeline/daz/g9_skin.py` (+ `face_blend.py`) | bakes the photo onto Genesis 9's head tile, picks the closest Genesis 9 skin, blends the photo's detail onto it, keeps Genesis 9's eyes/nostrils/lips/nose bridge; calibration brightness into the colour maps, its hue into Translucency Color | `.../Runtime/Textures/Ours/<Name>/` |
| `daz` | `pipeline/daz/g9_calibrate.py` → `g9_build.dsa` | **Daz Studio 6 in its own instance**: Genesis 9 + our morph, body morphs, exact height, skin preset + our maps, clothes (no head hair), saves the scene; small Iray renders measure the cheeks against the portrait and correct the skin (`--rounds`, resumes from the last calibration; `--no-iray`: build and save only) | `tests/daz/<name>_g9.duf`, `<name>_calibration.json` |
| `blender` | `pipeline/daz/daz_to_blender.py` → `blender_daz_render.py` → **`blender_hair.py`** | Daz exports `.dbz` (Diffeomorphic's exporter, made headless); Blender imports the scene; **hair grown** (curves on the scalp above a per-direction hairline, combed from the crown, gravity, locks, flicked tips, waves, cut per direction; colour from the portrait; a scalp cap; parented to the head bone); studio lights; EEVEE renders face / three-quarter / body | `tests/blender_daz/<name>_bl_{face,face34,body}.png`, `<name>_bl.blend` |

Every script prints one `..._OK` line on success (`MAKE_DAZ_CHARACTER_OK`, `G9_FACE_OK`, `G9_SKIN_OK`,
`G9_CALIBRATE_OK`, `DAZ_TO_BLENDER_OK`, ...) and `..._FAILED at <step>` otherwise. Each finished step appends a
line to the cast's `DAZ_PROGRESS.md`. A full run is about 15 minutes; `--from face` about 8, `--from blender` 3.

Optional, off by default (tried, the owner preferred the plain fit): `--likeness N` (the face matched to the portrait
in Blender with an eyelid fit, written back into the Daz morph: `blender_likeness.py`), `--colour N` in
`blender_daz_render.py` (skin and hair colour matched in Blender: `blender_colour.py`), `--daz-hair` (keep the head
hair in Daz; dForce/strand hair is then swapped for a light mesh hair).

## 3. Character Studio (edit a character without a terminal)

`node pipeline/studio/server.mjs 4280` → http://localhost:4280 → **Daz cast** in the left column (characters of
`../game001/characters`, or `DAZ_CAST`). A character's page: the portrait and latest renders; **Face** sliders
(`face_shape`) and **Hair** sliders and colour (`blender_hair`); **Rebuild face** (`--from face`) or **Regrow hair**
(`--from blender`), with progress on the page. Server side: `pipeline/studio/daz.mjs`; page: `public/daz.js`.
The values are saved in `char.json`, so every later rebuild keeps them.

## 4. Supporting scripts

- `pipeline/daz/landmarks2d.py`: MediaPipe points and colour patches of pictures, for Blender scripts.
- `pipeline/daz/daz_library.py`: index of installed Daz content → `tools/daz/library.json`; `--find "words"`.
- `pipeline/flame/flame_fit.py`, `flame_model.py`: FLAME plausibility check; loader without chumpy.
- `pipeline/humgen/hg_landmarks.py`: `surface_landmarks`, `similarity`, `rbf_moves`.

## 5. Daz and Blender facts (learned the hard way)

- Run Daz **one instance at a time**. Daz Studio 4.24 shows an update dialog that blocks scripts: use **Daz 6**.
- Iray on this machine (GTX 1050 Ti, 4 GB) falls back to the CPU and runs out of memory with strand hair: no head
  hair in Daz, keep texture compression on, keep Iray to the small calibration renders.
- "Not responding" during a scripted render is normal; check CPU use before calling it stuck.
- Actions with dialogs (Adjust Rigging to Shape, Diffeomorphic's export dialog) block scripts: rig changes go into
  the morph's formulas; the exporter is run with its dialog removed.
- Bone positions: `getWSPos()` (world, after morphs and height). New wearables: select the added node before its
  presets; a hair item adds several nodes; strand hair shows its Root/Tip Transmission Colors (tint those too).
- Diffeomorphic keeps Genesis 9's vertex order (25,182 vertices), so Blender-side shape changes can be written back
  into Daz morphs.

## 6. Older paths (still in the repo)

- **Human Generator** (`pipeline/humgen/`, needs `tools/Human`): HG characters, children ages, gown, Indian
  costumes, braid, long hair, glTF export. Character Studio's first pages still use it.
- **MPFB / video-to-motion** (`pipeline/*.py`, `pipeline/mpfb/`): the original video-to-animation pipeline; game001's
  first cast used MPFB.
