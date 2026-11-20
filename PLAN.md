# Plan: story to movie or game (personal use)

Owner's goal: design a complete **movie** and a complete **game** themselves, with this tool doing the 3D work by
script. Personal use first; licences (Daz Interactive, etc.) only if a project is released later. Budget is
tight: free tools first.

Last updated: 2026-11-20. Status marks: ✅ done · 🟡 partly · ⬜ not started.

---

## 1. Requirements (the checklist the owner keeps)

| # | Requirement | Status |
|---|---|---|
| R1 | The user enters a **story**. | ⬜ |
| R2 | The user picks **Movie** or **Game**; every later step adapts to it. | ⬜ |
| R3 | **Concept images** are generated automatically per character (poses too) with **Google Flow or Grok**, whichever is available. | 🟡 per character by script; no story-driven batch, no automatic fallback yet |
| R4 | The user clicks **Model**: the character is built in **Daz (Genesis 9)** with face, body, **skinning, texturing, hair**. If unhappy, **remodel** in Daz. | 🟡 one command builds it (`pipeline/daz/make_daz_character.py`); no button, no remodel UI |
| R5 | The character loads into **Character Studio** to be changed (face, body, skin, hair, clothes). | ⬜ studio still runs on Human Generator, not Daz |
| R6 | **Rigging, skinning, animation** by the LLM (scripts) or tools like **Cascadeur**. | 🟡 Genesis 9 is rigged and skinned; our face morphs carry their rig; no animation yet |
| R7 | **Props and environments** by Daz or Blender add-ons the owner installs. | ⬜ (library index exists: `pipeline/daz/daz_library.py`) |
| R8 | **Animation** driven by the LLM. | ⬜ (an older video-to-motion pipeline exists in `pipeline/`) |
| R9 | **Film lighting and camera** set up by the LLM in Daz or Blender. | ⬜ |
| R10 | **Rendering** started and managed by the pipeline. | 🟡 Daz check renders by script; no shot renders |
| R11 | **Game**: everything exported to the game engine (Godot for game001). | ⬜ |
| R12 | Movie vs Game: **high-poly, strand hair, 4K, path-traced** vs **low-poly + LODs, hair cards, baked 2K textures, baked lighting**. | ⬜ |
| R13 | Driven from **any LLM harness** (Antigravity, VS Code + Cline, Claude Code, Hermes Desktop) with **cheap models** (DeepSeek Flash, Haiku, a cheap GPT). | ⬜ (scripts are one-command already; no MCP server or runbook yet) |
| R14 | **Lowest cost**: free tools; paid licences only at release. | ✅ every tool used so far is free or already owned |

---

## 2. How it works (the design)

**Rule for cheap models:** the LLM never writes 3D code on the fly. It only chooses which tested tool to run and
with which settings. Every tool takes JSON or flags in, writes files out, and prints one `..._OK` or
`..._FAILED` line, so success is checkable by text.

### Project layout (target)

```
<project>/
  story.md            the story
  project.json        mode (movie|game), cast, sets, scenes, shots, and the status of every step
  characters/<name>/  concept images, face_run/ (5 views), char.json, flame/, fb/, build.json, exports
  sets/<name>/        environment and props
  shots/<id>/         camera, lighting, animation, renders (movie)
  game/               engine exports (game)
  PROGRESS.md         one line appended per finished step (what, when, result)
```

### Commands (target)

`story` → `concepts` → `model <name>` → `studio` → `rig` / `animate` → `set` → `light` → `render` (movie) or
`export-game` (game). Each reads `project.json`, does one job, updates it, and appends to `PROGRESS.md`.

### Stage by stage

| Stage | How | Built | To build |
|---|---|---|---|
| Story → plan | LLM turns the story into `project.json` (cast: age, sex, ethnicity, height, build, look, outfit; sets; scenes; shots) | — | prompt + schema + validator |
| Mode | `mode` in `project.json` picks `profiles/movie.json` or `profiles/game.json` (poly budgets, hair type, texture size, lighting) | — | profiles, used by every stage |
| Concepts | Flow or Grok (whichever is signed in): full-body concept → front portrait → turned views (left 45/75 made, right mirrored) | `engine/imagegen/` (`imagegen.mjs`, `face_views.mjs`) | story → prompts, pose sheets, automatic service fallback |
| Model (Daz) | Face fit (FLAME always; FaceBuilder when licensed) → Genesis 9 face morph with rig formulas → photo skin baked and blended (seamless "detail" mode, Genesis 9's own eyes/nostrils/lips kept) → body shape and exact height → hair (with tint) and outfit → Daz scene → check renders → colour calibration against the portrait | `pipeline/daz/` (see FLOW.md) | outfit/hair chosen from the description, "Remodel" button |
| Studio | Load the Daz character, edit, rebuild in Daz | studio UI (HG-based) | Daz → Blender → glTF preview; edits back to `char.json` |
| Rig / skin | Genesis 9 is rigged and skinned; face morphs move the head joints | ✅ | Daz → Blender bridge (free Diffeomorphic DAZ Importer), engine skeleton retarget |
| Props / environments | Daz sets and props, Blender add-ons (Poly Haven, BlenderKit free), procedural layouts (game001's level builder) | library index | set builder, prop placement |
| Animation | (a) video → motion (Flow/Grok clip → pose tracking → retarget), (b) Daz poses / Mixamo library, (c) Cascadeur free tier for polish; FACS expressions; lip-sync | old video-to-motion pipeline | retarget to Genesis 9, shot animation tool, lip-sync |
| Lighting / camera | presets as tools (three-point, sunset, night, interior), HDRI skies (Poly Haven), camera moves from shot text | — | lighting and camera tools |
| Render | **Movie: Blender Cycles on the GPU** (Daz Iray is CPU-only on a GTX 10-series card). Daz renders only as checks. | Daz check renders | shot render queue, frames → video |
| Game export | Daz → Blender → glTF/FBX: LODs, hair cards, baked 2K atlases, engine skeleton, collision → Godot | glTF export code from the HG path and game001 | Genesis 9 game profile, Godot import |

### Milestones

1. **M1: one character end to end, both modes.** Finish the game001 cast; Daz → Blender bridge; movie export
   (high detail) and game export (LODs, hair cards, baked textures) to Godot. *Done when Kira walks around in
   Godot and renders in Blender Cycles.*
2. **M2: project spine.** `project.json` schema, story → cast/sets/shots, mode profiles, CLI, `RUNBOOK.md`, MCP
   server. *Done when a cheap model in Cline turns a short story into a modelled cast using tools only.*
3. **M3: Character Studio on Daz.** Load, edit, "Remodel".
4. **M4: animation.** Video → motion on Genesis 9, pose/motion library, expressions, lip-sync, Cascadeur round trip.
5. **M5: sets, lighting, rendering.** Environment builder, lighting presets, cameras, Cycles queue, video assembly.
6. **M6: game export polish.** Godot scenes with characters, animation, collision, level data.

---

## 3. Status checklist

### Done ✅
- [x] Flow and Grok picture drivers; portraits and turned views by script (`engine/imagegen/`)
- [x] FLAME photo fitter, free, tuned to 0.037 vs FaceBuilder 0.034 on a test face (`pipeline/flame/flame_photofit.py`)
- [x] FaceBuilder bridge (KeenTools, Blender 4.2, trial) (`pipeline/keentools/`)
- [x] Genesis 9 face morph from our face, with rig formulas for 40 head/neck joints (`pipeline/daz/g9_face.py`)
- [x] Photo skin on Genesis 9: closest base skin, seamless detail blend, Genesis 9's own eyes/nostrils/lips kept,
      coloured-hair filter, facial-hair translucency mask (`pipeline/daz/g9_skin.py`, `pipeline/humgen/face_blend.py`)
- [x] Daz Studio driven by script in its own instance: body shape, exact height, presets, skin, hair (with tint),
      clothes, scene save, check renders with retries, CPU-friendly Iray settings (`pipeline/daz/g9_build.dsa`)
- [x] Colour calibration loop against the portrait, resumable (`pipeline/daz/g9_calibrate.py`)
- [x] One command per character (`pipeline/daz/make_daz_character.py`)
- [x] Daz library index (`pipeline/daz/daz_library.py` → `tools/daz/library.json`)
- [x] game001 cast built in Daz: Kira, Leo, Marcus, Recruiter (see `../game001/characters/DAZ_PROGRESS.md`)

### In progress 🟡
- [ ] Leo's blue hair re-dress (texture-compression fix after an Iray out-of-memory hang)
- [ ] Recruiter: hair style is the closest installed (spiky undercut, not slicked back)

### Remaining ⬜
- [ ] Outfits matching the concepts (jacket, raincoat, bomber, three-piece suit): add Genesis 9 outfits (Daz
      freebies or bought) or generate our own garments
- [ ] Guard android (armour: needs a sci-fi armour product or our own armour)
- [ ] Daz → Blender bridge; movie and game export profiles; Godot import (M1)
- [ ] project.json, mode profiles, CLI, RUNBOOK.md, MCP server (M2)
- [ ] Character Studio on Daz (M3)
- [ ] Animation (M4), sets/lighting/render queue (M5), game polish (M6)
- [ ] Remove the Human Generator path once Daz covers children and costumes (keep `tools/Human` until then)

---

## 4. Known limits

- **Hardware:** GTX 1050 Ti (4 GB), 24 GB RAM. Daz Iray no longer supports GeForce 10-series GPUs, so it renders on
  the CPU (minutes per picture; big textures can exhaust RAM). Final film renders → Blender Cycles (GPU).
- **Content:** the installed Daz library is small (Genesis 9 Starter Essentials, Worker Uniform, Viking armour,
  Eirgrid and Mavick hair, a few poses/HDRIs). Outfits and hair for each story need adding.
- **Licences:** fine for personal use and film renders. Shipping Daz content inside a game needs Daz Interactive
  Licences per product; keep game assets swappable.
- **FaceBuilder** runs on a KeenTools trial (ends about 2026-12-03); without a licence the free FLAME fit is used
  automatically.
