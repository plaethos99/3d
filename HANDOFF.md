# Handoff: AI 3D character tool (read this first)

Updated 2026-11-20. For a new session, possibly another account, working in this folder. Everything that matters
is in this folder and in the sibling game folder `../game001`; nothing depends on an old session's memory.
Read with: **PLAN.md** (requirements, milestones, checklist) and **FLOW.md** (how the code works).

## 1. What the owner wants

A tool that takes a **story** and makes a complete **movie** or **game** from it, the owner designing everything,
the tool doing the 3D work **by script**: concept pictures (Google Flow / Grok) → characters modelled in
**Daz Studio (Genesis 9)** with face, skin, hair, rig → Character Studio for edits → animation (LLM or
Cascadeur) → props and environments (Daz, Blender add-ons) → lighting and rendering (movie) or export to the
engine (game, Godot). Driven from any LLM harness (Antigravity, Cline, Claude Code, Hermes) with cheap models.
**Personal use first**; licences only at release. Full list: PLAN.md §1 (R1–R14).

## 2. How to work (important, the owner's rules)

- **Everything by script, which the agent writes and runs.** No hand-tweaking. Every fix goes into code with a
  note on what and why. Automate app steps too (Daz, Blender): no "click here" instructions unless a step is
  truly impossible to script (sign-ins, licences), and say why.
- **Judge by looks, not numbers:** render side-by-sides against the source pictures before saying it works.
- **No going in circles:** each step has a yes/no finish line; show renders at each one. Be blunt about limits
  (budget, hardware, licences). Reuse proven tools; don't rebuild from scratch.
- **Downloads:** state source, size and licence, and get approval **before** downloading. Never install
  content from unofficial sources (repacked .rar etc.).
- **Never** enter credentials or create accounts; the owner signs in (Flow, Grok, Daz, KeenTools) and accepts
  terms themselves.
- **Deleting:** list exactly what goes and get a yes first; permanent deletes say so.
- **Git:** the repo https://github.com/plaethos99/3d is **public**: push code and docs only (never `tools/`,
  clips, generated pictures, run outputs, faces or likenesses, browser profiles, Daz content). Commit or push
  only when asked; commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Keep
  personal details (emails, Windows user paths) out of committed files.
- **Never change the BookShorts folder** (read-only).

## 3. Environment

- Windows 11, **GTX 1050 Ti (4 GB)**, 24 GB RAM. NVIDIA driver 582.78, Intel UHD 630 driver 2022.
- **Daz Studio 6** (`C:\Program Files\DAZ 3D\DAZStudio6\DAZStudio.exe`); content library
  `C:\Users\Public\Documents\My DAZ 3D Library` with Genesis 9 Starter Essentials, Worker Uniform, dForce
  Viking armour, Eirgrid hair, Mavick hair and beard, HDRI starter pack, poses (installed officially with Daz
  Install Manager). Daz Studio 4.24 is also installed but shows an update dialog: **use Daz Studio 6**.
- **Daz Iray renders on the CPU only** (Iray dropped GeForce 10-series): slow; heavy scenes can run out of RAM.
- Blender 5.2.3 portable `tools/blender-5.2.3-windows-x64/` (main); Blender 4.2.23 LTS portable
  `tools/blender-4.2.23-windows-x64/` (only for KeenTools FaceBuilder).
- Python `tools/pose-env/Scripts/python.exe -I` (mediapipe, opencv, numpy; no scipy).
- Node 26 for `engine/imagegen/`. Flow (port 9434) and Grok (port 9460) run in their own Edge profiles under
  `tools/browser-profiles/`; `node engine/imagegen/imagegen.mjs state flow|grok` checks, `open` opens the window
  for the owner to sign in. Grok sometimes refuses edits ("did not start the picture"): use Flow.
- FLAME 2023 Open: `tools/flame/` (CC-BY-4.0, credit MPI-IS). KeenTools 2025.1 (owner's, **trial** in Blender 4.2,
  ends about 2026-12-03). Human Generator pack `tools/Human/` (kept until the studio runs on Daz).

## 4. Current state (what works)

The **Daz / Genesis 9 character path** works end to end by one command per character (details in FLOW.md):

```
python pipeline/daz/make_daz_character.py <char_dir> <Name> [--from flame|fb|face|skin|daz] [--rounds 2]
```

`<char_dir>/face_run/` holds five views of the person (front, left/right 45 and 75); `<char_dir>/char.json`
holds sex, height, body morphs, eye preset, hair (presets + tint) and clothes. Output: a Genesis 9 character with
our face (morph + rig formulas), calibrated skin, body and exact height, hair and clothes, saved as a Daz scene
with check renders in `tests/daz/<name>_g9*`. A step-by-step log is appended to `<char_dir>/../DAZ_PROGRESS.md`.

## 5. The game project in progress: `../game001` ("The Portal", Godot)

- Story: a stranger offers people a fortune to play a game in his portal; players fight for 7 Neural Keys in a
  20-storey tower. `../game001/README.md`, `game-001/DESIGN.md`, Godot project in `game-001/`.
- Cast (`characters/cast.json`, concepts in `concepts/`): **Kira** (female, 1.68 m), **Leo** (male, 1.72 m, blue
  hair, glasses), **Marcus** (male, 1.88 m, shaved head, beard), **the Recruiter** (male, 1.86 m, silver hair,
  suit), **the Guard** (android, not started).
- Daz versions live in `../game001/characters/<name>/daz/` (face_run, flame, fb, char.json, build.json) and
  `tests/daz/<name>_g9.duf` + renders. Comparison sheet: `../game001/characters/cast_daz.jpg`.
- Progress log: **`../game001/characters/DAZ_PROGRESS.md`** (one entry per finished step, newest at the bottom).

### What is going on right now (2026-11-20 afternoon)

- Leo's re-dress with blue hair tint was hanging (Iray "Memory allocation failed" on the CPU). Fixed with Iray
  texture compression (256/512) in `g9_build.dsa`; the re-run `make_daz_character.py .../leo/daz Leo --from daz
  --rounds 0` was rendering. Check `tests/daz/build_leo.txt` for `tint ... on N colour controls in M added nodes`
  and the renders `tests/daz/leo_g9_{face,face34,body}.png`.
- Recruiter re-dressed with a cool grey tint (#9EA3A8) on Mavick hair; still the closest installed style.
- Uncommitted code: `pipeline/daz/` (`make_daz_character.py`, `daz_library.py`, `g9_calibrate.py`, changes to
  `g9_build.dsa` and `g9_skin.py`), `pipeline/humgen/face_blend.py`, `pipeline/flame/*` (landmarks path), and
  PLAN.md / FLOW.md / HANDOFF.md. Commit only when the owner asks.

## 6. How to resume (another account or a fresh session)

1. Read PLAN.md, FLOW.md, this file, and `../game001/characters/DAZ_PROGRESS.md`.
2. Check nothing is running: no `DAZStudio.exe`, no `blender.exe` (Task Manager). Run Daz builds **one at a time**.
3. Check sign-ins only if pictures are needed: `node engine/imagegen/imagegen.mjs state flow` (and `grok`); if
   `signin`, run `open` and ask the owner to sign in.
4. To rebuild or re-dress a character: edit its `char.json`, then
   `python pipeline/daz/make_daz_character.py ../game001/characters/<name>/daz <Name> --from daz --rounds 0`
   (colours resume from the last calibration). New face pictures: `--from flame`.
5. To add a new character: make `face_run/` with Flow (`imagegen.mjs edit flow 3:4 <concept> front.png
   "<portrait prompt>"`, then `face_views.mjs flow front.png <face_run> left45,left75`, mirror for the right),
   write `char.json` (use `python pipeline/daz/daz_library.py --find "<words>"` to find hair and clothes), run the
   command with `--from flame`.
6. Judge every result by looking at the renders next to the portrait and the concept; log it in DAZ_PROGRESS.md.

## 7. Next steps (in order)

1. Finish Leo (blue hair visible?), then show the owner the full cast sheet.
2. Outfits closer to the concepts (Genesis 9 outfits from Daz freebies, or generate our own garments); the Guard
   android's armour.
3. **M1** (PLAN.md): Daz → Blender bridge (Diffeomorphic DAZ Importer, free; ask before downloading), movie export
   (Cycles on GPU) and game export (LODs, hair cards, baked 2K textures) to Godot. Finish line: Kira walking in
   Godot and rendering in Cycles.
4. **M2**: project.json + mode profiles + CLI + RUNBOOK.md + MCP server, so cheap models can drive it.

## 8. Lessons learned

- One shared topology (Genesis 9) beats gluing meshes from different sources.
- Faces: FaceBuilder (licensed) and our FLAME photo fit both beat the old scan route; the FLAME fit needs
  50 shape directions, strong shape regularisation and loose expression to be good.
- Skin: blend the photo's **detail** onto the base skin's own colour (no seams); keep the base's eyes, nostrils and
  lips (located from landmarks, not guessed from colours); calibrate colour against the portrait in renders.
- Daz scripting facts: FLOW.md §4. Blender exits 0 even when its script fails: every script prints `<NAME>_OK`.
