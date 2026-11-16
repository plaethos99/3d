# Scripts from blender-ai-plugin

Copied unchanged from https://github.com/proyecto26/blender-ai-plugin (MIT licence, see `LICENSE`;
commit in `UPSTREAM_COMMIT`), by Proyecto 26, based on blender-skills by arjun988.

Each script takes its arguments as a dict named `ARGS` and returns a JSON-safe dict named `result`.
Run them through `pipeline/blender_bridge.py`, or with `runpy.run_path(path, init_globals={"ARGS": {...}})`
inside Blender 5.2+.

Kept because they ran correctly on our Blender 5.2.3 (their own suite: 86 of 86 tests) and on our animated character:

| Script | Use here | Result on Joyce (dancing, 50 bones, 33 meshes) |
|---|---|---|
| `rig_check.py` | rig and skinning audit: unweighted vertices, influences per vertex, left/right symmetry | worked |
| `validate_asset.py` | game-ready checks: triangle budget, n-gons, non-manifold, UVs, materials, scale | worked: found the 132k-triangle head and the mouth cut's open edges |
| `topology_report.py`, `scene_audit.py` | mesh and scene reports | in their suite |
| `export_unity.py` | Unity FBX with axis and scale fixed | worked: rig, meshes, face shapes and the dance re-imported |
| `export_godot.py` | Godot GLB, read back to verify | worked, verified |
| `export_asset.py` | engine presets for FBX, glTF, USD, OBJ | USD worked, re-imported with animation |
| `export_unreal.py` | Unreal FBX with SM_/SK_, UCX_ and SOCKET_ naming checks | in their suite (needs Unreal-style names) |
| `make_collision.py`, `generate_lods.py` | game collision and lower-detail meshes | LODs fail on meshes with face shapes (Blender can't apply a modifier to them) |
| `light_rig.py`, `render_preset.py`, `comp_grade.py`, `camera_rig.py`, `turntable.py`, `background_render.py` | lighting presets, render settings, compositor grade, cameras, turntables, renders in a background process | worked |
| `silhouette_score.py` | outline overlap of objects against reference masks per camera, without rendering | for checking models against character sheets |
| `capture_views.py`, `ref_board.py`, `compare_to_refs.py` | view captures and colour/style match against a folder of references | for style checks |

Not kept: the 98 skill write-ups (agent instructions) and the style recipes. They're useful reading,
but our pipeline checks by measurement rather than by a model looking at screenshots.
