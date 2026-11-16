"""Run a list of engine scripts inside Blender on one file, and print their results as JSON.

Usage: blender -b <file.blend> -P pipeline/blender_bridge.py -- <jobs.json>
jobs.json: {"steps": [{"script": "<path>", "args": {...}, "name": "..."}], "save_as": optional .blend path,
            "render_still": optional {"path": ".png", "frame": 1}}
Each script is run with runpy and ARGS, as blender-ai-plugin's scripts expect. A failing step is
recorded and the rest still run. Prints one line: BRIDGE_RESULT {json}.
"""
import json
import runpy
import sys
import traceback

import bpy

jobs = json.load(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8"))
out = {}
for step in jobs.get("steps", []):
    name = step.get("name") or step["script"]
    try:
        out[name] = runpy.run_path(step["script"], init_globals={"ARGS": step.get("args", {})}).get("result")
    except Exception as e:
        out[name] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
if jobs.get("render_still"):
    r = jobs["render_still"]
    bpy.context.scene.frame_set(int(r.get("frame", 1)))
    bpy.context.scene.render.filepath = r["path"]
    bpy.ops.render.render(write_still=True)
    out["render_still"] = {"path": r["path"]}
if jobs.get("save_as"):
    bpy.ops.wm.save_as_mainfile(filepath=jobs["save_as"])
print("BRIDGE_RESULT " + json.dumps(out, default=str))
