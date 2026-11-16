"""The pipeline's tools: each wraps one of the project's scripts and returns scores and pass/fail.

Every tool returns a dict:
  {"tool", "args", "ok" (ran without error), "passed" (met its targets, or None if it has none),
   "metrics", "failed_targets", "files", "seconds", "log_tail"}
Targets live in targets.json, so they can be changed without touching code.
"""
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLENDER = ROOT / "tools" / "blender-5.2.3-windows-x64" / "blender.exe"
PY = ROOT / "tools" / "pose-env" / "Scripts" / "python.exe"
POSE_MODEL = ROOT / "tools" / "pose-model" / "pose_landmarker_heavy.task"
FACE_MODEL = ROOT / "tools" / "face-model" / "face_landmarker.task"
SCRIPTS = ROOT / "tests" / "joyce" / "scripts"
HEAD_SCRIPTS = ROOT / "tests" / "realhead" / "scripts"
RUNS = ROOT / "pipeline" / "runs"
TARGETS = json.loads((Path(__file__).parent / "targets.json").read_text(encoding="utf-8"))


def _run(cmd, timeout):
    start = time.time()
    try:
        done = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout,
                              cwd=ROOT, encoding="utf-8", errors="replace")
        text, code = done.stdout + "\n" + done.stderr, done.returncode
    except subprocess.TimeoutExpired:
        text, code = f"TIMEOUT after {timeout} s", -1
    return code, text, round(time.time() - start, 1)


def _blender(script, blend, args, timeout):
    return _run([BLENDER, "--background", blend, "--python-exit-code", "1", "--python", script, "--", *args], timeout)


def _python(script, args, timeout):
    return _run([PY, "-I", script, *args], timeout)


def _tail(text, lines=25):
    noise = ("TBBmalloc", "DeprecationWarning", "use_nodes = True", "XNNPACK", "InitGoogle", "W0000", "I0000",
             "inference_feedback_manager", "gl_context")
    keep = [l for l in text.splitlines() if l.strip() and not any(n in l for n in noise)]
    return "\n".join(keep[-lines:])


def _json_after(text, prefix):
    for line in text.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix):].strip())
    return None


def _check(tool, metrics):
    rules = TARGETS.get(tool)
    if not rules:
        return None, []
    failed = []
    for key, (op, limit) in rules.items():
        value = metrics.get(key)
        if value is None:
            failed.append(f"{key}: missing")
        elif (op == ">=" and value < limit) or (op == "<=" and value > limit):
            failed.append(f"{key} = {value}, needs {op} {limit}")
    return not failed, failed


def _result(tool, args, code, text, seconds, metrics, files):
    ok = code == 0 and "Traceback" not in text
    passed, failed = _check(tool, metrics) if ok else (False, ["the tool did not run successfully"])
    return {"tool": tool, "args": args, "ok": ok, "passed": passed, "metrics": metrics, "failed_targets": failed,
            "files": {k: str(v) for k, v in files.items()}, "seconds": seconds, "log_tail": _tail(text)}


def run_dir(run):
    path = RUNS / run
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------- motion from video
def track_video(run, video, max_seconds=30):
    """Find the performer's body points in every frame of a video."""
    d = run_dir(run)
    out = d / "source_pose.json"
    code, text, s = _python(SCRIPTS / "track_video.py", [POSE_MODEL, Path(video), out, str(max_seconds)], 1800)
    metrics = {}
    m = re.search(r"TRACK_OK frames (\d+) missing (\d+)", text)
    if m:
        frames, missing = int(m.group(1)), int(m.group(2))
        metrics = {"frames": frames, "tracked_share": round(1 - missing / max(frames, 1), 3)}
    (d / "video.txt").write_text(str(Path(video).resolve()), encoding="utf-8")
    return _result("track_video", {"run": run, "video": str(video), "max_seconds": max_seconds}, code, text, s, metrics,
                   {"pose": out})


def inspect_clip(run):
    """Check the clip suits motion capture: body in frame, camera steadiness, how clearly each part is seen."""
    d = run_dir(run)
    video = (d / "video.txt").read_text(encoding="utf-8").strip()
    out = d / "tracking_check.png"
    code, text, s = _python(SCRIPTS / "inspect_clip.py", [video, d / "source_pose.json", out], 900)
    metrics = {}
    m = re.search(r"WHOLE_BODY_IN_FRAME ([\d.]+)", text)
    if m:
        metrics["whole_body_share"] = round(float(m.group(1)) / 100, 3)
    m = re.search(r"CAMERA_DRIFT_PERCENT_OF_WIDTH x ([-\d.]+) y ([-\d.]+)", text)
    if m:
        metrics["camera_drift_percent"] = round(max(abs(float(m.group(1))), abs(float(m.group(2)))), 1)
    for part, vis in re.findall(r"^(\w+)\s+in frame\s+[\d.]+%\s+visibility ([\d.]+)", text, re.M):
        metrics[f"visibility_{part}"] = float(vis)
    return _result("inspect_clip", {"run": run}, code, text, s, metrics, {"sheet": out})


def clean_tracking(run):
    """Repair swapped left/right labels, remove outliers, and take shape from the 2D image."""
    d = run_dir(run)
    code, text, s = _python(SCRIPTS / "clean_tracking.py", [d / "source_pose.json", d / "clean_pose.json"], 900)
    metrics = _json_after(text, "CLEAN_REPORT") or {}
    return _result("clean_tracking", {"run": run}, code, text, s, metrics, {"pose": d / "clean_pose.json"})


def retarget(run, character):
    """Put the cleaned motion on the character's skeleton, with feet planted on the floor."""
    d = run_dir(run)
    code, text, s = _blender(SCRIPTS / "retarget_video.py", Path(character),
                             [d / "clean_pose.json", d / "motion.blend", d / "motion.glb", d / "retarget_report.json"], 3600)
    report = _json_after(text, "RETARGET_REPORT") or {}
    metrics = {k: report.get(k) for k in ("frames", "planted_foot_slip_mm", "lowest_shoe_mm", "highest_lowest_shoe_mm",
                                          "speed_m_per_s", "contact_share_L", "contact_share_R")}
    if metrics.get("planted_foot_slip_mm") is None:
        metrics["planted_foot_slip_mm"] = 0.0  # no foot stayed planted long enough to slip
    return _result("retarget", {"run": run, "character": str(character)}, code, text, s, metrics,
                   {"blend": d / "motion.blend", "glb": d / "motion.glb"})


def clear_arms(run, margin_mm=15):
    """Keep arms in front of the body where the performer's arms were seen (for rounder characters)."""
    d = run_dir(run)
    code, text, s = _blender(SCRIPTS / "arms_clear_body.py", d / "motion.blend",
                             [d / "clean_pose.json", d / "motion.blend", str(margin_mm)], 3600)
    metrics = {}
    m = re.search(r"ARMS_MOVED_FRAMES (\d+) of (\d+) move mm median ([\d.]+)", text)
    if m:
        metrics = {"frames_changed": int(m.group(1)), "frames": int(m.group(2)), "median_move_mm": float(m.group(3))}
    return _result("clear_arms", {"run": run, "margin_mm": margin_mm}, code, text, s, metrics, {"blend": d / "motion.blend"})


def render_motion(run):
    """Render the animation (slow: minutes per second of video on this computer)."""
    d = run_dir(run)
    code, text, s = _blender(SCRIPTS / "render_motion.py", d / "motion.blend", [d / "frames", d / "result.mp4"], 7200)
    return _result("render_motion", {"run": run}, code, text, s, {"rendered": int("RENDER_MOTION_OK" in text)},
                   {"video": d / "result.mp4"})


def score_motion(run):
    """Track the rendered character and compare its limbs with the performer's; makes a side-by-side video."""
    d = run_dir(run)
    video = (d / "video.txt").read_text(encoding="utf-8").strip()
    code1, t1, s1 = _python(SCRIPTS / "track_video.py", [POSE_MODEL, d / "result.mp4", d / "result_pose.json", "600"], 1800)
    code2, t2, s2 = _python(SCRIPTS / "compare_motion.py", [d / "source_pose.json", d / "result_pose.json", video,
                                                             d / "result.mp4", d / "side_by_side.mp4", d / "side_by_side.png"], 1800)
    code3, t3, s3 = _python(SCRIPTS / "score_motion.py",
                            [d / "source_pose.json", d / "result_pose.json", run], 600)
    text = t1 + t2 + t3
    metrics = {}
    m = re.search(r"LIMB_DIRECTION_ERROR_DEG \S+ ([\d.]+) (\{.*\})", t3)
    if m:
        metrics = {"overall_deg": float(m.group(1)), **json.loads(m.group(2).replace("'", '"'))}
    return _result("score_motion", {"run": run}, max(code1, code2, code3), text, s1 + s2 + s3, metrics,
                   {"side_by_side": d / "side_by_side.mp4", "sheet": d / "side_by_side.png"})


# ---------------------------------------------------------------- character checks
def render_compare(character, reference, pose="stand", azimuth=38, run="checks"):
    """Render the character like a reference image and measure how well the outlines overlap."""
    d = run_dir(run)
    prefix = d / f"compare_{Path(reference).stem}_{pose}"
    code, text, s = _blender(SCRIPTS / "render_compare.py", Path(character),
                             [Path(reference), prefix, str(azimuth), "--samples", "24", "--height", "1248", "--pose", pose], 1800)
    metrics = {}
    m = re.search(r"^IOU ([\d.]+)", text, re.M)
    if m:
        metrics["outline_overlap"] = float(m.group(1))
    return _result("render_compare", {"character": str(character), "reference": str(reference), "pose": pose,
                                      "azimuth": azimuth}, code, text, s, metrics,
                   {"compare": f"{prefix}_compare.png", "render": f"{prefix}_render.png"})


# ---------------------------------------------------------------- engine scripts (blender-ai-plugin, MIT)
VENDOR = ROOT / "engine" / "third_party" / "blender_ai_plugin"
BRIDGE = ROOT / "pipeline" / "blender_bridge.py"


def _bridge(blend, steps, name, timeout=3600, **extra):
    jobs = run_dir(name) / f"bridge_{steps[0]['name']}.json"
    jobs.write_text(json.dumps({"steps": steps, **extra}), encoding="utf-8")
    code, text, s = _blender(BRIDGE, Path(blend), [jobs], timeout)
    return code, text, s, (_json_after(text, "BRIDGE_RESULT") or {})


def _character_parts(blend, name):
    """The rig and every mesh it moves, read from the file."""
    probe = run_dir(name) / "parts.py"
    probe.write_text("import bpy, json\nrig = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)\n"
                     "meshes = [o.name for o in bpy.data.objects if o.type == 'MESH' and (o.parent == rig or any("
                     "m.type == 'ARMATURE' for m in o.modifiers))]\n"
                     "print('PARTS ' + json.dumps({'rig': rig.name if rig else None, 'meshes': meshes}))\n", encoding="utf-8")
    code, text, s = _run([BLENDER, "--background", Path(blend), "--python", probe], 600)
    return _json_after(text, "PARTS") or {"rig": None, "meshes": []}


def check_character(character, run="checks", max_influences=4):
    """Game-readiness audit of a rigged character: skinning, influences per vertex, symmetry, mesh problems."""
    parts = _character_parts(character, run)
    steps = [{"name": "rig_check", "script": str(VENDOR / "rig_check.py"),
              "args": {"armature": parts["rig"], "max_influences": max_influences}},
             {"name": "validate", "script": str(VENDOR / "validate_asset.py"),
              "args": {"objects": parts["meshes"], "prefix": ""}}]
    code, text, s, res = _bridge(character, steps, run)
    rig = res.get("rig_check", {})
    meshes = rig.get("meshes", {}) if isinstance(rig, dict) else {}
    val = res.get("validate", {}) if isinstance(res.get("validate"), dict) else {}
    failed = [f"{n}: {c}" for n, o in val.get("objects", {}).items() for c, r in o.get("checks", {}).items()
              if r.get("status") == "FAIL"]
    metrics = {"bones": rig.get("bones"), "meshes": len(parts["meshes"]),
               "unweighted_vertices": sum(m.get("unweighted_vertices", 0) for m in meshes.values()),
               "vertices_over_max_influences": sum(m.get("vertices_over_4_influences", 0) for m in meshes.values()),
               "unpaired_side_bones": len(rig.get("unpaired_side_bones", []) or []),
               "mesh_check_failures": len(failed)}
    result = _result("check_character", {"character": str(character)}, code, text, s, metrics, {})
    result["details"] = failed[:20]
    return result


def export_engines(character, run, animation=True, engines=("unity", "godot", "unreal", "usd")):
    """Export the character, with its animation, for Unity (FBX), Godot (GLB), Unreal (FBX) and USD."""
    d = run_dir(run)
    parts = _character_parts(character, run)
    objs = [parts["rig"]] + parts["meshes"] if parts["rig"] else parts["meshes"]
    stem = Path(character).stem
    plan = {"unity": ("export_unity.py", {"objects": objs, "out_path": str(d / f"{stem}_unity.fbx"), "animation": animation, "strict": False}),
            "godot": ("export_godot.py", {"objects": objs, "out_path": str(d / f"{stem}_godot.glb"), "animation": animation}),
            "unreal": ("export_asset.py", {"objects": objs, "engine": "unreal", "out_path": str(d / f"{stem}_unreal.fbx"), "animation": animation}),
            "usd": ("export_asset.py", {"objects": objs, "engine": "generic", "out_path": str(d / f"{stem}.usdc"), "animation": animation})}
    steps = [{"name": e, "script": str(VENDOR / plan[e][0]), "args": plan[e][1]} for e in engines]
    code, text, s, res = _bridge(character, steps, run)
    files, metrics = {}, {}
    for e in engines:
        r = res.get(e, {}) if isinstance(res.get(e), dict) else {}
        path = r.get("path") or plan[e][1]["out_path"]
        ok = bool(r.get("exported", r.get("exists"))) and not r.get("error")
        metrics[f"{e}_ok"] = int(ok)
        if ok:
            files[e] = path
            metrics[f"{e}_mb"] = round(Path(path).stat().st_size / 1e6, 1) if Path(path).exists() else None
    return _result("export_engines", {"character": str(character), "run": run}, code, text, s, metrics, files)


TOOLS = {f.__name__: f for f in (track_video, inspect_clip, clean_tracking, retarget, clear_arms, render_motion,
                                 score_motion, render_compare, check_character, export_engines)}

# Descriptions for the model, in the OpenAI-compatible "tools" format.
SCHEMAS = [
    {"name": "track_video", "description": track_video.__doc__, "parameters": {"type": "object", "properties": {
        "run": {"type": "string"}, "video": {"type": "string"}, "max_seconds": {"type": "number"}}, "required": ["run", "video"]}},
    {"name": "inspect_clip", "description": inspect_clip.__doc__, "parameters": {"type": "object", "properties": {
        "run": {"type": "string"}}, "required": ["run"]}},
    {"name": "clean_tracking", "description": clean_tracking.__doc__, "parameters": {"type": "object", "properties": {
        "run": {"type": "string"}}, "required": ["run"]}},
    {"name": "retarget", "description": retarget.__doc__, "parameters": {"type": "object", "properties": {
        "run": {"type": "string"}, "character": {"type": "string"}}, "required": ["run", "character"]}},
    {"name": "clear_arms", "description": clear_arms.__doc__, "parameters": {"type": "object", "properties": {
        "run": {"type": "string"}, "margin_mm": {"type": "number"}}, "required": ["run"]}},
    {"name": "render_motion", "description": render_motion.__doc__, "parameters": {"type": "object", "properties": {
        "run": {"type": "string"}}, "required": ["run"]}},
    {"name": "score_motion", "description": score_motion.__doc__, "parameters": {"type": "object", "properties": {
        "run": {"type": "string"}}, "required": ["run"]}},
    {"name": "render_compare", "description": render_compare.__doc__, "parameters": {"type": "object", "properties": {
        "character": {"type": "string"}, "reference": {"type": "string"}, "pose": {"type": "string"},
        "azimuth": {"type": "number"}}, "required": ["character", "reference"]}},
    {"name": "check_character", "description": check_character.__doc__, "parameters": {"type": "object", "properties": {
        "character": {"type": "string"}, "max_influences": {"type": "integer"}}, "required": ["character"]}},
    {"name": "export_engines", "description": export_engines.__doc__, "parameters": {"type": "object", "properties": {
        "character": {"type": "string"}, "run": {"type": "string"}, "animation": {"type": "boolean"}},
        "required": ["character", "run"]}},
]
