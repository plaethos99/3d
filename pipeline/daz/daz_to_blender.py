"""A Daz Studio scene (Genesis 9 character) in Blender: imported with the Diffeomorphic DAZ Importer, lit and rendered.

Usage: python pipeline/daz/daz_to_blender.py <scene.duf> <out_dir> [--name leo] [--engine eevee|cycles] [--samples 64]
       [--hair hair.json] [--photo portrait.png --morph <Daz morph .dsf>] [--likeness 3] [--reexport]
Steps (no clicks):
  1. Daz Studio exports <scene>.dbz next to the scene (the add-on's own export_to_blender.dsa, made headless here:
     its dialog and message box removed; written into tools/daz/, not the repo, it is GPL code of the add-on)
  2. the content folders Daz uses -> tools/daz/root_paths.json (the importer finds morphs, textures there)
  3. Blender 5.2 (blender_daz_render.py): Easy Import of the .duf, studio lights, camera on the eyes, renders
     <out_dir>/<name>_bl_{face,face34,body}.png and saves <out_dir>/<name>_bl.blend
Needs the add-on installed in the portable Blender (tools/blender-5.2.3-windows-x64, extension "import_daz", from
github.com/Diffeomorphic/import_daz, GPL). Prints DAZ_TO_BLENDER_OK.
"""
import gzip
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
B5 = os.path.join(ROOT, "tools", "blender-5.2.3-windows-x64", "blender.exe")
ADDON = os.path.join(os.path.dirname(B5), "portable", "extensions", "user_default", "import_daz")
DAZ = r"C:\Program Files\DAZ 3D\DAZStudio6\DAZStudio.exe"
TOOLS = os.path.join(ROOT, "tools", "daz")


def opt(k, default):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default


scene, out_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
name = opt("--name", os.path.splitext(os.path.basename(scene))[0])
os.makedirs(out_dir, exist_ok=True)
os.makedirs(TOOLS, exist_ok=True)
dbz = os.path.splitext(scene)[0] + ".dbz"

# 1. headless exporter from the add-on's own Daz script
src = open(os.path.join(ADDON, "to_daz_studio", "Scripts", "Diffeomorphic", "export_to_blender.dsa"), encoding="utf-8").read()
src = src.replace('MessageBox.information( msg, appName, "&OK" );', 'print(msg);')
lines = src.rstrip().splitlines()
assert lines[-1].strip() == "createDialog()", "export_to_blender.dsa changed: update daz_to_blender.py"
src = "\n".join(lines[:-1]) + r'''

// headless: scriptArgs = scene .duf, .dbz out, report
var rep = new DzFile(App.scriptArgs[2]); rep.open(DzFile.WriteOnly);
try {
    App.getContentMgr().openFile(App.scriptArgs[0], false);
    useExportHD = false; useHDConvention = false; useHDUVs = false;
    exportToBlender(App.scriptArgs[1]);
    rep.writeLine("exported " + App.scriptArgs[1]);
} catch (e) { rep.writeLine("ERROR " + e); }
rep.writeLine("done"); rep.close();
App.delayedExit();
'''
dsa = os.path.join(TOOLS, "export_dbz_headless.dsa")
open(dsa, "w", encoding="utf-8").write(src)

report = os.path.join(TOOLS, "export_dbz_report.txt")
t0 = time.time()
if os.path.exists(dbz) and os.path.getmtime(dbz) > os.path.getmtime(scene) and "--reexport" not in sys.argv:
    print("DBZ up to date:", dbz, flush=True)  # the scene has not changed since the last export
else:
    for f in (report, dbz):
        if os.path.exists(f):
            os.remove(f)
    proc = subprocess.Popen([DAZ, "-instanceName", "#", "-noPrompt", "-scriptArg", scene.replace("\\", "/"),
                             "-scriptArg", dbz.replace("\\", "/"), "-scriptArg", report.replace("\\", "/"),
                             dsa.replace("\\", "/")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while True:
        if os.path.exists(report) and "done" in open(report, errors="replace").read():
            break
        if proc.poll() is not None or time.time() - t0 > 900:
            raise SystemExit("DAZ_TO_BLENDER_FAILED: Daz did not export the .dbz")
        time.sleep(5)
    txt = open(report, errors="replace").read()
    if "ERROR" in txt or not os.path.exists(dbz):
        raise SystemExit("DAZ_TO_BLENDER_FAILED: " + txt.strip())
    print(f"DBZ {dbz} ({os.path.getsize(dbz) // 1024} KB, {time.time() - t0:.0f}s)", flush=True)
    try:
        proc.wait(120)  # Daz exits by itself; one Daz at a time
    except subprocess.TimeoutExpired:
        pass

# 2. Daz's content folders for the importer
try:
    data = json.load(gzip.open(dbz, "rt", encoding="utf-8"))
except OSError:
    data = json.load(open(dbz, encoding="utf-8"))
roots = os.path.join(TOOLS, "root_paths.json")
json.dump(data["rootpaths"], open(roots, "w"), indent=1)

# 3. Blender: import, light, render
r = subprocess.run([B5, "-b", "--factory-startup", "-P", os.path.join(HERE, "blender_daz_render.py"), "--",
                    scene, roots, out_dir, name, "--engine", opt("--engine", "eevee"), "--samples", opt("--samples", "64")]
                   + (["--hair", os.path.abspath(opt("--hair", ""))] if opt("--hair", "") else [])
                   + (["--photo", os.path.abspath(opt("--photo", "")), "--morph", opt("--morph", ""),
                       "--likeness", opt("--likeness", "3")] if opt("--photo", "") else []),
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
text = r.stdout + r.stderr
open(os.path.join(out_dir, f"{name}_bl_log.txt"), "w", encoding="utf-8").write(text)
for ln in text.splitlines():
    if ln.startswith(("IMPORTED", "RENDER", "CAMERA", "WARN", "LIKENESS", "LIDS", "COLOUR", "HAIR")):
        print("   ", ln[:200])
if "BLENDER_DAZ_RENDER_OK" not in text:
    print(text[-3000:])
    raise SystemExit("DAZ_TO_BLENDER_FAILED at blender_daz_render.py")
print("DAZ_TO_BLENDER_OK", name, f"({time.time() - t0:.0f}s)")
