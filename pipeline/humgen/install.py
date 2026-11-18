"""Install Human Generator 3D (add-on + .hgpack content packs) into the portable Blender.

Usage: blender -b -P pipeline/humgen/install.py -- <addon_dir> <content_dir> <pack.hgpack> [more ...]
  addon_dir   the HumGen3D add-on folder (tools/Human/HumGen3D)
  content_dir where the packs are unpacked (tools/Human/content); kept out of git (tools/ is ignored)

What it does, the same as the add-on's own "Install" button, but by script:
1. Copies the add-on into Blender's user add-on folder, applies compat.py's Blender 5.2 patches, enables it.
2. Sets the add-on's content folder preference and saves the preferences.
3. Unpacks each pack (skipping macOS __MACOSX junk), and writes the file list into the pack's json under
   content_packs/ (the add-on uses that list to uninstall a pack).
Packs already installed (their json exists) are skipped unless --force is given.
Prints HUMGEN_INSTALL_OK at the end (Blender exits 0 even when a script fails).
"""
import json
import os
import shutil
import sys
import zipfile

import addon_utils
import bpy

args = sys.argv[sys.argv.index("--") + 1:]
force = "--force" in args
args = [a for a in args if a != "--force"]
addon_src, content_dir, packs = os.path.abspath(args[0]), os.path.abspath(args[1]), args[2:]
content_dir = content_dir.rstrip("\\/") + os.sep  # add-on code joins paths with "+", needs the trailing sep

# 1. add-on
user_addons = bpy.utils.user_resource("SCRIPTS", path="addons", create=True)
dst = os.path.join(user_addons, "HumGen3D")
if os.path.isdir(dst):
    shutil.rmtree(dst)
shutil.copytree(addon_src, dst, ignore=shutil.ignore_patterns(".git*", "__pycache__", ".vscode"))
print("ADDON_COPIED", dst)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compat  # noqa: E402  (Blender 5.2 fixes, see compat.py)

compat.apply(dst)
if user_addons not in sys.path:  # folder may not have existed when Blender started
    sys.path.append(user_addons)
addon_utils.modules_refresh()
mod = addon_utils.enable("HumGen3D", default_set=True, persistent=True)
if mod is None:
    raise SystemExit("HUMGEN_INSTALL_FAILED add-on did not enable (see errors above)")
print("ADDON_ENABLED", getattr(mod, "bl_info", {}).get("version"))

# 2. content folder
os.makedirs(content_dir, exist_ok=True)
prefs = bpy.context.preferences.addons["HumGen3D"].preferences
prefs.filepath_ = content_dir
bpy.ops.wm.save_userpref()
print("CONTENT_DIR", prefs.filepath)

# 3. packs
for pack in packs:
    pack = os.path.abspath(pack)
    with zipfile.ZipFile(pack) as z:
        names = [n for n in z.namelist() if not n.startswith("__MACOSX") and not n.endswith("/")]
        jsons = [n for n in names if n.startswith("content_packs/") and n.endswith(".json")]
        if len(jsons) != 1:
            print("PACK_FAILED", os.path.basename(pack), "expected one content_packs/*.json, got", jsons)
            continue
        json_path = os.path.join(content_dir, jsons[0])
        if os.path.exists(json_path) and not force:
            print("PACK_SKIPPED (already installed)", os.path.basename(pack))
            continue
        for n in names:
            target = os.path.abspath(os.path.join(content_dir, n))
            if not target.startswith(content_dir):  # zip-slip guard
                raise SystemExit(f"HUMGEN_INSTALL_FAILED unsafe path in pack: {n}")
            z.extract(n, content_dir)
    with open(json_path) as f:
        data = json.load(f)
    data["files"] = names
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    print("PACK_OK", os.path.basename(pack), len(names), "files")

print("HUMGEN_INSTALL_OK")
