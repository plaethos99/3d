"""Install KeenTools FaceBuilder into the portable Blender 4.2 LTS (its core needs Python <= 3.11; Blender 5.2 has 3.13).

Usage: tools/blender-4.2.23-windows-x64/blender.exe -b -P pipeline/keentools/install_fb.py
Takes the add-on and its core library from tools/keentools/ (keentools/ and pykeentools_installation/, the user's
own KeenTools download; never in the repo), copies them into this Blender's add-ons (portable config), enables the
add-on and reports whether the core loads and the licence state. Prints FB_INSTALL_OK.
Only the FaceBuilder step runs in Blender 4.2 (fb_head.py); everything after it runs in Blender 5.2.
"""
import os
import shutil
import sys

import addon_utils
import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "tools", "keentools")
addons = bpy.utils.user_resource("SCRIPTS", path="addons", create=True)
dst = os.path.join(addons, "keentools")
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(os.path.join(SRC, "keentools"), dst)
# the core goes where the add-on's own installer unpacks it: pykeentools_loader/pykeentools/pykeentools_installation
core_dst = os.path.join(dst, "blender_independent_packages", "pykeentools_loader", "pykeentools",
                        "pykeentools_installation")
shutil.copytree(os.path.join(SRC, "pykeentools_installation"), core_dst)
print("COPIED to", dst)
bpy.utils.refresh_script_paths()
if addons not in sys.path:
    sys.path.append(addons)
mod = addon_utils.enable("keentools", default_set=True, persistent=True)
print("ENABLED", bool(mod))
bpy.ops.wm.save_userpref()
from keentools.blender_independent_packages import pykeentools_loader as pkt  # noqa: E402

print("CORE installed:", pkt.is_installed(), "| python supported:", pkt.is_python_supported())
try:
    m = pkt.module()
    print("CORE version:", m.__version__ if hasattr(m, "__version__") else "?")
    res = m.FaceBuilder.license_manager().perform_license_and_trial_check()
    r = res[0]
    print("FB_LICENSE", {a: getattr(r, a) for a in dir(r) if not a.startswith("_") and not callable(getattr(r, a))}, res[1])
except Exception as e:  # noqa: BLE001
    print("CORE load:", repr(e)[:400])
print("FB_INSTALL_OK")
