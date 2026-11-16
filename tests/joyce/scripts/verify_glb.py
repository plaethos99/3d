"""Load the exported GLB in an empty scene and confirm what actually survived.

Usage: blender -b --factory-startup -P verify_glb.py -- <file.glb>
"""
import sys

import bpy

path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=path)

meshes = [o for o in bpy.data.objects if o.type == "MESH"]
rigs = [o for o in bpy.data.objects if o.type == "ARMATURE"]
triangles = sum(sum(len(p.vertices) - 2 for p in m.data.polygons) for m in meshes)
skinned = sum(1 for m in meshes if any(mod.type == "ARMATURE" for mod in m.modifiers))
print("GLB_MESHES", len(meshes), "skinned", skinned, "triangles", triangles)
print("GLB_ARMATURES", len(rigs), "bones", len(rigs[0].data.bones) if rigs else 0)
print("GLB_ANIMATIONS", [(a.name, round(a.frame_range[1] - a.frame_range[0] + 1)) for a in bpy.data.actions])
print("GLB_IMAGES", sorted(i.name for i in bpy.data.images))
print("GLB_MATERIALS", sorted(m.name for m in bpy.data.materials))
print("VERIFY_OK")
