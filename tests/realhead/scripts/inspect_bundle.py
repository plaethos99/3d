"""List what the human base mesh bundle contains: objects, sizes, UVs, modifiers and materials."""
import bpy

for obj in bpy.data.objects:
    line = f"{obj.name:40s} {obj.type:9s}"
    if obj.type == "MESH":
        me = obj.data
        dims = tuple(round(v, 3) for v in obj.dimensions)
        line += f" verts {len(me.vertices):7d} faces {len(me.polygons):7d} uv {len(me.uv_layers)} dims {dims}"
        line += f" loc {tuple(round(v, 3) for v in obj.location)} scale {tuple(round(v, 3) for v in obj.scale)}"
        line += f" mods {[m.type for m in obj.modifiers]} mats {[m.name for m in me.materials if m]}"
        line += f" shapekeys {len(me.shape_keys.key_blocks) if me.shape_keys else 0} vgroups {len(obj.vertex_groups)}"
    line += f" parent {obj.parent.name if obj.parent else '-'} colls {[c.name for c in obj.users_collection]}"
    print("OBJ", line)
print("COLLECTIONS", [c.name for c in bpy.data.collections])
print("INSPECT_OK")
