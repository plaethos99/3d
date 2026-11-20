"""Patches that make Human Generator 3D 4.0.14 (written for Blender 3.2-3.6) run in Blender 5.2.

Applied by install.py to the add-on copy inside Blender (the original in tools/Human stays untouched), so a
reinstall always gives the same result. Each patch: file (relative to the add-on), the exact old text, the new
text, and why. A patch whose old text is missing fails loudly (the add-on changed), unless the new text is
already there (patched before).

Run standalone to patch an installed copy: python pipeline/humgen/compat.py <addon_dir>
"""
import os
import sys

PATCHES = [
    (
        "common/render.py",
        """    context.scene.eevee.use_gtao = True
    context.scene.render.hair_type = "STRIP"
""",
        """    # Blender 4.2+ removed eevee.use_gtao (AO is always on) and may drop hair_type: set only if present
    if hasattr(context.scene.eevee, "use_gtao"):
        context.scene.eevee.use_gtao = True
    if hasattr(context.scene.render, "hair_type"):
        context.scene.render.hair_type = "STRIP"
""",
        "eevee.use_gtao removed in Blender 4.2",
    ),
    # Blender 5 can't subclass bpy_prop_collection from Python any more (__init__ rejects the argument,
    # attribute access then crashes Blender). Make both wrappers plain proxies to the real collection.
    (
        "human/skin/skin.py",
        """class SkinNodes(bpy_prop_collection):
    \"\"\"Inherits from bpy_prop_collection to add custom methods to the collection.\"\"\"

    def __new__(cls, human: "Human") -> "SkinNodes":  # noqa D102
        skin_mat = human.objects.body.data.materials[0]
        nodes = skin_mat.node_tree.nodes
        return super().__new__(cls, nodes)  # type:ignore[call-arg]
""",
        """class _CollectionProxy:
    \"\"\"Forwards everything to a bpy collection (Blender 5 forbids subclassing it).\"\"\"

    def __init__(self, collection) -> None:  # noqa D107
        object.__setattr__(self, "_c", collection)

    def __getattr__(self, name):  # noqa D105
        return getattr(self._c, name)

    def __getitem__(self, key):  # noqa D105
        return self._c[key]

    def __iter__(self):  # noqa D105
        return iter(self._c)

    def __len__(self):  # noqa D105
        return len(self._c)

    def __contains__(self, key):  # noqa D105
        return key in self._c


class SkinNodes(_CollectionProxy):
    \"\"\"Skin material nodes with custom methods (proxy, see _CollectionProxy).\"\"\"

    def __init__(self, human: "Human") -> None:  # noqa D107
        super().__init__(human.objects.body.data.materials[0].node_tree.nodes)
""",
        "SkinNodes: Blender 5 forbids subclassing bpy_prop_collection",
    ),
    (
        "human/skin/skin.py",
        """class SkinLinks(bpy_prop_collection):
    \"\"\"Inherits from bpy_prop_collection to add custom methods to the collection.\"\"\"

    def __new__(cls, human: "Human") -> "SkinLinks":  # noqa D102
        skin_mat = human.objects.body.data.materials[0]
        links = skin_mat.node_tree.links
        return super().__new__(cls, links)  # type:ignore[call-arg]
""",
        """class SkinLinks(_CollectionProxy):
    \"\"\"Skin material links (proxy, see _CollectionProxy).\"\"\"

    def __init__(self, human: "Human") -> None:  # noqa D107
        super().__init__(human.objects.body.data.materials[0].node_tree.links)
""",
        "SkinLinks: Blender 5 forbids subclassing bpy_prop_collection",
    ),
    (
        "human/pose/pose.py",
        """            if bone.bone_group and bone.bone_group.name in SKIP_GROUPS:
""",
        """            # Blender 4.0 turned bone groups into bone collections (same names) when loading old files
            if any(c.name in SKIP_GROUPS for c in bone.bone.collections):
""",
        "bone groups became bone collections in Blender 4.0",
    ),
    (
        "human/pose/pose.py",
        """            posebone.bone.select = True
""",
        """            if hasattr(posebone, "select"):  # Blender 5 moved selection to the pose bone
                posebone.select = True
            else:
                posebone.bone.select = True
""",
        "Bone.select moved to PoseBone.select in Blender 5.0",
    ),
    # Not a Blender fix: clothes refit by smoothed local affine fits (addon_extra/smooth_deform.py) instead of
    # nearest-point offsets, which tear garments on big body changes (children).
    (
        "human/clothing/base_clothing.py",
        """        distance_dict = build_distance_dict(body_coords_world, cloth_coords_world)

        cloth_obj.parent = self._human.objects.rig

        body_eval_coords_world = world_coords_from_obj(
            body_obj,
            data=self._human.keys.all_deformation_shapekeys,
        )

        deform_obj_from_difference(
            "Body Proportions",
            distance_dict,
            body_eval_coords_world,
            cloth_obj,
            as_shapekey=True,
        )
""",
        """        cloth_obj.parent = self._human.objects.rig

        body_eval_coords_world = world_coords_from_obj(
            body_obj,
            data=self._human.keys.all_deformation_shapekeys,
        )

        from HumGen3D.common.smooth_deform import smooth_deform

        edges = np.empty(len(cloth_obj.data.edges) * 2, dtype=np.int64)
        cloth_obj.data.edges.foreach_get("vertices", edges)
        new_world = smooth_deform(
            np.asarray(body_coords_world, dtype=np.float64),
            np.asarray(body_eval_coords_world, dtype=np.float64),
            np.asarray(cloth_coords_world, dtype=np.float64),
            edges.reshape(-1, 2),
        )
        if not cloth_obj.data.shape_keys:
            basis = cloth_obj.shape_key_add(name="BASIS")
            basis.interpolation = "KEY_LINEAR"
        sk = cloth_obj.data.shape_keys.key_blocks.get("Body Proportions") or cloth_obj.shape_key_add(
            name="Body Proportions"
        )
        sk.interpolation = "KEY_LINEAR"
        inv = np.array(cloth_obj.matrix_world.inverted())
        local = new_world @ inv[:3, :3].T + inv[:3, 3]
        sk.data.foreach_set("co", local.astype(np.float32).ravel())
""",
        "clothes refit by smoothed local affine fits (children)",
    ),
    # Children: ages below 20 blend the child livekeys (special/age_child/<gender>_child_<age>.npz, made by
    # pipeline/humgen/child/build_ages.py) between the two nearest ages; skin ageing stays at its youngest.
    (
        "human/age.py",
        """        if age == self._current:
            return

        normal_value = min((age - 10) / 10, 4.0)
        young_value = (-0.1 * age + 3) if age < 30 else 0
""",
        """        if age == self._current:
            return

        child_age = age
        age = max(age, 20)  # below 20 the adult formulas stay at 20; the body comes from the child keys
        normal_value = min((age - 10) / 10, 4.0)
        young_value = (-0.1 * age + 3) if age < 30 else 0
        if child_age < 20:
            young_value = 0.0  # "aged_young" is an adult key: off for children
            normal_value = 0.6  # softer skin detail
""",
        "age below 20: child livekeys (1/2)",
    ),
    (
        "human/age.py",
        """        node_normal = nodes.get("Normal Map")
        node_normal.inputs["Strength"].default_value = normal_value

        self._human.objects.body["Age"] = age
""",
        """        node_normal = nodes.get("Normal Map")
        node_normal.inputs["Strength"].default_value = normal_value

        self._human.objects.body["Age"] = child_age
        self._set_child_shape(child_age)

    def _child_keys(self) -> dict:
        \"\"\"The child livekeys present for this human's gender: {age in years: key}.\"\"\"
        import re

        found = {}
        for key in self._human.keys.all_livekeys:
            m = re.fullmatch(r"child_(\\d+)", key.name)
            if m:
                found[int(m.group(1))] = key
        return found

    def _set_child_shape(self, age: int) -> None:
        \"\"\"Blend the two child keys nearest this age (none at 20 and over), then refit rig and clothes.\"\"\"
        import bpy

        keys = self._child_keys()
        if not keys:
            return
        want = {a: 0.0 for a in keys}
        ages = sorted(keys)
        if age < 20:
            if age <= ages[0]:
                want[ages[0]] = 1.0
            elif age >= ages[-1]:  # toward the adult body at 20
                want[ages[-1]] = 1.0 - (age - ages[-1]) / (20 - ages[-1])
            else:
                lo = max(a for a in ages if a <= age)
                hi = min(a for a in ages if a >= age)
                t = 0.0 if hi == lo else (age - lo) / (hi - lo)
                want[lo] += 1.0 - t
                want[hi] += t
        changed = False
        for a, key in keys.items():
            if abs(key.value - want[a]) > 1e-4:
                key.set_without_update(want[a])
                changed = True
        if changed:
            self._human.keys.update_human_from_key_change(bpy.context)
""",
        "age below 20: child livekeys (2/2)",
    ),
    (
        "backend/properties/scene_main_properties.py",
        """    age: IntProperty(
        default=30,
        min=20,
        max=70,
        step=10,""",
        """    age: IntProperty(
        default=30,
        min=2,  # children via pipeline/humgen child livekeys
        max=70,
        step=10,""",
        "age slider down to 2",
    ),
]

PATCHES += [
    # Blender 4.0 renamed Principled BSDF inputs ("Specular" -> "Specular IOR Level", ...): look names up through a map
    (
        "human/process/bake.py",
        '''    return next(
        (
            node_links.from_socket
            for node_links in target_node.inputs[target_socket].links  # type:ignore
        ),
        None,
    )
''',
        '''    return next(
        (
            node_links.from_socket
            for node_links in target_node.inputs[_socket_name(target_node, target_socket)].links  # type:ignore
        ),
        None,
    )


_RENAMED = {"Specular": "Specular IOR Level", "Subsurface": "Subsurface Weight", "Clearcoat": "Coat Weight",
            "Sheen": "Sheen Weight", "Transmission": "Transmission Weight", "Emission": "Emission Color"}


def _socket_name(node, name):
    """The input's name in this Blender (4.0 renamed several Principled BSDF inputs)."""
    if isinstance(name, str) and name not in node.inputs and _RENAMED.get(name) in node.inputs:
        return _RENAMED[name]
    return name
''',
        "Principled BSDF inputs renamed in Blender 4.0 (bake)",
    ),
    (
        "human/process/bake.py",
        '''            links.new(
                normal_node.outputs[0],  # type:ignore[index]
                principled.inputs[input_type],  # type:ignore
            )
        else:
            links.new(
                img_node.outputs[0], principled.inputs[input_type]  # type:ignore
            )''',
        '''            links.new(
                normal_node.outputs[0],  # type:ignore[index]
                principled.inputs[_socket_name(principled, input_type)],  # type:ignore
            )
        else:
            links.new(
                img_node.outputs[0], principled.inputs[_socket_name(principled, input_type)]  # type:ignore
            )''',
        "Principled BSDF inputs renamed in Blender 4.0 (baked materials)",
    ),
    (
        "human/process/bake.py",
        '''                mat.blend_method = "BLEND"
                mat.shadow_method = "CLIP"''',
        '''                if hasattr(mat, "surface_render_method"):  # Blender 4.2+
                    mat.surface_render_method = "BLENDED"
                else:
                    mat.blend_method = "BLEND"
                    mat.shadow_method = "CLIP"''',
        "blend_method/shadow_method replaced in Blender 4.2",
    ),
    # eye hair systems (brows) hidden: nothing converted, hc never set
    (
        "human/hair/basehair.py",
        '''        hair_objs: list[bpy.types.Object] = []
''',
        '''        hair_objs: list[bpy.types.Object] = []
        hc = None
''',
        "convert_to_haircards: no visible system left hc unset",
    ),
    (
        "human/hair/basehair.py",
        '''            hair_objs.append(cap_obj)

        hc.set_node_values(self._human)
''',
        '''            hair_objs.append(cap_obj)

        if hc:
            hc.set_node_values(self._human)
''',
        "convert_to_haircards: nothing converted, no node values to set",
    ),
]

# files added to the add-on: (source next to this file, destination inside the add-on)
EXTRA_FILES = [("addon_extra/smooth_deform.py", "common/smooth_deform.py")]


def apply(addon_dir):
    import shutil

    here = os.path.dirname(os.path.abspath(__file__))
    for src, dst in EXTRA_FILES:
        shutil.copyfile(os.path.join(here, src), os.path.join(addon_dir, dst))
        print("ADDED", dst)
    for rel, old, new, why in PATCHES:
        path = os.path.join(addon_dir, rel)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if new in text:
            print("PATCH_ALREADY", rel, "-", why)
            continue
        if old not in text:
            raise RuntimeError(f"PATCH_FAILED {rel}: old text not found ({why})")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new, 1))
        print("PATCHED", rel, "-", why)


if __name__ == "__main__":
    apply(sys.argv[1])
