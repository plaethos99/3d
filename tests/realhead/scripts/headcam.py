"""Shared camera and lighting helpers for rendering a head in Blender at known angles."""
import math

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector


def setup_render(width, height, samples):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    return scene


def soft_studio(scene, background=(0.62, 0.62, 0.62), strength=1.0, seen_backdrop=None, key=60.0, fill=30.0):
    """Even, soft light like a studio: a bright world plus a large key and fill.

    seen_backdrop, if given, is the colour the camera sees behind the subject, while the light
    falling on the subject stays the neutral background colour, as with a coloured paper backdrop.
    """
    world = bpy.data.worlds.new("studio")
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    node = nodes["Background"]
    node.inputs[0].default_value = (*background, 1)
    node.inputs[1].default_value = strength
    if seen_backdrop is not None:
        seen = nodes.new("ShaderNodeBackground")
        seen.inputs[0].default_value = (*seen_backdrop, 1)
        seen.inputs[1].default_value = 1.0
        path = nodes.new("ShaderNodeLightPath")
        mix = nodes.new("ShaderNodeMixShader")
        output = nodes["World Output"]
        links.new(path.outputs["Is Camera Ray"], mix.inputs[0])
        links.new(node.outputs[0], mix.inputs[1])
        links.new(seen.outputs[0], mix.inputs[2])
        links.new(mix.outputs[0], output.inputs["Surface"])
    scene.world = world
    for name, energy, size, az, el in (("key", key, 1.6, -25, 20), ("fill", fill, 2.0, 40, 5)):
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.size = energy, size
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        a, e = math.radians(az), math.radians(el)
        light.location = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))) * 1.6
        light.rotation_euler = (Vector((0, 0, 0)) - light.location).to_track_quat("-Z", "Y").to_euler()


def camera(scene, focal_mm=85.0, sensor_mm=36.0):
    data = bpy.data.cameras.new("cam")
    data.lens = focal_mm
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = sensor_mm
    cam = bpy.data.objects.new("cam", data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def place_orbit(cam, yaw_deg, pitch_deg, distance, target=Vector((0, 0, 0))):
    """Put the camera on a sphere around the target. Yaw 0 looks at the face (from -Y); positive yaw
    moves the camera toward the subject's left (+X), so the face turns to show its left side."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    offset = Vector((math.sin(yaw) * math.cos(pitch), -math.cos(yaw) * math.cos(pitch), math.sin(pitch))) * distance
    cam.location = target + offset
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()


def projection(scene, cam):
    """3x4 matrix taking world points to pixel coordinates (x right, y down), for the current render size."""
    width, height = scene.render.resolution_x, scene.render.resolution_y
    data = cam.data
    focal_px = data.lens / data.sensor_width * width
    k = np.array([[focal_px, 0, width / 2], [0, focal_px, height / 2], [0, 0, 1]])
    # Blender cameras look down -Z with +Y up; image rows run down, so flip Y and Z.
    world_to_cam = np.array(cam.matrix_world.inverted())
    flip = np.diag([1, -1, -1, 1])
    return k @ (flip @ world_to_cam)[:3, :]


def catchlights(scene, energy=40.0):
    """Two softboxes in front, seen only in reflections: the bright window shapes in the eyes and
    a gentle sheen on skin, without adding light to the face (the photo texture already has it)."""
    for name, az in (("softbox_left", -30), ("softbox_right", 30)):
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.shape, data.size = energy, "DISK", 0.22
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        a = math.radians(az)
        light.location = Vector((math.sin(a) * 1.4, -math.cos(a) * 1.4, 0.25))
        light.rotation_euler = (Vector((0, 0, 0)) - light.location).to_track_quat("-Z", "Y").to_euler()
        light.visible_diffuse = False
        light.visible_transmission = False
        light.visible_volume_scatter = False
        # Shine on the eyes only. On skin these reflections read as a wet, plastic sheen.
        eyes = bpy.data.collections.get("catchlight_receivers")
        if eyes is None:
            eyes = bpy.data.collections.new("catchlight_receivers")
            for obj in bpy.data.objects:
                if obj.name.startswith("eye."):
                    eyes.objects.link(obj)
        light.light_linking.receiver_collection = eyes
