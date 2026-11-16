"""Build Joyce in Blender. Usage: blender -b -P build_joyce.py -- <stage> <output.blend>

Stages: blockout (rough volumes, flat colours) or full (all detail and textures).
Measurements come from refs/stand.png: 940 px = 1.5 m, feet at pixel row 1105.
Joyce faces -Y, and her left side is +X.
"""
import math
import os
import random
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joyce_lib as L  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
STAGE = args[0]
OUT = args[1]
FULL = STAGE == "full"
TEX_DIR = os.path.join(os.path.dirname(OUT), "textures")

bpy.ops.wm.read_factory_settings(use_empty=True)
random.seed(7)
rng = np.random.default_rng(7)

HEAD_C = np.array((0.0, 0.0, 1.245))
HEAD_R = np.array((0.152, 0.150, 0.175))
SHOULDER = np.array((0.180, 0.0, 0.940))
ELBOW = np.array((0.216, 0.012, 0.735))
WRIST = np.array((0.224, -0.030, 0.565))
HIP = np.array((0.082, 0.0, 0.680))
KNEE = np.array((0.088, -0.006, 0.375))
ANKLE = np.array((0.092, 0.010, 0.085))


# ------------------------------------------------------------------ face sculpt
def gauss(x, z, cx, cz, sx, sz):
    return np.exp(-(((x - cx) / sx) ** 2 + ((z - cz) / sz) ** 2))


EYE_X, EYE_Y, EYE_Z, EYE_R = 0.062, -0.103, 1.243, 0.036


def mouth_line(x):
    return -0.110 + 0.009 * (np.abs(x) / 0.056) ** 2


def face_offset(x, z, front):
    """Sculpt offset in metres at head-local x, z. Positions come from the reference grid."""
    ax = np.abs(x)
    d = np.zeros_like(x)
    # Nose: a wide, round ball with soft wings and a short bridge.
    d += 0.029 * gauss(x, z, 0.0, -0.052, 0.024, 0.021)
    d += 0.0040 * gauss(x, z, 0.0, -0.012, 0.012, 0.036)
    d += 0.0085 * gauss(ax, z, 0.024, -0.065, 0.011, 0.012)
    d -= 0.0030 * gauss(ax, z, 0.017, -0.079, 0.0055, 0.0035)
    d -= 0.0030 * gauss(x, z, 0.0, -0.084, 0.008, 0.004)
    # Apple cheeks that lift towards the eye, and soft jowls beneath.
    d += 0.028 * gauss(ax, z, 0.094, -0.084, 0.044, 0.048)
    d += 0.015 * gauss(ax, z, 0.088, -0.138, 0.040, 0.026)
    # Smile fold from the nose wing to the mouth corner.
    fold = np.exp(-(((z - (-0.060 - 0.95 * (ax - 0.038))) / 0.0065) ** 2))
    fold = fold * np.clip((ax - 0.030) / 0.012, 0, 1) * np.clip((0.072 - ax) / 0.012, 0, 1)
    d -= 0.0030 * fold
    # Eye sockets and the pad under each eye.
    d -= 0.021 * gauss(ax, z, EYE_X, 0.000, 0.034, 0.032)
    d += 0.0050 * gauss(ax, z, EYE_X + 0.006, -0.040, 0.030, 0.0085)
    under = np.exp(-(((z - (-0.031 - 3.0 * (ax - EYE_X) ** 2)) / 0.0035) ** 2))
    d -= 0.0030 * under * gauss(ax, z, EYE_X, -0.031, 0.040, 0.02)
    # Brow ridge and forehead.
    d += 0.008 * gauss(ax, z, 0.062, 0.050, 0.044, 0.014)
    d += 0.006 * gauss(x, z, 0.0, 0.092, 0.085, 0.040)
    # Lips: thin, closed and smiling.
    upper = np.exp(-(((z - (mouth_line(x) + 0.0060)) / 0.0058) ** 2)) * np.exp(-((x / 0.054) ** 4))
    lower = np.exp(-(((z - (mouth_line(x) - 0.0075)) / 0.0068) ** 2)) * np.exp(-((x / 0.042) ** 4))
    d += 0.0016 * upper + 0.0030 * lower
    crease = np.exp(-(((z - mouth_line(x)) / 0.0030) ** 2)) * np.exp(-((x / 0.058) ** 6))
    d -= 0.0038 * crease
    d -= 0.0032 * gauss(ax, z, 0.058, -0.1012, 0.0060, 0.0080)
    # Chin ball, with a soft dip above it.
    d += 0.0060 * gauss(x, z, 0.0, -0.152, 0.036, 0.018)
    d -= 0.0022 * gauss(x, z, 0.0, -0.132, 0.034, 0.007)
    return d * front


def shape_head(x, y, z):
    """Base skull: rounder and wide through the jaw, tapering softly at the chin."""
    jaw = np.clip((-0.110 - z) / 0.065, 0, 1) ** 1.5
    lower = np.clip((-0.020 - z) / 0.070, 0, 1)
    forward = np.where(y < 0, 1 + 0.035 * lower, 1.0)
    return x * (1 - 0.10 * jaw), y * forward * (1 - 0.07 * jaw)


def build_head_rings(rows, cols):
    rings = L.ellipsoid_rings((0, 0, 0), HEAD_R, rows, cols)
    x, y, z = rings[..., 0], rings[..., 1], rings[..., 2]
    x, y = shape_head(x, y, z)
    # A fuller chin: the front of the jaw drops towards the measured chin height.
    z = z - 0.024 * np.clip((-y - 0.04) / 0.06, 0, 1) * np.clip((-0.085 - z) / 0.05, 0, 1)
    if FULL:
        front = np.clip((-y - 0.035) / 0.05, 0, 1)
        d = face_offset(x, z, front)
        # Push mostly forward, so a feature stays over its measured x and z position.
        length = np.sqrt(x * x + y * y + z * z)
        x = x + d * (x / np.maximum(length, 1e-6)) * 0.45
        y = y - d * 0.9
    return np.stack([x + HEAD_C[0], y + HEAD_C[1], z + HEAD_C[2]], axis=-1)


def head_surface_y(x, z):
    """Find the y of the sculpted front surface at head-local x, z."""
    u = np.clip(x / HEAD_R[0], -0.98, 0.98)
    w = np.clip(z / HEAD_R[2], -0.98, 0.98)
    y = -HEAD_R[1] * math.sqrt(max(1 - u * u - w * w, 0.01))
    d = face_offset(np.array([x]), np.array([z]), np.array([1.0]))[0]
    return y - d * 0.9


# ------------------------------------------------------------------ textures
def skin_texture():
    """Paint the skin in the head's spherical UV layout."""
    h, w = 1024, 2048
    v, u = np.mgrid[0:h, 0:w]
    polar = 0.02 + (v / (h - 1)) * (math.pi - 0.04)
    theta = (u / (w - 1)) * 2 * math.pi
    x = HEAD_R[0] * np.sin(polar) * np.sin(theta)
    y = -HEAD_R[1] * np.sin(polar) * np.cos(theta)
    z = HEAD_R[2] * np.cos(polar)
    x, y = shape_head(x, y, z)
    front = np.clip((-y - 0.03) / 0.05, 0, 1)
    ax = np.abs(x)

    base = np.array([0.955, 0.775, 0.655])
    rgb = np.ones((h, w, 3)) * base
    mottling = rng.normal(0, 1, (h // 8, w // 8))
    mottling = L.resize_nearest(mottling[..., None], h, w)[..., 0]
    rgb += (mottling * 0.010)[..., None]

    def blend(mask, colour):
        nonlocal rgb
        rgb = rgb * (1 - mask[..., None]) + np.array(colour) * mask[..., None]

    # Warm flush on the cheeks, nose and chin; cooler shadow round the eyes.
    blend(0.50 * gauss(ax, z, 0.090, -0.070, 0.042, 0.038) * front, (0.925, 0.530, 0.455))
    blend(0.50 * gauss(x, z, 0.0, -0.056, 0.030, 0.024) * front, (0.930, 0.570, 0.490))
    blend(0.26 * gauss(x, z, 0.0, -0.152, 0.052, 0.026) * front, (0.930, 0.640, 0.540))
    blend(0.34 * gauss(ax, z, EYE_X, -0.004, 0.044, 0.034) * front, (0.820, 0.610, 0.540))
    # Deep nostril shadow and the shadow beneath the nose.
    blend(0.52 * gauss(ax, z, 0.017, -0.079, 0.0060, 0.0036) * front, (0.470, 0.220, 0.180))
    blend(0.24 * gauss(x, z, 0.0, -0.085, 0.024, 0.006) * front, (0.750, 0.480, 0.410))

    bump = np.zeros((h, w))
    lines = np.zeros((h, w))

    def line(zc_fn, width, xmin, xmax, depth):
        nonlocal lines
        mask = np.exp(-(((z - zc_fn(x)) / width) ** 2))
        mask = mask * np.clip((ax - xmin) / 0.008, 0, 1) * np.clip((xmax - ax) / 0.016, 0, 1)
        lines = np.maximum(lines, mask * depth * front)

    # Forehead lines: fine, shallow and slightly uneven.
    for i, zc in enumerate((0.086, 0.098, 0.110, 0.122)):
        line(lambda xx, c=zc, n=i: c + 0.0035 * np.sin(xx * 38 + n * 1.7), 0.0015, 0.0, 0.082 - i * 0.006, 0.55)
    # Crow's feet fanning from the outer corner of the eye.
    for angle, start in ((-0.62, 0.104), (-0.28, 0.102), (0.05, 0.104), (0.34, 0.106)):
        line(lambda xx, a=angle: -0.006 + a * (np.abs(xx) - 0.100), 0.0014, start, 0.134, 0.62)
    # Stacked arcs under the eye.
    for depth_z, curve, strength in ((-0.031, 3.0, 0.72), (-0.042, 2.2, 0.55), (-0.052, 1.6, 0.34)):
        line(lambda xx, c=depth_z, k=curve: c - k * (np.abs(xx) - EYE_X) ** 2, 0.0016, 0.030, 0.108, strength)
    # Smile fold, then the fine lines outside it and round the mouth.
    line(lambda xx: -0.060 - 0.95 * (np.abs(xx) - 0.038), 0.0038, 0.030, 0.068, 0.60)
    for offset, strength in ((0.012, 0.42), (0.022, 0.32), (0.032, 0.24)):
        line(lambda xx, o=offset: -0.058 - 1.05 * (np.abs(xx) - 0.046 - o), 0.0015, 0.050 + offset, 0.096 + offset, strength)
    # Lower-cheek creases and jowl lines.
    for zc, slope, strength in ((-0.128, 0.75, 0.42), (-0.138, 0.82, 0.36), (-0.148, 0.90, 0.30)):
        line(lambda xx, c=zc, s=slope: c + s * (np.abs(xx) - 0.066), 0.0016, 0.056, 0.104, strength)

    bump -= lines
    blend(lines * 0.42, (0.700, 0.440, 0.370))

    # Soft rose lips, with a clear dark line where they meet.
    upper = np.exp(-(((z - (mouth_line(x) + 0.0058)) / 0.0052) ** 2)) * np.exp(-((x / 0.054) ** 4))
    lower = np.exp(-(((z - (mouth_line(x) - 0.0072)) / 0.0072) ** 2)) * np.exp(-((x / 0.042) ** 4))
    blend(np.clip(upper * 0.48 * front, 0, 1), (0.780, 0.440, 0.400))
    blend(np.clip(lower * 0.52 * front, 0, 1), (0.830, 0.490, 0.450))
    crease = np.exp(-(((z - mouth_line(x)) / 0.0020) ** 2)) * np.exp(-((x / 0.060) ** 6)) * front
    blend(crease * 0.92, (0.340, 0.120, 0.110))
    bump -= crease

    pores = rng.normal(0, 1, (h, w)) * 0.010
    bump = bump + pores + 0.5
    # Row 0 is the top pole, matching v = 0 at the first ring of the head mesh.
    color_img = L.array_to_image("joyce_skin", np.clip(rgb, 0, 1), TEX_DIR)
    bump_img = L.array_to_image("joyce_skin_bump", np.dstack([bump, bump, bump]), TEX_DIR)
    return color_img, bump_img


def knit_texture():
    """Weave the Fair Isle pattern from the reference, stitch by stitch."""
    cream = (0.855, 0.775, 0.665)
    wheat = (0.790, 0.620, 0.450)
    maroon = (0.470, 0.120, 0.165)
    red = (0.610, 0.190, 0.175)
    rust = (0.650, 0.300, 0.170)
    teal = (0.180, 0.420, 0.395)
    plum = (0.340, 0.130, 0.190)
    sage = (0.480, 0.580, 0.500)

    def motif(kind, r, c, rows):
        mid = rows // 2
        if kind == "zig":
            return abs((c % 6) - 3) == abs(r - mid) % 4
        if kind == "dia":
            return abs((c % 12) - 6) + abs(r - mid) == 3 or (c % 12 == 6 and r == mid)
        if kind == "dot":
            return (c + r) % 2 == 0
        if kind == "tooth":
            return (c % 4) < (r % 4) + 1
        if kind == "x":
            return abs((c % 8) - 4) == abs(r - mid)
        return False

    bands = [
        (5, maroon, cream, "x"), (2, rust, None, None), (5, rust, cream, "zig"),
        (2, teal, None, None), (4, teal, wheat, "tooth"), (2, red, None, None),
        (5, red, cream, "zig"), (1, plum, None, None), (8, teal, cream, "dia"),
        (2, rust, None, None), (3, wheat, rust, "dot"), (2, maroon, None, None),
        (6, plum, red, "dia"), (2, sage, None, None), (5, rust, cream, "tooth"),
        (2, teal, None, None), (6, teal, wheat, "x"), (2, maroon, None, None),
        (4, wheat, maroon, "zig"), (2, red, None, None),
    ]
    stitch_rows, stitch_cols = 128, 128
    stitches = np.zeros((stitch_rows, stitch_cols, 3))
    row = 0
    while row < stitch_rows:
        for count, bg, fg, kind in bands:
            for r in range(count):
                if row >= stitch_rows:
                    break
                for c in range(stitch_cols):
                    use_fg = fg is not None and motif(kind, r, c, count)
                    stitches[row, c] = fg if use_fg else bg
                row += 1

    cell = 16
    size = stitch_rows * cell
    y, x = np.mgrid[0:size, 0:size]
    fx = (x % cell) / cell - 0.5
    fy = (y % cell) / cell - 0.5
    # A rounded V for each stitch reads as knitted yarn at this scale.
    v_shape = np.abs(fx) * 1.1 + fy * 0.55
    yarn = np.cos(np.clip(v_shape * 2.8, -math.pi, math.pi)) * 0.5 + 0.5
    height = yarn * (1 - (np.abs(fx) * 2) ** 4 * 0.7)
    fuzz = rng.normal(0, 1, (size, size)) * 0.045

    colours = stitches[y // cell, x // cell]
    # Offset each stitch's tone slightly so colours do not look printed.
    tone = rng.normal(0, 0.035, (stitch_rows, stitch_cols))[y // cell, x // cell]
    shade = 0.70 + 0.42 * height + fuzz + tone
    rgb = np.clip(colours * shade[..., None], 0, 1)
    bump = np.clip(height + fuzz * 0.6, 0, 1)
    return (
        L.array_to_image("joyce_knit", rgb, TEX_DIR),
        L.array_to_image("joyce_knit_bump", np.dstack([bump, bump, bump]), TEX_DIR),
    )


def rib_bump():
    size = 512
    _, x = np.mgrid[0:size, 0:size]
    rib = np.cos((x % 32) / 32 * 2 * math.pi) * 0.5 + 0.5
    return L.array_to_image("joyce_rib_bump", np.dstack([rib, rib, rib]), TEX_DIR)


def iris_texture():
    """Measured in the reference: iris radius 0.47 and pupil 0.27 of the eyeball radius."""
    size = 1024
    y, x = np.mgrid[0:size, 0:size]
    dx = (x - size / 2) / (size / 2)
    dy = (y - size / 2) / (size / 2)
    r = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)
    rgb = np.ones((size, size, 3)) * np.array((0.975, 0.955, 0.935))
    # A little warmth and shadow towards the corners of the eye.
    rgb = rgb * (1 - 0.14 * np.clip((r - 0.62) / 0.38, 0, 1))[..., None]
    rays = 0.5 + 0.5 * np.sin(angle * 58 + 1.3) * np.sin(angle * 23 + 0.4)
    outer = np.array((0.430, 0.250, 0.090))
    inner = np.array((0.780, 0.540, 0.210))
    mix = np.clip((r - 0.27) / 0.20, 0, 1)[..., None]
    iris = (inner * (1 - mix) + outer * mix) * (0.72 + 0.42 * rays)[..., None]
    edge = np.clip((0.47 - r) / 0.012, 0, 1)[..., None]
    rgb = rgb * (1 - edge) + iris * edge
    ring = np.exp(-(((r - 0.462) / 0.026) ** 2))[..., None]
    rgb = rgb * (1 - ring * 0.90) + np.array((0.10, 0.055, 0.025)) * ring * 0.90
    pupil = np.clip((0.27 - r) / 0.010, 0, 1)[..., None]
    rgb = rgb * (1 - pupil) + np.array((0.012, 0.010, 0.010)) * pupil
    # Painted catchlight: a stylized highlight that also survives export to GLB.
    glint = np.exp(-(((dx - 0.13) ** 2 + (dy - 0.19) ** 2) / (0.075**2)) ** 2)[..., None]
    rgb = rgb * (1 - glint) + np.array((1.0, 1.0, 1.0)) * glint
    return L.array_to_image("joyce_eye", rgb, TEX_DIR)


# ------------------------------------------------------------------ materials
def make_materials():
    m = {}
    m["skin"] = L.new_material("skin", L.srgb("#EFC1A2"), 0.48)
    m["skin_plain"] = L.new_material("skin_plain", L.srgb("#EDBFA3"), 0.50)
    m["hair"] = L.new_material("hair", L.srgb("#C4C4C8"), 0.46)
    m["hair_inner"] = L.new_material("hair_inner", L.srgb("#8F8F94"), 0.8)
    m["lash"] = L.new_material("lash", L.srgb("#241A16"), 0.55)
    m["crease"] = L.new_material("crease", L.srgb("#B97E6A"), 0.65)
    m["knit"] = L.new_material("knit", L.srgb("#8A4A3C"), 0.92)
    m["band"] = L.new_material("band", L.srgb("#7B2330"), 0.88)
    m["blouse"] = L.new_material("blouse", L.srgb("#E9E0D2"), 0.72)
    m["pants"] = L.new_material("pants", L.srgb("#1B2238"), 0.78)
    m["shoe"] = L.new_material("shoe", L.srgb("#242A3E"), 0.55)
    m["sole"] = L.new_material("sole", L.srgb("#6A5D52"), 0.7)
    m["eye"] = L.new_material("eye", L.srgb("#F2EDE6"), 0.08)
    m["brow"] = L.new_material("brow", L.srgb("#B2AEA8"), 0.8)
    m["button"] = L.new_material("button", L.srgb("#D8CDB8"), 0.35)
    if not FULL:
        return m

    skin_color, skin_bump = skin_texture()
    L.attach_color_image(m["skin"], skin_color)
    L.attach_bump_image(m["skin"], skin_bump, 0.55, 0.0035)
    bsdf = m["skin"].node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Subsurface Weight"].default_value = 0.35
    bsdf.inputs["Subsurface Radius"].default_value = (0.9, 0.35, 0.2)
    bsdf.inputs["Subsurface Scale"].default_value = 0.012

    plain = m["skin_plain"].node_tree.nodes["Principled BSDF"]
    plain.inputs["Subsurface Weight"].default_value = 0.35
    plain.inputs["Subsurface Radius"].default_value = (0.9, 0.35, 0.2)
    plain.inputs["Subsurface Scale"].default_value = 0.012

    knit_color, knit_bump = knit_texture()
    L.attach_color_image(m["knit"], knit_color)
    L.attach_bump_image(m["knit"], knit_bump, 0.8, 0.004)
    m["knit"].node_tree.nodes["Principled BSDF"].inputs["Sheen Weight"].default_value = 0.2

    rib = rib_bump()
    L.attach_bump_image(m["band"], rib, 0.8, 0.004)
    m["band"].node_tree.nodes["Principled BSDF"].inputs["Sheen Weight"].default_value = 0.1

    hair_bsdf = m["hair"].node_tree.nodes["Principled BSDF"]
    hair_bsdf.inputs["Roughness"].default_value = 0.48
    hair_bsdf.inputs["Sheen Weight"].default_value = 0.15

    L.attach_color_image(m["eye"], iris_texture())
    m["pants"].node_tree.nodes["Principled BSDF"].inputs["Sheen Weight"].default_value = 0.05
    return m


# ------------------------------------------------------------------ body parts
parts = {}


def register(key, obj):
    parts[key] = obj
    return obj


def build_head(m):
    rows, cols = (160, 240) if FULL else (28, 40)
    rings = build_head_rings(rows, cols)
    register("head", L.grid_object("head", rings, m["skin"], True, True, True))
    neck = [(0, 0.012, 1.03), (0, 0.010, 1.08), (0, 0.006, 1.13)]
    register("neck", L.tube("neck", neck, [(0.062, 0.060)] * 3, m["skin_plain"], 10, 24))

    # Ears, measured from the reference: flat ovals with a bowl and a raised outer rim.
    for sign, label in ((1, "L"), (-1, "R")):
        center = np.array((sign * 0.150, 0.038, 1.181))
        ear = L.ellipsoid_rings(center, (0.011, 0.028, 0.037), 30 if FULL else 12, 36 if FULL else 14)
        if FULL:
            local_y = ear[..., 1] - center[1]
            local_z = ear[..., 2] - center[2]
            bowl = np.exp(-((local_y / 0.016) ** 2 + (local_z / 0.024) ** 2))
            outward = np.clip((ear[..., 0] - center[0]) * sign / 0.011, 0, 1)
            ear[..., 0] -= sign * bowl * outward * 0.011
        register(f"ear_{label}", L.grid_object(f"ear_{label}", ear, m["skin_plain"], True, True, True))
        if FULL:
            rim = []
            for a in np.radians(np.linspace(-40, 235, 16)):
                rim.append((center[0] + sign * 0.006, center[1] + 0.024 * math.cos(a), center[2] + 0.033 * math.sin(a)))
            register(
                f"ear_rim_{label}",
                L.tube(f"ear_rim_{label}", rim, [(0.0045, 0.0045)] * 16, m["skin_plain"], 40, 10, side=(1, 0, 0)),
            )


def lid_edge(center, radius, axis_deg, spread_deg, keep=0.08, samples=240):
    """Trace a lid's cut edge, keeping the part that faces forward, in x order."""
    tilt = math.radians(axis_deg)
    axis = np.array((0.0, -math.cos(tilt), math.sin(tilt)))
    side_a = np.array((1.0, 0.0, 0.0))
    side_b = np.cross(axis, side_a)
    spread = math.radians(spread_deg)
    points = []
    for theta in np.linspace(0, 2 * math.pi, samples, endpoint=False):
        direction = math.cos(spread) * axis + math.sin(spread) * (math.cos(theta) * side_a + math.sin(theta) * side_b)
        if direction[1] < -keep:
            points.append(direction)
    points = np.array(points)
    points = points[np.argsort(points[:, 0])]
    return center + points * radius


def build_eyes(m):
    eye_r = EYE_R
    for sign, label in ((1, "L"), (-1, "R")):
        center = np.array((sign * EYE_X, EYE_Y, EYE_Z))
        rows, cols = (48, 64) if FULL else (10, 14)
        # A pole-at-the-front sphere makes the iris texture simple and clean.
        polar = np.linspace(0.02, math.pi - 0.02, rows)
        theta = L.ring_angles(cols)
        rings = np.zeros((rows, cols, 3))
        for i, p in enumerate(polar):
            rings[i, :, 0] = eye_r * np.sin(p) * np.cos(theta)
            rings[i, :, 2] = eye_r * np.sin(p) * np.sin(theta)
            rings[i, :, 1] = -eye_r * np.cos(p)
        rings = rings + center
        obj = L.grid_object(f"eye_{label}", rings, m["eye"], True, True, True)
        # Planar UVs through the front of the eye.
        mesh = obj.data
        uv = []
        for poly in mesh.polygons:
            for vid in poly.vertices:
                co = np.array(mesh.vertices[vid].co) - center
                uv.append((0.5 + co[0] / (2 * eye_r), 0.5 + co[2] / (2 * eye_r)))
        mesh.uv_layers.active.uv.foreach_set("vector", np.array(uv, dtype=np.float32).ravel())
        register(f"eye_{label}", obj)

        if not FULL:
            continue
        # Lids are sphere caps cut by tilted planes. These angles put the opening where
        # the reference has it: 42 degrees up, 35 down, closing about 75 either side.
        lids = (("upper", 147.0, 109.0), ("lower", -131.0, 96.0))
        for lid, axis_angle, spread in lids:
            rows2, cols2 = 44, 64
            rad = eye_r + 0.0012
            lid_rings = np.zeros((rows2, cols2, 3))
            tilt = math.radians(axis_angle)
            for i, a in enumerate(np.radians(np.linspace(0.5, spread, rows2))):
                ring_theta = L.ring_angles(cols2)
                lx = rad * math.sin(a) * np.cos(ring_theta)
                lz = rad * math.sin(a) * np.sin(ring_theta)
                ly = -rad * math.cos(a) * np.ones(cols2)
                lid_rings[i, :, 0] = lx
                lid_rings[i, :, 1] = ly * math.cos(tilt) + lz * math.sin(tilt)
                lid_rings[i, :, 2] = -ly * math.sin(tilt) + lz * math.cos(tilt)
            lid_rings = lid_rings + center
            register(
                f"lid_{lid}_{label}",
                L.grid_object(f"lid_{lid}_{label}", lid_rings, m["skin_plain"], True, True, False),
            )

        # A dark lash line defines the eye, thickest towards the outer corner.
        upper_edge = lid_edge(center, eye_r + 0.0018, 147.0, 109.0)
        upper_edge = upper_edge[:: max(len(upper_edge) // 12, 1)]
        count = len(upper_edge)
        lash_radii = []
        for i in range(count):
            t = i / (count - 1)
            outer = t if sign > 0 else 1 - t
            lash_radii.append((0.0009 + 0.0022 * outer, 0.0009 + 0.0011 * outer))
        register(f"lash_{label}", L.tube(f"lash_{label}", upper_edge, lash_radii, m["lash"], 44, 10, side=(0, 0, 1)))
        # A fine, softer line for the lower lid.
        lower_edge = lid_edge(center, eye_r + 0.0015, -131.0, 96.0)
        lower_edge = lower_edge[:: max(len(lower_edge) // 10, 1)]
        register(
            f"lash_low_{label}",
            L.tube(f"lash_low_{label}", lower_edge, [(0.0007, 0.0006)] * len(lower_edge), m["crease"], 34, 8, side=(0, 0, 1)),
        )
        # The fold above the upper lid.
        fold = lid_edge(center, eye_r + 0.0030, 147.0, 96.0, keep=0.30)
        fold = fold[:: max(len(fold) // 9, 1)]
        count = len(fold)
        register(
            f"crease_{label}",
            L.tube(f"crease_{label}", fold,
                   [(0.0011 * math.sin((i + 0.5) / count * math.pi) + 0.0003, 0.0007) for i in range(count)],
                   m["crease"], 30, 8, side=(0, 0, 1)),
        )

        # Eyebrow: a soft tapered base with many small strands laid along it.
        def brow_at(t):
            z = 0.052 + 0.009 * math.sin(t * math.pi * 0.95) - 0.020 * t * t
            x = 0.022 + 0.094 * t
            return x, z

        points = []
        for t in np.linspace(0, 1, 9):
            bx, bz = brow_at(t)
            points.append((sign * bx, head_surface_y(bx, bz) - 0.0012, HEAD_C[2] + bz))
        radii = [(0.0016, 0.0048 * math.sin((i + 0.6) / 9.6 * math.pi) + 0.0012) for i in range(9)]
        register(f"brow_{label}", L.tube(f"brow_{label}", points, radii, m["brow"], 20, 8, side=(0, -1, 0)))
        strand_rings = []
        for _ in range(70):
            t = random.random()
            bx, bz = brow_at(t)
            bz += random.uniform(-0.0052, 0.0052) * math.sin((t + 0.06) / 1.12 * math.pi)
            angle = math.radians(random.uniform(8, 34) * (1 - t) + random.uniform(-22, 2) * t)
            length = random.uniform(0.006, 0.011)
            path = []
            for k in np.linspace(0, 1, 4):
                sx = bx + math.cos(angle) * length * k
                sz = bz + math.sin(angle) * length * k
                path.append((sign * sx, head_surface_y(sx, sz) - 0.0024, HEAD_C[2] + sz))
            strand_rings.append(L.tube_rings(
                path, [(0.0006, 0.0006), (0.0005, 0.0005), (0.0004, 0.0004), (0.00015, 0.00015)], 6, 4, side=(0, -1, 0)
            ))
        faces = []
        verts = []
        for rings_ in strand_rings:
            start = len(verts)
            verts.extend(rings_.reshape(-1, 3).tolist())
            faces.extend(tuple(v + start for v in f) for f in L.grid_faces(6, 4, True))
        register(f"brow_hair_{label}", L.make_object(f"brow_hair_{label}", verts, faces, None, m["brow"]))


def build_torso(m):
    # Blouse: a closed form under the open cardigan.
    blouse = [
        (0.70, 0.150, 0.112, 0.000), (0.80, 0.150, 0.112, 0.000), (0.90, 0.153, 0.116, 0.000),
        (0.97, 0.150, 0.105, 0.004), (1.02, 0.115, 0.088, 0.006), (1.06, 0.072, 0.070, 0.010),
        (1.075, 0.060, 0.062, 0.012),
    ]
    rows = 40 if FULL else 14
    cols = 64 if FULL else 24
    sections = L.catmull(blouse, rows)
    theta = L.ring_angles(cols)
    rings = np.zeros((rows, cols, 3))
    for i, (z, rx, ry, cy) in enumerate(sections):
        rings[i, :, 0] = rx * np.sin(theta)
        rings[i, :, 1] = cy - ry * np.cos(theta)
        rings[i, :, 2] = z
    register("blouse", L.grid_object("blouse", rings, m["blouse"], True, True, False))

    # Cardigan: an open shell whose gap follows the band down the front.
    cardigan = [
        (0.510, 0.188, 0.158, 0.000, 0.072),
        (0.600, 0.184, 0.154, 0.000, 0.068),
        (0.750, 0.176, 0.150, 0.000, 0.060),
        (0.880, 0.176, 0.142, 0.000, 0.050),
        (0.950, 0.166, 0.122, 0.006, 0.040),
        (1.005, 0.132, 0.100, 0.010, 0.034),
        (1.050, 0.090, 0.082, 0.014, 0.030),
        (1.080, 0.072, 0.072, 0.016, 0.028),
    ]
    rows = 72 if FULL else 18
    cols = 110 if FULL else 28
    sections = L.catmull(cardigan, rows)
    rings = np.zeros((rows, cols, 3))
    left_edge, right_edge = [], []
    for i, (z, rx, ry, cy, gap) in enumerate(sections):
        phi0 = math.asin(min(gap / rx, 0.95))
        phi = np.linspace(phi0, 2 * math.pi - phi0, cols)
        rings[i, :, 0] = rx * np.sin(phi)
        rings[i, :, 1] = cy - ry * np.cos(phi)
        rings[i, :, 2] = z
        left_edge.append(rings[i, 0].copy())
        right_edge.append(rings[i, -1].copy())
    # UVs are in real-world stitch scale so the pattern matches on all parts.
    uv_u = np.linspace(0, 1.08 * 1.18, cols)
    uv_v = (sections[:, 0] - 0.51) * 1.18
    register("cardigan", L.grid_object("cardigan", rings, m["knit"], False, uv_u=uv_u, uv_v=uv_v))

    # Ribbed front band and collar, one continuous strip like a real cardigan.
    collar_back = []
    last = sections[-1]
    for a in np.linspace(math.asin(last[4] / last[1]) + 0.25, 2 * math.pi - math.asin(last[4] / last[1]) - 0.25, 14):
        collar_back.append((last[1] * math.sin(a), last[3] - last[2] * math.cos(a), last[0] + 0.004))
    left_path = np.array(left_edge)[::6]
    right_path = np.array(right_edge)[::6]
    path = np.vstack([left_path, np.array(collar_back), right_path[::-1]])
    radii = [(0.024, 0.0095)] * len(path)
    rows2 = 170 if FULL else 50
    band_rings = L.tube_rings(path, radii, rows2, 14 if FULL else 8, side=(1, 0, 0), power=3.2)
    uv_u = np.linspace(0, 3.0, band_rings.shape[1] + 1)
    uv_v = np.linspace(0, 12.0, rows2)
    register("band", L.grid_object("band", band_rings, m["band"], True, True, True, uv_u, uv_v))

    # Hem ribbing all the way around the back, at the bottom of the cardigan.
    hem = sections[0]
    hem_rings = np.zeros((10, cols, 3))
    phi0 = math.asin(hem[4] / hem[1])
    phi = np.linspace(phi0, 2 * math.pi - phi0, cols)
    for i, (dz, grow) in enumerate(
        zip(np.linspace(-0.038, 0.012, 10), (0.002, 0.006, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.006, 0.0))
    ):
        hem_rings[i, :, 0] = (hem[1] + grow) * np.sin(phi)
        hem_rings[i, :, 1] = hem[3] - (hem[2] + grow) * np.cos(phi)
        hem_rings[i, :, 2] = hem[0] + dz
    register(
        "hem",
        L.grid_object("hem", hem_rings, m["band"], False, uv_u=np.linspace(0, 40, cols),
                      uv_v=np.linspace(0, 1, 10)),
    )

    if FULL:
        # Patch pockets, as in the reference.
        for sign, label in ((1, "L"), (-1, "R")):
            pocket_rows, pocket_cols = 12, 20
            pocket = np.zeros((pocket_rows, pocket_cols, 3))
            for i, z in enumerate(np.linspace(0.540, 0.635, pocket_rows)):
                t = (z - 0.51) / 0.09
                rx = 0.188 - t * 0.004 + 0.005
                ry = 0.158 - t * 0.004 + 0.005
                phi_values = np.linspace(0.52, 1.18, pocket_cols)
                pocket[i, :, 0] = sign * rx * np.sin(phi_values)
                pocket[i, :, 1] = -ry * np.cos(phi_values)
                pocket[i, :, 2] = z
            register(
                f"pocket_{label}",
                L.grid_object(f"pocket_{label}", pocket, m["knit"], False,
                              uv_u=np.linspace(0.2, 0.32, pocket_cols),
                              uv_v=np.linspace(0.02, 0.10, pocket_rows)),
            )
            trim_path = pocket[-1]
            register(
                f"pocket_trim_{label}",
                L.tube(f"pocket_trim_{label}", trim_path[::3], [(0.012, 0.005)] * len(trim_path[::3]),
                       m["band"], 22, 8, side=(0, 0, 1), power=3),
            )

        # Blouse details: placket, buttons and a soft collar.
        for z in (0.790, 0.860, 0.930):
            ry = np.interp(z, [0.80, 0.90, 0.97, 1.02, 1.06], [0.112, 0.116, 0.105, 0.088, 0.070])
            cy = np.interp(z, [0.80, 0.97, 1.02], [0.0, 0.004, 0.006])
            L_btn = L.ellipsoid(f"button_{z}", (0, cy - ry - 0.002, z), (0.0082, 0.0032, 0.0082),
                                m["button"], 10, 16)
            register(f"button_{z}", L_btn)
        placket = [(0.0, -0.1135, 0.72), (0.0, -0.1175, 0.85), (0.0, -0.1105, 0.96), (0.0, -0.1015, 1.00)]
        register("placket", L.tube("placket", placket, [(0.0145, 0.0022)] * 4, m["blouse"], 24, 8, power=3))
        v_path = [(0.0, -0.0705, 1.078), (0.0, -0.0905, 1.040), (0.0, -0.1005, 0.998)]
        register(
            "chest_v",
            L.tube("chest_v", v_path, [(0.052, 0.003), (0.031, 0.003), (0.004, 0.002)], m["skin_plain"], 16, 16, power=3),
        )
        for sign, label in ((1, "L"), (-1, "R")):
            collar = [
                (sign * 0.006, -0.101, 0.998), (sign * 0.030, -0.092, 1.038),
                (sign * 0.054, -0.070, 1.068), (sign * 0.068, -0.030, 1.082),
            ]
            register(
                f"collar_{label}",
                L.tube(f"collar_{label}", collar, [(0.020, 0.0045), (0.024, 0.0045), (0.019, 0.0045), (0.016, 0.0045)],
                       m["blouse"], 20, 12, side=(0, 0, 1), power=2.5),
            )


def finger_path(base, direction, length, curl):
    direction = L.normalized(np.array(direction, dtype=float))
    down = np.array((0.0, -0.20, -0.98))
    points = [base]
    current = np.array(base, dtype=float)
    heading = direction.copy()
    for _ in range(3):
        current = current + heading * length / 3
        points.append(current.copy())
        heading = L.normalized(heading + curl * np.array((0.0, -1.0, -0.2)))
    return points, down


def build_arm(m, sign, label):
    S = SHOULDER * (sign, 1, 1)
    E = ELBOW * (sign, 1, 1)
    W = WRIST * (sign, 1, 1)
    upper = S + np.array((-sign * 0.050, 0.0, 0.040))
    path = [upper, S, (S + E) / 2 + np.array((sign * 0.002, 0, 0)), E, (E + W) / 2, W + (W - E) * 0.02]
    radii = [(0.030, 0.040), (0.063, 0.070), (0.067, 0.073), (0.064, 0.068), (0.060, 0.064), (0.056, 0.059)]
    rows = 54 if FULL else 14
    cols = 44 if FULL else 18
    rings = L.tube_rings(path, radii, rows, cols, side=(1, 0, 0))
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(rings.mean(axis=1), axis=0), axis=1))])
    uv_v = (arc.max() - arc) * 1.18 + 0.03
    uv_u = np.linspace(0, 0.46 * 1.18, cols + 1)
    register(f"sleeve_{label}", L.grid_object(f"sleeve_{label}", rings, m["knit"], True, False, False, uv_u, uv_v))

    # Ribbed cuff with a folded, slightly fatter edge.
    direction = L.normalized(W - E)
    cuff_path = [W - direction * 0.018, W + direction * 0.015, W + direction * 0.040]
    cuff = L.tube_rings(cuff_path, [(0.060, 0.063), (0.062, 0.065), (0.057, 0.059)], 10, cols, side=(1, 0, 0))
    register(
        f"cuff_{label}",
        L.grid_object(f"cuff_{label}", cuff, m["band"], True, False, True,
                      uv_u=np.linspace(0, 14, cols + 1), uv_v=np.linspace(0, 1, 10)),
    )

    # Hand: a chubby palm with short, soft fingers.
    wrist_center = W + direction * 0.035
    palm_center = wrist_center + direction * 0.040
    palm_path = [wrist_center - direction * 0.012, wrist_center + direction * 0.006, palm_center, palm_center + direction * 0.032]
    palm_rad = [(0.040, 0.032), (0.045, 0.033), (0.050, 0.030), (0.048, 0.026)]
    register(
        f"palm_{label}",
        L.tube(f"palm_{label}", palm_path, palm_rad, m["skin_plain"], 18 if FULL else 8, 22 if FULL else 10, side=(0, 1, 0)),
    )
    if not FULL:
        return
    # Local hand axes: `across` runs from little finger to thumb side (front of body).
    across = np.array((0.0, -1.0, 0.0))
    knuckles = palm_center + direction * 0.028
    for i, (offset, length, radius) in enumerate(
        ((-0.033, 0.030, 0.0125), (-0.011, 0.038, 0.0135), (0.011, 0.040, 0.0138))
    ):
        base = knuckles + across * (-offset)
        pts, _ = finger_path(base, direction, length, 0.13)
        pts = [p + np.array((0, 0, 0)) for p in pts]
        register(
            f"finger{i}_{label}",
            L.tube(f"finger{i}_{label}", pts, [(radius, radius * 0.92), (radius, radius * 0.92), (radius * 0.90, radius * 0.84), (radius * 0.50, radius * 0.46)], m["skin_plain"], 14, 12),
        )
    index_base = knuckles + across * 0.033
    pts, _ = finger_path(index_base, direction, 0.036, 0.13)
    register(f"finger3_{label}", L.tube(f"finger3_{label}", pts, [(0.0132, 0.0122), (0.0132, 0.0122), (0.0118, 0.0110), (0.0066, 0.0061)], m["skin_plain"], 14, 12))
    thumb_base = wrist_center + direction * 0.018 + across * 0.038
    thumb_dir = L.normalized(direction * 0.75 + across * 0.62 + np.array((-sign * 0.20, 0, 0)))
    pts, _ = finger_path(thumb_base, thumb_dir, 0.044, 0.05)
    register(f"thumb_{label}", L.tube(f"thumb_{label}", pts, [(0.0165, 0.0148), (0.0150, 0.0140), (0.0132, 0.0124), (0.0072, 0.0068)], m["skin_plain"], 14, 12))


def build_lower(m):
    hips = [
        (0.56, 0.158, 0.118), (0.62, 0.170, 0.128), (0.68, 0.170, 0.130),
        (0.74, 0.160, 0.122), (0.785, 0.152, 0.114),
    ]
    rows = 22 if FULL else 10
    cols = 48 if FULL else 20
    sections = L.catmull(hips, rows)
    theta = L.ring_angles(cols)
    rings = np.zeros((rows, cols, 3))
    for i, (z, rx, ry) in enumerate(sections):
        rings[i, :, 0] = rx * np.sin(theta)
        rings[i, :, 1] = -ry * np.cos(theta)
        rings[i, :, 2] = z
    register("hips", L.grid_object("hips", rings, m["pants"], True, True, True))

    for sign, label in ((1, "L"), (-1, "R")):
        H = HIP * (sign, 1, 1)
        K = KNEE * (sign, 1, 1)
        A = ANKLE * (sign, 1, 1)
        path = [H + np.array((0, 0, 0.05)), H, (H + K) / 2, K, (K + A) / 2, A, A + np.array((0, 0, -0.030))]
        radii = [(0.088, 0.104), (0.092, 0.106), (0.090, 0.104), (0.082, 0.090), (0.079, 0.086), (0.078, 0.085), (0.079, 0.086)]
        register(
            f"leg_{label}",
            L.tube(f"leg_{label}", path, radii, m["pants"], 44 if FULL else 14, 36 if FULL else 16, cap_start=False),
        )
        build_shoe(m, sign, label, A)


def build_shoe(m, sign, label, ankle):
    # A soft ballet flat, lofted heel to toe. The opening stays under the trouser cuff.
    length = 0.215
    heel_y = ankle[1] + 0.065
    x0 = ankle[0] + sign * 0.006
    splay = sign * 0.028
    path = []
    radii = []
    for t, width, height in (
        (0.00, 0.020, 0.018), (0.08, 0.042, 0.040), (0.30, 0.050, 0.052), (0.56, 0.055, 0.043),
        (0.78, 0.054, 0.032), (0.93, 0.040, 0.024), (1.00, 0.016, 0.012),
    ):
        path.append((x0 + splay * t, heel_y - length * t, height + 0.012))
        radii.append((width, height))
    rows = 40 if FULL else 12
    cols = 32 if FULL else 14
    rings = L.tube_rings(path, radii, rows, cols, side=(1, 0, 0))
    rings[..., 2] = np.maximum(rings[..., 2], 0.012) - 0.012  # flat underside, on the floor
    register(f"shoe_{label}", L.grid_object(f"shoe_{label}", rings, m["shoe"], True, True, True))


# ------------------------------------------------------------------ hair
HAIR_C = np.array((0.0, 0.004, 1.285))
HAIR_R = np.array((0.200, 0.205, 0.200))


def polar_limit(azimuth):
    """Largest polar angle hair can reach before covering her face or going too low."""
    a = abs(math.degrees(azimuth)) % 360
    a = min(a, 360 - a)
    if a < 25:
        return math.radians(46)
    if a < 70:
        return math.radians(46 + (a - 25) / 45 * 40)
    if a < 110:
        return math.radians(86 + (a - 70) / 40 * 46)
    return math.radians(134)


def envelope(azimuth, polar):
    """The hair's overall mass: full and low at the back, tucked in at its edge."""
    back = 0.5 - 0.5 * math.cos(azimuth)
    fullness = 0.20 * math.exp(-(((polar - math.radians(102)) / math.radians(32)) ** 2)) * (0.42 + 0.58 * back)
    tuck = 22.0 if abs(math.degrees(azimuth)) < 80 else 9.0
    edge = min(max((polar_limit(azimuth) - polar) / math.radians(tuck), 0.0), 1.0)
    edge = edge * edge * (3 - 2 * edge)
    inset = 0.60 if abs(math.degrees(azimuth)) < 80 else 0.72
    return (1.0 + fullness) * (inset + (1.0 - inset) * edge)


def hair_point(azimuth, polar, radial=1.0):
    """azimuth 0 is the face direction and positive runs to her left (+X)."""
    radial = radial * envelope(azimuth, polar)
    low = 1.0 + 0.24 * max(-math.cos(polar), 0.0)
    return HAIR_C + radial * HAIR_R * np.array(
        (math.sin(polar) * math.sin(azimuth), -math.sin(polar) * math.cos(azimuth), math.cos(polar) * low)
    )


LIMIT_TABLE_AZ = np.radians(np.linspace(-360, 360, 1441))
LIMIT_TABLE = np.array([polar_limit(a) for a in LIMIT_TABLE_AZ])


def limit_array(azimuth):
    return np.interp(azimuth, LIMIT_TABLE_AZ, LIMIT_TABLE)


def paint_hair_flow():
    """Paint every strand into az/polar maps, using flow paths traced from the references.

    Each strand writes a tone and a height. Neighbouring strands share a wave phase, so
    their heights add up to continuous ridges across the flow, which become real shape.
    Later strands overwrite earlier ones, the layering an artist paints by hand.
    """
    h, w = 1536, 3072
    tone = np.full((h, w), 0.70)
    height = np.zeros((h, w))
    samples = 170
    t = np.linspace(0, 1, samples)
    rad = math.radians

    def R(a, b):
        return random.uniform(a, b)

    def stamp(az, polar, values, heights):
        polar = np.minimum(polar, limit_array(az))
        cols = ((az + math.pi) / (2 * math.pi) * w).astype(int) % w
        rows = np.clip((polar / math.pi * h).astype(int), 0, h - 1)
        for dr, dc in ((0, 0), (1, 0), (0, 1), (1, 1)):
            r = np.clip(rows + dr, 0, h - 1)
            c = (cols + dc) % w
            tone[r, c] = values
            height[r, c] = heights

    def shade(s, seed):
        # Strands in a group share a tone, so highlights gather into soft bands.
        group = 0.045 * math.sin(s * 47 + seed) + 0.060 * math.sin(s * 131 + seed * 2.1) + 0.050 * math.sin(s * 317 + seed * 0.7)
        return float(np.clip(0.76 + group + random.gauss(0, 0.055), 0.40, 1.0))

    def family(path_a, path_b, count, waves, tangential, radial, phase, phase_step, seed):
        path_a = np.array(path_a, dtype=float)
        path_b = np.array(path_b, dtype=float)
        knots = np.linspace(0, 1, len(path_a))
        for _ in range(count):
            s = random.random()
            mixed = path_a * (1 - s) + path_b * s
            az = np.interp(t, knots, mixed[:, 0]) + R(-0.6, 0.6)
            polar = np.interp(t, knots, mixed[:, 1])
            wave = np.sin(2 * math.pi * waves * t + phase + s * phase_step + R(-0.04, 0.04))
            fade = np.clip(t * 3.0, 0, 1)
            polar = polar + tangential * wave * fade
            stamp(np.radians(az), np.radians(polar), shade(s, seed), radial * wave * fade)

    # 2) The crown: a soft whorl turning out from the top of the head, so no bare patch shows
    #    where the sweeps below meet. It has its own random numbers, so adding it leaves every
    #    other strand of the hair exactly where it was.
    crown = random.Random(77)
    for _ in range(9000):
        start = math.radians(crown.uniform(-180, 180))
        reach = crown.uniform(14, 30)
        turn = math.radians(crown.uniform(55, 85))
        az = start + turn * t
        az = (az + math.pi) % (2 * math.pi) - math.pi
        polar = np.radians(0.2 + reach * t)
        value = float(np.clip(0.76 + 0.07 * math.sin(start * 9) + crown.gauss(0, 0.055), 0.40, 1.0))
        stamp(az, polar, value, 0.0)

    # 3) Lower layers on each side, from the temple, sweeping back around the ear.
    family([(68, 84), (86, 100), (108, 113), (136, 122)], [(58, 72), (82, 84), (112, 94), (142, 104)],
           4200, 1.15, 3.0, 0.042, 1.0, 0.7, 1.0)
    family([(-66, 86), (-88, 104), (-112, 118), (-142, 126)], [(-48, 66), (-76, 78), (-110, 88), (-146, 98)],
           6500, 1.25, 3.5, 0.060, 0.4, 0.8, 2.0)

    # 4) From the part, under the main sweep, on both sides.
    family([(16, 46), (50, 60), (84, 74), (116, 84), (148, 90)], [(16, 12), (66, 24), (112, 38), (146, 52), (170, 64)],
           6000, 1.05, 3.0, 0.040, 2.0, 0.9, 3.0)
    family([(16, 46), (-26, 53), (-66, 64), (-104, 72), (-140, 80)], [(16, 10), (-58, 22), (-108, 34), (-148, 46), (-176, 58)],
           7000, 1.05, 3.0, 0.044, 0.2, 1.0, 5.0)

    # 5) Main sweep: rising from the front hairline, then flowing up and back over the head.
    family([(22, 45), (40, 34), (78, 34), (122, 44), (164, 62)], [(62, 80), (82, 76), (106, 78), (134, 88), (162, 100)],
           7500, 1.10, 4.5, 0.052, 1.2, 0.9, 6.0)
    family([(20, 45), (-6, 31), (-56, 22), (-116, 30), (-166, 50)], [(-60, 80), (-84, 73), (-108, 74), (-138, 82), (-168, 95)],
           16000, 1.15, 5.5, 0.066, 3.6, 1.1, 7.0)
    # 6) The back of the head, combed down from the crown over the side sweeps.
    for _ in range(13000):
        a = R(118, 242) + random.gauss(0, 4)
        a = a - 360 if a > 180 else a
        end = 74 + 7 * math.sin(math.radians(a) * 6) + R(-6, 6)
        wave = np.sin(2 * math.pi * 1.1 * t + rad(a) * 0.9)
        az = np.radians(a + R(-2, 2) + 2.5 * wave)
        polar = np.radians(R(3, 14) + (end - 8) * t)
        stamp(az, polar, shade(a / 90.0, 4.0), 0.030 * wave * np.clip(t * 3, 0, 1))

    # 7) Rolled tiers across the lower back, on top. Height depends on the position
    #    within a tier, so each one reads as a round roll of hair.
    for center, half, amp, count in ((124, 8.0, 0.060, 3400), (110, 8.0, 0.075, 3800), (96, 7.5, 0.075, 3600), (83, 7.0, 0.055, 3000)):
        for _ in range(count):
            a = R(64, 296)
            a = a - 360 if a > 180 else a
            offset = R(-1, 1)
            polar_start = center + offset * half + 5.0 * math.sin(math.radians(a) * 5.0 + center * 0.2)
            az = np.radians(a + (t - 0.5) * R(34, 52))
            slope = 1.0 if a < 0 else -1.0
            droop = 3.0 * np.sin(t * math.pi) + slope * 7.0 * (t - 0.5)
            polar = np.radians(polar_start + droop)
            roll = amp * math.cos(offset * math.pi / 2)
            lobes = 0.016 * np.sin(az * 5.0 + center * 0.13)
            stamp(az, polar, shade((offset + 1) / 2, center), roll + lobes)

    return tone, height


def blur_map(array, passes):
    out = array
    for _ in range(passes):
        out = (
            out * 0.4
            + (np.roll(out, 1, 1) + np.roll(out, -1, 1)) * 0.2
            + (np.roll(out, 1, 0) + np.roll(out, -1, 0)) * 0.1
        )
    return out


def build_hair(m):
    rows, cols = (300, 480) if FULL else (24, 36)
    if FULL:
        tone, painted = paint_hair_flow()
        ridges = blur_map(painted, 11) * 2.0
        ridges = 0.085 * np.tanh(ridges / 0.085)
        h, w = painted.shape
        # Ease the ridges into one height toward the top of the head. Without this the ridges
        # stop abruptly around the crown and leave a flat disc inside a raised ring.
        blend_rows = int(math.radians(30) / math.pi * h)
        crown_height = float(ridges[blend_rows].mean())
        for row in range(blend_rows):
            s = row / blend_rows
            s = s * s * (3 - 2 * s)
            ridges[row] = crown_height + (ridges[row] - crown_height) * s
    polar = np.linspace(0.02, math.pi - 0.02, rows)
    azimuth = np.linspace(-math.pi, math.pi, cols, endpoint=False)
    shell = np.zeros((rows, cols, 3))
    for i, pol in enumerate(polar):
        for j, az in enumerate(azimuth):
            lift = 0.0
            if FULL:
                lift = ridges[min(int(pol / math.pi * h), h - 1), int((az + math.pi) / (2 * math.pi) * w) % w]
            edge_fade = min(max((polar_limit(az) - pol) / math.radians(12), 0.0), 1.0)
            shell[i, j] = hair_point(az, min(pol, polar_limit(az) + math.radians(1)), 1.0 + lift * edge_fade)
    obj = L.grid_object("hair_base", shell, m["hair"], True, True, False,
                        uv_u=np.linspace(0, 1, cols + 1), uv_v=polar / math.pi)
    mesh = obj.data
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(mesh)
    remove = []
    index_polar = np.repeat(polar, cols)
    index_az = np.tile(azimuth, rows)
    bm.verts.ensure_lookup_table()
    for face in bm.faces:
        vertex = min(v.index for v in face.verts)
        if index_polar[vertex] > polar_limit(index_az[vertex]) + math.radians(1):
            remove.append(face)
    bmesh.ops.delete(bm, geom=remove, context="FACES")
    bm.to_mesh(mesh)
    bm.free()
    register("hair_base", obj)
    if not FULL:
        return

    inner = np.zeros((rows // 3, cols // 3, 3))
    for i, pol in enumerate(np.linspace(0.02, math.pi - 0.02, rows // 3)):
        for j, az in enumerate(np.linspace(-math.pi, math.pi, cols // 3, endpoint=False)):
            clamped = min(pol, polar_limit(az) - math.radians(16))
            inner[i, j] = HEAD_C + HEAD_R * 1.03 * np.array(
                (math.sin(clamped) * math.sin(az), -math.sin(clamped) * math.cos(az), math.cos(clamped))
            )
    register("hair_inner", L.grid_object("hair_inner", inner, m["hair_inner"], True, False, False))

    # Valleys between ridges stay darker, like the soft shadow in the reference.
    span = max(float(np.abs(ridges).max()), 1e-6)
    depth = 0.86 + 0.26 * (ridges / span)
    soft = blur_map(tone, 1)
    value = np.clip(soft * depth, 0, 1)
    rgb = np.dstack([value * 0.975, value * 0.975, value * 1.0])
    color = L.array_to_image("joyce_hair", rgb, TEX_DIR)
    fine = np.clip(0.5 + (tone - blur_map(tone, 5)) * 1.6, 0, 1)
    bump = L.array_to_image("joyce_hair_bump", np.dstack([fine, fine, fine]), TEX_DIR)
    L.attach_color_image(m["hair"], color)
    L.attach_bump_image(m["hair"], bump, 0.35, 0.0015)


# ------------------------------------------------------------------ build
materials = make_materials()
build_head(materials)
build_eyes(materials)
build_torso(materials)
build_arm(materials, 1, "L")
build_arm(materials, -1, "R")
build_lower(materials)
build_hair(materials)

HEAD_SHIFT = np.array((0.0, -0.018, 0.0))
for key, obj in parts.items():
    if key.split("_")[0] in ("head", "ear", "eye", "lid", "brow", "hair", "lash", "crease"):
        count = len(obj.data.vertices)
        coords = np.empty(count * 3)
        obj.data.vertices.foreach_get("co", coords)
        coords = coords.reshape(-1, 3) + HEAD_SHIFT
        obj.data.vertices.foreach_set("co", coords.ravel())
        obj.data.update()

for material in bpy.data.materials:
    material.use_backface_culling = False

os.makedirs(os.path.dirname(OUT), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=OUT, compress=False)

triangles = sum(
    sum(len(p.vertices) - 2 for p in o.data.polygons) for o in bpy.data.objects if o.type == "MESH"
)
print("BUILD_OK", STAGE, "objects", len(parts), "triangles", triangles)
