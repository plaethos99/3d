"""Indian textile prints as Human Generator clothing patterns (pose-env python, -I).

Usage: python -I pipeline/humgen/indian_patterns.py <HG content folder>
Writes <content>/patterns/Indian/<name>.png (2048 px, seamless) and <name>.jpg (256 px preview). HG's pattern
format: the red, green and blue channels are three masks, each painted in one of the garment's pattern colours
(PC1, PC2, PC3 on its HG_Control node) over its main colour. Prints PATTERNS_OK.
Prints: paisley_butta (paisley motifs, a dot inside, scattered small dots), bandhani (tie-dye dot clusters),
leheriya (diagonal wave stripes), brocade (diamond lattice with flowers), block_floral (five-petal flowers, leaves).
"""
import math
import os
import sys

import cv2
import numpy as np

S = 2048
out_dir = os.path.join(os.path.abspath(sys.argv[1]), "patterns", "Indian")
os.makedirs(out_dir, exist_ok=True)


def canvas():
    return [np.zeros((S, S), np.uint8) for _ in range(3)]


def tiled(draw, x, y):
    """Draw a motif at (x, y) and its copies across the edges, so the pattern tiles without seams."""
    for dx in (-S, 0, S):
        for dy in (-S, 0, S):
            draw(x + dx, y + dy)


def paisley(ch, x, y, size, angle):
    """A paisley (teardrop with a curled tip) filled into channel ch."""
    pts = []
    for t in np.linspace(0, 2 * math.pi, 80):
        r = size * (0.55 + 0.45 * math.cos(t / 2) ** 2)
        px, py = r * math.cos(t) * 0.75, r * math.sin(t)
        pts.append((px, py))
    tip = [(size * 0.2 + size * 0.5 * math.cos(a), -size * 0.9 + size * 0.5 * math.sin(a))
           for a in np.linspace(0.2, 2.6, 20)]
    pts = np.array(pts + tip)
    c, s = math.cos(angle), math.sin(angle)
    pts = pts @ np.array([[c, -s], [s, c]]).T + [x, y]
    cv2.fillPoly(ch, [pts.astype(np.int32)], 255, cv2.LINE_AA)


def save(name, ch):
    img = np.dstack([ch[2], ch[1], ch[0]])  # cv2 writes BGR: R, G, B masks in that order
    cv2.imwrite(os.path.join(out_dir, f"{name}.png"), img)
    # preview: maroon base, gold / green / cream pattern colours
    base = np.array([40, 20, 110], np.float32)
    cols = [np.array([60, 180, 230], np.float32), np.array([80, 140, 40], np.float32),
            np.array([220, 235, 245], np.float32)]
    prev = np.ones((S, S, 3), np.float32) * base
    for m, col in zip(ch, cols):
        a = (m.astype(np.float32) / 255)[..., None]
        prev = prev * (1 - a) + col * a
    cv2.imwrite(os.path.join(out_dir, f"{name}.jpg"), cv2.resize(prev.astype(np.uint8), (256, 256),
                                                                    interpolation=cv2.INTER_AREA))
    print("PATTERN", name)


rng = np.random.default_rng(7)

# paisley butta: offset rows of paisleys
ch = canvas()
step = S // 6
for row in range(6):
    for col in range(6):
        x = col * step + (step // 2 if row % 2 else 0)
        y = row * step + step // 2
        ang = 0.5 if row % 2 else -0.5
        tiled(lambda a, b: paisley(ch[0], a, b, step * 0.28, ang), x, y)
        tiled(lambda a, b: cv2.circle(ch[1], (int(a), int(b + step * 0.05)), int(step * 0.07), 255, -1, cv2.LINE_AA), x, y)
for _ in range(500):
    x, y = rng.integers(0, S, 2)
    tiled(lambda a, b: cv2.circle(ch[2], (int(a), int(b)), 6, 255, -1, cv2.LINE_AA), x, y)
ch[2][ch[0] > 0] = 0
save("paisley_butta", ch)

# bandhani: diamond grid of dot clusters (small rings with a centre)
ch = canvas()
step = S // 16
for row in range(16):
    for col in range(16):
        x = col * step + (step // 2 if row % 2 else 0)
        y = row * step
        tiled(lambda a, b: cv2.circle(ch[0], (int(a), int(b)), int(step * 0.18), 255, 5, cv2.LINE_AA), x, y)
        tiled(lambda a, b: cv2.circle(ch[1], (int(a), int(b)), int(step * 0.06), 255, -1, cv2.LINE_AA), x, y)
        if (row + col) % 4 == 0:
            for k in range(6):
                a_ = k * math.pi / 3
                tiled(lambda a, b: cv2.circle(ch[2], (int(a + step * 0.32 * math.cos(a_)), int(b + step * 0.32 * math.sin(a_))),
                                              8, 255, -1, cv2.LINE_AA), x, y)
save("bandhani", ch)

# leheriya: diagonal waves in three alternating colours
ch = canvas()
yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
period = S / 24
phase = (xx + yy) / period + 0.35 * np.sin((xx - yy) * 2 * math.pi / (S / 4))
band = np.floor(phase).astype(int) % 6
for i in range(3):
    ch[i] = np.where(band == 2 * i, 255, 0).astype(np.uint8)
save("leheriya", ch)

# brocade: diamond lattice, flowers at the crossings, dots in the middle
ch = canvas()
step = S // 8
for k in range(-8, 17):
    for sgn in (1, -1):
        p0 = (k * step, 0) if sgn == 1 else (k * step, 0)
        p1 = (k * step + S * sgn, S)
        cv2.line(ch[0], p0, p1, 255, 10, cv2.LINE_AA)
for row in range(9):
    for col in range(9):
        x, y = col * step, row * step
        for k in range(8):
            a_ = k * math.pi / 4
            tiled(lambda a, b: cv2.ellipse(ch[1], (int(a + step * 0.09 * math.cos(a_)), int(b + step * 0.09 * math.sin(a_))),
                                           (int(step * 0.08), int(step * 0.035)), math.degrees(a_), 0, 360, 255, -1,
                                           cv2.LINE_AA), x, y)
        tiled(lambda a, b: cv2.circle(ch[2], (int(a + step / 2), int(b + step / 2)), int(step * 0.06), 255, -1,
                                      cv2.LINE_AA), x, y)
save("brocade", ch)

# block floral: five-petal flowers with two leaves, small dots between
ch = canvas()
step = S // 5
for row in range(5):
    for col in range(5):
        x = col * step + (step // 2 if row % 2 else 0)
        y = row * step + step // 2
        for k in range(5):
            a_ = k * 2 * math.pi / 5
            tiled(lambda a, b: cv2.ellipse(ch[0], (int(a + step * 0.12 * math.cos(a_)), int(b + step * 0.12 * math.sin(a_))),
                                           (int(step * 0.11), int(step * 0.06)), math.degrees(a_), 0, 360, 255, -1,
                                           cv2.LINE_AA), x, y)
        tiled(lambda a, b: cv2.circle(ch[2], (int(a), int(b)), int(step * 0.05), 255, -1, cv2.LINE_AA), x, y)
        for sgn in (-1, 1):
            tiled(lambda a, b: cv2.ellipse(ch[1], (int(a + sgn * step * 0.3), int(b + step * 0.22)),
                                           (int(step * 0.13), int(step * 0.045)), 35 * sgn, 0, 360, 255, -1,
                                           cv2.LINE_AA), x, y)
ch[0][ch[2] > 0] = 0
save("block_floral", ch)
print("PATTERNS_OK", out_dir)
