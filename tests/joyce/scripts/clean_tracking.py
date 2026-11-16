"""Repair raw tracker output before it is used to animate a character.

Usage: clean_tracking.py <source_pose.json> <clean_pose.json> [--no-labels] [--no-image]

Three repairs, each measured and reported:
1. Left/right labels. Seen from the side, the tracker often calls whichever leg is in front
   "right", so the labels swap at every step and neither leg ever completes a stride. Real
   limbs cannot jump, so the labelling that gives the smoothest motion over time is chosen.
2. Outliers. A point that leaves its neighbours and comes straight back is replaced.
3. Body shape. The tracker's 3D estimate pulls poses toward an average body, which shortens
   strides and keeps knees from straightening. Its 2D image positions don't have that
   problem, so they supply left-right and up-down, and the 3D estimate supplies depth only.
"""
import json
import sys

import numpy as np

source, out = sys.argv[1:3]
USE_LABELS = "--no-labels" not in sys.argv
USE_IMAGE = "--no-image" not in sys.argv

with open(source, encoding="utf-8") as handle:
    data = json.load(handle)
aspect = data["width"] / data["height"]
frames = data["frames"]
count = len(frames)
tracked = [i for i, f in enumerate(frames) if f is not None]
if len(tracked) < 3:
    raise SystemExit("Too few tracked frames to clean.")

world = np.array([frames[i]["world"] for i in tracked], dtype=float)
image = np.array([frames[i]["image"] for i in tracked], dtype=float)
n = len(tracked)

LEG_PAIRS = [(25, 26), (27, 28), (29, 30), (31, 32)]
LEG_WEIGHTS = [0.6, 1.0, 1.0, 1.0]
ARM_PAIRS = [(13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]
ARM_WEIGHTS = [0.6, 1.0, 0.5, 0.5, 0.3]


def relative_points():
    """Image positions centred on the hips and measured in torso lengths, so camera moves don't count."""
    xy = np.stack([image[:, :, 0] * aspect, image[:, :, 1]], axis=2)
    hips = (xy[:, 23] + xy[:, 24]) / 2
    shoulders = (xy[:, 11] + xy[:, 12]) / 2
    torso = np.median(np.linalg.norm(shoulders - hips, axis=1))
    return (xy - hips[:, None, :]) / max(torso, 1e-6)


def best_labels(rel, pairs, weights):
    """Viterbi over (previous, current) swap states, minimising acceleration of the limb points."""
    def position(t, swapped, side):
        ids = [p[side ^ swapped] for p in pairs]
        return rel[t, ids]

    w = np.array(weights)[:, None]

    def acceleration(t, a, b, c):
        total = 0.0
        for side in (0, 1):
            step = position(t, c, side) - 2 * position(t - 1, b, side) + position(t - 2, a, side)
            # Capped, so one bad point can't decide the labelling of everything around it.
            total += float((w * np.minimum(step * step, 0.6)).sum())
        return total

    # Start from either labelling, then extend. States are (s[t-1], s[t]).
    cost = {(a, b): 0.0 for a in (0, 1) for b in (0, 1)}
    back = []
    for t in range(2, n):
        new_cost, pointers = {}, {}
        for b in (0, 1):
            for c in (0, 1):
                options = [(cost[(a, b)] + acceleration(t, a, b, c), a) for a in (0, 1)]
                value, choice = min(options)
                new_cost[(b, c)] = value
                pointers[(b, c)] = choice
        cost = new_cost
        back.append(pointers)
    state = min(cost, key=cost.get)
    labels = [state[1], state[0]]
    for pointers in reversed(back):
        state = (pointers[state], state[0])
        labels.append(state[0])
    return np.array(labels[::-1], dtype=int)


def apply_labels(pairs, swapped):
    """Swap labelled pairs on the chosen frames, then choose which track is which side."""
    if swapped.mean() > 0.5:
        swapped = 1 - swapped  # keep the label the tracker used most often
    for a, b in pairs:
        # The tracker's depth belongs to the label, not the limb: in a side view it always puts
        # "left" on the far side. Carry that constant across, or the feet would jump sideways.
        depth_gap = float(np.median(world[:, a, 2] - world[:, b, 2]))
        for t in np.nonzero(swapped)[0]:
            w_a, w_b = world[t, a].copy(), world[t, b].copy()
            i_a, i_b = image[t, a].copy(), image[t, b].copy()
            world[t, a], world[t, b] = w_b, w_a
            world[t, a, 2] = w_b[2] + depth_gap
            world[t, b, 2] = w_a[2] - depth_gap
            image[t, a], image[t, b] = i_b, i_a
    return swapped


def count_runs(values):
    return int((np.diff(values) != 0).sum())


def swap_range(pairs, gaps, start, stop):
    """Swap labels from frame `start` up to but not including `stop`, keeping depth with the label."""
    for (a, b), gap in zip(pairs, gaps):
        for t in range(start, stop):
            w_a, w_b = world[t, a].copy(), world[t, b].copy()
            i_a, i_b = image[t, a].copy(), image[t, b].copy()
            world[t, a], world[t, b] = w_b, w_a
            world[t, a, 2] = w_b[2] + gap
            world[t, b, 2] = w_a[2] - gap
            image[t, a], image[t, b] = i_b, i_a


def line_fit(values, frames):
    """Straight line through a foot's positions: returns its value at frame 0 and its slope."""
    slope = np.array([np.polyfit(frames, values[:, axis], 1)[0] for axis in range(values.shape[1])])
    return values.mean(axis=0) - slope * np.mean(frames), slope


FOOT_IDS = ((27, 29, 31), (28, 30, 32))
TOUCH_DISTANCE = 0.35   # feet closer than this, in torso lengths, can't be told apart reliably
LONGEST_TOUCH = 8       # longer than this and the feet are simply together, as when standing
FIT_FRAMES = 4


def resolve_crossings():
    """Where the feet pass, decide from their motion on each side whether they crossed.

    The tracker stalls both feet as they pass, so frame-to-frame smoothness can prefer two feet
    bouncing off each other. Which foot continues where is clearer from a few frames before and
    after: a swinging foot keeps swinging and a planted foot keeps sliding back. Those few frames
    of stalled readings are then replaced with straight motion between the good ones.
    """
    gaps = [float(np.median(world[:, a, 2] - world[:, b, 2])) for a, b in LEG_PAIRS]
    fixed, smoothed, by_rhythm = 0, 0, 0
    passes = []
    t = 0
    while t < n:
        rel = relative_points()
        foot = [rel[:, list(ids)].mean(axis=1) for ids in FOOT_IDS]
        if np.linalg.norm(foot[0][t] - foot[1][t]) >= TOUCH_DISTANCE:
            t += 1
            continue
        start = t
        while t < n and np.linalg.norm(foot[0][t] - foot[1][t]) < TOUCH_DISTANCE:
            t += 1
        stop = t  # frames start .. stop - 1 touch
        if stop - start > LONGEST_TOUCH or start < 2 or stop > n - 2:
            continue
        before = np.arange(max(start - FIT_FRAMES, 0), start)
        after = np.arange(stop, min(stop + FIT_FRAMES, n))
        middle = (start + stop - 1) / 2
        pre = [line_fit(foot[k][before], before) for k in (0, 1)]
        post = [line_fit(foot[k][after], after) for k in (0, 1)]

        def join_cost(a, b):
            position = pre[a][0] + pre[a][1] * middle - (post[b][0] + post[b][1] * middle)
            turn = (pre[a][1] - post[b][1]) * FIT_FRAMES
            return float((position * position).sum() + (turn * turn).sum())

        stay = join_cost(0, 0) + join_cost(1, 1)
        cross = join_cost(0, 1) + join_cost(1, 0)
        swap_after = cross < stay
        # Do the feet end up having changed places? When the last passes have come at a steady
        # rhythm, a pass due at this moment is a crossing, even if too few frames follow to show it.
        apart_before = foot[0][start - 1] - foot[1][start - 1]
        apart_after = foot[0][stop] - foot[1][stop]
        changed_places = float(np.dot(apart_before, apart_after)) < 0
        if swap_after:
            changed_places = not changed_places
        if not changed_places and len(passes) >= 2:
            rhythm = float(np.median(np.diff(passes)))
            if abs(middle - (passes[-1] + rhythm)) <= 3:
                swap_after = not swap_after
                changed_places = True
                by_rhythm += 1
        if changed_places:
            passes.append(middle)
        if swap_after:
            swap_range(LEG_PAIRS, gaps, stop, n)
            fixed += 1
        # Straight motion across the stalled frames, for each leg point and in both data sets.
        for a, b in LEG_PAIRS:
            for joint in (a, b):
                span = stop - start + 1
                for k, frame in enumerate(range(start, stop), 1):
                    mix = k / span
                    world[frame, joint, :3] = world[start - 1, joint, :3] * (1 - mix) + world[stop, joint, :3] * mix
                    xy_now = (image[frame, 23, :2] + image[frame, 24, :2]) / 2
                    xy_a = image[start - 1, joint, :2] - (image[start - 1, 23, :2] + image[start - 1, 24, :2]) / 2
                    xy_b = image[stop, joint, :2] - (image[stop, 23, :2] + image[stop, 24, :2]) / 2
                    image[frame, joint, :2] = xy_now + xy_a * (1 - mix) + xy_b * mix
        smoothed += stop - start
    report["crossings_decided_by_rhythm"] = by_rhythm
    return fixed, smoothed


report = {"frames": count, "tracked": n}
if USE_LABELS and n >= 5:
    rel = relative_points()
    legs = apply_labels(LEG_PAIRS, best_labels(rel, LEG_PAIRS, LEG_WEIGHTS))
    report["crossings_corrected"], report["stalled_frames_replaced"] = resolve_crossings()
    rel = relative_points()
    arms = apply_labels(ARM_PAIRS, best_labels(rel, ARM_PAIRS, ARM_WEIGHTS))
    report["leg_label_swaps"] = count_runs(legs)
    report["leg_frames_relabelled"] = int(legs.sum())
    report["arm_label_swaps"] = count_runs(arms)
    report["arm_frames_relabelled"] = int(arms.sum())


# ---------------------------------------------------------------- outliers
def remove_outliers(values, limit):
    """Replace points that sit far from the median of their neighbours, per coordinate."""
    fixed = values.copy()
    replaced = np.zeros(len(values), dtype=bool)
    radius = 2
    for t in range(len(values)):
        lo, hi = max(t - radius, 0), min(t + radius + 1, len(values))
        centre = np.median(values[lo:hi], axis=0)
        if np.linalg.norm(values[t] - centre) > limit:
            fixed[t] = centre
            replaced[t] = True
    return fixed, replaced


rel = relative_points()
xy_full = np.stack([image[:, :, 0] * aspect, image[:, :, 1]], axis=2)
hips_full = (xy_full[:, 23] + xy_full[:, 24]) / 2
shoulders_full = (xy_full[:, 11] + xy_full[:, 12]) / 2
torso_image = float(np.median(np.linalg.norm(shoulders_full - hips_full, axis=1)))
replaced_total = 0
for joint in (13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32):
    fixed, replaced = remove_outliers(rel[:, joint], 0.30)
    if replaced.any():
        replaced_total += int(replaced.sum())
        points = fixed * torso_image + hips_full
        image[replaced, joint, 0] = points[replaced, 0] / aspect
        image[replaced, joint, 1] = points[replaced, 1]
        # The 3D estimate at that moment came from the same bad reading, so follow its neighbours.
        for t in np.nonzero(replaced)[0]:
            lo, hi = max(t - 2, 0), min(t + 3, n)
            world[t, joint, :3] = np.median(world[lo:hi, joint, :3], axis=0)
report["outlier_points_replaced"] = replaced_total

# ---------------------------------------------------------------- 2D positions, tracker depth
if USE_IMAGE:
    xy = np.stack([image[:, :, 0] * aspect, image[:, :, 1]], axis=2)
    hips = (xy[:, 23] + xy[:, 24]) / 2
    shoulders = (xy[:, 11] + xy[:, 12]) / 2
    world_hips = (world[:, 23, :2] + world[:, 24, :2]) / 2
    world_shoulders = (world[:, 11, :2] + world[:, 12, :2]) / 2
    image_length = np.linalg.norm(shoulders - hips, axis=1)
    world_length = np.linalg.norm(world_shoulders - world_hips, axis=1)
    # One scale for the clip's median body size, then eased over time as she nears or leaves the lens.
    ratio = world_length / np.maximum(image_length, 1e-6)
    window = max(int(data["fps"]), 3)
    scale = np.array([np.median(ratio[max(t - window, 0):t + window + 1]) for t in range(n)])
    report["image_to_metres_scale"] = round(float(np.median(scale)), 2)
    centred = (xy - hips[:, None, :]) * scale[:, None, None]
    world[:, :, :2] = centred + world_hips[:, None, :]

# ---------------------------------------------------------------- write
cleaned = [None] * count
for k, i in enumerate(tracked):
    cleaned[i] = {"world": world[k].tolist(), "image": image[k].tolist()}
with open(out, "w", encoding="utf-8") as handle:
    json.dump({"fps": data["fps"], "width": data["width"], "height": data["height"], "frames": cleaned, "clean_report": report}, handle)
print("CLEAN_REPORT", json.dumps(report))
print("CLEAN_OK")
