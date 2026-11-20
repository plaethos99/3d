"""How plausible is a reconstructed head? Fit FLAME to its face landmarks and see where it can't follow.

Usage: tools/pose-env/Scripts/python.exe -I pipeline/flame/flame_fit.py <subject_fit.json> [<subject_fit.json> ...]
       [--model tools/flame/flame2023_Open.pkl] [--landmarks tools/flame/flame_landmarks.json] [--shapes 100]
       [--reg 1.0] [--out fit.json] [--save-head heads.npz]  (with one subject_fit: the fitted head, for flame_head.py)
FLAME (3,800 real head scans) is fitted to the head's MediaPipe landmarks: a similarity (scale, turn, move) and
the first --shapes identity values, kept near real faces (ridge --reg on the values, which are in standard
deviations). What it can't follow is what no real head does: residual per landmark (mm) and per face region.
A clean reconstruction leaves a few mm everywhere; a region with a big residual (a jaw the photos barely saw, a
crushed nose) is where the reconstruction is suspect. Also reports how unusual the fitted face is (|values|).
Prints FLAME_FIT_OK.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from flame_model import load  # noqa: E402


def similarity(src, dst):
    """Umeyama: s, R, t with dst ~ s R src + t (as hg_landmarks.similarity, without Blender)."""
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, S_, Vt = np.linalg.svd(xd.T @ xs / len(src))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ d @ Vt
    s = float(np.trace(np.diag(S_) @ d) / (xs ** 2).sum(1).mean())
    return s, R, md - s * R @ ms


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


fits = [a for i, a in enumerate(sys.argv[1:], 1) if a.endswith(".json") and sys.argv[i - 1] not in
        ("--landmarks", "--out", "--save-head")]
m = load(os.path.join(ROOT, opt("--model", "tools/flame/flame2023_Open.pkl")))
lm = np.array(json.load(open(os.path.join(ROOT, opt("--landmarks", "tools/flame/flame_landmarks.json"))))["landmarks"])
K = int(opt("--shapes", "100"))
reg = float(opt("--reg", "1.0"))
# MediaPipe face regions (landmark ids), for the per-region report
REGIONS = {
    "jaw/outline": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
                    148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "nose": [1, 2, 4, 5, 6, 19, 94, 97, 98, 326, 327, 168, 195, 197, 48, 278, 64, 294, 129, 358],
    "eyes": [33, 7, 163, 144, 145, 153, 154, 155, 133, 263, 249, 390, 373, 374, 380, 381, 382, 362, 159, 386],
    "mouth": [61, 291, 0, 17, 37, 267, 84, 314, 40, 270, 91, 321, 185, 409, 146, 375],
    "brows": [70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
    "cheeks": [50, 280, 101, 330, 205, 425, 187, 411, 123, 352, 117, 346],
}
T = m["v_template"]
S = m["shapedirs"][:, :, :K]
report = {}
for path in fits:
    fit = json.load(open(path))
    ids = np.array(fit["ids"])
    ok = (ids < len(lm)) & (lm[np.minimum(ids, len(lm) - 1)] >= 0)
    ids, L = ids[ok], np.array(fit["points"])[ok]
    # FLAME's frame is y up, facing +z: the head runs face -y, z up
    vi = lm[ids]
    A0 = T[vi]
    Sv = S[vi]  # (n,3,K)
    b = np.zeros(K)
    for it in range(15):
        X = A0 + Sv @ b
        s, R, t = similarity(X, L)
        # in FLAME's frame: L' = R^T (L - t) / s ~ A0 + Sv b
        Lf = ((L - t) @ R) / s
        M = Sv.reshape(-1, K)
        rhs = (Lf - A0).reshape(-1)
        # ridge: values in standard deviations; reg scaled by the landmark count (per-mm units: metres -> 1e-3)
        lam = reg * (1e-3 ** 2) * len(ids)
        b = np.linalg.solve(M.T @ M + lam * np.eye(K), M.T @ rhs)
    X = A0 + Sv @ b
    s, R, t = similarity(X, L)
    Xw = s * X @ R.T + t
    res = np.linalg.norm(Xw - L, axis=1) * 1000
    reg_res = {}
    for name, rid in REGIONS.items():
        sel = np.isin(ids, rid)
        if sel.any():
            reg_res[name] = round(float(np.median(res[sel])), 2)
    report[path] = {"median_mm": round(float(np.median(res)), 2), "p90_mm": round(float(np.percentile(res, 90)), 2),
                    "regions_mm": reg_res, "unusualness": round(float(np.sqrt((b ** 2).mean())), 2),
                    "largest_values": [round(float(x), 2) for x in b[np.argsort(-np.abs(b))[:5]]]}
    print("FIT", os.path.relpath(path, ROOT), json.dumps(report[path]))
    if "--save-head" in sys.argv:  # the fitted FLAME head, for flame_head.py (FLAME's frame)
        heads = opt("--save-head", "")
        names = [os.path.basename(os.path.dirname(os.path.abspath(path)))]
        np.savez(heads, faces=m["faces"], names=np.array(["fit_" + names[0]]),
                 verts=(T + m["shapedirs"][:, :, :K] @ b)[None])
        print("SAVED", heads)
if "--out" in sys.argv:
    json.dump(report, open(opt("--out", ""), "w"), indent=1)
print("FLAME_FIT_OK")
