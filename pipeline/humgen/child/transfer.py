"""Carry MPFB's adult-to-child change onto Human Generator's body: a child livekey for HG.

Usage: blender -b -P pipeline/humgen/child/transfer.py -- <hg.npz> <mpfb_adult.npz> <mpfb_child.npz> <out.npz>
       [--height 1.15] [--livekey <content>/livekeys/special/age_child/child_6_female.npz] [--fit-iters 30]
hg.npz from export_hg.py; the MPFB pair from export_mpfb.py (same settings and stance, only the age differs).

How (deformation transfer, Sumner & Popovic 2004):
1. Wrap: MPFB's adult is scaled to HG's height and pulled onto HG's surface (non-rigid ICP: nearest points both
   ways, normals must agree, the pull smoothed over the mesh, less smoothing each round).
2. Each HG triangle finds the wrapped MPFB triangle under its centre.
3. Each MPFB triangle's change from adult to child is a 3x3 matrix (its two edges and normal, before and after).
4. HG's child: the points whose triangle edges best follow those matrices applied to HG's own edges (a Poisson
   solve, conjugate gradients in numpy), so HG's face and body detail is kept but reshaped like MPFB's child.
5. Scaled about the floor to --height (MakeHuman's children run short), feet on the floor, centred.
Writes out.npz (child points, the wrap error) and, with --livekey, the livekey HG reads (offsets from HG's
rest, as HG's own save_to_library writes them). Prints TRANSFER_OK.
"""
import os
import sys
import time

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:]
hg, A_npz, C_npz = (np.load(os.path.abspath(p)) for p in argv[:3])
out = os.path.abspath(argv[3])


def opt(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


height = float(opt("--height", "0"))
fit_iters = int(opt("--fit-iters", "30"))
t0 = time.time()

H, TH = hg["rest"].astype(np.float64), hg["tris"]
A, C, TA = A_npz["points"].astype(np.float64), C_npz["points"].astype(np.float64), A_npz["tris"]


def edges_of(T):
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    e.sort(1)
    return np.unique(e, axis=0)


def vnormals(P, T):
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    n = np.zeros_like(P)
    for k in range(3):
        np.add.at(n, T[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def smooth(D, E, n):
    deg = np.bincount(E.ravel(), minlength=len(D)).astype(float)[:, None]
    for _ in range(n):
        S = np.zeros_like(D)
        np.add.at(S, E[:, 0], D[E[:, 1]])
        np.add.at(S, E[:, 1], D[E[:, 0]])
        D = 0.5 * D + 0.5 * S / np.maximum(deg, 1)
    return D


def bvh(P, T):
    return BVHTree.FromPolygons([tuple(p) for p in P], [tuple(t) for t in T])


def nearest(tree, P):
    loc, nrm, idx = np.zeros_like(P), np.zeros_like(P), np.full(len(P), -1)
    for i, p in enumerate(P):
        r = tree.find_nearest(Vector(p))
        if r[0] is not None:
            loc[i], nrm[i], idx[i] = r[0], r[1], r[2]
    return loc, nrm, idx


# 1. wrap MPFB's adult onto HG
s = (H[:, 2].max() - H[:, 2].min()) / (A[:, 2].max() - A[:, 2].min())
W = A * s
W[:, 2] -= W[:, 2].min() - H[:, 2].min()
W[:, 0] += H[:, 0].mean() - W[:, 0].mean()
W[:, 1] += H[:, 1].mean() - W[:, 1].mean()
EA = edges_of(TA)
treeH = bvh(H, TH)
nH = vnormals(H, TH)
for it in range(fit_iters):
    f = it / max(fit_iters - 1, 1)
    nW = vnormals(W, TA)
    # MPFB -> HG
    loc, nrm, _ = nearest(treeH, W)
    ok = (np.einsum("ij,ij->i", nrm, nW) > 0.5)
    D = np.where(ok[:, None], loc - W, 0.0)
    wgt = ok.astype(float)
    # HG -> MPFB: each HG point pulls the nearest wrapped triangle's corners
    treeW = bvh(W, TA)
    locw, nrmw, idxw = nearest(treeW, H)
    okh = (idxw >= 0) & (np.einsum("ij,ij->i", nrmw, nH) > 0.5)
    for k in range(3):
        np.add.at(D, TA[idxw[okh], k], (H[okh] - locw[okh]))
        np.add.at(wgt, TA[idxw[okh], k], 1.0)
    D = D / np.maximum(wgt, 1)[:, None]
    n_smooth = int(80 * (1 - f) + 4)
    W += 0.7 * smooth(D, EA, n_smooth)
    if it % 5 == 0 or it == fit_iters - 1:
        err = np.linalg.norm(nearest(treeH, W)[0] - W, axis=1)
        print(f"WRAP {it} smooth={n_smooth} mean={err.mean() * 1000:.1f}mm p95={np.percentile(err, 95) * 1000:.1f}mm"
              f" ({time.time() - t0:.0f}s)")

# 2. HG triangle -> MPFB triangle
cent = H[TH].mean(1)
treeW = bvh(W, TA)
_, _, corr = nearest(treeW, cent)


# 3. MPFB adult -> child, per triangle
def frames(P, T):
    e1, e2 = P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]
    n = np.cross(e1, e2)
    n /= np.sqrt(np.linalg.norm(n, axis=1, keepdims=True)) + 1e-12
    return np.stack([e1, e2, n], axis=2)


G = frames(C, TA) @ np.linalg.inv(frames(A, TA))
Gt = G[corr]

# 4. Poisson solve on HG's mesh: X_j - X_i ~= G (H_j - H_i) for each triangle edge
I = np.concatenate([TH[:, 0], TH[:, 1], TH[:, 2]])
J = np.concatenate([TH[:, 1], TH[:, 2], TH[:, 0]])
GG = np.concatenate([Gt, Gt, Gt])
d = np.einsum("kij,kj->ki", GG, H[J] - H[I])
nV = len(H)
deg = (np.bincount(I, minlength=nV) + np.bincount(J, minlength=nV)).astype(float)
b = np.zeros_like(H)
for c in range(3):
    b[:, c] = np.bincount(J, d[:, c], nV) - np.bincount(I, d[:, c], nV)


def Lmul(x):
    return deg * x - np.bincount(I, x[J], nV) - np.bincount(J, x[I], nV)


X = np.zeros_like(H)
for c in range(3):
    x = H[:, c] * (np.ptp(C[:, 2]) / np.ptp(A[:, 2]))  # start: HG shrunk like MPFB's child
    r = b[:, c] - Lmul(x)
    r -= r.mean()
    z = r / np.maximum(deg, 1)
    p = z.copy()
    rz = r @ z
    for k in range(4000):
        Ap = Lmul(p)
        a = rz / (p @ Ap)
        x += a * p
        r -= a * Ap
        if np.linalg.norm(r) < 1e-7 * np.linalg.norm(b[:, c]):
            break
        z = r / np.maximum(deg, 1)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    X[:, c] = x
    print(f"SOLVE axis {c}: {k} iterations, residual {np.linalg.norm(r):.2e}")
res = np.linalg.norm((X[J] - X[I]) - d, axis=1)
print(f"EDGE_FIT mean {res.mean() * 1000:.2f}mm max {res.max() * 1000:.1f}mm")

# 5. size and place
X[:, 2] -= X[:, 2].min() - H[:, 2].min()
if height:  # uniform scale about the floor: proportions kept
    X *= height / np.ptp(X[:, 2])
    X[:, 2] -= X[:, 2].min() - H[:, 2].min()
X[:, 0] += H[:, 0].mean() - X[:, 0].mean()
X[:, 1] += H[:, 1].mean() - X[:, 1].mean()
print("CHILD height", round(float(np.ptp(X[:, 2])), 3), "width", round(float(np.ptp(X[:, 0])), 3))
np.savez(out, child=X, wrapped=W, corr=corr, scale=s)
if "--livekey" in argv:
    lk = os.path.abspath(opt("--livekey", ""))
    os.makedirs(os.path.dirname(lk), exist_ok=True)
    # offsets from the body as shown (for a man the "Male" key stays on underneath), rounded as HG saves them
    rel = (X - H).ravel().round(4)
    idx = np.nonzero(rel)
    np.savez(lk, indices=idx, relative_coordinates=rel[idx])
    print("LIVEKEY", lk, len(idx[0]), "values")
print(f"TRANSFER_OK {out} ({time.time() - t0:.0f}s)")
