"""Clothes follow a reshaped body by local affine fits, smoothed over the garment (added by pipeline/humgen).

Replaces HG's own refit (each cloth point moved by its single nearest body point, offset unscaled), which tears
garments when the body changes a lot, as for a child. Here each cloth point takes the best affine map (stretch,
turn, shift) of its K nearest body points from the old body to the new one, weighted by distance; the resulting
moves are then smoothed along the garment's own edges so neighbouring points can't part.
"""
import numpy as np
from mathutils import Vector, kdtree


def smooth_deform(body_old, body_new, cloth, cloth_edges, k=16, smooth_iters=12, reg=1e-4):
    """New cloth points (world). body_old/body_new (N,3) same body points before/after; cloth (M,3)."""
    kd = kdtree.KDTree(len(body_old))
    for i, co in enumerate(body_old):
        kd.insert(co, i)
    kd.balance()
    idx = np.empty((len(cloth), k), np.int64)
    dist = np.empty((len(cloth), k))
    for j, co in enumerate(cloth):
        found = kd.find_n(Vector(co), k)
        for m, (_, i, d) in enumerate(found):
            idx[j, m], dist[j, m] = i, d
    w = 1.0 / (dist + 0.005) ** 2  # (M,K)
    w /= w.sum(1, keepdims=True)
    B = body_old[idx]  # (M,K,3)
    N = body_new[idx]
    bc = (w[..., None] * B).sum(1)
    nc = (w[..., None] * N).sum(1)
    Bd, Nd = B - bc[:, None], N - nc[:, None]
    # affine part: Nd ~= A Bd, weighted least squares, regularised toward the mean scale
    M = np.einsum("mk,mki,mkj->mij", w, Bd, Bd)
    R = np.einsum("mk,mki,mkj->mij", w, Nd, Bd)
    s = np.sqrt(np.einsum("mk,mki,mki->m", w, Nd, Nd) / np.maximum(np.einsum("mk,mki,mki->m", w, Bd, Bd), 1e-12))
    lam = reg + 0.05 * np.trace(M, axis1=1, axis2=2)
    eye = np.eye(3)[None]
    A = np.linalg.solve((M + lam[:, None, None] * eye).transpose(0, 2, 1),
                        (R + (lam * s)[:, None, None] * eye).transpose(0, 2, 1)).transpose(0, 2, 1)
    new = nc + np.einsum("mij,mj->mi", A, cloth - bc)
    D = new - cloth
    if len(cloth_edges) and smooth_iters:
        e = np.asarray(cloth_edges)
        deg = np.bincount(e.ravel(), minlength=len(cloth)).astype(float)[:, None]
        for _ in range(smooth_iters):
            S = np.zeros_like(D)
            np.add.at(S, e[:, 0], D[e[:, 1]])
            np.add.at(S, e[:, 1], D[e[:, 0]])
            D = np.where(deg > 0, 0.5 * D + 0.5 * S / np.maximum(deg, 1), D)
    return cloth + D
