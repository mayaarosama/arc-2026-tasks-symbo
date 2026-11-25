"""Geometric error of a generated mesh against the Google Scanned Objects ground truth.

Runs on the operator laptop, never on a machine under test, so it cannot touch the timings.

Pipeline:
 1. Sample N points uniformly from each surface (area-weighted).
 2. Normalise each cloud: centre on its centroid, scale to unit RMS radius. Both steps are
    rotation-invariant, so normalising before alignment is safe. Absolute scale is not
    recoverable from one image, so we compare shape only.
 3. Align: the output's axis convention is unknown (GLB is Y-up, GSO OBJ is Z-up, and models
    pick their own canonical front), so try all 24 proper axis-aligned rotations, refine each
    with rigid ICP (no scaling, no reflection) and keep the best.
 4. Report symmetric Chamfer distance (mean nearest-neighbour distance, both directions,
    ×100 for display, lower is better) and F-score at tau = 0.05 (higher is better).

Sanity checks in tests/test_scoring.py: truth vs itself after a random rotation scores ~0.
"""
from __future__ import annotations

import itertools
import json
import sys
from dataclasses import asdict, dataclass

import numpy as np
import trimesh

try:                                    # scoring runs in the conductor's env; the llm / t2i / music venvs
    from scipy.spatial import cKDTree   # have no scipy, yet import the conductor's task modules (scripts)
except ImportError:
    cKDTree = None


@dataclass
class Score:
    chamfer: float          # mean NN distance, both directions, in units of RMS radius
    chamfer_x100: float     # display number
    fscore: float           # F-score at tau
    tau: float
    rotation_index: int
    points: int

    def as_dict(self) -> dict:
        return asdict(self)


def load_mesh(path: str) -> trimesh.Trimesh:
    # force="mesh" flattens a GLB scene graph into one mesh with node transforms applied
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"{path}: no triangle geometry")
    return mesh


def sample(mesh: trimesh.Trimesh, n: int, seed: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, dtype=np.float64)


def normalise(p: np.ndarray) -> np.ndarray:
    p = p - p.mean(axis=0)
    r = np.sqrt((p ** 2).sum(axis=1).mean())
    return p / (r if r > 0 else 1.0)


def proper_rotations() -> list[np.ndarray]:
    """The 24 rotation matrices that map axes to axes (det = +1)."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                m[i, j] = s
            if np.isclose(np.linalg.det(m), 1.0):
                mats.append(m)
    return mats


def _icp(src: np.ndarray, dst_tree: cKDTree, dst: np.ndarray, iters: int = 30) -> tuple[np.ndarray, float]:
    """Rigid point-to-point ICP (Kabsch). Returns transformed src and final mean NN distance."""
    cur = src.copy()
    prev = np.inf
    for _ in range(iters):
        d, idx = dst_tree.query(cur, workers=-1)
        m = dst[idx]
        mu_s, mu_d = cur.mean(0), m.mean(0)
        h = (cur - mu_s).T @ (m - mu_d)
        u, _, vt = np.linalg.svd(h)
        dfix = np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))])
        r = vt.T @ dfix @ u.T
        cur = (cur - mu_s) @ r.T + mu_d
        err = d.mean()
        if abs(prev - err) < 1e-6:
            break
        prev = err
    d, _ = dst_tree.query(cur, workers=-1)
    return cur, float(d.mean())


def score_points(pred: np.ndarray, truth: np.ndarray, tau: float = 0.05,
                 icp_points: int = 1000, seed: int = 0) -> Score:
    pred, truth = normalise(pred), normalise(truth)
    rng = np.random.default_rng(seed)
    pick = lambda a, n: a[rng.choice(len(a), min(n, len(a)), replace=False)]  # noqa: E731

    # Coarse: 24 starts on small subsamples (this is where the time goes).
    sub, t_sub = pick(pred, icp_points), pick(truth, 4 * icp_points)
    t_sub_tree = cKDTree(t_sub)
    best = (np.inf, -1, None)
    for i, rot in enumerate(proper_rotations()):
        _, err = _icp(sub @ rot.T, t_sub_tree, t_sub, iters=20)
        if err < best[0]:
            best = (err, i, rot)
    _, best_i, rot = best

    # Fine: one ICP on the full clouds from the best start, then measure.
    t_tree = cKDTree(truth)
    full, _ = _icp(pred @ rot.T, t_tree, truth, iters=50)
    d_pt, _ = t_tree.query(full, workers=-1)               # pred -> truth (accuracy)
    d_tp, _ = cKDTree(full).query(truth, workers=-1)       # truth -> pred (completeness)
    chamfer = 0.5 * (d_pt.mean() + d_tp.mean())
    precision = (d_pt < tau).mean()
    recall = (d_tp < tau).mean()
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return Score(chamfer=float(chamfer), chamfer_x100=round(float(chamfer) * 100, 2),
                 fscore=round(float(f), 3), tau=tau, rotation_index=best_i, points=len(pred))


def score_files(pred_path: str, truth_path: str, n: int = 20000, seed: int = 0) -> Score:
    pred = sample(load_mesh(pred_path), n, seed)
    truth = sample(load_mesh(truth_path), n, seed + 1)
    return score_points(pred, truth, seed=seed)


if __name__ == "__main__":
    print(json.dumps(score_files(sys.argv[1], sys.argv[2]).as_dict(), indent=2))
