"""Acceptance check from the Task 2 spec: the scoring script must return ~0 for the ground
truth against itself — including after an arbitrary rotation, translation and scale, because
generated meshes come back in their own frame."""
import numpy as np
import trimesh

from arcdemo.scoring import proper_rotations, sample, score_points


def mug() -> trimesh.Trimesh:
    """Asymmetric test object with a thin handle, like the 'hard' tier."""
    body = trimesh.creation.cylinder(radius=0.4, height=1.0, sections=48)
    handle = trimesh.creation.torus(major_radius=0.25, minor_radius=0.05)
    handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    handle.apply_translation([0.55, 0, 0.1])
    return trimesh.util.concatenate([body, handle])


def test_24_proper_rotations():
    rots = proper_rotations()
    assert len(rots) == 24
    assert all(np.isclose(np.linalg.det(r), 1) for r in rots)


def test_truth_vs_itself_transformed_scores_near_zero():
    m = mug()
    truth = sample(m, 8000, seed=1)
    moved = m.copy()
    # a Y-up/Z-up style axis swap plus a small extra tilt, a shift and a different scale
    t = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    t = t @ trimesh.transformations.rotation_matrix(0.15, [0, 1, 0])
    moved.apply_transform(t)
    moved.apply_translation([3, -2, 1])
    moved.apply_scale(2.7)
    s = score_points(sample(moved, 8000, seed=2), truth)
    assert s.chamfer_x100 < 3.0, s
    assert s.fscore > 0.95, s


def test_different_shape_scores_clearly_worse():
    truth = sample(mug(), 8000, seed=1)
    blob = sample(trimesh.creation.icosphere(subdivisions=3), 8000, seed=2)
    good = score_points(sample(mug(), 8000, seed=3), truth)
    bad = score_points(blob, truth)
    assert bad.chamfer > 3 * good.chamfer, (good, bad)
