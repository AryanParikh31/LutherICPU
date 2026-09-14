"""Unit tests for lutherICPU Core Mathematics and Epipolar Geometry."""
import math
import numpy as np
import pytest

from luther_core.types import CameraIntrinsics, CameraView
from luther_core.camera import (
    quat_to_rot_matrix,
    project_points,
    unproject_pixels,
    compute_fundamental_matrix,
    triangulate_two_views
)


def test_quaternion_orthonormality():
    """Verifies that quat_to_rot_matrix outputs valid SO(3) orthonormal rotation matrices."""
    # Test arbitrary non-trivial quaternion
    q = np.array([math.cos(math.pi / 6), 0.0, math.sin(math.pi / 6), 0.0], dtype=np.float32)
    R = quat_to_rot_matrix(q)

    # Check R * R^T = I
    identity = R @ R.T
    assert np.allclose(identity, np.eye(3), atol=1e-5)

    # Check det(R) = 1
    det = np.linalg.det(R)
    assert pytest.approx(det, rel=1e-5) == 1.0


def test_projection_and_unprojection_roundtrip():
    """Verifies that 3D world points project to pixels and unproject back to identical 3D points."""
    intrinsics = CameraIntrinsics(width=1920, height=1080, fx=1200.0, fy=1200.0, cx=960.0, cy=540.0)
    view = CameraView(
        image_id=1,
        name="test_cam",
        qvec=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        tvec=np.array([0.0, 0.0, 5.0], dtype=np.float32),
        intrinsics=intrinsics
    )

    # Points in front of camera (Z > 0 in camera space)
    orig_pts = np.array([
        [0.0, 0.0, 2.0],
        [1.5, -0.5, 3.0],
        [-2.0, 1.0, 4.0]
    ], dtype=np.float32)

    pixels, depths, valid_mask = project_points(orig_pts, view)
    assert np.all(valid_mask)
    assert np.all(depths > 0)

    reconstructed_pts = unproject_pixels(pixels, depths, view)
    assert np.allclose(orig_pts, reconstructed_pts, atol=1e-4)


def test_fundamental_matrix_epipolar_constraint():
    """Verifies that calibrated stereo views satisfy epipolar constraint x2^T * F * x1 = 0."""
    intrinsics = CameraIntrinsics(width=1000, height=1000, fx=800.0, fy=800.0, cx=500.0, cy=500.0)
    view1 = CameraView(
        image_id=1,
        name="cam1",
        qvec=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        tvec=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        intrinsics=intrinsics
    )
    view2 = CameraView(
        image_id=2,
        name="cam2",
        qvec=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        tvec=np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Baseline along X
        intrinsics=intrinsics
    )

    F = compute_fundamental_matrix(view1, view2)
    assert F.shape == (3, 3)

    # 3D test point
    X = np.array([[0.5, 0.2, 3.0]], dtype=np.float32)
    pix1, _, _ = project_points(X, view1)
    pix2, _, _ = project_points(X, view2)

    x1_h = np.array([pix1[0, 0], pix1[0, 1], 1.0])
    x2_h = np.array([pix2[0, 0], pix2[0, 1], 1.0])

    epipolar_error = abs(x2_h @ F @ x1_h)
    assert epipolar_error < 1e-3
