"""lutherICPU Rigorous Camera and Epipolar Mathematics.

Provides projection, ray unprojection, epipolar geometry, triangulation,
and multi-view geometry primitives.
"""
import math
import numpy as np
from typing import Tuple, List, Optional
from luther_core.types import CameraIntrinsics, CameraView


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Converts a quaternion [w, x, y, z] to a 3x3 orthonormal rotation matrix."""
    w, x, y, z = q
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm > 1e-12:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]
    ], dtype=np.float32)


def project_points(points_3d: np.ndarray, view: CameraView) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projects 3D world coordinates into 2D pixel coordinates.

    Args:
        points_3d: (N, 3) float32 array of world coordinates.
        view: CameraView object.

    Returns:
        pixels: (N, 2) pixel coordinates (u, v).
        depths: (N,) positive depths along camera Z axis.
        valid_mask: (N,) boolean mask indicating points in front of camera and inside sensor bounds.
    """
    R = view.R
    t = view.tvec
    # Camera space: P_cam = R * P_world + t
    P_cam = (R @ points_3d.T).T + t  # (N, 3)
    z = P_cam[:, 2]

    eps = 1e-6
    u = (view.intrinsics.fx * P_cam[:, 0] / np.maximum(z, eps)) + view.intrinsics.cx
    v = (view.intrinsics.fy * P_cam[:, 1] / np.maximum(z, eps)) + view.intrinsics.cy

    pixels = np.column_stack([u, v]).astype(np.float32)
    valid_mask = (
        (z > 0.1) &
        (u >= 0) & (u < view.intrinsics.width) &
        (v >= 0) & (v < view.intrinsics.height)
    )

    return pixels, z, valid_mask


def unproject_pixels(
    pixels: np.ndarray, depths: np.ndarray, view: CameraView
) -> np.ndarray:
    """Unprojects 2D image coordinates with depths back into 3D world coordinates.

    Args:
        pixels: (N, 2) pixel coordinates (u, v).
        depths: (N,) depth values along optical Z axis.
        view: CameraView object.

    Returns:
        points_3d: (N, 3) world coordinates.
    """
    u = pixels[:, 0]
    v = pixels[:, 1]
    x_cam = (u - view.intrinsics.cx) * depths / view.intrinsics.fx
    y_cam = (v - view.intrinsics.cy) * depths / view.intrinsics.fy
    z_cam = depths

    P_cam = np.column_stack([x_cam, y_cam, z_cam])  # (N, 3)
    # P_world = R^T * (P_cam - t)
    R_inv = view.R.T
    points_3d = (R_inv @ (P_cam - view.tvec).T).T
    return points_3d.astype(np.float32)


def get_camera_rays(view: CameraView, subsample: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Generates ray origins and normalized direction vectors for every sensor pixel.

    Returns:
        origins: (H, W, 3) world space ray origins (camera center).
        directions: (H, W, 3) world space normalized ray direction vectors.
    """
    H, W = view.intrinsics.height // subsample, view.intrinsics.width // subsample
    fx, fy = view.intrinsics.fx / subsample, view.intrinsics.fy / subsample
    cx, cy = view.intrinsics.cx / subsample, view.intrinsics.cy / subsample

    u, v = np.meshgrid(
        np.arange(W, dtype=np.float32) + 0.5,
        np.arange(H, dtype=np.float32) + 0.5
    )

    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = np.ones_like(x_cam)

    dirs_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)
    norm = np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    dirs_cam = dirs_cam / np.maximum(norm, 1e-8)

    # Rotate into world space
    R_inv = view.R.T
    dirs_world = np.einsum("ij,hwj->hwi", R_inv, dirs_cam).astype(np.float32)
    origins = np.broadcast_to(view.center, (H, W, 3)).astype(np.float32)

    return origins, dirs_world


def compute_fundamental_matrix(view1: CameraView, view2: CameraView) -> np.ndarray:
    """Computes the 3x3 Fundamental Matrix F such that x2^T * F * x1 = 0."""
    # Relative pose from 1 to 2
    R1, t1 = view1.R, view1.tvec
    R2, t2 = view2.R, view2.tvec

    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1

    # Skew-symmetric matrix [t]_x
    tx, ty, tz = t_rel
    S = np.array([
        [0.0, -tz, ty],
        [tz, 0.0, -tx],
        [-ty, tx, 0.0]
    ], dtype=np.float32)

    # Essential matrix E = [t]_x * R
    E = S @ R_rel

    # Fundamental matrix F = K2^(-T) * E * K1^(-1)
    K1_inv = view1.intrinsics.K_inv
    K2_inv_T = view2.intrinsics.K_inv.T

    F = K2_inv_T @ E @ K1_inv
    return F.astype(np.float32)


def triangulate_two_views(
    pt1: np.ndarray, pt2: np.ndarray, view1: CameraView, view2: CameraView
) -> np.ndarray:
    """Triangulates a 3D point using Direct Linear Transform (DLT) from 2 calibrated views."""
    P1 = view1.projection_matrix
    P2 = view2.projection_matrix

    A = np.array([
        pt1[0] * P1[2] - P1[0],
        pt1[1] * P1[2] - P1[1],
        pt2[0] * P2[2] - P2[0],
        pt2[1] * P2[2] - P2[1],
    ])

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / (X[3] + 1e-12)
    return X.astype(np.float32)
