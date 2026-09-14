"""lutherICPU Novel View and Camera Trajectory Generator.

Generates smooth 360-degree forensic orbit paths, flythroughs, and novel camera viewpoints.
"""
import math
from typing import List, Tuple, Optional
import numpy as np

from luther_core.types import CameraView, CameraIntrinsics


def generate_orbit_path(
    center: np.ndarray,
    radius: float,
    elevation_deg: float = 15.0,
    num_frames: int = 60,
    intrinsics: Optional[CameraIntrinsics] = None
) -> List[CameraView]:
    """Generates a smooth circular 360-degree orbit camera trajectory around the scene center."""
    if intrinsics is None:
        intrinsics = CameraIntrinsics(width=1280, height=720, fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)

    elevation_rad = math.radians(elevation_deg)
    views = []

    for i in range(num_frames):
        azimuth_rad = 2.0 * math.pi * (i / num_frames)

        # Eye position in world coordinates (assuming +Y is down/ground or +Z is up)
        eye_x = center[0] + radius * math.cos(azimuth_rad) * math.cos(elevation_rad)
        eye_z = center[2] + radius * math.sin(azimuth_rad) * math.cos(elevation_rad)
        eye_y = center[1] - radius * math.sin(elevation_rad)  # Camera height

        eye_pos = np.array([eye_x, eye_y, eye_z], dtype=np.float32)

        # Camera LookAt matrix: forward vector points from eye to center
        forward = center - eye_pos
        forward /= np.linalg.norm(forward)

        # World up is [0, -1, 0] in COLMAP convention
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= r_norm
        actual_up = np.cross(right, forward)

        # Rotation matrix R: transforms world to camera
        R = np.vstack([right, actual_up, forward]).astype(np.float32)
        tvec = -R @ eye_pos

        # Convert R to quaternion
        tr = np.trace(R)
        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        else:
            qw = 1.0
            qx = 0.0
            qy = 0.0
            qz = 0.0

        qvec = np.array([qw, qx, qy, qz], dtype=np.float32)

        views.append(CameraView(
            image_id=1000 + i,
            name=f"orbit_{i:04d}",
            qvec=qvec,
            tvec=tvec,
            intrinsics=intrinsics
        ))

    return views
