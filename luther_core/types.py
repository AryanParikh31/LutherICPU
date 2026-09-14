"""lutherICPU Core Types and Data Models.

Defines point clouds, cameras, meshes, texture atlases, and forensic markers.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole and radial distortion camera parameters."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "PINHOLE"
    params: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    @property
    def K(self) -> np.ndarray:
        """3x3 Intrinsic calibration matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

    @property
    def K_inv(self) -> np.ndarray:
        """Inverse intrinsic matrix."""
        return np.linalg.inv(self.K)


@dataclass
class CameraView:
    """Single camera pose and corresponding image metadata."""
    image_id: int
    name: str
    qvec: np.ndarray  # Quaternion (w, x, y, z)
    tvec: np.ndarray  # Translation vector (tx, ty, tz)
    intrinsics: CameraIntrinsics
    image_path: Optional[str] = None
    point3d_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    xys: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))

    @property
    def R(self) -> np.ndarray:
        """3x3 Rotation matrix. Uses _R override if set, otherwise computes from quaternion."""
        if hasattr(self, '_R') and self._R is not None:
            return self._R
        w, x, y, z = self.qvec
        return np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
        ], dtype=np.float32)

    @property
    def center(self) -> np.ndarray:
        """3D Camera optical center in world coordinates: C = -R^T * t."""
        return -self.R.T @ self.tvec

    @property
    def viewing_direction(self) -> np.ndarray:
        """Forward viewing direction vector in world coordinates (COLMAP: +Z is forward)."""
        return self.R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

    @property
    def up_vector(self) -> np.ndarray:
        """Up vector in world coordinates (COLMAP: -Y is up)."""
        return self.R.T @ np.array([0.0, -1.0, 0.0], dtype=np.float32)

    @property
    def projection_matrix(self) -> np.ndarray:
        """3x4 Projection matrix P = K * [R | t]."""
        Rt = np.hstack([self.R, self.tvec.reshape(3, 1)])
        return self.intrinsics.K @ Rt


@dataclass
class PointCloud:
    """3D point cloud with positions, normals, colors, and confidence."""
    positions: np.ndarray  # (N, 3) float32
    colors: np.ndarray  # (N, 3) uint8 or float32 [0..1]
    normals: Optional[np.ndarray] = None  # (N, 3) float32
    errors: Optional[np.ndarray] = None  # (N,) float32
    confidence: Optional[np.ndarray] = None  # (N,) float32

    def __len__(self) -> int:
        return len(self.positions)


@dataclass
class SurfaceMesh:
    """Continuous 3D triangular manifold mesh."""
    vertices: np.ndarray  # (V, 3) float32
    faces: np.ndarray  # (F, 3) int32
    normals: Optional[np.ndarray] = None  # (V, 3) float32
    uvs: Optional[np.ndarray] = None  # (F, 3, 2) or (V, 2) float32
    vertex_colors: Optional[np.ndarray] = None  # (V, 3) float32 [0..1]
    texture_map: Optional[np.ndarray] = None  # (H, W, 3) uint8

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)


@dataclass
class ForensicEvidence:
    """Forensic evidence tag for courtroom and insurance analysis."""
    evidence_id: str
    title: str
    description: str
    position_3d: np.ndarray  # (3,) float32
    category: str  # e.g., 'IMPACT_DAMAGE', 'TIRE_SKID', 'BALLISTICS', 'DEBRIS'
    measurements: Dict[str, float] = field(default_factory=dict)
    confidence_score: float = 1.0
    source_camera_ids: List[int] = field(default_factory=list)
    timestamp: str = ""
