"""lutherICPU Core Package."""
from luther_core.types import (
    CameraIntrinsics,
    CameraView,
    PointCloud,
    SurfaceMesh,
    ForensicEvidence
)
from luther_core.safe_memory import MemoryGuardian, global_guardian

__all__ = [
    "CameraIntrinsics",
    "CameraView",
    "PointCloud",
    "SurfaceMesh",
    "ForensicEvidence",
    "MemoryGuardian",
    "global_guardian",
    "quat_to_rot_matrix",
    "project_points",
    "unproject_pixels",
    "get_camera_rays",
    "compute_fundamental_matrix",
    "triangulate_two_views",
    "load_colmap_model",
    "export_colmap_sparse",
    "ImageQualityController",
    "compute_laplacian_variance",
    "compute_sha256",
    "StructureFromMotionEngine",
    "extract_sift_features",
    "match_sift_features",
    "triangulate_dlt_point",
    "GaussianSplattingEngine",
]


def __getattr__(name: str):
    if name in (
        "quat_to_rot_matrix",
        "project_points",
        "unproject_pixels",
        "get_camera_rays",
        "compute_fundamental_matrix",
        "triangulate_two_views",
    ):
        from luther_core import camera
        val = getattr(camera, name)
        globals()[name] = val
        return val
    elif name in ("load_colmap_model", "export_colmap_sparse"):
        from luther_core import colmap_loader
        val = getattr(colmap_loader, name)
        globals()[name] = val
        return val
    elif name in ("ImageQualityController", "compute_laplacian_variance", "compute_sha256"):
        from luther_core import ingestion
        val = getattr(ingestion, name)
        globals()[name] = val
        return val
    elif name in ("StructureFromMotionEngine", "extract_sift_features", "match_sift_features", "triangulate_dlt_point"):
        from luther_core import sfm
        val = getattr(sfm, name)
        globals()[name] = val
        return val
    elif name == "GaussianSplattingEngine":
        from luther_core.gaussian_splatting_engine import GaussianSplattingEngine
        globals()[name] = GaussianSplattingEngine
        return GaussianSplattingEngine
    raise AttributeError(f"module 'luther_core' has no attribute '{name}'")
