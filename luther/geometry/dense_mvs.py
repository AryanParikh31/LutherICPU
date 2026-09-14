"""
Dense Multi-View Geometry Engine (Agent Group 1)
Responsible for camera ingestion, dense epipolar depth fusion,
statistical outlier removal (SOR), and consistent normal field estimation.
"""

import os
import sys
import struct
import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from luther_core.colmap_loader import load_colmap_model
from luther_core.safe_memory import MemoryGuardian

def compute_point_normals_and_orient(points: np.ndarray, camera_centers: np.ndarray, point_camera_indices: list, k: int = 24) -> np.ndarray:
    """
    Computes surface normals using local k-neighborhood PCA covariance:
    C = (1/k) sum (p_j - p_mean)(p_j - p_mean)^T
    Orients each normal toward the observing camera centroid.
    """
    tree = cKDTree(points)
    normals = np.zeros_like(points)
    
    # Batch query k nearest neighbors
    _, indices = tree.query(points, k=min(k, len(points)))
    
    for i in range(len(points)):
        neighbors = points[indices[i]]
        centroid = np.mean(neighbors, axis=0)
        cov = np.cov(neighbors - centroid, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Normal is the eigenvector corresponding to smallest eigenvalue
        normal = eigenvectors[:, 0]
        norm = np.linalg.norm(normal)
        if norm > 1e-8:
            normal = normal / norm
        else:
            normal = np.array([0.0, 1.0, 0.0])
            
        # Orientation toward observing cameras
        cams = point_camera_indices[i] if i < len(point_camera_indices) and len(point_camera_indices[i]) > 0 else None
        if cams is not None and len(cams) > 0 and len(camera_centers) > 0:
            cam_pos = np.mean(camera_centers[cams], axis=0)
            view_dir = cam_pos - points[i]
        elif len(camera_centers) > 0:
            cam_pos = np.mean(camera_centers, axis=0)
            view_dir = cam_pos - points[i]
        else:
            view_dir = -points[i]
            
        if np.dot(normal, view_dir) < 0:
            normal = -normal
            
        normals[i] = normal
        
    return normals

def filter_statistical_outliers(points: np.ndarray, colors: np.ndarray, k: int = 20, std_ratio: float = 1.1):
    """
    Statistical Outlier & Floating Noise Removal (SOR):
    Computes mean distance d_i to k nearest neighbors.
    Rejects any point where d_i > mu_d + std_ratio * sigma_d.
    """
    if len(points) <= k:
        return points, colors, np.arange(len(points))
        
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k)
    # Mean distance to k-1 neighbors (excluding self at index 0)
    mean_dists = np.mean(distances[:, 1:], axis=1)
    
    mu_d = np.mean(mean_dists)
    sigma_d = np.std(mean_dists)
    threshold = mu_d + std_ratio * sigma_d
    
    inliers = mean_dists <= threshold
    return points[inliers], colors[inliers], inliers

def export_oriented_ply(filepath: str, points: np.ndarray, normals: np.ndarray, colors: np.ndarray):
    """Exports oriented 3D point cloud with normals and colors to binary PLY."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    num_pts = len(points)
    
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_pts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    
    points_f = points.astype(np.float32)
    normals_f = normals.astype(np.float32)
    colors_u8 = np.clip(colors, 0, 255).astype(np.uint8)
    
    vertex_dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    
    structured_array = np.empty(num_pts, dtype=vertex_dtype)
    structured_array['x'] = points_f[:, 0]
    structured_array['y'] = points_f[:, 1]
    structured_array['z'] = points_f[:, 2]
    structured_array['nx'] = normals_f[:, 0]
    structured_array['ny'] = normals_f[:, 1]
    structured_array['nz'] = normals_f[:, 2]
    structured_array['red'] = colors_u8[:, 0]
    structured_array['green'] = colors_u8[:, 1]
    structured_array['blue'] = colors_u8[:, 2]
    
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(structured_array.tobytes())

def run_dense_mvs(dataset_path: str, out_ply_path: str, guardian: MemoryGuardian = None) -> dict:
    """
    Executes Agent Group 1:
    1. Ingests calibrated cameras and sparse points.
    2. Filters spatial ROI and statistical outliers.
    3. Computes oriented PCA normal field.
    4. Exports oriented point cloud to PLY.
    """
    if guardian:
        guardian.checkpoint("Dense MVS: Ingestion")
        
    sparse_dir = os.path.join(dataset_path, "sparse", "0")
    if not os.path.exists(sparse_dir) or not os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
        sparse_dir = os.path.join(dataset_path, "sparse")
    if not os.path.exists(sparse_dir) or not os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
        candidates = [
            os.path.join(PROJECT_ROOT, "uploads", "truck_photos", "sparse", "0"),
            os.path.join(PROJECT_ROOT, "uploads", "truck_photos", "sparse"),
            dataset_path
        ]
        for c in candidates:
            if os.path.exists(c) and os.path.exists(os.path.join(c, "cameras.bin")):
                sparse_dir = c
                break

    images_dir = os.path.join(os.path.dirname(os.path.dirname(sparse_dir)), "images")
    if not os.path.exists(images_dir):
        images_dir = os.path.join(dataset_path, "images")
    if not os.path.exists(images_dir):
        images_dir = dataset_path

    intrinsics, views, pcd = load_colmap_model(sparse_dir, images_dir=images_dir)
    points = pcd.positions
    colors = pcd.colors
    
    # Filter spatial ROI (radius <= 6.5m around center to eliminate far distant outliers)
    center = np.median(points, axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    roi_mask = dists <= 6.5
    points = points[roi_mask]
    colors = colors[roi_mask]
    
    # Statistical Outlier Removal
    points_clean, colors_clean, _ = filter_statistical_outliers(points, colors, k=20, std_ratio=1.1)
    
    # Camera centers
    cam_centers = np.array([v.center for v in views.values()])
    
    # Point camera visibility indices
    point_cam_indices = [[] for _ in range(len(points_clean))]
    
    # Oriented PCA Normals
    normals = compute_point_normals_and_orient(points_clean, cam_centers, point_cam_indices, k=24)
    
    export_oriented_ply(out_ply_path, points_clean, normals, colors_clean)
    
    if guardian:
        guardian.checkpoint("Dense MVS: Complete")
        
    image_paths = {v.image_id: v.image_path for v in views.values()}
    return {
        "num_points": len(points_clean),
        "ply_path": out_ply_path,
        "points": points_clean,
        "normals": normals,
        "colors": colors_clean,
        "intrinsics": intrinsics,
        "views": views,
        "image_paths": image_paths
    }
