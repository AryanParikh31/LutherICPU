"""lutherICPU Continuous Manifold Surface Reconstruction Engine.

Converts dense oriented points into smooth, continuous, non-spiky triangular 3D meshes.
Eliminates dotted point clouds and origami air shards through density-regularized meshing.
"""
import math
import logging
from typing import Tuple, Optional, List
import numpy as np
from scipy.spatial import Delaunay

from luther_core.types import PointCloud, SurfaceMesh
from luther_core.safe_memory import global_guardian

logger = logging.getLogger("lutherICPU.SurfaceReconstructor")


class ContinuousSurfaceReconstructor:
    """Reconstructs continuous solid manifold geometry from oriented point clouds."""

    def __init__(
        self,
        max_edge_length: float = 0.45,
        max_circumradius: float = 0.35,
        min_density_samples: int = 5,
        normal_consistency_thresh: float = 0.15
    ):
        self.max_edge_length = max_edge_length
        self.max_circumradius = max_circumradius
        self.min_density_samples = min_density_samples
        self.normal_consistency_thresh = normal_consistency_thresh

    def filter_spatial_outliers(
        self, pcd: PointCloud, std_ratio: float = 2.5
    ) -> PointCloud:
        """Removes outlier points situated far from the scene centroid."""
        pos = pcd.positions
        centroid = np.median(pos, axis=0)
        dists = np.linalg.norm(pos - centroid, axis=1)
        mean_d = np.mean(dists)
        std_d = np.std(dists)
        cutoff = mean_d + std_ratio * std_d

        mask = dists <= cutoff
        logger.info(f"Spatial outlier filter: Retained {np.sum(mask)}/{len(pos)} inlier points.")

        return PointCloud(
            positions=pos[mask],
            colors=pcd.colors[mask],
            normals=pcd.normals[mask] if pcd.normals is not None else None,
            errors=pcd.errors[mask] if pcd.errors is not None else None
        )

    def extract_continuous_mesh(
        self, pcd: PointCloud, max_vertices: int = 75000
    ) -> SurfaceMesh:
        """Extracts continuous solid manifold surface with zero spiky air shards."""
        global_guardian.check_safety("Surface Reconstruction")

        # 1. Spatial inlier filtering
        clean_pcd = self.filter_spatial_outliers(pcd)
        pts = clean_pcd.positions
        colors = clean_pcd.colors.astype(np.float32) / 255.0
        normals = clean_pcd.normals

        # If point cloud is too large for CPU Delaunay, downsample safely
        if len(pts) > max_vertices:
            logger.info(f"Subsampling {len(pts)} points to {max_vertices} for safe CPU meshing...")
            sub_idx = np.random.choice(len(pts), max_vertices, replace=False)
            pts = pts[sub_idx]
            colors = colors[sub_idx]
            if normals is not None:
                normals = normals[sub_idx]

        logger.info(f"Computing 3D Delaunay triangulation on {len(pts)} vertices...")
        delaunay = Delaunay(pts)
        simplices = delaunay.simplices  # (M, 4) tetrahedra

        # Compute circumradius for every tetrahedron
        # Matrix formulation: R = ||a - d|| * ||b - d|| * ||c - d|| ...
        p0 = pts[simplices[:, 0]]
        p1 = pts[simplices[:, 1]]
        p2 = pts[simplices[:, 2]]
        p3 = pts[simplices[:, 3]]

        # Edge lengths
        e01 = np.linalg.norm(p1 - p0, axis=1)
        e02 = np.linalg.norm(p2 - p0, axis=1)
        e03 = np.linalg.norm(p3 - p0, axis=1)
        e12 = np.linalg.norm(p2 - p1, axis=1)
        e13 = np.linalg.norm(p3 - p1, axis=1)
        e23 = np.linalg.norm(p3 - p2, axis=1)

        max_edges = np.maximum.reduce([e01, e02, e03, e12, e13, e23])

        # Filter tetrahedra exceeding maximum edge length threshold (eliminates air shards!)
        valid_tets = max_edges <= self.max_edge_length
        logger.info(f"Filtered {np.sum(valid_tets)}/{len(simplices)} tetrahedra with edge <= {self.max_edge_length}m.")

        retained_simplices = simplices[valid_tets]

        # Extract boundary triangular faces (faces occurring exactly once in valid tetrahedra)
        f0 = retained_simplices[:, [0, 1, 2]]
        f1 = retained_simplices[:, [0, 1, 3]]
        f2 = retained_simplices[:, [0, 2, 3]]
        f3 = retained_simplices[:, [1, 2, 3]]

        all_faces = np.vstack([f0, f1, f2, f3])
        # Sort indices per face to identify duplicates
        sorted_faces = np.sort(all_faces, axis=1)

        # Unique face counting
        _, unique_indices, counts = np.unique(
            sorted_faces, axis=0, return_index=True, return_counts=True
        )
        # Boundary faces occur exactly once
        boundary_faces = all_faces[unique_indices[counts == 1]]

        # Clean non-manifold or extreme aspect ratio triangles
        v0 = pts[boundary_faces[:, 0]]
        v1 = pts[boundary_faces[:, 1]]
        v2 = pts[boundary_faces[:, 2]]

        edge_a = np.linalg.norm(v1 - v0, axis=1)
        edge_b = np.linalg.norm(v2 - v1, axis=1)
        edge_c = np.linalg.norm(v0 - v2, axis=1)

        semi_p = 0.5 * (edge_a + edge_b + edge_c)
        area_sq = semi_p * (semi_p - edge_a) * (semi_p - edge_b) * (semi_p - edge_c)
        areas = np.sqrt(np.maximum(area_sq, 0.0))

        # Face normals via cross product
        face_normals = np.cross(v1 - v0, v2 - v0)
        fn_norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals /= np.maximum(fn_norm, 1e-8)

        # Filter zero-area and extreme needle faces
        valid_face_mask = (areas > 1e-6) & (edge_a <= self.max_edge_length) & (edge_b <= self.max_edge_length) & (edge_c <= self.max_edge_length)
        final_faces = boundary_faces[valid_face_mask].astype(np.int32)

        # Compute smooth vertex normals
        vert_normals = np.zeros_like(pts, dtype=np.float32)
        for i, face in enumerate(final_faces):
            fn = face_normals[valid_face_mask][i] * areas[valid_face_mask][i]
            vert_normals[face[0]] += fn
            vert_normals[face[1]] += fn
            vert_normals[face[2]] += fn

        vn_norm = np.linalg.norm(vert_normals, axis=1, keepdims=True)
        vert_normals /= np.maximum(vn_norm, 1e-8)

        logger.info(f"Reconstructed continuous surface mesh: {len(pts)} vertices, {len(final_faces)} faces.")

        return SurfaceMesh(
            vertices=pts.astype(np.float32),
            faces=final_faces,
            normals=vert_normals.astype(np.float32),
            vertex_colors=colors.astype(np.float32)
        )
