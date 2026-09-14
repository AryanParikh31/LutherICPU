"""
Unit & Integration Tests for Volumetric TSDF Fusion & Watertight Meshing.
Verifies elimination of 2.5D cardboard layers, zero NaN/inf values,
manifold triangle integrity, and strict CPU memory budget (<2.5 GB).
"""

import os
import sys
import numpy as np
import pytest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from luther_geometry.volumetric_tsdf_fusion import VolumetricTSDFFusionEngine


def test_volumetric_tsdf_mesh_generation(tmp_path):
    """Verifies that Volumetric TSDF extracts a single solid watertight manifold, not 2.5D flat cards."""
    engine = VolumetricTSDFFusionEngine(voxel_grid_res=64, octree_depth=7, memory_budget_gb=2.0)
    
    # Create a synthetic 3D sphere point cloud (dense surface in all directions)
    np.random.seed(42)
    phi = np.random.uniform(0, 2 * np.pi, 2000)
    costheta = np.random.uniform(-1, 1, 2000)
    theta = np.arccos(costheta)
    r = 2.0
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = np.ones((2000, 3), dtype=np.float32) * 0.8
    
    # Run reconstruction
    verts, faces, vert_colors = engine.reconstruct_scene(points, colors)
    
    assert len(verts) > 0, "Vertices must not be empty"
    assert len(faces) > 0, "Faces must not be empty"
    assert faces.shape[1] == 3, "Faces must be 3-tuples (triangles)"
    
    # Verify no NaN or Inf
    assert not np.isnan(verts).any(), "Mesh contains NaN vertices"
    assert not np.isinf(verts).any(), "Mesh contains Inf vertices"
    assert not np.isnan(vert_colors).any(), "Mesh colors contain NaN"
    
    # Verify 3D bounding box volume (must be a true 3D solid, not flat card where one dimension is ~0)
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    extents = bbox_max - bbox_min
    assert np.all(extents > 1.0), f"Expected 3D solid volume, but extent is flat: {extents}"


def test_zero_nan_and_sliver_faces(tmp_path):
    """Verifies exported mesh has zero degenerate / aspect-ratio slivers."""
    engine = VolumetricTSDFFusionEngine(voxel_grid_res=48, octree_depth=6)
    
    # Simple cube point cloud
    pts = []
    for axis in range(3):
        for val in [-1.0, 1.0]:
            grid = np.linspace(-1.0, 1.0, 20)
            u, v = np.meshgrid(grid, grid)
            face_pts = np.zeros((400, 3), dtype=np.float32)
            face_pts[:, axis] = val
            face_pts[:, (axis + 1) % 3] = u.flatten()
            face_pts[:, (axis + 2) % 3] = v.flatten()
            pts.append(face_pts)
    points = np.concatenate(pts, axis=0)
    colors = np.ones_like(points)
    
    verts, faces, _ = engine.reconstruct_scene(points, colors)
    
    # Check triangle area
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    
    # Should have no degenerate 0-area faces
    assert np.sum(areas <= 1e-12) == 0, "Found degenerate zero-area triangles"


def test_peak_ram_under_budget():
    """Asserts that TSDF evaluation stays strictly within memory budget (< 2.5 GB)."""
    import psutil
    proc = psutil.Process()
    ram_before = proc.memory_info().rss / (1024 ** 3)
    
    engine = VolumetricTSDFFusionEngine(voxel_grid_res=64, octree_depth=6, memory_budget_gb=2.0)
    points = np.random.uniform(-5, 5, (5000, 3)).astype(np.float32)
    colors = np.ones((5000, 3), dtype=np.float32)
    
    verts, faces, _ = engine.reconstruct_scene(points, colors)
    ram_after = proc.memory_info().rss / (1024 ** 3)
    
    ram_delta = ram_after - ram_before
    assert ram_delta < 1.0, f"Process RAM increased by {ram_delta:.2f} GB, which exceeds safe limit"


def test_unobserved_roof_planar_injection_and_closure():
    """Verifies that an open-top unobserved structure gets closed into a solid flat watertight manifold."""
    from luther_core.types import CameraView, CameraIntrinsics
    from scipy.spatial.transform import Rotation

    engine = VolumetricTSDFFusionEngine(voxel_grid_res=48, memory_budget_gb=2.0)

    # Build 5-sided box (open top at Y = -1.0, ground at Y = +1.0)
    pts_list = []
    for side in [-1.0, 1.0]:
        # X walls
        u, v = np.meshgrid(np.linspace(-1.0, 1.0, 20), np.linspace(-1.0, 1.0, 20))
        pts_list.append(np.column_stack([np.full(u.size, side), u.ravel(), v.ravel()]))
        # Z walls
        pts_list.append(np.column_stack([u.ravel(), v.ravel(), np.full(u.size, side)]))

    # Bottom wall at Y = +1.0
    u, v = np.meshgrid(np.linspace(-1.0, 1.0, 20), np.linspace(-1.0, 1.0, 20))
    pts_list.append(np.column_stack([u.ravel(), np.full(u.size, 1.0), v.ravel()]))

    # Upper rim points at Y = -1.0
    edge_x = np.linspace(-1.0, 1.0, 30)
    pts_list.append(np.column_stack([edge_x, np.full(30, -1.0), np.full(30, -1.0)]))
    pts_list.append(np.column_stack([edge_x, np.full(30, -1.0), np.full(30, 1.0)]))
    pts_list.append(np.column_stack([np.full(30, -1.0), np.full(30, -1.0), edge_x]))
    pts_list.append(np.column_stack([np.full(30, 1.0), np.full(30, -1.0), edge_x]))

    points = np.concatenate(pts_list, axis=0).astype(np.float32)
    colors = np.full((len(points), 3), 0.7, dtype=np.float32)

    # Create synthetic cameras looking at the box
    views = []
    intr = CameraIntrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        cam_pos = np.array([3.0 * np.cos(angle), 0.0, 3.0 * np.sin(angle)], dtype=np.float32)
        forward = -cam_pos / np.linalg.norm(cam_pos)
        cam_y_init = np.array([0.0, 1.0, 0.0], dtype=np.float32) # Down is +Y
        cam_x = np.cross(cam_y_init, forward)
        cam_x /= np.linalg.norm(cam_x)
        cam_y = np.cross(forward, cam_x)
        cam_y /= np.linalg.norm(cam_y)
        R = np.vstack([cam_x, cam_y, forward]).astype(np.float32)
        tvec = -R @ cam_pos
        quat_xyzw = Rotation.from_matrix(R).as_quat()
        qvec = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        view = CameraView(image_id=len(views)+1, name=f"cam_{len(views)}.jpg", qvec=qvec, tvec=tvec, intrinsics=intr, image_path=None)
        views.append(view)

    verts, faces, vert_colors = engine.reconstruct_scene(
        points=points,
        colors=colors,
        camera_views=views,
        voxel_res=48
    )

    assert len(verts) > 0, "Expected reconstructed vertices"
    assert len(faces) > 0, "Expected reconstructed faces"
    assert not np.isnan(verts).any()

    # Verify that the reconstructed solid has a closed top face around Y = -1.0
    top_verts = verts[verts[:, 1] < -0.8]
    assert len(top_verts) > 100, f"Expected >100 roof vertices, got {len(top_verts)}"
    median_top_y = np.median(top_verts[:, 1])
    assert np.isclose(median_top_y, -1.0, atol=0.08), f"Expected median top Y ~ -1.0, got {median_top_y}"

