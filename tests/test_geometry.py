"""Unit tests for lutherICPU Continuous Surface Reconstruction and Manifold Meshing."""
import numpy as np
import pytest

from luther_core.types import PointCloud, SurfaceMesh
from luther_geometry.surface_reconstructor import ContinuousSurfaceReconstructor
from luther_geometry.manifold_cleaner import build_boundary_cage


def test_surface_reconstruction_from_cube_points():
    """Verifies that surface reconstructor generates continuous manifold mesh without air shards."""
    # Generate points on the surface of a 3D unit sphere/cube
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    u_grid, v_grid = np.meshgrid(u, v)

    r = 1.0
    x = (r * np.sin(v_grid) * np.cos(u_grid)).flatten()
    y = (r * np.sin(v_grid) * np.sin(u_grid)).flatten()
    z = (r * np.cos(v_grid)).flatten()

    pts = np.column_stack([x, y, z]).astype(np.float32)
    cols = np.full((len(pts), 3), 200, dtype=np.uint8)
    normals = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    pcd = PointCloud(positions=pts, colors=cols, normals=normals.astype(np.float32))

    reconstructor = ContinuousSurfaceReconstructor(max_edge_length=0.6, max_circumradius=0.5)
    mesh = reconstructor.extract_continuous_mesh(pcd, max_vertices=1000)

    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert mesh.faces.shape[1] == 3  # Valid triangular faces
    assert mesh.normals.shape == mesh.vertices.shape


def test_boundary_cage_bounds():
    """Verifies that 3D boundary cage strictly encloses all scene geometry."""
    pts = np.array([
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0]
    ], dtype=np.float32)

    cage = build_boundary_cage(pts, margin_multiplier=1.2)
    assert "cage_vertices" in cage
    assert "cage_faces" in cage
    assert cage["safe_radius"] > np.max(np.linalg.norm(pts, axis=1))


def test_estimate_scene_up_vector():
    """Verifies that scene up-vector is accurately recovered from camera extrinsics."""
    from scipy.spatial.transform import Rotation
    from luther_core.types import CameraView, CameraIntrinsics
    from luther_geometry.manifold_cleaner import estimate_scene_up_vector

    # Create synthetic cameras looking at origin around a circle in X-Z plane, with Y pointing down in camera frame
    views = []
    intr = CameraIntrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        cam_pos = np.array([3.0 * np.cos(angle), 0.0, 3.0 * np.sin(angle)], dtype=np.float32)
        forward = -cam_pos / np.linalg.norm(cam_pos)
        world_up = np.array([0.0, -1.0, 0.0], dtype=np.float32) # -Y is up in world
        # In camera frame: +X is right, +Y is down, +Z is forward
        # If camera looks forward along 'forward' with top pointing along world_up:
        # cam_x (right) = forward x (-world_up)
        cam_y_init = -world_up
        cam_x = np.cross(cam_y_init, forward)
        cam_x /= np.linalg.norm(cam_x)
        cam_y = np.cross(forward, cam_x) # Camera +Y points towards world_down
        cam_y /= np.linalg.norm(cam_y)
        R = np.vstack([cam_x, cam_y, forward]).astype(np.float32)
        tvec = -R @ cam_pos
        quat_xyzw = Rotation.from_matrix(R).as_quat()
        qvec = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        view = CameraView(image_id=len(views)+1, name=f"cam_{len(views)}.jpg", qvec=qvec, tvec=tvec, intrinsics=intr, image_path=None)
        views.append(view)

    up_vec = estimate_scene_up_vector(views)
    assert np.allclose(up_vec, [0.0, -1.0, 0.0], atol=1e-3), f"Expected [0, -1, 0], got {up_vec}"


def test_extract_and_synthesize_roof_surface():
    """Verifies RANSAC roof plane extraction and 2D convex hull synthesis on open-top vehicle geometry."""
    from luther_geometry.manifold_cleaner import (
        extract_roof_plane_and_boundary,
        synthesize_planar_roof_points,
        inject_unobserved_roof_surface
    )

    # Generate synthetic truck body (box open at top Y = -1.0, ground at Y = +1.0)
    # 4 side walls:
    pts_list = []
    for side in [-1.0, 1.0]:
        # X sides
        u, v = np.meshgrid(np.linspace(-1.0, 1.0, 25), np.linspace(-1.0, 1.0, 25))
        wall_x = np.column_stack([np.full(u.size, side), u.ravel(), v.ravel()])
        pts_list.append(wall_x)
        # Z sides
        wall_z = np.column_stack([u.ravel(), v.ravel(), np.full(u.size, side)])
        pts_list.append(wall_z)

    # Bottom at Y = +1.0
    u, v = np.meshgrid(np.linspace(-1.0, 1.0, 25), np.linspace(-1.0, 1.0, 25))
    bot = np.column_stack([u.ravel(), np.full(u.size, 1.0), v.ravel()])
    pts_list.append(bot)

    # Upper perimeter rim points at Y = -1.0 (edges of the open roof)
    edge_x = np.linspace(-1.0, 1.0, 40)
    rim1 = np.column_stack([edge_x, np.full(40, -1.0), np.full(40, -1.0)])
    rim2 = np.column_stack([edge_x, np.full(40, -1.0), np.full(40, 1.0)])
    rim3 = np.column_stack([np.full(40, -1.0), np.full(40, -1.0), edge_x])
    rim4 = np.column_stack([np.full(40, 1.0), np.full(40, -1.0), edge_x])
    pts_list.extend([rim1, rim2, rim3, rim4])

    points = np.concatenate(pts_list, axis=0).astype(np.float32)
    colors = np.full((len(points), 3), 0.7, dtype=np.float32)

    up_vector = np.array([0.0, -1.0, 0.0], dtype=np.float32) # -Y is up

    # 1. Extract roof plane
    roof_info = extract_roof_plane_and_boundary(
        points=points,
        colors=colors,
        up_vector=up_vector,
        percentile_min=88.0,
        percentile_max=100.0,
        ransac_thresh=0.05
    )

    assert roof_info is not None, "Failed to extract roof plane"
    # Normal should be [0, -1, 0]
    assert np.dot(roof_info["plane_normal"], up_vector) > 0.95
    # Roof height Y should be approximately -1.0
    roof_y = roof_info["origin_p0"][1]
    assert np.isclose(roof_y, -1.0, atol=0.08), f"Expected roof Y ~ -1.0, got {roof_y}"

    # 2. Synthesize roof points
    synth_pts, synth_cols, synth_norms = synthesize_planar_roof_points(roof_info, grid_step=0.08)
    assert len(synth_pts) > 50, f"Expected >50 roof points, got {len(synth_pts)}"
    assert not np.isnan(synth_pts).any()
    assert np.allclose(synth_pts[:, 1], -1.0, atol=0.08)

    # 3. Test end-to-end injection
    aug_pts, aug_cols = inject_unobserved_roof_surface(
        points, colors, camera_views=None, grid_step=0.08, percentile_min=88.0
    )
    assert len(aug_pts) > len(points)
    # Check that roof interior now has points (e.g. at [0.0, -1.0, 0.0])
    dists_to_center = np.linalg.norm(aug_pts - np.array([0.0, -1.0, 0.0]), axis=1)
    assert np.min(dists_to_center) < 0.15, "Roof center has no injected points"

