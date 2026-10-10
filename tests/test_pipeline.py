"""
tests/test_pipeline.py — Master Production Test Suite for LutherICPU 3D Gaussian Splatting Pipeline.

Verifies:
1. test_camera_matrix_projection: World-to-camera transformation and perspective intrinsics projection.
2. test_gaussian_covariance_positive_semidefinite: 3D covariance construction produces symmetric positive-definite matrices.
3. test_forward_backward_gradient_flow: 5 real optimization steps on a synthetic 64x64 RGB patch verifying non-zero gradients.
4. test_memory_boundary: Memory residency bounds and zero memory leakage across iterations.
5. test_ply_export_integrity: Standard Inria 3DGS binary PLY format header, properties, and payload integrity.
"""

import os
import sys
import math
import struct
import tempfile
import psutil
import pytest
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from luther_core.camera import quat_to_rot_matrix, project_points, unproject_pixels
from luther_core.types import CameraView, CameraIntrinsics
from luther_core.cuda_gaussian_trainer import (
    CUDAGaussianTrainer,
    GaussianModel,
    compute_covariance_3d,
    quat_to_rot_matrix_torch,
    get_ssim_torch,
    SH_C0
)
from luther_core.gaussian_splatting_engine import GaussianSplattingEngine


def test_camera_matrix_projection():
    """
    Test 1: Verifies world-to-camera transformation and perspective projection matrix math.
    """
    width, height = 640, 480
    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0
    intrinsics = CameraIntrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    # Camera at origin looking along +Z
    qvec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity rotation
    tvec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cam = CameraView(image_id=1, name="test_cam", qvec=qvec, tvec=tvec, intrinsics=intrinsics)
    cam._R = np.eye(3, dtype=np.float32)

    # 1. Point on optical axis at Z = 5.0 should project exactly to principal point (cx, cy)
    p_center = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
    pixels, depths, valid = project_points(p_center, cam)
    assert valid[0], "Point in front of camera should be valid"
    assert math.isclose(depths[0], 5.0, rel_tol=1e-5)
    assert math.isclose(pixels[0, 0], cx, abs_tol=1e-4), f"Expected cx={cx}, got {pixels[0, 0]}"
    assert math.isclose(pixels[0, 1], cy, abs_tol=1e-4), f"Expected cy={cy}, got {pixels[0, 1]}"

    # 2. Point displaced by delta_x = 1.0 at Z = 5.0
    # Expected u = fx * 1.0 / 5.0 + cx = 500 * 0.2 + 320 = 420
    p_right = np.array([[1.0, 0.0, 5.0]], dtype=np.float32)
    pixels_r, _, _ = project_points(p_right, cam)
    assert math.isclose(pixels_r[0, 0], 420.0, abs_tol=1e-4)
    assert math.isclose(pixels_r[0, 1], cy, abs_tol=1e-4)

    # 3. Roundtrip unprojection consistency: X_world == unproject(project(X_world))
    pts_3d = np.array([
        [0.5, -0.3, 4.0],
        [-1.2, 0.8, 6.5],
        [0.0, 0.0, 2.5]
    ], dtype=np.float32)
    proj_pix, proj_z, val = project_points(pts_3d, cam)
    reconstructed_3d = unproject_pixels(proj_pix, proj_z, cam)
    assert np.allclose(pts_3d, reconstructed_3d, atol=1e-4), "Unprojection did not match original 3D coordinates"

    # 4. Point behind camera (Z < 0) must be flagged as invalid
    p_behind = np.array([[0.0, 0.0, -2.0]], dtype=np.float32)
    _, _, val_behind = project_points(p_behind, cam)
    assert not val_behind[0], "Points with negative depth must not be marked valid"


def test_gaussian_covariance_positive_semidefinite():
    """
    Test 2: Verifies that scale and quaternion conversions produce valid, non-singular 3D covariance matrices.
    Sigma = R * S * S^T * R^T must be symmetric positive-definite (all eigenvalues > 0).
    """
    torch.manual_seed(42)
    N = 32

    # Generate random linear scales > 0 and arbitrary quaternions
    linear_scales = torch.exp(torch.randn(N, 3) * 0.5 - 2.0)
    raw_quats = torch.randn(N, 4)
    norm_quats = F.normalize(raw_quats, p=2, dim=-1)

    covs = compute_covariance_3d(linear_scales, norm_quats)  # [N, 3, 3]

    assert covs.shape == (N, 3, 3), f"Expected shape (N, 3, 3), got {covs.shape}"

    for i in range(N):
        sigma = covs[i]
        # 1. Symmetry check: Sigma == Sigma^T
        diff_sym = torch.max(torch.abs(sigma - sigma.t())).item()
        assert diff_sym < 1e-6, f"Covariance matrix is not symmetric (max error: {diff_sym})"

        # 2. Positive-definite check: all eigenvalues must be strictly positive
        eigenvalues = torch.linalg.eigvalsh(sigma)
        min_eig = torch.min(eigenvalues).item()
        assert min_eig > 0.0, f"Covariance matrix is not positive-definite (min eigenvalue: {min_eig})"

        # 3. Determinant check: det(Sigma) == sx^2 * sy^2 * sz^2
        det_computed = torch.linalg.det(sigma).item()
        s = linear_scales[i]
        det_expected = (s[0] * s[1] * s[2]) ** 2
        det_expected = det_expected.item()
        assert det_computed > 0.0, "Determinant must be strictly positive"
        assert math.isclose(det_computed, det_expected, rel_tol=1e-4), (
            f"Determinant mismatch: computed {det_computed} vs expected {det_expected}"
        )


def test_forward_backward_gradient_flow():
    """
    Test 3: Runs 5 real optimization steps on a synthetic 64x64 RGB patch and verifies
    that non-zero gradients propagate to positions, rotations, scales, opacities, and colors.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(123)
    np.random.seed(123)

    # 1. Synthetic 64x64 Ground Truth Image
    H, W = 64, 64
    gt_canvas = torch.zeros((3, H, W), device=device, dtype=torch.float32)
    gt_canvas[0, 16:48, 16:48] = 0.85
    gt_canvas[1, 24:40, 24:40] = 0.65
    gt_canvas[2, :, :] = 0.20

    # 2. Camera View
    fx = fy = 60.0
    cx, cy = 32.0, 32.0
    w2c = torch.eye(4, dtype=torch.float32, device=device)
    w2c[2, 3] = 2.0  # Camera placed at Z = -2 looking towards origin

    view = {
        "height": H,
        "width": W,
        "fx": fx, "fy": fy,
        "cx": cx, "cy": cy,
        "w2c": w2c,
        "cam_pos": torch.tensor([0.0, 0.0, -2.0], dtype=torch.float32, device=device),
        "image": gt_canvas.cpu()
    }

    # 3. Initialize Model with 16 Gaussians centered around origin
    init_pts = np.random.uniform(-0.3, 0.3, (16, 3)).astype(np.float32)
    init_pts[:, 2] += 2.0  # Position in front of camera
    init_colors = np.random.uniform(0.2, 0.8, (16, 3)).astype(np.float32)

    model = GaussianModel(sh_degree=3)
    model.create_from_pcd(init_pts, init_colors, device)
    optimizer = model.setup_optimizer(lr_pos=0.01)

    trainer = CUDAGaussianTrainer(output_dir=tempfile.gettempdir(), device=device)

    initial_loss = None
    losses = []

    # 4. Run 5 real optimization steps
    for step in range(5):
        optimizer.zero_grad()
        rendered = trainer._render_view(model, view)

        l1_val = F.l1_loss(rendered, gt_canvas)
        ssim_val = get_ssim_torch(rendered, gt_canvas)
        loss = 0.8 * l1_val + 0.2 * (1.0 - ssim_val)

        losses.append(loss.item())
        if step == 0:
            initial_loss = loss.item()

        loss.backward()

        # Verify non-zero gradient propagation across all parameters
        assert model.get_xyz.grad is not None, "Position parameter missing gradients"
        assert torch.norm(model.get_xyz.grad).item() > 0.0, "Zero gradient on positions"

        assert model._scaling.grad is not None, "Scale parameter missing gradients"
        assert torch.norm(model._scaling.grad).item() > 0.0, "Zero gradient on scales"

        assert model._rotation.grad is not None, "Rotation parameter missing gradients"
        assert torch.norm(model._rotation.grad).item() > 0.0, "Zero gradient on rotations"

        assert model._opacity.grad is not None, "Opacity parameter missing gradients"
        assert torch.norm(model._opacity.grad).item() > 0.0, "Zero gradient on opacities"

        assert model._features_dc.grad is not None, "Color SH DC parameter missing gradients"
        assert torch.norm(model._features_dc.grad).item() > 0.0, "Zero gradient on SH DC colors"

        optimizer.step()

    # Verify that optimization proceeded and parameters were actively updated
    assert len(losses) == 5
    assert all(math.isfinite(l) for l in losses), "Loss contains non-finite values (NaN / Inf)"
    assert losses[-1] < initial_loss, f"Loss did not decrease: initial {initial_loss:.4f} vs final {losses[-1]:.4f}"


def test_memory_boundary():
    """
    Test 4: Verifies that allocations do not leak between iterations.
    Checks memory residency across iterations to guarantee strict memory bounding.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(99)

    H, W = 32, 32
    gt_canvas = torch.rand((3, H, W), device=device, dtype=torch.float32)

    w2c = torch.eye(4, dtype=torch.float32, device=device)
    w2c[2, 3] = 2.0
    view = {
        "height": H, "width": W,
        "fx": 40.0, "fy": 40.0, "cx": 16.0, "cy": 16.0,
        "w2c": w2c,
        "cam_pos": torch.tensor([0.0, 0.0, -2.0], dtype=torch.float32, device=device)
    }

    init_pts = np.random.uniform(-0.2, 0.2, (8, 3)).astype(np.float32)
    init_pts[:, 2] += 2.0
    init_cols = np.ones((8, 3), dtype=np.float32) * 0.5

    model = GaussianModel(sh_degree=1)
    model.create_from_pcd(init_pts, init_cols, device)
    optimizer = model.setup_optimizer()
    trainer = CUDAGaussianTrainer(output_dir=tempfile.gettempdir(), device=device)

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        rendered = trainer._render_view(model, view)
        loss = F.l1_loss(rendered, gt_canvas)
        loss.backward()
        optimizer.step()

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        mem_start = torch.cuda.memory_allocated(device)
        for _ in range(25):
            optimizer.zero_grad()
            rendered = trainer._render_view(model, view)
            loss = F.l1_loss(rendered, gt_canvas)
            loss.backward()
            optimizer.step()
        mem_end = torch.cuda.memory_allocated(device)
        mem_diff_mb = (mem_end - mem_start) / (1024 * 1024)
        assert mem_diff_mb < 2.0, f"CUDA memory grew unbounded by {mem_diff_mb:.2f} MB"
    else:
        proc = psutil.Process()
        mem_start = proc.memory_info().rss
        for _ in range(25):
            optimizer.zero_grad()
            rendered = trainer._render_view(model, view)
            loss = F.l1_loss(rendered, gt_canvas)
            loss.backward()
            optimizer.step()
        mem_end = proc.memory_info().rss
        mem_diff_mb = (mem_end - mem_start) / (1024 * 1024)
        # Bounded memory delta under 25 MB
        assert mem_diff_mb < 25.0, f"Host RSS grew unbounded by {mem_diff_mb:.2f} MB"


def test_ply_export_integrity():
    """
    Test 5: Verifies that exported .ply contains valid Inria 3DGS headers,
    vertex counts, properties, and matching coordinate values.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ply_path = os.path.join(tmpdir, "test_gaussians.ply")
        N = 100

        # Construct synthetic field
        np.random.seed(77)
        pos = np.random.uniform(-5.0, 5.0, (N, 3)).astype(np.float32)
        scales = np.random.uniform(0.005, 0.05, (N, 3)).astype(np.float32)
        scales_log = np.log(scales)
        quats = np.zeros((N, 4), dtype=np.float32)
        quats[:, 0] = 1.0  # Identity
        opac = np.random.uniform(0.1, 0.9, N).astype(np.float32)
        opac_logit = np.log(opac / (1.0 - opac))
        colors = np.random.uniform(0.0, 1.0, (N, 3)).astype(np.float32)
        f_dc = (colors - 0.5) / SH_C0

        field = {
            "positions": pos,
            "scales_linear": scales,
            "scales_log": scales_log,
            "rotations": quats,
            "opacities_linear": opac,
            "opacities_logit": opac_logit,
            "colors_rgb": colors,
            "sh_dc": f_dc
        }

        gs_engine = GaussianSplattingEngine()
        gs_engine.export_inria_ply(field, ply_path)

        assert os.path.exists(ply_path), "Exported PLY file does not exist"
        file_size = os.path.getsize(ply_path)
        assert file_size > 0, "Exported PLY file is empty"

        # Read header and verify structure
        with open(ply_path, "rb") as f:
            header_bytes = b""
            while b"end_header\n" not in header_bytes:
                line = f.readline()
                header_bytes += line

            header_text = header_bytes.decode("ascii")
            assert "ply\n" in header_text
            assert "format binary_little_endian 1.0" in header_text
            assert f"element vertex {N}" in header_text

            expected_props = [
                "property float x", "property float y", "property float z",
                "property float nx", "property float ny", "property float nz",
                "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
                "property float opacity",
                "property float scale_0", "property float scale_1", "property float scale_2",
                "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3"
            ]
            for prop in expected_props:
                assert prop in header_text, f"Missing expected property: {prop}"

            # Check binary payload size: 68 bytes per vertex (17 float32 properties) for degree 0
            remaining_bytes = f.read()
            expected_payload_bytes = N * 68
            assert len(remaining_bytes) == expected_payload_bytes, (
                f"Binary payload length mismatch: got {len(remaining_bytes)}, expected {expected_payload_bytes}"
            )

            # Unpack first vertex and verify coordinate match
            first_vertex_values = struct.unpack("<17f", remaining_bytes[:68])
            x_read, y_read, z_read = first_vertex_values[0], first_vertex_values[1], first_vertex_values[2]
            assert math.isclose(x_read, pos[0, 0], rel_tol=1e-5)
            assert math.isclose(y_read, pos[0, 1], rel_tol=1e-5)
            assert math.isclose(z_read, pos[0, 2], rel_tol=1e-5)
