"""
Test Suite for lutherICPU 5-Stage Mathematical Simulation Model.

Verifies:
1. Stage 1 Ingestion & QC (SHA-256 cryptographic hashing & Laplacian variance Var(∇²I)).
2. Stage 2 Structure-from-Motion (SIFT keypoints, epipolar matching, DLT triangulation).
3. Stage 3 Dense MVS (PatchMatch Normalized Cross-Correlation depth estimation & dense fusion).
4. Stage 4 Watertight Surface Reconstruction (TSDF voxel meshing, Taubin smoothing).
5. Stage 5 Photographic texturing & Supreme continuous radiance simulation synthesis.
6. End-to-End LutherICPU model execution.
"""

import os
import tempfile
import shutil
import numpy as np
import cv2
import pytest
from PIL import Image

from luther_core.types import CameraView, CameraIntrinsics, PointCloud, SurfaceMesh
from luther_core.ingestion import compute_sha256, compute_laplacian_variance, ImageQualityController
from luther_core.sfm import extract_sift_features, match_sift_features, triangulate_dlt_point, StructureFromMotionEngine
from luther_pipeline.dense_mvs_engine import compute_patch_ncc, LutherDenseMVSEngine
from luther_geometry.volumetric_tsdf_fusion import VolumetricTSDFFusionEngine
from luther_texture.projective_baker import ProjectiveTextureBaker
from luther_renderer.supreme_simulation_engine import SupremeSimulationEngine
from luther_pipeline.simulation_model import LutherICPU


def create_synthetic_textured_image(filepath: str, width: int = 320, height: int = 240, blur: bool = False):
    """Creates a synthetic textured test image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Add high-frequency checkerboard / gradient pattern
    for y in range(0, height, 16):
        for x in range(0, width, 16):
            if (x // 16 + y // 16) % 2 == 0:
                img[y:y+16, x:x+16] = [200, 150, 80]
            else:
                img[y:y+16, x:x+16] = [40, 90, 180]

    cv2.circle(img, (width // 2, height // 2), 40, (255, 255, 255), -1)
    cv2.putText(img, "lutherICPU", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if blur:
        img = cv2.GaussianBlur(img, (21, 21), 0)

    Image.fromarray(img).save(filepath)
    return img


def test_stage_1_ingestion_and_laplacian_qc():
    """Verifies Stage 1 SHA-256 and Laplacian variance Var(∇²I) blur computation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sharp_path = os.path.join(tmpdir, "sharp.png")
        blurry_path = os.path.join(tmpdir, "blurry.png")

        create_synthetic_textured_image(sharp_path, blur=False)
        create_synthetic_textured_image(blurry_path, blur=True)

        # 1. SHA-256
        hash_sharp = compute_sha256(sharp_path)
        assert len(hash_sharp) == 64
        assert isinstance(hash_sharp, str)

        # 2. Laplacian Variance Var(∇²I)
        img_sharp = cv2.imread(sharp_path)
        img_blurry = cv2.imread(blurry_path)

        score_sharp = compute_laplacian_variance(img_sharp)
        score_blurry = compute_laplacian_variance(img_blurry)

        assert score_sharp > score_blurry
        assert score_sharp > 100.0
        assert score_blurry < 50.0

        # 3. ImageQualityController
        qc = ImageQualityController(blur_threshold=60.0)
        report = qc.ingest_dataset(tmpdir)
        assert report["total_images"] == 2
        assert report["flagged_images"] == 1  # The blurry one flagged


def test_stage_2_sfm_sift_and_dlt():
    """Verifies Stage 2 SIFT feature extraction, matching, and DLT triangulation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img1_p = os.path.join(tmpdir, "view1.png")
        img2_p = os.path.join(tmpdir, "view2.png")
        create_synthetic_textured_image(img1_p)
        create_synthetic_textured_image(img2_p)

        # SIFT Extraction
        kps1, desc1 = extract_sift_features(img1_p, max_features=500)
        kps2, desc2 = extract_sift_features(img2_p, max_features=500)
        assert len(kps1) > 0
        assert desc1.shape[1] == 128

        # SIFT Matching
        matches = match_sift_features(desc1, desc2)
        assert len(matches) > 0

        # DLT Triangulation
        P1 = np.array([[500, 0, 160, 0], [0, 500, 120, 0], [0, 0, 1, 0]], dtype=np.float64)
        P2 = np.array([[500, 0, 160, -100], [0, 500, 120, 0], [0, 0, 1, 0]], dtype=np.float64)
        X = triangulate_dlt_point(P1, P2, np.array([160.0, 120.0]), np.array([140.0, 120.0]))
        assert len(X) == 3
        assert np.isfinite(X).all()


def test_stage_3_patchmatch_ncc():
    """Verifies Stage 3 PatchMatch Normalized Cross-Correlation scoring."""
    p1 = np.full((7, 7, 3), 100, dtype=np.float32)
    p1[2:5, 2:5] = 200
    p2_identical = p1.copy()
    p2_inverted = 255.0 - p1

    ncc_same = compute_patch_ncc(p1, p2_identical)
    ncc_diff = compute_patch_ncc(p1, p2_inverted)

    assert ncc_same > 0.99
    assert ncc_diff < -0.90


def test_stage_4_tsdf_volumetric_meshing():
    """Verifies Stage 4 Volumetric TSDF Isosurface Extraction & Taubin smoothing."""
    # Synthetic cube points
    np.random.seed(42)
    pts = np.random.uniform(-1.0, 1.0, (2000, 3)).astype(np.float32)
    cols = np.full((2000, 3), 0.8, dtype=np.float32)
    cams = np.array([[0, 0, 3], [0, 0, -3], [3, 0, 0]], dtype=np.float32)

    tsdf = VolumetricTSDFFusionEngine()
    verts, faces, vert_cols = tsdf.reconstruct_scene(points=pts, colors=cols, camera_centers=cams, voxel_res=32)

    assert len(verts) > 0
    assert len(faces) > 0
    assert faces.shape[1] == 3


def test_stage_5_supreme_simulation_rendering():
    """Verifies Stage 5 Supreme continuous photorealistic radiance synthesis."""
    pts = np.random.uniform(-1.0, 1.0, (1000, 3)).astype(np.float32)
    intrinsics = CameraIntrinsics(width=640, height=360, fx=400.0, fy=400.0, cx=320.0, cy=180.0)
    cam = CameraView(
        image_id=1,
        name="test_cam",
        qvec=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        tvec=np.array([0.0, 0.0, 3.0], dtype=np.float32),
        intrinsics=intrinsics
    )
    cam._R = np.eye(3, dtype=np.float32)

    src_img = np.full((360, 640, 3), 0.7, dtype=np.float32)
    sources = [(cam, src_img)]

    renderer = SupremeSimulationEngine()
    frame = renderer.synthesize_continuous_view(
        points=pts,
        target_cam=cam,
        source_views=sources,
        width=320,
        height=180,
        infill_radius=12
    )

    assert frame.shape == (180, 320, 3)
    assert frame.dtype == np.uint8
    assert np.mean(frame) > 0


def test_luthericpu_end_to_end_simulation():
    """Verifies complete end-to-end LutherICPU pure model execution."""
    with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
        # Create 3 synthetic calibrated views
        for i in range(3):
            img_p = os.path.join(tmp_in, f"photo_{i:02d}.png")
            create_synthetic_textured_image(img_p, width=160, height=120)

        model = LutherICPU(output_dir=tmp_out, max_ram_gb=2.0)
        res = model.simulate(
            images_path=tmp_in,
            iterations=1000,
            scene_name="test_sim",
            texture_resolution=512,
            render_simulation=True,
            max_views=3
        )

        assert res["status"] == "SUCCESS"
        assert res["scene_name"] == "test_sim"
        assert os.path.exists(res["obj_path"])
        assert os.path.exists(res["mtl_path"])
        assert os.path.exists(res["diffuse_png_path"])
        assert os.path.exists(res["mesh_ply_path"])
        assert os.path.exists(res["dense_ply_path"])
        assert os.path.exists(res["qc_manifest"])
        assert "hero_1080p" in res["renders"]
        assert os.path.exists(res["renders"]["hero_1080p"])
        assert os.path.exists(res["renders"]["turntable_gif"])
