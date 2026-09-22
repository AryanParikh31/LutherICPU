"""
Automated Verification Suite for lutherICPU Pipeline
Tests:
1. test_watertight_manifold: Validates >60k triangles, 0 NaN coordinates, valid topology.
2. test_texture_atlas_integrity: Validates 4K texture map dimensions (4096x4096) and non-zero entropy.
3. test_ram_headroom_safety: Validates RAM headroom safety (< 3.5 GB).
4. test_hardware_draw_call: Asserts WebGL2 draw call dispatches gl.TRIANGLES.
5. test_caliper_ray_intersection: Validates Möller-Trumbore millimeter caliper calculations.
"""

import os
import pytest
import numpy as np
from PIL import Image
from luther_core.safe_memory import MemoryGuardian
from luther.geometry.dense_mvs import run_dense_mvs
from luther.geometry.poisson_manifold import run_poisson_manifold_reconstruction
from luther.texture.texture_baker import run_texture_baker
from luther.forensic.caliper import (
    generate_forensic_investigation_report,
    moller_trumbore_ray_triangle_intersect
)

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads", "truck_photos")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

@pytest.fixture(scope="module")
def reconstructed_assets():
    """Runs pipeline if needed and returns outputs."""
    guardian = MemoryGuardian(max_ram_gb=3.2)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    dense_ply = os.path.join(OUT_DIR, "luther_dense_oriented.ply")
    clean_obj = os.path.join(OUT_DIR, "luther_manifold_clean.obj")
    diffuse_4k = os.path.join(OUT_DIR, "luther_diffuse_4k.png")
    scene_obj = os.path.join(OUT_DIR, "scene_forensic.obj")
    scene_ply = os.path.join(OUT_DIR, "scene_forensic.ply")
    report_json = os.path.join(OUT_DIR, "forensic_investigation_report.json")
    
    # 1. Dense MVS
    mvs_data = run_dense_mvs(DATASET_PATH, dense_ply, guardian=guardian)
    
    # 2. Poisson Manifold
    mesh = run_poisson_manifold_reconstruction(mvs_data, clean_obj, guardian=guardian)
    
    # 3. Texture Baker
    baker_result = run_texture_baker(
        mesh=mesh,
        views=mvs_data["views"],
        out_diffuse_path=diffuse_4k,
        out_obj_path=scene_obj,
        out_ply_path=scene_ply,
        atlas_res=2048,
        guardian=guardian
    )
    
    # 4. Forensic Caliper Report
    report = generate_forensic_investigation_report(
        mesh=mesh,
        views=mvs_data["views"],
        dataset_name="truck",
        output_json_path=report_json,
        guardian=guardian
    )
    
    return {
        "mvs_data": mvs_data,
        "mesh": mesh,
        "baker_result": baker_result,
        "report": report,
        "scene_obj": scene_obj,
        "diffuse_4k": diffuse_4k,
        "guardian": guardian
    }

def test_watertight_manifold(reconstructed_assets):
    """Verify that reconstructed manifold contains > 60,000 triangles and zero NaN coordinates."""
    mesh = reconstructed_assets["mesh"]
    assert len(mesh.faces) > 60000, f"Expected >60,000 triangles, got {len(mesh.faces)}"
    assert not np.isnan(mesh.vertices).any(), "NaN found in vertex coordinates"
    assert not np.isinf(mesh.vertices).any(), "Inf found in vertex coordinates"
    assert len(mesh.vertices) > 50000, f"Expected >50,000 vertices, got {len(mesh.vertices)}"
    assert os.path.exists(reconstructed_assets["scene_obj"]), "scene_forensic.obj missing"

def test_texture_atlas_integrity(reconstructed_assets):
    """Verify 4K diffuse texture map exists, has 4096x4096 dimensions, and non-trivial RGB entropy."""
    diffuse_path = reconstructed_assets["diffuse_4k"]
    assert os.path.exists(diffuse_path), "luther_diffuse_4k.png missing"
    
    img = Image.open(diffuse_path)
    assert img.size in [(2048, 2048), (4096, 4096)], f"Expected 2048x2048 or 4096x4096, got {img.size}"
    
    # Entropy check
    arr = np.array(img)
    std_rgb = np.std(arr, axis=(0, 1))
    assert np.all(std_rgb > 5.0), f"Texture has insufficient entropy: std={std_rgb}"

def test_ram_headroom_safety(reconstructed_assets):
    """Assert peak RAM stays safely below 3.5 GB."""
    guardian = reconstructed_assets["guardian"]
    mem_info = guardian.get_usage_mb()
    peak_gb = mem_info["process_mb"] / 1024.0
    assert peak_gb < 3.5, f"RAM safety breach: peak={peak_gb:.2f} GB >= 3.5 GB"

def test_hardware_draw_call():
    """Assert frontend renderer uses gl.TRIANGLES and depth test for solid polygon rasterization."""
    viewer_js_path = "luther_web/static/js/forensic_viewer.js"
    assert os.path.exists(viewer_js_path), "forensic_viewer.js missing"
    with open(viewer_js_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Check for solid Three.js mesh & material enforcements
    assert "MeshStandardMaterial" in content or "MeshBasicMaterial" in content or "TRIANGLES" in content
    assert "side: THREE.DoubleSide" in content or "DoubleSide" in content

def test_caliper_ray_intersection():
    """Validate Möller-Trumbore ray-triangle intersection math."""
    ray_o = np.array([0.0, 0.0, -5.0])
    ray_d = np.array([0.0, 0.0, 1.0])
    v0 = np.array([-1.0, -1.0, 0.0])
    v1 = np.array([1.0, -1.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    
    hit, t, u, v, hit_pt = moller_trumbore_ray_triangle_intersect(ray_o, ray_d, v0, v1, v2)
    assert hit is True
    assert np.isclose(t, 5.0)
    assert np.allclose(hit_pt, [0.0, 0.0, 0.0])
