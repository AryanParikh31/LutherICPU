import os
import sys
import time
import shutil
import numpy as np
from PIL import Image

PROJECT_ROOT = r"c:\Users\AARYAN\OneDrive\Desktop\cpunew"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from luther_core.colmap_loader import load_colmap_model
from luther_core.gaussian_splatting_engine import GaussianSplattingEngine
from luther_renderer.supreme_simulation_engine import SupremeSimulationEngine

def run_simulation():
    images_path = os.path.join(PROJECT_ROOT, "uploads", "truck_photos", "images")
    colmap_path = os.path.join(PROJECT_ROOT, "uploads", "truck_photos", "sparse", "0")
    output_dir = os.path.join(PROJECT_ROOT, "output", "truck_3dgs_simulation")
    artifact_dir = r"C:\Users\AARYAN\.gemini\antigravity-ide\brain\13224361-4efd-4356-804b-748717b4ad7e"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(" [*] INGESTING CALIBRATED CAMERAS & RECONSTRUCTING 3D GAUSSIAN FIELD")
    print("=" * 80)

    cameras, views_dict, sparse_pcd = load_colmap_model(sparse_dir=colmap_path, images_dir=images_path)
    all_views = list(views_dict.values())
    print(f"Ingested {len(all_views)} calibrated cameras, {len(sparse_pcd):,} 3D points.")

    # Generate 3D Gaussian Field with Curvature-Aware Covariance (CAC) & surface wafers
    gs_engine = GaussianSplattingEngine()
    field = gs_engine.generate_gaussian_field(
        positions=sparse_pcd.positions,
        colors=sparse_pcd.colors
    )
    ply_path = os.path.join(output_dir, "truck_3dgs.ply")
    splat_path = os.path.join(output_dir, "truck_3dgs.splat")
    gs_engine.export_inria_ply(field, ply_path)
    gs_engine.export_binary_splat(field, splat_path)

    # Copy to root output/
    shutil.copy2(splat_path, os.path.join(PROJECT_ROOT, "output", "truck_photos_3dgs.splat"))
    shutil.copy2(ply_path, os.path.join(PROJECT_ROOT, "output", "truck_photos_3dgs.ply"))
    shutil.copy2(splat_path, os.path.join(PROJECT_ROOT, "output", "scene.splat"))

    # Write manifest
    manifest_path = os.path.join(PROJECT_ROOT, "output", "latest_simulation.json")
    manifest_data = {
        "scene_id": "truck",
        "scene_name": "Tanks and Temples Truck",
        "output_dir": output_dir,
        "splat_path": splat_path,
        "ply_3dgs_path": ply_path,
        "cameras_json_path": os.path.join(PROJECT_ROOT, "output", "truck_cameras.json"),
        "splat_count": len(sparse_pcd)
    }
    import json
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print("\n" + "=" * 80)
    print(" [*] SYNTHESIZING CONTINUOUS 3D GAUSSIAN SPLATTING SIBR RADIANCE SIMULATION")
    print("=" * 80)

    renderer = SupremeSimulationEngine(output_dir=output_dir)
    res = renderer.synthesize_simulation_suite(
        points=sparse_pcd.positions,
        point_colors=sparse_pcd.colors,
        camera_views=all_views,
        scene_name="proof_3dgs_truck",
        num_turntable_frames=24
    )

    print("\n" + "=" * 80)
    print(" [*] SYNCHRONIZING ARTIFACTS")
    print("=" * 80)

    for key, path in res.items():
        if os.path.exists(path):
            dst = os.path.join(artifact_dir, os.path.basename(path))
            shutil.copy2(path, dst)
            print(f"  -> Synchronized {key}: {os.path.basename(path)}")

    # Copy keyframes
    proof_dir = os.path.join(output_dir, "proof_renders")
    for kf in os.listdir(proof_dir):
        if kf.endswith(".png") or kf.endswith(".gif"):
            shutil.copy2(os.path.join(proof_dir, kf), os.path.join(artifact_dir, kf))
            print(f"  -> Keyframe/Proof: {kf}")

    print("\n[SUCCESS] All 3D Gaussian Splatting SIBR simulation proof renders generated!")

if __name__ == "__main__":
    run_simulation()
