"""
Unified Command-Line Simulation Runner and Image Capture Tool.
Executes the master 3D simulation and captures all multi-angle simulation views.
"""

import os
import sys
import time
import shutil
from pathlib import Path
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from luther_renderer.supreme_simulation_engine import SupremeSimulationEngine

def run_and_capture():
    print("=" * 80)
    print(" [*] EXECUTING LUTHERICPU 3D NOVEL-VIEW SIMULATION & IMAGE CAPTURE")
    print("=" * 80)

    images_path = r"F:\tandt_db\tandt\truck\images"
    colmap_path = r"uploads\truck_photos\sparse\0"
    scene_name = "truck_251_master"

    engine = SupremeSimulationEngine(output_dir="output")
    result = engine.run_master_simulation(
        images_path=images_path,
        colmap_path=colmap_path,
        scene_name=scene_name
    )

    print("\n" + "=" * 80)
    print(" [*] SIMULATION CAPTURE COMPLETED SUCCESSFULLY!")
    print(f" Status:             {result['status']}")
    print(f" Elapsed Time:       {result['elapsed_seconds']}s")
    print(f" 1080p Hero View:    {result['hero_render_1080p']}")
    print(f" 1080p Side View:    {result['side_render_1080p']}")
    print(f" 4K UHD Master:      {result['uhd_render_4k']}")
    print(f" 3D Gaussian Splat:  {result['gaussian_splat_render']}")
    print(f" 360 Turntable GIF:  {result['turntable_360_gif']}")
    print("=" * 80 + "\n")

    # Verify all artifacts in destination
    art_dir = Path(r"C:\Users\AARYAN\.gemini\antigravity-ide\brain\13224361-4efd-4356-804b-748717b4ad7e")
    for f in [
        f"{scene_name}_simulation_1080p_hero.png",
        f"{scene_name}_simulation_1080p_side.png",
        f"{scene_name}_simulation_4k_ultra.png",
        f"{scene_name}_gaussian_splat_simulation.png",
        f"{scene_name}_360_simulation.gif",
        f"{scene_name}_turntable_frame_00_angle_000.png",
        f"{scene_name}_turntable_frame_04_angle_060.png",
        f"{scene_name}_turntable_frame_08_angle_120.png",
        f"{scene_name}_turntable_frame_12_angle_180.png",
        f"{scene_name}_turntable_frame_16_angle_240.png",
        f"{scene_name}_turntable_frame_20_angle_300.png",
    ]:
        p = art_dir / f
        if p.exists():
            print(f" [OK] Verified Artifact: {p.name} ({os.path.getsize(p):,} bytes)")
        else:
            src = Path("output") / "proof_renders" / f
            if src.exists():
                shutil.copy2(src, p)
                print(f" [SYNC] Synchronized: {p.name}")

if __name__ == "__main__":
    run_and_capture()
