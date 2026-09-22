"""
lutherICPU — Master Unified 3D Reconstruction & Simulation Runner.

Runs the complete 5-stage CPU photogrammetry and simulation model from a single command,
and automatically opens the interactive SIBR-like 3D simulation popup window upon completion.

Usage Example:
  python run.py --images "path/to/your/images" --iterations 30000

Short Options:
  python run.py -i "path/to/your/images" -n 30k
"""

import os
import sys
import argparse
import logging
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from luther_pipeline.simulation_model import LutherICPU
from luther_pipeline.iterative_engine import parse_iteration_count
from luther_renderer.display_window import display_simulation_window

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lutherICPU.Run")


def main():
    parser = argparse.ArgumentParser(
        description="lutherICPU: CPU-Native 3D Reconstruction Model with Instant SIBR-Like Simulation Popup"
    )
    parser.add_argument(
        "--images", "-i",
        type=str,
        default=None,
        help="Path to folder containing multi-angle source photographs (Required)"
    )
    parser.add_argument(
        "--colmap", "-c",
        type=str,
        default=None,
        help="Path to COLMAP sparse folder (containing cameras.bin, images.bin, points3D.bin). Auto-detected if omitted."
    )
    parser.add_argument(
        "--iterations", "-n",
        type=str,
        default="30000",
        help="Number of iterations for reconstruction and densification (e.g. 30000, 30k)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for generated 3D simulation assets and renders"
    )
    parser.add_argument(
        "--resolution", "--res",
        type=int,
        default=2048,
        help="Texture atlas resolution (e.g. 2048, 4096)"
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=32,
        help="Maximum camera views for dense stereo matching (default: 32)"
    )
    parser.add_argument(
        "--max-ram", "--ram",
        type=float,
        default=3.2,
        help="Maximum RAM limit in GB (default: 3.2 GB for 8GB RAM CPU safety)"
    )
    parser.add_argument(
        "--no-popup",
        action="store_true",
        help="Do not automatically open the interactive SIBR simulation popup window upon completion"
    )

    args = parser.parse_args()

    if not args.images:
        print("[ERROR] Please provide the path to your input images using '--images <path>' or '-i <path>'.")
        parser.print_help()
        sys.exit(1)

    images_path = os.path.abspath(args.images)
    if not os.path.exists(images_path):
        print(f"[Error] Images directory not found: {images_path}")
        sys.exit(1)

    iters = parse_iteration_count(args.iterations)
    scene_name = os.path.basename(os.path.normpath(images_path))
    if scene_name.lower() in ["images", "img", "photos", ""]:
        scene_name = os.path.basename(os.path.dirname(os.path.normpath(images_path)))
    scene_name = scene_name.lower()

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output or os.path.join(PROJECT_ROOT, "output", f"{scene_name}_{timestamp_str}")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f" Source Images Path: {images_path}")
    print(f" Reconstruction Mode: Native CPU SIFT SfM (Directly From Raw Images)")
    print(f" Iterations Target:  {iters:,}")
    print(f" Output Directory:   {out_dir}")
    print(f" SIBR Popup Display: {'ENABLED (Auto-Launch)' if not args.no_popup else 'DISABLED'}")
    print("=" * 80 + "\n")

    # 1. Instantiate Pure Simulation Model
    model = LutherICPU(
        output_dir=out_dir,
        max_ram_gb=args.max_ram
    )

    # 2. Run All 5 Mathematical Stages
    result = model.simulate(
        images_path=images_path,
        colmap_path=args.colmap,
        iterations=iters,
        scene_name=scene_name,
        texture_resolution=args.resolution,
        render_simulation=True,
        max_views=args.max_views
    )

    print("\n" + "=" * 80)
    print(f" [SUCCESS] 3D SIMULATION GENERATED FOR '{scene_name.upper()}'")
    print(f" Total Processing Time: {result['elapsed_seconds']}s on CPU")
    print(f" Output Folder:         {result['output_dir']}")
    print(f" Reconstructed Mesh:    {result['num_vertices']:,} vertices, {result['num_faces']:,} triangles")
    print(f" Dense Point Cloud:     {result['num_dense_points']:,} surface points")
    print(f" Textured 3D Model:     {result['obj_path']}")
    print(f" Photographic Atlas:    {result['diffuse_png_path']}")
    print("=" * 80 + "\n")

    # 3. Automatically Open Interactive 3D SIBR Simulation Viewer
    if not args.no_popup:
        print("[*] Launching SIBR-Style Interactive 3D Simulation Display Window...")
        try:
            from luther_display import launch_interactive_3d_viewport
            launch_interactive_3d_viewport(port=8080)
        except Exception as e:
            logger.warning(f"Failed to launch 3D viewport: {e}, falling back to turntable window.")
            if result.get("renders"):
                display_simulation_window(
                    proof_dir_or_dict=result["renders"],
                    scene_name=scene_name,
                    window_title=f"lutherICPU SIBR Simulation Viewer — [{scene_name.upper()}]"
                )


if __name__ == "__main__":
    main()
