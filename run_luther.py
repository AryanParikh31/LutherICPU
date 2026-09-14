"""
lutherICPU — Master CLI & Simulation Model Launcher.

Pure CPU-native 3D reconstruction and simulation model:
1. Stage 1: Ingestion & Quality Control (SHA-256 hash & Laplacian Var(∇²I) blur check)
2. Stage 2: Structure-from-Motion (DoG SIFT / Epipolar Essential Matrix / DLT Triangulation / COLMAP)
3. Stage 3: Dense Multi-View Stereo (PatchMatch NCC Stereo & Dense Point Cloud Fusion)
4. Stage 4: Watertight Surface Reconstruction (Volumetric TSDF / Screened Poisson & Taubin Smoothing)
5. Stage 5: Photographic Texturing & Supreme Continuous Radiance Simulation Rendering

Usage Examples:
  # 1. Run simulation on multi-angle images with auto-discovered COLMAP:
  python run_luther.py --images "F:/tandt_db/tandt/truck/images" --iterations 30000

  # 2. Short options syntax:
  python run_luther.py -i "F:/tandt_db/tandt/truck/images" -c "F:/tandt_db/tandt/truck/sparse/0" -n 30k

  # 3. Predefined benchmark scene:
  python run_luther.py --scene truck

  # 4. Automated verification test suite:
  python run_luther.py --test
"""

import os
import sys
import argparse
import logging
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from luther_pipeline.simulation_model import LutherICPU
from luther_pipeline.iterative_engine import parse_iteration_count
from benchmark.run_all_scenes import run_full_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lutherICPU.Launcher")


def main():
    parser = argparse.ArgumentParser(
        description="lutherICPU: CPU-Native 3D Reconstruction & Photorealistic Simulation Model"
    )
    parser.add_argument(
        "--images", "--image_path", "-i",
        type=str,
        default=None,
        help="Path to folder containing multi-angle source photographs"
    )
    parser.add_argument(
        "--colmap", "--colmap_path", "--sparse_path", "-c",
        type=str,
        default=None,
        help="Path to COLMAP sparse folder (containing cameras.bin, images.bin, points3D.bin)"
    )
    parser.add_argument(
        "--iterations", "--iters", "-n",
        type=str,
        default="30000",
        help="Number of reconstruction iterations (e.g. 30000, 30k, 7000)"
    )
    parser.add_argument(
        "--output", "--out_dir", "-o",
        type=str,
        default=os.path.join(PROJECT_ROOT, "output"),
        help="Output directory for generated 3D simulation assets and renders"
    )
    parser.add_argument(
        "--scene", "-s",
        type=str,
        default=None,
        help="Predefined benchmark scene shortcut (truck, train, drjohnson, playroom)"
    )
    parser.add_argument(
        "--resolution", "--res",
        type=int,
        default=2048,
        help="Texture atlas resolution (e.g. 2048, 4096)"
    )
    parser.add_argument(
        "--max-ram", "--ram",
        type=float,
        default=3.2,
        help="Maximum RAM limit in GB (default: 3.2 GB for 8GB RAM CPU safety)"
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=40,
        help="Maximum key views for dense stereo matching (default: 40)"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip synthesis of photographic simulation proof renders and turntable GIF"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run pytest automated verification test suite"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run full multi-dataset simulation benchmark"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Launch native desktop interactive 3D simulation player window"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Optional: Launch WebGL2 local viewer server upon completion"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port number (default: 8080)"
    )

    args = parser.parse_args()

    # Import desktop display player
    from luther_renderer.display_window import display_simulation_window

    # 1. Test Suite Option
    if args.test:
        logger.info("Running complete verification test suite with pytest...")
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], cwd=PROJECT_ROOT)
        return

    # 2. Benchmark Option
    if args.benchmark:
        run_full_benchmark(output_dir=args.output)
        return

    # 3. Scene Shortcut Mapping
    scene_map = {
        "truck": (r"F:\tandt_db\tandt\truck\images", r"F:\tandt_db\tandt\truck\sparse\0"),
        "train": (r"F:\tandt_db\tandt\train\images", r"F:\tandt_db\tandt\train\sparse\0"),
        "drjohnson": (r"F:\tandt_db\db\drjohnson\images", r"F:\tandt_db\db\drjohnson\sparse\0"),
        "playroom": (r"F:\tandt_db\db\playroom\images", r"F:\tandt_db\db\playroom\sparse\0")
    }

    if args.scene and not args.images:
        name_key = args.scene.lower()
        if name_key in scene_map:
            args.images, args.colmap = scene_map[name_key]
        else:
            args.images = args.scene

    # Default fallback to truck dataset if no arguments provided
    if not args.images:
        if os.path.exists(r"F:\tandt_db\tandt\truck\images"):
            args.images, args.colmap = scene_map["truck"]
            args.scene = "truck"
        else:
            parser.print_help()
            return

    iters = parse_iteration_count(args.iterations)
    scene_name = args.scene.lower() if args.scene else os.path.basename(os.path.normpath(args.images))
    if scene_name in ["images", "img", "photos", ""]:
        scene_name = os.path.basename(os.path.dirname(os.path.normpath(args.images)))

    # Auto-discover COLMAP calibration if colmap path not found or missing
    if not args.colmap or not os.path.exists(args.colmap):
        candidates = [
            os.path.join(os.path.dirname(args.images), "sparse", "0"),
            os.path.join(os.path.dirname(args.images), "sparse"),
            os.path.join(PROJECT_ROOT, "uploads", f"{scene_name}_photos", "sparse", "0"),
            os.path.join(PROJECT_ROOT, "uploads", "truck_photos", "sparse", "0"),
        ]
        for c in candidates:
            if os.path.exists(c):
                args.colmap = c
                logger.info(f"Auto-discovered sparse calibration model: '{c}'")
                break

    # Execute Master lutherICPU Pure Simulation Model
    model = LutherICPU(
        output_dir=args.output,
        max_ram_gb=args.max_ram
    )

    result = model.simulate(
        images_path=args.images,
        colmap_path=args.colmap,
        iterations=iters,
        scene_name=scene_name,
        texture_resolution=args.resolution,
        render_simulation=not args.no_render,
        max_views=args.max_views
    )

    print("\n" + "=" * 80)
    print(f" [LUTHERICPU SIMULATION COMPLETE] Scene: {result['scene_name'].upper()}")
    print(f" Elapsed Time:     {result['elapsed_seconds']}s on CPU")
    print(f" Reconstructed 3D: {result['num_vertices']:,} vertices, {result['num_faces']:,} triangles")
    print(f" Textured OBJ:     {result['obj_path']}")
    print(f" Texture Atlas:    {result['diffuse_png_path']}")
    if result.get("renders"):
        for r_name, r_path in result["renders"].items():
            print(f" Simulation Frame: [{r_name}] -> {r_path}")
    print("=" * 80 + "\n")

    if args.display:
        display_simulation_window(result.get("renders", {}), scene_name=scene_name)

    if args.serve:
        from luther_web.server import start_server
        start_server(port=args.port)


if __name__ == "__main__":
    main()
