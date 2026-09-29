"""
lutherICPU Unified 3D Reconstruction, Simulation & Interactive Player.

Usage Examples:
  # 1. Train model on user-provided images folder and display simulation:
  python simulate.py --images "path/to/your/images" --iterations 30000 --output "output/my_scene"

  # 2. Short flags syntax:
  python simulate.py -i "path/to/your/images" -n 30k -o "output/my_scene"

  # 3. Quick interactive viewer for latest simulation:
  python simulate.py

  # 4. WebGL browser 3D viewport:
  python simulate.py --web
"""

import os
import sys
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from luther_pipeline.simulation_model import LutherICPU
from luther_pipeline.iterative_engine import parse_iteration_count
import shutil
from luther_renderer.display_window import display_simulation_window

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lutherICPU.Simulate")


def safe_copy(src: str, dst: str):
    """Safely copies a file, preventing self-copy and handling Windows file locks."""
    try:
        if not src or not os.path.exists(src):
            return
        abs_src = os.path.abspath(src)
        abs_dst = os.path.abspath(dst)
        if abs_src == abs_dst:
            return
        os.makedirs(os.path.dirname(abs_dst), exist_ok=True)
        if os.path.exists(abs_dst):
            try:
                os.remove(abs_dst)
            except Exception:
                pass
        shutil.copy2(abs_src, abs_dst)
    except Exception as e:
        logger.debug(f"[SafeCopy] Notice copying {src} -> {dst}: {e}")


def configure_os_environment(windows: bool = False, linux: bool = False):
    """Configures OS-specific thread limits, high-DPI scaling, and windowing backends."""
    if windows or (not linux and sys.platform.startswith("win")):
        try:
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
        os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() or 4))
        logger.info("[OS Init] Mode: Windows Native (High-DPI Aware / Multi-Threaded CPU).")
    elif linux or sys.platform.startswith("linux"):
        if "QT_QPA_PLATFORM" not in os.environ:
            os.environ["QT_QPA_PLATFORM"] = "xcb"
        os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() or 4))
        logger.info("[OS Init] Mode: Linux Native (X11/Wayland / Multi-Threaded CPU).")


def main():
    parser = argparse.ArgumentParser(
        description="lutherICPU: Train 3D Model on Input Images and Display Interactive Simulation"
    )
    parser.add_argument(
        "--images", "--image_path", "-i",
        type=str,
        default=None,
        help="Path to folder containing multi-angle source photographs"
    )
    parser.add_argument(
        "--iterations", "--iters", "-n",
        type=str,
        default="30000",
        help="Number of reconstruction/training iterations (e.g. 30000, 30k, 7000)"
    )
    parser.add_argument(
        "--windows", "-w",
        action="store_true",
        help="Select Windows native OS platform mode (High-DPI aware, DirectX/Win32 display)"
    )
    parser.add_argument(
        "--linux", "-l",
        action="store_true",
        help="Select Linux native OS platform mode (X11/Wayland display)"
    )
    parser.add_argument(
        "--output", "--out_dir", "-o",
        type=str,
        default=None,
        help="Output directory path for 3D model assets and simulation renders"
    )
    parser.add_argument(
        "--colmap", "--colmap_path", "-c",
        type=str,
        default=None,
        help="Optional path to COLMAP sparse directory (auto-detected if omitted)"
    )
    parser.add_argument(
        "--scene", "-s",
        type=str,
        default=None,
        help="Scene name (e.g. truck, train, drjohnson)"
    )
    parser.add_argument(
        "--max-ram",
        type=float,
        default=3.2,
        help="Maximum RAM limit in GB (default: 3.2 GB)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run training without launching the desktop interactive display window"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch browser WebGL viewport instead of native desktop popup window"
    )
    parser.add_argument(
        "--import-bundle", "-b",
        type=str,
        default=None,
        help="Path to downloaded Colab simulation bundle (luther_simulation_bundle.zip or .splat)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for 3D simulation server (default: 8080)"
    )

    parser.add_argument(
        "--software", "--cv",
        action="store_true",
        help="Launch pure CPU software projection window fallback instead of hardware WebGL viewport"
    )

    args = parser.parse_args()
    configure_os_environment(windows=args.windows, linux=args.linux)

    # Handle Colab bundle import directly
    if args.import_bundle:
        raw_bundle_path = args.import_bundle.strip('\'"')
        bundle_path = os.path.abspath(raw_bundle_path)
        
        # Check if user passed path without .zip extension but .zip exists
        if not os.path.exists(bundle_path) and os.path.exists(bundle_path + ".zip"):
            bundle_path = bundle_path + ".zip"

        if not os.path.exists(bundle_path):
            print(f"[ERROR] Specified bundle path does not exist: {raw_bundle_path}")
            sys.exit(1)

        out_dir = args.output or os.path.join(PROJECT_ROOT, "output", "colab_imported_simulation")
        os.makedirs(out_dir, exist_ok=True)
        root_output_dir = os.path.join(PROJECT_ROOT, "output")
        os.makedirs(root_output_dir, exist_ok=True)

        if os.path.isfile(bundle_path) and bundle_path.endswith(".zip"):
            import zipfile
            print(f"[*] Extracting Colab simulation bundle: {os.path.basename(bundle_path)}...")
            with zipfile.ZipFile(bundle_path, 'r') as zf:
                zf.extractall(out_dir)
        elif os.path.isdir(bundle_path):
            print(f"[*] Copying simulation bundle files from directory: {os.path.basename(bundle_path)}...")
            for root, dirs, files in os.walk(bundle_path):
                for f in files:
                    src_f = os.path.join(root, f)
                    rel_f = os.path.relpath(src_f, bundle_path)
                    dst_f = os.path.join(out_dir, rel_f)
                    safe_copy(src_f, dst_f)
        elif bundle_path.endswith((".splat", ".ply")):
            safe_copy(bundle_path, os.path.join(out_dir, os.path.basename(bundle_path)))

        # Auto-discover extracted splat, mesh, and texture files
        found_splat = None
        found_ply = None
        found_obj = None
        found_atlas = None
        found_bin = None

        for root, dirs, files in os.walk(out_dir):
            for f in files:
                full_p = os.path.join(root, f)
                lower_f = f.lower()
                if lower_f.endswith(".splat") and not found_splat:
                    found_splat = full_p
                elif lower_f.endswith(".ply") and "3dgs" in lower_f and not found_ply:
                    found_ply = full_p
                elif lower_f.endswith(".obj") and not found_obj:
                    found_obj = full_p
                elif lower_f.endswith((".png", ".jpg", ".jpeg")) and ("atlas" in lower_f or "diffuse" in lower_f) and not found_atlas:
                    found_atlas = full_p
                elif lower_f.endswith(".bin") and not found_bin:
                    found_bin = full_p

        # Fallbacks if specific filenames not matched
        if not found_ply:
            for root, dirs, files in os.walk(out_dir):
                for f in files:
                    if f.lower().endswith(".ply") and "mesh" not in f.lower():
                        found_ply = os.path.join(root, f)
                        break

        root_splat_dst = os.path.join(root_output_dir, "scene.splat")
        if found_splat:
            safe_copy(found_splat, root_splat_dst)
            safe_copy(found_splat, os.path.join(out_dir, "scene.splat"))
        elif found_ply:
            safe_copy(found_ply, os.path.join(root_output_dir, "scene.ply"))

        if found_obj:
            safe_copy(found_obj, os.path.join(root_output_dir, "solid_mesh.obj"))
            safe_copy(found_obj, os.path.join(out_dir, "solid_mesh.obj"))

        if found_atlas:
            safe_copy(found_atlas, os.path.join(root_output_dir, "diffuse_atlas.png"))
            safe_copy(found_atlas, os.path.join(out_dir, "diffuse_atlas.png"))

        manifest_data = {
            "scene_name": "drjohnson" if "drjohnson" in bundle_path.lower() else "Cloud GPU Trained Scene",
            "output_dir": out_dir,
            "splat_path": root_splat_dst if found_splat else (found_ply or ""),
            "obj_path": found_obj or "",
            "diffuse_png_path": found_atlas or "",
            "binary_mesh_path": found_bin or "",
            "source": "Google Colab 3DGS Model"
        }
        import json
        with open(os.path.join(root_output_dir, "latest_simulation.json"), "w") as f:
            json.dump(manifest_data, f, indent=2)

        print("\n" + "=" * 80)
        print(" [SUCCESS] COLAB 3D SIMULATION BUNDLE IMPORTED!")
        print(f" Target Directory: {out_dir}")
        print(f" 3D Splat Asset:   {found_splat or found_ply or 'Auto-Detected'}")
        print(f" OBJ Mesh Asset:   {found_obj or 'Auto-Detected'}")
        print(f" Diffuse 4K Atlas: {found_atlas or 'Auto-Detected'}")
        print("=" * 80 + "\n")

        if not args.no_display:
            if args.software:
                display_simulation_window(out_dir, scene_name="Colab Scene")
            else:
                from luther_display import launch_interactive_3d_viewport
                launch_interactive_3d_viewport(port=args.port)
        return

    # Direct launch of interactive 3D simulation player if no training args provided
    if not args.images and not args.scene:
        proof_dir = os.path.join(PROJECT_ROOT, "output")
        latest_json = os.path.join(PROJECT_ROOT, "output", "latest_simulation.json")
        scene_name = "simulation"
        if os.path.exists(latest_json):
            try:
                import json
                with open(latest_json, "r") as f:
                    lat_data = json.load(f)
                    if lat_data.get("output_dir"):
                        proof_dir = lat_data["output_dir"]
                    if lat_data.get("scene_name"):
                        scene_name = lat_data["scene_name"]
            except Exception:
                pass
        if args.software:
            display_simulation_window(proof_dir, scene_name=scene_name)
        else:
            from luther_display import launch_interactive_3d_viewport
            launch_interactive_3d_viewport(port=args.port)
        return

    # Process user-specified image path
    if args.scene and not args.images:
        if os.path.isdir(args.scene):
            args.images = args.scene

    if not args.images:
        print("[ERROR] Please provide the path to your input images using '--images <path>' or '-i <path>'.")
        sys.exit(1)

    # Normalize images directory path
    if os.path.isdir(args.images):
        sub_img = os.path.join(args.images, "images")
        if os.path.isdir(sub_img) and not any(f.lower().endswith((".jpg", ".png", ".jpeg")) for f in os.listdir(args.images)):
            args.images = sub_img
    elif not os.path.exists(args.images):
        print(f"[ERROR] The specified images path does not exist: {args.images}")
        sys.exit(1)

    iters = parse_iteration_count(args.iterations)
    scene_name = args.scene.lower() if args.scene else os.path.basename(os.path.normpath(args.images))
    if scene_name in ["images", "img", "photos", ""]:
        scene_name = os.path.basename(os.path.dirname(os.path.normpath(args.images)))

    # Output directory handling
    out_dir = args.output or os.path.join(PROJECT_ROOT, "output", f"{scene_name}_sim")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(" [*] TRAINING LUTHERICPU 3D MODEL & SYNTHESIZING SIMULATION")
    print(f" Images Path:     {args.images}")
    print(f" Iterations:      {iters:,}")
    print(f" OS Platform:     {'Windows (-w)' if args.windows else ('Linux (-l)' if args.linux else sys.platform)}")
    print(f" Output Path:     {out_dir}")
    print(f" SfM Engine:      {'Native Pure CPU SIFT SfM' if not args.colmap else 'External (' + args.colmap + ')'}")
    print(f" RAM Safety Cap:  {args.max_ram:.1f} GB (Pure CPU)")
    print("=" * 80 + "\n")

    model = LutherICPU(output_dir=out_dir, max_ram_gb=args.max_ram)
    result = model.simulate(
        images_path=args.images,
        colmap_path=args.colmap,
        iterations=iters,
        scene_name=scene_name,
        render_simulation=True
    )

    print("\n" + "=" * 80)
    print(" [*] TRAINING & SIMULATION COMPLETE!")
    print(f" Elapsed Time:     {result['elapsed_seconds']}s")
    print(f" Mesh Vertices:    {result['num_vertices']:,}")
    print(f" Mesh Triangles:   {result['num_faces']:,}")
    print(f" 3D OBJ Model:     {result['obj_path']}")
    print(f" Texture Atlas:    {result['diffuse_png_path']}")
    print("=" * 80 + "\n")

    # Automatically launch native desktop viewport displaying 60 FPS 3D simulation
    if not args.no_display:
        out_dir_path = result.get("output_dir") or out_dir
        if args.software:
            display_simulation_window(out_dir_path, scene_name=scene_name)
        else:
            from luther_display import launch_interactive_3d_viewport
            launch_interactive_3d_viewport(port=args.port)


if __name__ == "__main__":
    main()
