"""lutherICPU Multi-Dataset Benchmark and Quality Verification Suite.

Runs end-to-end CPU reconstruction on Truck, Train, Dr. Johnson, and Playroom datasets,
benchmarking PSNR, SSIM, RAM safety, and rendering continuous non-dotted proof images.
"""
import os
import sys
import time
import json
import logging
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from luther_pipeline.reconstructor import LutherCPUReconstructor
from benchmark.evaluator import calculate_psnr, calculate_ssim, generate_side_by_side_comparison
from luther_core.safe_memory import global_guardian

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lutherICPU.BenchmarkSuite")

DATASETS = [
    {
        "name": "truck",
        "path": r"F:\tandt_db\tandt\truck",
        "images_subfolder": "images",
        "sparse_subfolder": "sparse/0",
        "target_view_name": "000001.jpg"
    },
    {
        "name": "train",
        "path": r"F:\tandt_db\tandt\train",
        "images_subfolder": "images",
        "sparse_subfolder": "sparse/0",
        "target_view_name": "000001.jpg"
    },
    {
        "name": "drjohnson",
        "path": r"F:\tandt_db\db\drjohnson",
        "images_subfolder": "images",
        "sparse_subfolder": "sparse/0",
        "target_view_name": "000001.jpg"
    },
    {
        "name": "playroom",
        "path": r"F:\tandt_db\db\playroom",
        "images_subfolder": "images",
        "sparse_subfolder": "sparse/0",
        "target_view_name": "000001.jpg"
    }
]


def run_full_benchmark(output_dir: str = "output") -> Dict[str, Any]:
    """Runs reconstruction and quality benchmarking on all available datasets."""
    os.makedirs(output_dir, exist_ok=True)
    reconstructor = LutherCPUReconstructor(output_dir=output_dir)
    results = {}

    for item in DATASETS:
        scene_name = item["name"]
        scene_path = item["path"]

        if not os.path.exists(scene_path):
            logger.warning(f"Dataset path does not exist: {scene_path}. Skipping...")
            continue

        logger.info(f"\n=======================================================")
        logger.info(f"   STARTING BENCHMARK: {scene_name.upper()}")
        logger.info(f"=======================================================\n")

        start_t = time.time()
        res = reconstructor.process_scene(
            scene_dir=scene_path,
            images_subfolder=item["images_subfolder"],
            sparse_subfolder=item["sparse_subfolder"],
            scene_name=scene_name,
            max_views_to_mesh=16,
            texture_resolution=2048,
            run_cpu_proof_render=True
        )
        elapsed = time.time() - start_t

        # Quality evaluation against reference ground truth photo
        proof_img_path = res.get("proof_img_path")
        psnr_val = 0.0
        ssim_val = 0.0

        if proof_img_path and os.path.exists(proof_img_path):
            ref_gt_path = os.path.join(scene_path, item["images_subfolder"], item["target_view_name"])
            if not os.path.exists(ref_gt_path):
                # Try finding first jpg in images
                img_dir = os.path.join(scene_path, item["images_subfolder"])
                first_imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.JPG'))]
                if first_imgs:
                    ref_gt_path = os.path.join(img_dir, first_imgs[0])

            if os.path.exists(ref_gt_path):
                pred_img = Image.open(proof_img_path).convert("RGB")
                gt_img = Image.open(ref_gt_path).convert("RGB").resize(pred_img.size, Image.Resampling.BILINEAR)

                pred_arr = np.array(pred_img)
                gt_arr = np.array(gt_img)

                psnr_val = calculate_psnr(pred_arr, gt_arr)
                ssim_val = calculate_ssim(pred_arr, gt_arr)

                comp_path = os.path.join(output_dir, f"{scene_name}_comparison.png")
                generate_side_by_side_comparison(
                    gt_img_path=ref_gt_path,
                    pred_img=pred_img,
                    output_path=comp_path,
                    scene_name=scene_name,
                    psnr_val=psnr_val,
                    ssim_val=ssim_val
                )

        ram_mb = global_guardian.process_memory_gb * 1024

        scene_metrics = {
            "scene_name": scene_name,
            "elapsed_seconds": round(elapsed, 2),
            "num_vertices": res["num_vertices"],
            "num_faces": res["num_faces"],
            "num_cameras": res["num_cameras"],
            "psnr_db": round(psnr_val, 2),
            "ssim": round(ssim_val, 4),
            "ram_mb": round(ram_mb, 1),
            "ply_path": res["ply_path"],
            "scene_json_path": res["scene_json_path"],
            "proof_img_path": proof_img_path,
            "dossier_path": res["dossier_path"]
        }

        results[scene_name] = scene_metrics
        logger.info(f"Finished {scene_name}: PSNR={psnr_val:.2f}dB, SSIM={ssim_val:.4f}, Elapsed={elapsed:.1f}s, RAM={ram_mb:.1f}MB")

    # Save summary report
    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n=== Benchmark Complete across all datasets. Summary saved to {summary_path} ===")
    return results


if __name__ == "__main__":
    run_full_benchmark()
