"""
Stage 1: Ingestion & Quality Control (QC) Module for lutherICPU.

Implements:
1. SHA-256 cryptographic hashing for every input photograph (chain-of-custody anchor).
2. Laplacian kernel edge variance calculation for blur and focus detection:
       [ 0   1   0 ]
   ∇²I = [ 1  -4   1 ]
       [ 0   1   0 ]
   blur_score = Var(∇²I)
3. Dynamic range, exposure, resolution analysis, and QC scoring.
"""

import os
import sys
import json
import argparse
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger("lutherICPU.IngestionQC")


def compute_sha256(filepath: str) -> str:
    """Computes cryptographic SHA-256 hash of an image file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_laplacian_variance(image_gray_or_rgb: np.ndarray) -> float:
    """
    Computes the variance of the Laplacian of the grayscale image:
        blur_score = Var(∇²I)

    A sharp photo has high-frequency edge content -> high variance.
    A blurry photo suppresses those edges -> low variance.
    """
    if len(image_gray_or_rgb.shape) == 3:
        gray = cv2.cvtColor(image_gray_or_rgb, cv2.COLOR_BGR2GRAY if image_gray_or_rgb.dtype == np.uint8 else cv2.COLOR_RGB2GRAY)
    else:
        gray = image_gray_or_rgb

    # 3x3 Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)
    variance = float(np.var(lap))
    return variance


class ImageQualityController:
    """Stage 1: Image Ingestion and Quality Control Validator."""

    def __init__(self, blur_threshold: float = 60.0):
        self.blur_threshold = blur_threshold

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Performs complete QC analysis on a single input photograph."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        file_size_bytes = os.path.getsize(image_path)
        sha256_hash = compute_sha256(image_path)

        # Load image with OpenCV
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Failed to decode image: {image_path}")

        height, width, channels = img_bgr.shape
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        blur_score = compute_laplacian_variance(gray)
        is_blurry = blur_score < self.blur_threshold

        # Mean brightness and contrast
        mean_brightness = float(np.mean(gray))
        contrast_std = float(np.std(gray))

        # Dynamic range saturation checks
        underexposed_pct = float(np.sum(gray < 10) / gray.size * 100.0)
        overexposed_pct = float(np.sum(gray > 245) / gray.size * 100.0)

        # Composite quality rating [0.0 - 100.0]
        sharpness_factor = min(1.0, blur_score / 250.0)
        contrast_factor = min(1.0, contrast_std / 50.0)
        exposure_factor = max(0.0, 1.0 - (underexposed_pct + overexposed_pct) / 100.0)
        overall_qc_score = float(round((0.5 * sharpness_factor + 0.3 * contrast_factor + 0.2 * exposure_factor) * 100.0, 1))

        status = "PASSED"
        warnings = []
        if is_blurry:
            status = "FLAGGED_BLUR"
            warnings.append(f"Image has low edge sharpness (blur score: {blur_score:.1f} < {self.blur_threshold})")
        if underexposed_pct > 25.0:
            warnings.append(f"High underexposure: {underexposed_pct:.1f}% dark pixels")
        if overexposed_pct > 25.0:
            warnings.append(f"High overexposure: {overexposed_pct:.1f}% clipped highlight pixels")

        return {
            "filename": os.path.basename(image_path),
            "filepath": os.path.abspath(image_path),
            "sha256": sha256_hash,
            "width": width,
            "height": height,
            "aspect_ratio": round(width / max(1, height), 3),
            "file_size_bytes": file_size_bytes,
            "blur_score": round(blur_score, 2),
            "mean_brightness": round(mean_brightness, 2),
            "contrast_std": round(contrast_std, 2),
            "underexposed_pct": round(underexposed_pct, 2),
            "overexposed_pct": round(overexposed_pct, 2),
            "overall_qc_score": overall_qc_score,
            "status": status,
            "warnings": warnings
        }

    def ingest_dataset(
        self,
        images_dir: str,
        supported_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    ) -> Dict[str, Any]:
        """
        Ingests all photographs from a folder, runs QC, and generates manifest.
        """
        if not os.path.isdir(images_dir):
            raise NotADirectoryError(f"Directory not found: {images_dir}")

        # Check if images are inside an 'images' subfolder
        sub_img = os.path.join(images_dir, "images")
        if os.path.isdir(sub_img) and not any(f.lower().endswith(supported_extensions) for f in os.listdir(images_dir)):
            images_dir = sub_img

        image_files = sorted([
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith(supported_extensions)
        ])

        if not image_files:
            raise ValueError(f"No valid images found in {images_dir} with extensions {supported_extensions}")

        logger.info(f"Stage 1: Ingesting and QC analyzing {len(image_files)} photographs from '{images_dir}'...")

        qc_records = []
        total_blur_score = 0.0
        flagged_count = 0

        for img_p in image_files:
            qc = self.analyze_image(img_p)
            qc_records.append(qc)
            total_blur_score += qc["blur_score"]
            if qc["status"] != "PASSED":
                flagged_count += 1
                logger.warning(f"QC Flag on [{qc['filename']}]: {', '.join(qc['warnings'])}")

        avg_blur = total_blur_score / len(image_files)
        logger.info(
            f"Stage 1 Complete: {len(image_files)} images ingested. "
            f"Avg blur score: {avg_blur:.1f}, Flagged: {flagged_count}/{len(image_files)}."
        )

        return {
            "total_images": len(image_files),
            "flagged_images": flagged_count,
            "average_blur_score": round(avg_blur, 2),
            "records": qc_records,
            "image_paths": [r["filepath"] for r in qc_records]
        }


def main():
    """CLI entrypoint for Stage 1 Camera Ingestion and QC."""
    parser = argparse.ArgumentParser(description="LutherICPU Native Image Ingestion & Quality Control (QC)")
    parser.add_argument("--images", "-i", required=True, help="Path to input dataset images folder")
    parser.add_argument("--output", "-o", default="./scene_output", help="Path to output directory for QC reports")
    parser.add_argument("--blur-threshold", "-b", type=float, default=60.0, help="Laplacian variance threshold for blur detection (default: 60.0)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 80)
    print("[lutherICPU] Stage 1: Native Camera Ingestion & Forensic Quality Control")
    print("=" * 80)
    print(f"[*] Ingesting images from: {args.images}")
    print(f"[*] Output directory:     {args.output}")
    print(f"[*] Blur threshold:       {args.blur_threshold}")

    controller = ImageQualityController(blur_threshold=args.blur_threshold)
    manifest = controller.ingest_dataset(args.images)

    manifest_path = os.path.join(args.output, "ingestion_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("[SUCCESS] Stage 1 Ingestion & Quality Control Completed!")
    print(f"Total Ingested Images: {manifest['total_images']}")
    print(f"Flagged (Blurry/Warn): {manifest['flagged_images']}")
    print(f"Average Blur Score:    {manifest['average_blur_score']:.2f}")
    print(f"Ingestion Manifest:    {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

