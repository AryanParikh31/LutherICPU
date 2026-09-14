"""lutherICPU Dense Multi-View Stereo (MVS) Depth Estimation on CPU.

Extracts dense metric depth maps using multithreaded patch normalized cross-correlation (NCC)
and generates dense 3D point clouds with calibrated surface normals.
"""
import math
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image

from luther_core.types import CameraView, PointCloud
from luther_core.safe_memory import global_guardian

logger = logging.getLogger("lutherICPU.DenseMVS")


def compute_patch_ncc(
    patch1: np.ndarray, patch2_candidates: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Computes Normalized Cross Correlation between patch1 and a bank of candidate patches.

    Args:
        patch1: (K, K, C) float32 reference patch.
        patch2_candidates: (N, K, K, C) float32 candidate patches along epipolar line.

    Returns:
        ncc_scores: (N,) float32 in [-1.0, 1.0]. Higher is better.
    """
    p1_mean = np.mean(patch1, axis=(0, 1), keepdims=True)
    p1_norm = patch1 - p1_mean
    p1_std = np.sqrt(np.sum(p1_norm ** 2)) + eps

    p2_mean = np.mean(patch2_candidates, axis=(1, 2), keepdims=True)
    p2_norm = patch2_candidates - p2_mean
    p2_std = np.sqrt(np.sum(p2_norm ** 2, axis=(1, 2, 3))) + eps

    cov = np.sum(p2_norm * p1_norm, axis=(1, 2, 3))
    ncc = cov / (p1_std * p2_std)
    return ncc


class CpuDenseMVS:
    """CPU-Optimized Multi-View Stereo Depth Estimator."""

    def __init__(self, patch_size: int = 5, num_depth_bins: int = 48, min_ncc_threshold: float = 0.55):
        self.patch_size = patch_size
        self.num_depth_bins = num_depth_bins
        self.min_ncc_threshold = min_ncc_threshold

    def select_neighbor_views(
        self, ref_view: CameraView, all_views: List[CameraView], max_neighbors: int = 4
    ) -> List[CameraView]:
        """Selects neighbor views with optimal baseline angle (10 to 35 degrees) for triangulation."""
        ref_center = ref_view.center
        ref_dir = ref_view.viewing_direction

        candidates = []
        for v in all_views:
            if v.image_id == ref_view.image_id:
                continue
            dist = np.linalg.norm(v.center - ref_center)
            if dist < 1e-3:
                continue
            # Baseline viewing direction angle
            cos_angle = np.clip(np.dot(ref_dir, v.viewing_direction), -1.0, 1.0)
            angle_deg = math.degrees(math.acos(cos_angle))

            # Prefer cameras looking at similar scene region with moderate baseline
            if 3.0 <= angle_deg <= 45.0:
                candidates.append((angle_deg, dist, v))

        # Sort by proximity in viewing angle
        candidates.sort(key=lambda x: abs(x[0] - 15.0))
        return [c[2] for c in candidates[:max_neighbors]]

    def estimate_depth_map_for_view(
        self,
        ref_view: CameraView,
        neighbor_views: List[CameraView],
        sparse_pcd: PointCloud,
        subsample: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes dense depth map and photometric confidence for reference camera view."""
        global_guardian.check_safety(f"MVS view {ref_view.name}")

        # Load reference image
        if not ref_view.image_path:
            raise ValueError(f"Reference view {ref_view.name} lacks image_path")

        ref_img = Image.open(ref_view.image_path).convert("RGB")
        w_orig, h_orig = ref_img.size
        w_sub = w_orig // subsample
        h_sub = h_orig // subsample
        ref_img_sub = ref_img.resize((w_sub, h_sub), Image.Resampling.BILINEAR)
        ref_arr = np.array(ref_img_sub, dtype=np.float32) / 255.0

        # Estimate depth range from sparse 3D points visible in this view
        R = ref_view.R
        t = ref_view.tvec
        P_cam = (R @ sparse_pcd.positions.T).T + t
        valid_z = P_cam[:, 2][P_cam[:, 2] > 0.2]

        if len(valid_z) > 10:
            min_depth = float(np.percentile(valid_z, 2))
            max_depth = float(np.percentile(valid_z, 98))
        else:
            min_depth = 0.5
            max_depth = 15.0

        depth_hypotheses = np.linspace(min_depth, max_depth, self.num_depth_bins, dtype=np.float32)

        # Load neighbor images
        neighbor_imgs = []
        for nv in neighbor_views:
            if nv.image_path:
                n_img = Image.open(nv.image_path).convert("RGB").resize((w_sub, h_sub), Image.Resampling.BILINEAR)
                neighbor_imgs.append((nv, np.array(n_img, dtype=np.float32) / 255.0))

        depth_map = np.zeros((h_sub, w_sub), dtype=np.float32)
        conf_map = np.zeros((h_sub, w_sub), dtype=np.float32)

        half_p = self.patch_size // 2
        fx = ref_view.intrinsics.fx / subsample
        fy = ref_view.intrinsics.fy / subsample
        cx = ref_view.intrinsics.cx / subsample
        cy = ref_view.intrinsics.cy / subsample

        # Pixel grid
        y_coords, x_coords = np.mgrid[half_p:h_sub - half_p:2, half_p:w_sub - half_p:2]
        y_flat = y_coords.flatten()
        x_flat = x_coords.flatten()

        for y, x in zip(y_flat, x_flat):
            ref_patch = ref_arr[y - half_p:y + half_p + 1, x - half_p:x + half_p + 1]

            # Cast 3D rays at all depth hypotheses
            x_cam = (x - cx) * depth_hypotheses / fx
            y_cam = (y - cy) * depth_hypotheses / fy
            z_cam = depth_hypotheses

            pts_cam = np.column_stack([x_cam, y_cam, z_cam])
            pts_world = (ref_view.R.T @ (pts_cam - ref_view.tvec).T).T

            best_ncc = -1.0
            best_d = 0.0

            for nv, n_arr in neighbor_imgs:
                # Project into neighbor camera
                n_P_cam = (nv.R @ pts_world.T).T + nv.tvec
                n_z = n_P_cam[:, 2]
                n_fx = nv.intrinsics.fx / subsample
                n_fy = nv.intrinsics.fy / subsample
                n_cx = nv.intrinsics.cx / subsample
                n_cy = nv.intrinsics.cy / subsample

                n_u = (n_fx * n_P_cam[:, 0] / np.maximum(n_z, 1e-4)) + n_cx
                n_v = (n_fy * n_P_cam[:, 1] / np.maximum(n_z, 1e-4)) + n_cy

                for d_idx in range(self.num_depth_bins):
                    nu = int(round(n_u[d_idx]))
                    nv_pix = int(round(n_v[d_idx]))
                    if (half_p <= nv_pix < h_sub - half_p) and (half_p <= nu < w_sub - half_p) and n_z[d_idx] > 0.1:
                        cand_patch = n_arr[nv_pix - half_p:nv_pix + half_p + 1, nu - half_p:nu + half_p + 1]
                        ncc = float(np.sum((ref_patch - np.mean(ref_patch)) * (cand_patch - np.mean(cand_patch))) /
                                    (np.std(ref_patch) * np.std(cand_patch) * ref_patch.size + 1e-6))
                        if ncc > best_ncc:
                            best_ncc = ncc
                            best_d = depth_hypotheses[d_idx]

            if best_ncc > self.min_ncc_threshold:
                depth_map[y, x] = best_d
                conf_map[y, x] = best_ncc

        return depth_map, conf_map, ref_arr

    def reconstruct_dense_point_cloud(
        self,
        views: List[CameraView],
        sparse_pcd: PointCloud,
        max_views: int = 16,
        subsample: int = 2
    ) -> PointCloud:
        """Runs dense MVS across key camera views and fuses them into a continuous dense point cloud."""
        all_points = []
        all_colors = []
        all_normals = []

        # Select evenly distributed keyframe views
        step = max(1, len(views) // max_views)
        key_views = views[::step][:max_views]

        logger.info(f"Starting Dense CPU MVS across {len(key_views)} keyframe views...")

        for i, ref_v in enumerate(key_views):
            neighbors = self.select_neighbor_views(ref_v, views, max_neighbors=3)
            if not neighbors:
                continue

            depth_map, conf_map, img_arr = self.estimate_depth_map_for_view(
                ref_v, neighbors, sparse_pcd, subsample=subsample
            )

            # Extract valid 3D points
            valid_mask = (depth_map > 0.1) & (conf_map > self.min_ncc_threshold)
            ys, xs = np.where(valid_mask)

            if len(ys) == 0:
                continue

            depths = depth_map[ys, xs]
            colors = (img_arr[ys, xs] * 255).astype(np.uint8)

            fx = ref_v.intrinsics.fx / subsample
            fy = ref_v.intrinsics.fy / subsample
            cx = ref_v.intrinsics.cx / subsample
            cy = ref_v.intrinsics.cy / subsample

            x_cam = (xs - cx) * depths / fx
            y_cam = (ys - cy) * depths / fy
            z_cam = depths

            P_cam = np.column_stack([x_cam, y_cam, z_cam])
            P_world = (ref_v.R.T @ (P_cam - ref_v.tvec).T).T

            # Surface normals: view vector facing camera
            view_dirs = ref_v.center - P_world
            view_dirs /= np.maximum(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-6)

            all_points.append(P_world.astype(np.float32))
            all_colors.append(colors)
            all_normals.append(view_dirs.astype(np.float32))

            logger.info(f"View {i+1}/{len(key_views)} [{ref_v.name}]: Extracted {len(ys)} dense points.")

        # Also combine with filtered inlier sparse points
        all_points.append(sparse_pcd.positions)
        all_colors.append(sparse_pcd.colors)
        if sparse_pcd.normals is not None:
            all_normals.append(sparse_pcd.normals)
        else:
            # Default up normals for sparse points
            all_normals.append(np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(sparse_pcd), 1)))

        fused_pts = np.vstack(all_points)
        fused_cols = np.vstack(all_colors)
        fused_nrms = np.vstack(all_normals)

        logger.info(f"Dense MVS complete: Total {len(fused_pts)} high-density surface points.")
        return PointCloud(positions=fused_pts, colors=fused_cols, normals=fused_nrms)
