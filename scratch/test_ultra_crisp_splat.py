import os
import sys
import time
import math
import numpy as np
import cv2
from PIL import Image
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from luther_core.colmap_loader import load_colmap_model
from luther_core.types import CameraView

def test_ultra_crisp_simulation():
    t0 = time.time()
    colmap_path = r"uploads\truck_photos\sparse\0"
    images_path = r"F:\tandt_db\tandt\truck\images"
    
    cameras, views_dict, sparse_pcd = load_colmap_model(sparse_dir=colmap_path, images_dir=images_path)
    all_views = list(views_dict.values())
    hero_cam = all_views[0]

    # Pre-cache top source images
    cached_source_views = []
    for i, v in enumerate(all_views[:40]):
        if v.image_path and os.path.exists(v.image_path):
            img = Image.open(v.image_path).convert("RGB")
            if img.width > 1920:
                img = img.resize((1920, int(1920 * img.height / img.width)), Image.Resampling.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            cached_source_views.append((v, arr))
    print(f"Cached {len(cached_source_views)} source views in {time.time() - t0:.2f}s")

    # Filter points
    scene_centroid = np.median(sparse_pcd.positions, axis=0)
    p_dists = np.linalg.norm(sparse_pcd.positions - scene_centroid, axis=1)
    radius_limit = max(18.0, float(np.percentile(p_dists, 98.0) * 1.15))
    valid_pts_mask = p_dists <= radius_limit
    points = sparse_pcd.positions[valid_pts_mask]
    base_colors = sparse_pcd.colors[valid_pts_mask].astype(np.float32) / 255.0

    print(f"Total points: {len(points)}")

    # 1. Cleanse isolated black dots & dark noise in 3D Point Cloud
    t_clean = time.time()
    tree = cKDTree(points)
    dists, knn_idx = tree.query(points, k=6)
    local_spacing = np.mean(dists[:, 1:4], axis=1)
    local_spacing = np.clip(local_spacing, 0.015, 0.25)

    # Neighborhood color consensus
    neighbor_colors = base_colors[knn_idx[:, 1:]] # (N, 5, 3)
    median_neighbor_c = np.median(neighbor_colors, axis=1) # (N, 3)
    color_dist = np.linalg.norm(base_colors - median_neighbor_c, axis=1)
    
    # Identify black / dark outlier points
    is_dark_outlier = (color_dist > 0.35) & (np.mean(base_colors, axis=1) < 0.15) & (np.mean(median_neighbor_c, axis=1) > 0.25)
    print(f"Found and cleansed {np.sum(is_dark_outlier)} dark outlier points in {time.time() - t_clean:.2f}s")
    clean_colors = base_colors.copy()
    clean_colors[is_dark_outlier] = median_neighbor_c[is_dark_outlier]

    W, H = 1920, 1080
    scale_x = W / hero_cam.intrinsics.width
    scale_y = H / hero_cam.intrinsics.height
    fx = hero_cam.intrinsics.fx * scale_x
    fy = hero_cam.intrinsics.fy * scale_y
    cx = hero_cam.intrinsics.cx * scale_x
    cy = hero_cam.intrinsics.cy * scale_y

    # Project to target camera space
    R = hero_cam.R
    t = hero_cam.tvec
    P_cam = (R @ points.T).T + t
    z = P_cam[:, 2]

    in_front = (z > 0.15) & (z < 60.0)
    pts_w = points[in_front]
    pts_c = P_cam[in_front]
    base_c = clean_colors[in_front]
    z_c = pts_c[:, 2]
    spacing_c = local_spacing[in_front]

    u_scr = (fx * pts_c[:, 0] / z_c) + cx
    v_scr = (fy * pts_c[:, 1] / z_c) + cy

    # Crisp adaptive radius: smaller radius for sharp high-density regions, seamless overlap
    r_pix = np.clip((fx / np.maximum(z_c, 0.1)) * spacing_c * 1.15, 1.5, 9.0)

    onscreen = (
        (u_scr >= -r_pix) & (u_scr < W + r_pix) &
        (v_scr >= -r_pix) & (v_scr < H + r_pix)
    )
    u_scr = u_scr[onscreen]
    v_scr = v_scr[onscreen]
    z_c = z_c[onscreen]
    pts_w = pts_w[onscreen]
    base_c = base_c[onscreen]
    r_pix = r_pix[onscreen]

    num_visible = len(u_scr)
    print(f"Visible Gaussians: {num_visible}")

    # 2. Multi-View Projective Radiance with Cosine-Weighted Consensus
    t_rad = time.time()
    best_colors = base_c.copy()
    best_scores = np.full(num_visible, -1.0, dtype=np.float32)

    tgt_rays = pts_w - hero_cam.center
    tgt_dirs = tgt_rays / np.maximum(np.linalg.norm(tgt_rays, axis=1, keepdims=True), 1e-6)

    for src_view, src_rgb in cached_source_views:
        src_h, src_w, _ = src_rgb.shape
        src_P_cam = (src_view.R @ pts_w.T).T + src_view.tvec
        src_z = src_P_cam[:, 2]

        src_valid = src_z > 0.2
        if not np.any(src_valid):
            continue

        src_fx = src_view.intrinsics.fx * (src_w / src_view.intrinsics.width)
        src_fy = src_view.intrinsics.fy * (src_h / src_view.intrinsics.height)
        src_cx = src_view.intrinsics.cx * (src_w / src_view.intrinsics.width)
        src_cy = src_view.intrinsics.cy * (src_h / src_view.intrinsics.height)

        src_u = (src_fx * src_P_cam[:, 0] / np.maximum(src_z, 1e-4)) + src_cx
        src_v = (src_fy * src_P_cam[:, 1] / np.maximum(src_z, 1e-4)) + src_cy

        in_b = src_valid & (src_u >= 1) & (src_u < src_w - 2) & (src_v >= 1) & (src_v < src_h - 2)
        valid_idx = np.where(in_b)[0]
        if len(valid_idx) == 0:
            continue

        src_rays = pts_w[valid_idx] - src_view.center
        src_dirs = src_rays / np.maximum(np.linalg.norm(src_rays, axis=1, keepdims=True), 1e-6)
        cos_ang = np.sum(src_dirs * tgt_dirs[valid_idx], axis=1)

        # Bilinear sample
        u_p = src_u[valid_idx]
        v_p = src_v[valid_idx]
        u0 = np.floor(u_p).astype(int)
        u1 = u0 + 1
        v0 = np.floor(v_p).astype(int)
        v1 = v0 + 1
        du = (u_p - u0)[:, np.newaxis]
        dv = (v_p - v0)[:, np.newaxis]

        c00 = src_rgb[v0, u0]
        c10 = src_rgb[v0, u1]
        c01 = src_rgb[v1, u0]
        c11 = src_rgb[v1, u1]
        sampled = c00 * (1 - du) * (1 - dv) + c10 * du * (1 - dv) + c01 * (1 - du) * dv + c11 * du * dv

        # Outlier rejection with base albedo
        color_diff = np.linalg.norm(sampled - base_c[valid_idx], axis=1)
        consensus = np.exp(- (color_diff ** 2) / (2.0 * (0.30 ** 2)))
        score = (cos_ang ** 2) * consensus

        beats = score > best_scores[valid_idx]
        idx_beats = valid_idx[beats]
        best_scores[idx_beats] = score[beats]
        best_colors[idx_beats] = sampled[beats]

    print(f"Projective Radiance evaluated in {time.time() - t_rad:.2f}s")

    # 3. High-Fidelity Continuous Splatting with Depth Peeling / Sharp Depth Weighting
    t_rast = time.time()
    accum_rgb = np.zeros((H, W, 3), dtype=np.float32)
    accum_w = np.zeros((H, W), dtype=np.float32)

    # Sort front-to-back
    sort_order = np.argsort(z_c)
    u_scr = u_scr[sort_order]
    v_scr = v_scr[sort_order]
    z_c = z_c[sort_order]
    best_colors = best_colors[sort_order]
    r_pix = r_pix[sort_order]

    u_int = np.round(u_scr).astype(int)
    v_int = np.round(v_scr).astype(int)
    r_int = np.ceil(r_pix).astype(int)
    max_r = int(np.max(r_int))

    for dy in range(-max_r, max_r + 1):
        for dx in range(-max_r, max_r + 1):
            d2 = dx**2 + dy**2
            valid = d2 <= (r_pix ** 2)
            if not np.any(valid):
                continue
            
            vx = np.clip(v_int[valid] + dy, 0, H - 1)
            ux = np.clip(u_int[valid] + dx, 0, W - 1)

            # Crisp Gaussian kernel
            w_vals = np.exp(-2.0 * d2 / (r_pix[valid] ** 2)).astype(np.float32)
            # Front-biased depth weight: exp(-0.15 * z)
            z_w = np.exp(-0.10 * (z_c[valid] - np.min(z_c)))
            w_total = w_vals * z_w

            np.add.at(accum_rgb[:, :, 0], (vx, ux), best_colors[valid, 0] * w_total)
            np.add.at(accum_rgb[:, :, 1], (vx, ux), best_colors[valid, 1] * w_total)
            np.add.at(accum_rgb[:, :, 2], (vx, ux), best_colors[valid, 2] * w_total)
            np.add.at(accum_w, (vx, ux), w_total)

    # Normalized reconstruction
    has_cov = accum_w > 1e-4
    rendered = np.zeros((H, W, 3), dtype=np.float32)
    rendered[has_cov] = accum_rgb[has_cov] / accum_w[has_cov, np.newaxis]

    # Watertight Screen-Space Convex Hull infilling for sky/boundary continuity
    points_2d = np.column_stack([u_int, v_int])
    hull = cv2.convexHull(points_2d)
    solid_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(solid_mask, [hull], 1)

    is_unrendered = (solid_mask > 0) & (~has_cov)
    if np.any(is_unrendered):
        rendered_u8 = (np.clip(rendered, 0.0, 1.0) * 255).astype(np.uint8)
        rendered_u8 = cv2.inpaint(rendered_u8, is_unrendered.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        rendered = rendered_u8.astype(np.float32) / 255.0

    print(f"Gaussian Splat rasterization done in {time.time() - t_rast:.2f}s")

    # Anti-Speckle Cleanse
    out_u8 = (np.clip(rendered, 0.0, 1.0) * 255).astype(np.uint8)
    med = cv2.medianBlur(out_u8, 3)
    gray = cv2.cvtColor(out_u8, cv2.COLOR_RGB2GRAY)
    gray_med = cv2.cvtColor(med, cv2.COLOR_RGB2GRAY)
    is_speckle = (solid_mask > 0) & ((gray_med.astype(np.int16) - gray.astype(np.int16)) > 15)
    out_u8[is_speckle] = med[is_speckle]

    Image.fromarray(out_u8).save(r"scratch\test_ultra_crisp_splat.png")
    print("Saved test_ultra_crisp_splat.png")

if __name__ == "__main__":
    test_ultra_crisp_simulation()
