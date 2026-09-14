import os
import sys
import time
import math
import numpy as np
import cv2
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from luther_core.colmap_loader import load_colmap_model
from luther_core.types import CameraView

def test_smooth_continuous_sibr():
    colmap_path = r"uploads\truck_photos\sparse\0"
    images_path = r"F:\tandt_db\tandt\truck\images"
    
    cameras, views_dict, sparse_pcd = load_colmap_model(sparse_dir=colmap_path, images_dir=images_path)
    all_views = list(views_dict.values())
    hero_cam = all_views[0]

    # Pre-cache source DSLR images
    cached_source_views = []
    for i, v in enumerate(all_views[:28]):
        if v.image_path and os.path.exists(v.image_path):
            img = Image.open(v.image_path).convert("RGB")
            if img.width > 1920:
                img = img.resize((1920, int(1920 * img.height / img.width)), Image.Resampling.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            cached_source_views.append((v, arr))
    print(f"Cached {len(cached_source_views)} source views.")

    # Filter points
    scene_centroid = np.median(sparse_pcd.positions, axis=0)
    p_dists = np.linalg.norm(sparse_pcd.positions - scene_centroid, axis=1)
    radius_limit = max(18.0, float(np.percentile(p_dists, 98.0) * 1.15))
    valid_pts_mask = p_dists <= radius_limit
    points = sparse_pcd.positions[valid_pts_mask]
    point_colors = sparse_pcd.colors[valid_pts_mask].astype(np.float32) / 255.0

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
    pts_c = P_cam[in_front]
    cols_c = point_colors[in_front]
    z_c = pts_c[:, 2]

    u_scr = (fx * pts_c[:, 0] / z_c) + cx
    v_scr = (fy * pts_c[:, 1] / z_c) + cy

    onscreen = (u_scr >= 0) & (u_scr < W) & (v_scr >= 0) & (v_scr < H)
    u_scr = u_scr[onscreen]
    v_scr = v_scr[onscreen]
    z_c = z_c[onscreen]
    cols_c = cols_c[onscreen]

    u_pix = np.clip(np.round(u_scr).astype(int), 0, W - 1)
    v_pix = np.clip(np.round(v_scr).astype(int), 0, H - 1)

    # 1. Compute smooth continuous depth & albedo via Gaussian Splat accumulation
    t0 = time.time()
    accum_z = np.zeros((H, W), dtype=np.float32)
    accum_rgb = np.zeros((H, W, 3), dtype=np.float32)
    accum_w = np.zeros((H, W), dtype=np.float32)

    r_splat = 3
    for dy in range(-r_splat, r_splat + 1):
        for dx in range(-r_splat, r_splat + 1):
            d2 = dx**2 + dy**2
            if d2 > (r_splat**2):
                continue
            w_val = float(np.exp(-1.5 * d2 / (r_splat**2)))
            vx = np.clip(v_pix + dy, 0, H - 1)
            ux = np.clip(u_pix + dx, 0, W - 1)
            
            # Closer points get slightly higher priority to resolve occlusions correctly
            inv_z = 1.0 / np.maximum(z_c, 0.1)
            w_total = w_val * inv_z

            np.add.at(accum_z, (vx, ux), z_c * w_total)
            np.add.at(accum_rgb[:, :, 0], (vx, ux), cols_c[:, 0] * w_total)
            np.add.at(accum_rgb[:, :, 1], (vx, ux), cols_c[:, 1] * w_total)
            np.add.at(accum_rgb[:, :, 2], (vx, ux), cols_c[:, 2] * w_total)
            np.add.at(accum_w, (vx, ux), w_total)

    has_data = accum_w > 1e-4
    smooth_z = np.zeros((H, W), dtype=np.float32)
    smooth_z[has_data] = accum_z[has_data] / accum_w[has_data]

    smooth_albedo = np.zeros((H, W, 3), dtype=np.float32)
    smooth_albedo[has_data] = accum_rgb[has_data] / accum_w[has_data, np.newaxis]

    # Infill remaining sub-pixel holes using smooth bilateral depth inpainting
    # (NOT raw nearest-neighbor Voronoi!)
    points_2d = np.column_stack([u_pix, v_pix])
    hull = cv2.convexHull(points_2d)
    solid_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(solid_mask, [hull], 1)

    # Inpaint depth with smooth Navier-Stokes / Telea inpainting to guarantee smooth surface continuity
    inpaint_mask = ((has_data == 0) & (solid_mask > 0)).astype(np.uint8)
    
    # Normalize depth to 16-bit for OpenCV inpainting
    z_min, z_max = np.min(smooth_z[has_data]), np.max(smooth_z[has_data])
    norm_z = np.clip((smooth_z - z_min) / (z_max - z_min + 1e-5) * 65535, 0, 65535).astype(np.uint16)
    
    # Inpaint in float/8-bit or 16-bit
    norm_z_8 = (norm_z >> 8).astype(np.uint8)
    infilled_z_8 = cv2.inpaint(norm_z_8, inpaint_mask, 5, cv2.INPAINT_TELEA)
    # Recover smooth depth
    continuous_z = (infilled_z_8.astype(np.float32) / 255.0) * (z_max - z_min) + z_min

    # Inpaint albedo smoothly
    albedo_u8 = (np.clip(smooth_albedo, 0.0, 1.0) * 255).astype(np.uint8)
    infilled_albedo = cv2.inpaint(albedo_u8, inpaint_mask, 5, cv2.INPAINT_TELEA)
    continuous_albedo = infilled_albedo.astype(np.float32) / 255.0

    print(f"Smooth continuous depth field constructed in {time.time() - t0:.2f}s")

    # 2. Reconstruct Continuous 3D World Positions
    y_coords, x_coords = np.where(solid_mask > 0)
    pix_z = continuous_z[y_coords, x_coords]
    x_c = (x_coords + 0.5 - cx) * pix_z / fx
    y_c = (y_coords + 0.5 - cy) * pix_z / fy
    pix_P_cam = np.column_stack([x_c, y_c, pix_z])
    pix_P_world = (hero_cam.R.T @ (pix_P_cam - hero_cam.tvec).T).T

    pix_albedo = continuous_albedo[y_coords, x_coords]

    # 3. Continuous SIBR Radiance Sampling
    t_sibr = time.time()
    num_pix = len(y_coords)
    best_colors = np.zeros((num_pix, 3), dtype=np.float32)
    best_scores = np.full(num_pix, -1.0, dtype=np.float32)
    second_colors = np.zeros((num_pix, 3), dtype=np.float32)
    second_scores = np.full(num_pix, -1.0, dtype=np.float32)

    tgt_rays = pix_P_world - hero_cam.center
    tgt_dirs = tgt_rays / np.maximum(np.linalg.norm(tgt_rays, axis=1, keepdims=True), 1e-6)

    for src_view, src_rgb in cached_source_views:
        src_h, src_w, _ = src_rgb.shape
        src_P_cam = (src_view.R @ pix_P_world.T).T + src_view.tvec
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
        valid_p_idx = np.where(in_b)[0]
        if len(valid_p_idx) == 0:
            continue

        src_rays = pix_P_world[valid_p_idx] - src_view.center
        src_dirs = src_rays / np.maximum(np.linalg.norm(src_rays, axis=1, keepdims=True), 1e-6)
        cos_ang = np.sum(src_dirs * tgt_dirs[valid_p_idx], axis=1)

        # Bilinear sampling
        u_p = src_u[valid_p_idx]
        v_p = src_v[valid_p_idx]
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

        # Consensus factor
        color_diff = np.linalg.norm(sampled - pix_albedo[valid_p_idx], axis=1)
        consensus = np.exp(- (color_diff ** 2) / (2.0 * (0.35 ** 2)))
        score = cos_ang * consensus

        cur_best = best_scores[valid_p_idx]
        cur_second = second_scores[valid_p_idx]
        beats_best = score > cur_best
        beats_second = (~beats_best) & (score > cur_second)

        idx_bb = valid_p_idx[beats_best]
        second_scores[idx_bb] = best_scores[idx_bb]
        second_colors[idx_bb] = best_colors[idx_bb]
        best_scores[idx_bb] = score[beats_best]
        best_colors[idx_bb] = sampled[beats_best]

        idx_bs = valid_p_idx[beats_second]
        second_scores[idx_bs] = score[beats_second]
        second_colors[idx_bs] = sampled[beats_second]

    # Blend top 2 views
    w1 = np.maximum(0.0, best_scores) ** 4
    w2 = np.where(second_scores > 0.05, np.maximum(0.0, second_scores) ** 4, 0.0)
    tot_w = w1 + w2

    final_rgb = np.zeros((num_pix, 3), dtype=np.float32)
    valid_w = tot_w > 1e-4
    final_rgb[valid_w] = (
        best_colors[valid_w] * w1[valid_w, np.newaxis] +
        second_colors[valid_w] * w2[valid_w, np.newaxis]
    ) / tot_w[valid_w, np.newaxis]

    # Fallback to smooth albedo for any low-confidence pixels
    conf = np.clip(best_scores / 0.4, 0.0, 1.0)[:, np.newaxis]
    final_rgb = final_rgb * conf + pix_albedo * (1.0 - conf)

    # Assemble framebuffer
    framebuffer = np.full((H, W, 3), (15, 20, 30), dtype=np.uint8)
    framebuffer[y_coords, x_coords] = (np.clip(final_rgb, 0.0, 1.0) * 255).astype(np.uint8)

    print(f"Continuous SIBR rendering done in {time.time() - t_sibr:.2f}s")
    Image.fromarray(framebuffer).save(r"scratch\test_continuous_telea_sibr.png")
    print("Saved test_continuous_telea_sibr.png")

if __name__ == "__main__":
    test_smooth_continuous_sibr()
