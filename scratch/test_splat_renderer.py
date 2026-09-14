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

def test_gaussian_splatting():
    colmap_path = r"uploads\truck_photos\sparse\0"
    images_path = r"F:\tandt_db\tandt\truck\images"
    
    cameras, views_dict, sparse_pcd = load_colmap_model(sparse_dir=colmap_path, images_dir=images_path)
    all_views = list(views_dict.values())
    hero_cam = all_views[0]
    
    # Filter points
    scene_centroid = np.median(sparse_pcd.positions, axis=0)
    p_dists = np.linalg.norm(sparse_pcd.positions - scene_centroid, axis=1)
    radius_limit = max(18.0, float(np.percentile(p_dists, 98.0) * 1.15))
    valid_pts_mask = p_dists <= radius_limit
    points = sparse_pcd.positions[valid_pts_mask]
    colors = sparse_pcd.colors[valid_pts_mask].astype(np.float32) / 255.0

    print(f"Total points for rendering: {len(points)}")

    # Compute adaptive point radii using k-NN in 3D
    t0 = time.time()
    tree = cKDTree(points[::4]) # sub-sampled for fast kNN
    dists, _ = tree.query(points, k=4)
    # Average distance to 3 nearest neighbors is local point spacing
    local_spacing = np.mean(dists[:, 1:4], axis=1)
    local_spacing = np.clip(local_spacing, 0.02, 0.35)
    print(f"k-NN spacing computed in {time.time() - t0:.2f}s, mean spacing: {local_spacing.mean():.4f}")

    # Render at 1920x1080
    W, H = 1920, 1080
    scale_x = W / hero_cam.intrinsics.width
    scale_y = H / hero_cam.intrinsics.height
    fx = hero_cam.intrinsics.fx * scale_x
    fy = hero_cam.intrinsics.fy * scale_y
    cx = hero_cam.intrinsics.cx * scale_x
    cy = hero_cam.intrinsics.cy * scale_y

    # Transform to camera space
    R = hero_cam.R
    t = hero_cam.tvec
    P_cam = (R @ points.T).T + t
    z = P_cam[:, 2]

    in_front = z > 0.15
    pts_c = P_cam[in_front]
    cols_c = colors[in_front]
    z_c = pts_c[:, 2]
    spacing_c = local_spacing[in_front]

    u_scr = (fx * pts_c[:, 0] / z_c) + cx
    v_scr = (fy * pts_c[:, 1] / z_c) + cy

    # Projected pixel radius for each Gaussian: r_pix = (fx / z) * spacing
    r_pix = np.clip((fx / np.maximum(z_c, 0.1)) * spacing_c * 1.5, 2.0, 15.0)

    onscreen = (
        (u_scr >= -r_pix) & (u_scr < W + r_pix) &
        (v_scr >= -r_pix) & (v_scr < H + r_pix)
    )
    u_scr = u_scr[onscreen]
    v_scr = v_scr[onscreen]
    z_c = z_c[onscreen]
    cols_c = cols_c[onscreen]
    r_pix = r_pix[onscreen]

    print(f"Onscreen Gaussians: {len(u_scr)}")

    # Sort front-to-back or back-to-front
    sort_order = np.argsort(z_c) # front to back
    u_scr = u_scr[sort_order]
    v_scr = v_scr[sort_order]
    z_c = z_c[sort_order]
    cols_c = cols_c[sort_order]
    r_pix = r_pix[sort_order]

    # Render with fast continuous accumulation
    accum_rgb = np.zeros((H, W, 3), dtype=np.float32)
    accum_w = np.zeros((H, W), dtype=np.float32)

    # Let's vectorize by splat radius bins or tile rasterizer
    t_render = time.time()
    
    # Rasterize each point with Gaussian kernel
    u_int = np.round(u_scr).astype(int)
    v_int = np.round(v_scr).astype(int)
    r_int = np.ceil(r_pix).astype(int)

    # Fast multi-radius splatting in numpy
    max_r = int(np.max(r_int))
    print(f"Max splat radius: {max_r}")

    for dy in range(-max_r, max_r + 1):
        for dx in range(-max_r, max_r + 1):
            d2 = dx**2 + dy**2
            # Mask points where this offset falls within point's footprint
            valid_pts = d2 <= (r_pix ** 2)
            if not np.any(valid_pts):
                continue
            
            vx = np.clip(v_int[valid_pts] + dy, 0, H - 1)
            ux = np.clip(u_int[valid_pts] + dx, 0, W - 1)
            
            # Gaussian weight: exp(-2.0 * d^2 / r^2)
            w_vals = np.exp(-1.5 * d2 / (r_pix[valid_pts] ** 2)).astype(np.float32)
            # Depth weight: closer points have slightly higher weight
            z_w = 1.0 / np.sqrt(np.maximum(z_c[valid_pts], 0.1))
            w_total = w_vals * z_w

            np.add.at(accum_rgb[:, :, 0], (vx, ux), cols_c[valid_pts, 0] * w_total)
            np.add.at(accum_rgb[:, :, 1], (vx, ux), cols_c[valid_pts, 1] * w_total)
            np.add.at(accum_rgb[:, :, 2], (vx, ux), cols_c[valid_pts, 2] * w_total)
            np.add.at(accum_w, (vx, ux), w_total)

    # Normalized reconstruction
    has_cov = accum_w > 1e-4
    rendered = np.zeros((H, W, 3), dtype=np.float32)
    rendered[has_cov] = accum_rgb[has_cov] / accum_w[has_cov, np.newaxis]

    print(f"Rasterization completed in {time.time() - t_render:.2f}s, coverage: {has_cov.mean()*100:.1f}%")

    out_u8 = (np.clip(rendered, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(out_u8).save(r"scratch\test_adaptive_splat.png")
    print("Saved test_adaptive_splat.png")

if __name__ == "__main__":
    test_gaussian_splatting()
