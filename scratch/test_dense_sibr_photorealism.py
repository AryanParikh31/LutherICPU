import os
import sys
import time
import math
import numpy as np
import cv2
from PIL import Image
from scipy.spatial import Delaunay, cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from luther_core.colmap_loader import load_colmap_model
from luther_core.types import CameraView

def test_dense_sibr_photorealism():
    t0 = time.time()
    colmap_path = r"uploads\truck_photos\sparse\0"
    images_path = r"F:\tandt_db\tandt\truck\images"
    
    cameras, views_dict, sparse_pcd = load_colmap_model(sparse_dir=colmap_path, images_dir=images_path)
    all_views = list(views_dict.values())
    
    # Target camera: View 16 (the exact hero side view with SAN PEDRO SQUARE MARKET logo)
    target_cam = views_dict[16]
    print(f"Target Cam: {target_cam.name}, path: {target_cam.image_path}")

    # Load source views
    source_views = []
    for vid, v in list(views_dict.items())[:60]:
        if v.image_path and os.path.exists(v.image_path):
            img = Image.open(v.image_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            source_views.append((v, arr))
    print(f"Loaded {len(source_views)} source views in {time.time() - t0:.2f}s")

    # Filter points
    scene_centroid = np.median(sparse_pcd.positions, axis=0)
    p_dists = np.linalg.norm(sparse_pcd.positions - scene_centroid, axis=1)
    radius_limit = max(18.0, float(np.percentile(p_dists, 98.0) * 1.15))
    valid_pts_mask = p_dists <= radius_limit
    points = sparse_pcd.positions[valid_pts_mask]
    point_colors = sparse_pcd.colors[valid_pts_mask].astype(np.float32) / 255.0

    W, H = 1920, 1080
    scale_x = W / target_cam.intrinsics.width
    scale_y = H / target_cam.intrinsics.height
    fx = target_cam.intrinsics.fx * scale_x
    fy = target_cam.intrinsics.fy * scale_y
    cx = target_cam.intrinsics.cx * scale_x
    cy = target_cam.intrinsics.cy * scale_y

    # Project to target camera space
    R = target_cam.R
    t = target_cam.tvec
    P_cam = (R @ points.T).T + t
    z = P_cam[:, 2]

    in_front = (z > 0.15) & (z < 60.0)
    pts_c = P_cam[in_front]
    pts_w = points[in_front]
    cols_c = point_colors[in_front]
    z_c = pts_c[:, 2]

    u_scr = (fx * pts_c[:, 0] / z_c) + cx
    v_scr = (fy * pts_c[:, 1] / z_c) + cy

    onscreen = (u_scr >= 0) & (u_scr < W) & (v_scr >= 0) & (v_scr < H)
    u_scr = u_scr[onscreen]
    v_scr = v_scr[onscreen]
    z_c = z_c[onscreen]
    pts_w = pts_w[onscreen]
    cols_c = cols_c[onscreen]

    # 1. Triangulate into continuous 3D surface mesh
    t_del = time.time()
    pts_2d = np.column_stack([u_scr, v_scr])
    tri = Delaunay(pts_2d)
    simplices = tri.simplices

    # Filter invalid triangles
    v0, v1, v2 = simplices[:, 0], simplices[:, 1], simplices[:, 2]
    z0, z1, z2 = z_c[v0], z_c[v1], z_c[v2]
    max_z_diff = np.maximum(np.abs(z0 - z1), np.maximum(np.abs(z1 - z2), np.abs(z2 - z0)))
    min_z = np.minimum(z0, np.minimum(z1, z2))
    valid_depth_tri = (max_z_diff / np.maximum(min_z, 0.1)) < 0.40

    p0_2d, p1_2d, p2_2d = pts_2d[v0], pts_2d[v1], pts_2d[v2]
    e01 = np.linalg.norm(p0_2d - p1_2d, axis=1)
    e12 = np.linalg.norm(p1_2d - p2_2d, axis=1)
    e20 = np.linalg.norm(p2_2d - p0_2d, axis=1)
    max_edge = np.maximum(e01, np.maximum(e12, e20))
    valid_edge_tri = max_edge < 60.0

    valid_tri_mask = valid_depth_tri & valid_edge_tri
    valid_simplices = simplices[valid_tri_mask]
    print(f"Delaunay mesh constructed with {len(valid_simplices)} triangles in {time.time() - t_del:.2f}s")

    # 2. Dense Continuous Z-Buffer & World Coordinate Map Rasterization
    t_rast = time.time()
    z_buffer = np.full((H, W), np.inf, dtype=np.float32)
    world_pos_map = np.zeros((H, W, 3), dtype=np.float32)

    # Sort triangles back-to-front
    tri_z = (z_c[valid_simplices[:, 0]] + z_c[valid_simplices[:, 1]] + z_c[valid_simplices[:, 2]]) / 3.0
    sort_order = np.argsort(tri_z) # front to back
    sorted_simplices = valid_simplices[sort_order]

    # Render each triangle into dense continuous z-buffer and world coordinates
    # We can rasterize depth efficiently:
    mask_rendered = np.zeros((H, W), dtype=np.uint8)

    # Draw continuous depth using OpenCV polyfill for solid coverage
    for s in sorted_simplices:
        pts = pts_2d[s].astype(np.int32)
        mean_depth = float(np.mean(z_c[s]))
        # Draw on mask
        cv2.fillConvexPoly(mask_rendered, pts, 1)

    # Reconstruct continuous world positions from smooth z-buffer
    # Build continuous depth field via distance transform + bilateral filter
    pts_int = np.clip(np.round(pts_2d).astype(int), 0, [W-1, H-1])
    # Splat sparse depth
    sparse_z = np.full((H, W), np.nan, dtype=np.float32)
    # Front-to-back assign
    sort_pts = np.argsort(z_c)[::-1]
    for idx in sort_pts:
        ux, vy = pts_int[idx, 0], pts_int[idx, 1]
        sparse_z[vy, ux] = z_c[idx]

    # Smooth continuous depth interpolation
    hull = cv2.convexHull(pts_int)
    solid_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(solid_mask, [hull], 1)

    # Inpaint / interpolate depth smoothly
    valid_z_mask = ~np.isnan(sparse_z)
    from scipy.ndimage import distance_transform_edt
    _, ind = distance_transform_edt(~valid_z_mask, return_indices=True)
    dense_z = sparse_z[ind[0], ind[1]]
    # Bilateral smoothing of depth field
    dense_z = cv2.bilateralFilter(dense_z.astype(np.float32), d=15, sigmaColor=0.25, sigmaSpace=15.0)

    # Compute continuous 3D world coordinates for every pixel
    y_grid, x_grid = np.where(solid_mask > 0)
    pix_z = dense_z[y_grid, x_grid]
    x_c = (x_grid + 0.5 - cx) * pix_z / fx
    y_c = (y_grid + 0.5 - cy) * pix_z / fy
    pix_P_cam = np.column_stack([x_c, y_c, pix_z])
    pix_P_world = (target_cam.R.T @ (pix_P_cam - target_cam.tvec).T).T

    print(f"Continuous 3D world field synthesized ({len(y_grid)} pixels) in {time.time() - t_rast:.2f}s")

    # 3. Dense Multi-View Projective Radiance Synthesis
    t_sibr = time.time()
    num_pix = len(y_grid)
    accum_color = np.zeros((num_pix, 3), dtype=np.float32)
    accum_weight = np.zeros(num_pix, dtype=np.float32)

    tgt_rays = pix_P_world - target_cam.center
    tgt_dirs = tgt_rays / np.maximum(np.linalg.norm(tgt_rays, axis=1, keepdims=True), 1e-6)

    # Select top source cameras closest to target view
    cam_dists = []
    for src_view, src_rgb in source_views:
        # Distance between optical centers
        center_dist = np.linalg.norm(src_view.center - target_cam.center)
        # Angular difference between optical axes
        src_fwd = src_view.R[2, :]
        tgt_fwd = target_cam.R[2, :]
        cos_fwd = np.dot(src_fwd, tgt_fwd)
        cam_dists.append((center_dist, cos_fwd, src_view, src_rgb))

    # Sort by closest optical alignment
    cam_dists.sort(key=lambda x: -x[1]) # highest cosine alignment first

    for center_dist, cos_fwd, src_view, src_rgb in cam_dists[:12]:
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
        valid_idx = np.where(in_b)[0]
        if len(valid_idx) == 0:
            continue

        src_rays = pix_P_world[valid_idx] - src_view.center
        src_dirs = src_rays / np.maximum(np.linalg.norm(src_rays, axis=1, keepdims=True), 1e-6)
        cos_ang = np.sum(src_dirs * tgt_dirs[valid_idx], axis=1)

        # High-precision bilinear sampling from high-res source photograph
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

        # Angular weight (cos theta)^12 for razor-sharp selection of best camera
        w = np.maximum(0.0, cos_ang) ** 12

        accum_color[valid_idx] += sampled * w[:, np.newaxis]
        accum_weight[valid_idx] += w

    has_rad = accum_weight > 1e-5
    final_rgb = np.zeros((num_pix, 3), dtype=np.float32)
    final_rgb[has_rad] = accum_color[has_rad] / accum_weight[has_rad, np.newaxis]

    framebuffer = np.full((H, W, 3), (15, 20, 30), dtype=np.uint8)
    framebuffer[y_grid[has_rad], x_grid[has_rad]] = (np.clip(final_rgb[has_rad], 0.0, 1.0) * 255).astype(np.uint8)

    print(f"Dense Projective Radiance completed in {time.time() - t_sibr:.2f}s")
    out_path = r"scratch\photoreal_cam16_rendered.png"
    Image.fromarray(framebuffer).save(out_path)
    print(f"Saved {out_path}")

    # Compare with real 000016.jpg
    gt16 = cv2.imread(target_cam.image_path)
    gt16_resized = cv2.resize(gt16, (W, H))
    psnr = cv2.PSNR(gt16_resized, framebuffer)
    mae = np.mean(np.abs(gt16_resized.astype(float) - framebuffer.astype(float)))
    print(f"--- PHOTOREALISM QUALITY METRICS ---")
    print(f"PSNR vs Real DSLR Photo: {psnr:.2f} dB")
    print(f"Mean Absolute Error: {mae:.2f}")

if __name__ == "__main__":
    test_dense_sibr_photorealism()
