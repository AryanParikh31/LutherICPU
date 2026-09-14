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

def test_projective_triangle_texture_mapping():
    t0 = time.time()
    colmap_path = r"uploads\truck_photos\sparse\0"
    images_path = r"F:\tandt_db\tandt\truck\images"
    
    cameras, views_dict, sparse_pcd = load_colmap_model(sparse_dir=colmap_path, images_dir=images_path)
    all_views = list(views_dict.values())
    
    # Target camera: View 16 (the hero side view with SAN PEDRO SQUARE MARKET logo)
    target_cam = views_dict[16]
    print(f"Target Cam: {target_cam.name}")

    # Load high-resolution source views
    source_views = {}
    for vid, v in list(views_dict.items())[:60]:
        if v.image_path and os.path.exists(v.image_path):
            img = cv2.imread(v.image_path) # BGR
            source_views[vid] = (v, img)
    print(f"Loaded {len(source_views)} source images in {time.time() - t0:.2f}s")

    # Filter points
    scene_centroid = np.median(sparse_pcd.positions, axis=0)
    p_dists = np.linalg.norm(sparse_pcd.positions - scene_centroid, axis=1)
    radius_limit = max(18.0, float(np.percentile(p_dists, 98.0) * 1.15))
    valid_pts_mask = p_dists <= radius_limit
    points = sparse_pcd.positions[valid_pts_mask]
    point_colors = sparse_pcd.colors[valid_pts_mask]

    W, H = 1920, 1080
    scale_x = W / target_cam.intrinsics.width
    scale_y = H / target_cam.intrinsics.height
    fx = target_cam.intrinsics.fx * scale_x
    fy = target_cam.intrinsics.fy * scale_y
    cx = target_cam.intrinsics.cx * scale_x
    cy = target_cam.intrinsics.cy * scale_y

    # Project points to target camera
    R = target_cam.R
    t = target_cam.tvec
    P_cam = (R @ points.T).T + t
    z = P_cam[:, 2]

    in_front = (z > 0.15) & (z < 60.0)
    pts_w = points[in_front]
    pts_c = P_cam[in_front]
    cols_c = point_colors[in_front]
    z_c = pts_c[:, 2]

    u_scr = (fx * pts_c[:, 0] / z_c) + cx
    v_scr = (fy * pts_c[:, 1] / z_c) + cy

    onscreen = (u_scr >= -20) & (u_scr < W + 20) & (v_scr >= -20) & (v_scr < H + 20)
    u_scr = u_scr[onscreen]
    v_scr = v_scr[onscreen]
    z_c = z_c[onscreen]
    pts_w = pts_w[onscreen]
    cols_c = cols_c[onscreen]

    # 1. Triangulate into 2D screen mesh
    t_del = time.time()
    pts_2d = np.column_stack([u_scr, v_scr])
    tri = Delaunay(pts_2d)
    simplices = tri.simplices

    # Strict depth-step filtering to prevent foreground-background bridge faces
    v0, v1, v2 = simplices[:, 0], simplices[:, 1], simplices[:, 2]
    z0, z1, z2 = z_c[v0], z_c[v1], z_c[v2]
    max_z_diff = np.maximum(np.abs(z0 - z1), np.maximum(np.abs(z1 - z2), np.abs(z2 - z0)))
    min_z = np.minimum(z0, np.minimum(z1, z2))
    # No depth bridge across > 25% relative depth jump
    valid_depth = (max_z_diff / np.maximum(min_z, 0.1)) < 0.25

    p0, p1, p2 = pts_2d[v0], pts_2d[v1], pts_2d[v2]
    e01 = np.linalg.norm(p0 - p1, axis=1)
    e12 = np.linalg.norm(p1 - p2, axis=1)
    e20 = np.linalg.norm(p2 - p0, axis=1)
    max_edge = np.maximum(e01, np.maximum(e12, e20))
    valid_edge = max_edge < 75.0

    valid_mask = valid_depth & valid_edge
    valid_simplices = simplices[valid_mask]
    print(f"Retained {len(valid_simplices)} valid surface triangles in {time.time() - t_del:.2f}s")

    # 2. Projective Texture Warping per Triangle
    t_warp = time.time()
    framebuffer = np.full((H, W, 3), (15, 20, 30), dtype=np.uint8)
    rendered_mask = np.zeros((H, W), dtype=np.uint8)

    # Sort triangles back-to-front by mean depth
    tri_z = (z_c[valid_simplices[:, 0]] + z_c[valid_simplices[:, 1]] + z_c[valid_simplices[:, 2]]) / 3.0
    sort_order = np.argsort(tri_z)[::-1]
    sorted_simplices = valid_simplices[sort_order]

    # Pre-project 3D vertices into all candidate source cameras
    src_proj = {}
    for vid, (src_v, src_img) in source_views.items():
        src_h, src_w, _ = src_img.shape
        src_P_cam = (src_v.R @ pts_w.T).T + src_v.tvec
        src_z = src_P_cam[:, 2]
        valid_z = src_z > 0.1
        src_fx = src_v.intrinsics.fx * (src_w / src_v.intrinsics.width)
        src_fy = src_v.intrinsics.fy * (src_h / src_v.intrinsics.height)
        src_cx = src_v.intrinsics.cx * (src_w / src_v.intrinsics.width)
        src_cy = src_v.intrinsics.cy * (src_h / src_v.intrinsics.height)
        src_u = np.where(valid_z, (src_fx * src_P_cam[:, 0] / np.maximum(src_z, 1e-4)) + src_cx, -999.0)
        src_v_coord = np.where(valid_z, (src_fy * src_P_cam[:, 1] / np.maximum(src_z, 1e-4)) + src_cy, -999.0)
        src_proj[vid] = (np.column_stack([src_u, src_v_coord]), src_z > 0.1, src_v.center)

    # Select best camera for each triangle
    # For novel view synthesis, choose source camera with highest ray alignment
    tgt_cam_center = target_cam.center

    for tri_idx in sorted_simplices:
        tri_pts_tgt = pts_2d[tri_idx].astype(np.float32)
        tri_3d = pts_w[tri_idx]
        face_center = np.mean(tri_3d, axis=0)

        tgt_ray = face_center - tgt_cam_center
        tgt_dir = tgt_ray / np.maximum(np.linalg.norm(tgt_ray), 1e-6)

        best_vid = None
        best_score = -1.0

        for vid, (proj_2d, valid_z_arr, cam_c) in src_proj.items():
            if not np.all(valid_z_arr[tri_idx]):
                continue
            src_ray = face_center - cam_c
            src_dir = src_ray / np.maximum(np.linalg.norm(src_ray), 1e-6)
            cos_a = np.dot(src_dir, tgt_dir)
            if cos_a > best_score:
                best_score = cos_a
                best_vid = vid

        if best_vid is None or best_score < 0.2:
            # Fallback to Gouraud color
            pts_int = tri_pts_tgt.astype(np.int32)
            c = np.mean(cols_c[tri_idx], axis=0).astype(int).tolist()
            cv2.fillConvexPoly(framebuffer, pts_int, (c[0], c[1], c[2]), lineType=cv2.LINE_AA)
            cv2.fillConvexPoly(rendered_mask, pts_int, 1)
            continue

        # Affine Warp Triangle from Best Source Camera to Target Screen
        src_v, src_img = source_views[best_vid]
        tri_pts_src = src_proj[best_vid][0][tri_idx].astype(np.float32)

        # Bounding box on target screen
        min_x = int(max(0, np.floor(np.min(tri_pts_tgt[:, 0]))))
        max_x = int(min(W - 1, np.ceil(np.max(tri_pts_tgt[:, 0]))))
        min_y = int(max(0, np.floor(np.min(tri_pts_tgt[:, 1]))))
        max_y = int(min(H - 1, np.ceil(np.max(tri_pts_tgt[:, 1]))))

        if max_x <= min_x or max_y <= min_y:
            continue

        # Crop offsets
        tri_tgt_local = tri_pts_tgt.copy()
        tri_tgt_local[:, 0] -= min_x
        tri_tgt_local[:, 1] -= min_y

        # Bounding box in source image
        src_h, src_w, _ = src_img.shape
        s_min_x = int(max(0, np.floor(np.min(tri_pts_src[:, 0]))))
        s_max_x = int(min(src_w - 1, np.ceil(np.max(tri_pts_src[:, 0]))))
        s_min_y = int(max(0, np.floor(np.min(tri_pts_src[:, 1]))))
        s_max_y = int(min(src_h - 1, np.ceil(np.max(tri_pts_src[:, 1]))))

        if s_max_x <= s_min_x or s_max_y <= s_min_y:
            continue

        tri_src_local = tri_pts_src.copy()
        tri_src_local[:, 0] -= s_min_x
        tri_src_local[:, 1] -= s_min_y

        src_crop = src_img[s_min_y:s_max_y+1, s_min_x:s_max_x+1]
        if src_crop.size == 0:
            continue

        try:
            M = cv2.getAffineTransform(tri_src_local, tri_tgt_local)
            w_box = max_x - min_x + 1
            h_box = max_y - min_y + 1
            warped = cv2.warpAffine(src_crop, M, (w_box, h_box), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            # Mask for local triangle
            mask_local = np.zeros((h_box, w_box), dtype=np.uint8)
            cv2.fillConvexPoly(mask_local, tri_tgt_local.astype(np.int32), 1, lineType=cv2.LINE_AA)

            # Blend onto framebuffer
            roi_fb = framebuffer[min_y:max_y+1, min_x:max_x+1]
            roi_mask = rendered_mask[min_y:max_y+1, min_x:max_x+1]

            m_idx = mask_local > 0
            roi_fb[m_idx] = warped[m_idx]
            roi_mask[m_idx] = 1
        except Exception:
            pts_int = tri_pts_tgt.astype(np.int32)
            c = np.mean(cols_c[tri_idx], axis=0).astype(int).tolist()
            cv2.fillConvexPoly(framebuffer, pts_int, (c[0], c[1], c[2]), lineType=cv2.LINE_AA)

    print(f"Projective texture mapped in {time.time() - t_warp:.2f}s")

    # Infill any remaining sub-pixel micro-gaps within convex hull
    hull = cv2.convexHull(pts_2d.astype(np.int32))
    solid_hull = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(solid_hull, [hull], 1)

    unrendered = (solid_hull > 0) & (rendered_mask == 0)
    if np.any(unrendered):
        framebuffer = cv2.inpaint(framebuffer, unrendered.astype(np.uint8), 3, cv2.INPAINT_TELEA)

    out_path = r"scratch\test_projective_texture_warping.png"
    cv2.imwrite(out_path, framebuffer)
    print(f"Saved {out_path}")

    # Compare with ground truth
    gt = cv2.imread(target_cam.image_path)
    gt_res = cv2.resize(gt, (W, H))
    psnr = cv2.PSNR(gt_res, framebuffer)
    mae = np.mean(np.abs(gt_res.astype(float) - framebuffer.astype(float)))
    print(f"--- PROJECTIVE TEXTURE WARPING QUALITY ---")
    print(f"PSNR vs Real DSLR: {psnr:.2f} dB, MAE: {mae:.2f}")

if __name__ == "__main__":
    test_projective_triangle_texture_mapping()
