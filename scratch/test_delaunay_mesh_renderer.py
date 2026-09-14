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

def test_screen_delaunay_rasterizer():
    t0 = time.time()
    colmap_path = r"uploads\truck_photos\sparse\0"
    images_path = r"F:\tandt_db\tandt\truck\images"
    
    cameras, views_dict, sparse_pcd = load_colmap_model(sparse_dir=colmap_path, images_dir=images_path)
    all_views = list(views_dict.values())
    hero_cam = all_views[0]

    # Pre-cache source DSLR images
    cached_source_views = []
    for i, v in enumerate(all_views[:20]):
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

    print(f"Projected {len(u_scr)} onscreen vertices in {time.time() - t0:.2f}s")

    # 1. 2D Screen-Space Delaunay Triangulation
    t_del = time.time()
    pts_2d = np.column_stack([u_scr, v_scr])
    tri = Delaunay(pts_2d)
    simplices = tri.simplices
    print(f"Constructed Delaunay triangulation with {len(simplices)} triangles in {time.time() - t_del:.2f}s")

    # Filter out invalid triangles:
    # a) triangles with large depth jumps (connecting foreground to background)
    # b) triangles with long 2D edge lengths (connecting distant silhouette edges)
    v0 = simplices[:, 0]
    v1 = simplices[:, 1]
    v2 = simplices[:, 2]

    z0 = z_c[v0]
    z1 = z_c[v1]
    z2 = z_c[v2]

    max_z_diff = np.maximum(np.abs(z0 - z1), np.maximum(np.abs(z1 - z2), np.abs(z2 - z0)))
    min_z = np.minimum(z0, np.minimum(z1, z2))
    # Relative depth ratio < 35%
    valid_depth_tri = (max_z_diff / np.maximum(min_z, 0.1)) < 0.35

    # Edge lengths
    p0_2d = pts_2d[v0]
    p1_2d = pts_2d[v1]
    p2_2d = pts_2d[v2]
    e01 = np.linalg.norm(p0_2d - p1_2d, axis=1)
    e12 = np.linalg.norm(p1_2d - p2_2d, axis=1)
    e20 = np.linalg.norm(p2_2d - p0_2d, axis=1)
    max_edge = np.maximum(e01, np.maximum(e12, e20))
    valid_edge_tri = max_edge < 45.0 # pixels

    valid_tri_mask = valid_depth_tri & valid_edge_tri
    valid_simplices = simplices[valid_tri_mask]
    print(f"Retained {len(valid_simplices)} valid surface triangles ({np.mean(valid_tri_mask)*100:.1f}%)")

    # 2. Rasterize Triangles with Smooth Gouraud / Bilinear Barycentric Color
    t_rast = time.time()
    framebuffer = np.full((H, W, 3), (15, 20, 30), dtype=np.uint8)
    
    # Sort triangles back to front by mean depth
    tri_z = (z_c[valid_simplices[:, 0]] + z_c[valid_simplices[:, 1]] + z_c[valid_simplices[:, 2]]) / 3.0
    sort_tri = np.argsort(tri_z)[::-1]
    sorted_simplices = valid_simplices[sort_tri]

    # Draw shaded triangles using OpenCV fillConvexPoly for fast CPU rasterization
    # Compute mean color per triangle or Gouraud
    for tri_idx in sorted_simplices:
        pts = pts_2d[tri_idx].astype(np.int32)
        mean_c = np.mean(cols_c[tri_idx], axis=0)
        c_u8 = (np.clip(mean_c, 0.0, 1.0) * 255).astype(int).tolist()
        # OpenCV expects BGR
        cv2.fillConvexPoly(framebuffer, pts, (c_u8[0], c_u8[1], c_u8[2]), lineType=cv2.LINE_AA)

    print(f"Triangles rasterized in {time.time() - t_rast:.2f}s")
    Image.fromarray(framebuffer).save(r"scratch\test_delaunay_mesh.png")
    print("Saved test_delaunay_mesh.png")

if __name__ == "__main__":
    test_screen_delaunay_rasterizer()
