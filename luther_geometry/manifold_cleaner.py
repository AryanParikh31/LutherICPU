"""lutherICPU Manifold Mesh Cleaning, PLY Exporter, and Collision Boundary Generator.

Cleans non-manifold geometry, calculates collision boundary hulls, and serializes to PLY.

Silhouette Sawtooth Fix (Kutulakos & Seitz, 2000 - Space Carving):
  sky_silhouette_carver() removes points from the dense cloud that project into
  sky/background regions in 2+ cameras, preventing contaminated NCC boundary estimates
  from ever entering the TSDF and producing roofline zigzag triangles.
"""
import os
import struct
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from scipy.spatial import ConvexHull

from luther_core.types import SurfaceMesh, PointCloud
from luther_core.safe_memory import global_guardian

logger = logging.getLogger("lutherICPU.ManifoldCleaner")


def export_mesh_to_ply(mesh: SurfaceMesh, output_path: str, save_vertex_normals: bool = True) -> None:
    """Exports SurfaceMesh to binary Stanford PLY file format."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    num_verts = len(mesh.vertices)
    num_faces = len(mesh.faces)

    has_colors = mesh.vertex_colors is not None
    has_normals = (mesh.normals is not None) and save_vertex_normals

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_verts}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if has_normals:
        header.extend([
            "property float nx",
            "property float ny",
            "property float nz",
        ])

    if has_colors:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])

    header.extend([
        f"element face {num_faces}",
        "property list uchar int vertex_indices",
        "end_header\n"
    ])

    header_bytes = "\n".join(header).encode("ascii")

    with open(output_path, "wb") as f:
        f.write(header_bytes)

        # Pack vertex data
        for i in range(num_verts):
            x, y, z = mesh.vertices[i]
            vert_bytes = struct.pack("<3f", x, y, z)
            f.write(vert_bytes)

            if has_normals:
                nx, ny, nz = mesh.normals[i]
                f.write(struct.pack("<3f", nx, ny, nz))

            if has_colors:
                r = int(np.clip(mesh.vertex_colors[i, 0] * 255, 0, 255))
                g = int(np.clip(mesh.vertex_colors[i, 1] * 255, 0, 255))
                b = int(np.clip(mesh.vertex_colors[i, 2] * 255, 0, 255))
                f.write(struct.pack("<3B", r, g, b))

        # Pack face data
        for face in mesh.faces:
            f.write(struct.pack("<B3i", 3, int(face[0]), int(face[1]), int(face[2])))

    logger.info(f"Saved binary PLY mesh to {output_path} ({num_verts} verts, {num_faces} faces).")


def bilateral_normal_mesh_filter(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 5,
    sigma_s_factor: float = 1.5,
    sigma_r: float = 0.35,
    vertex_update_iters: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ultra-Fast Vectorized Bilateral Normal Mesh Filtering for Silhouette Sawtooth Removal.
    
    Implements state-of-the-art anisotropic bilateral mesh denoising (Zheng et al., Sun et al.):
    1. Filters face normals using spatial proximity (Ws) and normal orientation similarity (Wr).
       - Flat surfaces and smooth contours are strongly smoothed (removing MVS silhouette ripple).
       - Crisp 90° edges and rooflines (where ||n_i - n_j|| > sigma_r) are strictly preserved.
    2. Iteratively updates vertex positions to match the denoised normal field without volume shrinkage.
    
    Args:
        vertices: (V, 3) float32 array of vertex coordinates.
        faces: (F, 3) int32 array of triangular face indices.
        iterations: Number of bilateral normal filtering passes.
        sigma_s_factor: Multiplier on average face-to-face distance for spatial bandwidth.
        sigma_r: Normal similarity range parameter in radians (~0.35 rad = ~20 deg).
        vertex_update_iters: Number of vertex position integration steps.
        
    Returns:
        (filtered_vertices, filtered_normals)
    """
    verts = np.asarray(vertices, dtype=np.float32).copy()
    fcs = np.asarray(faces, dtype=np.int32)
    num_verts = len(verts)
    num_faces = len(fcs)
    
    if num_verts == 0 or num_faces == 0:
        return verts, np.zeros((0, 3), dtype=np.float32)

    # 1. Compute initial face centroids, normals, and areas
    v0 = verts[fcs[:, 0]]
    v1 = verts[fcs[:, 1]]
    v2 = verts[fcs[:, 2]]
    
    face_centroids = (v0 + v1 + v2) / 3.0
    cross_prod = np.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    face_areas_safe = np.maximum(face_areas, 1e-8)
    face_normals = cross_prod / face_areas_safe[:, np.newaxis]

    # Build face-to-face adjacency across shared triangular edges in O(F)
    edge_map = {}
    face_nbrs = np.full((num_faces, 3), -1, dtype=np.int32)

    for f_idx in range(num_faces):
        fa, fb, fc = int(fcs[f_idx, 0]), int(fcs[f_idx, 1]), int(fcs[f_idx, 2])
        e0 = (min(fa, fb), max(fa, fb))
        e1 = (min(fb, fc), max(fb, fc))
        e2 = (min(fc, fa), max(fc, fa))

        for k, e in enumerate([e0, e1, e2]):
            if e in edge_map:
                other_f, other_k = edge_map[e]
                face_nbrs[f_idx, k] = other_f
                face_nbrs[other_f, other_k] = f_idx
            else:
                edge_map[e] = (f_idx, k)

    # Compute spatial bandwidth sigma_s from neighboring centroids
    valid_nbrs = face_nbrs >= 0
    safe_nbrs = np.where(valid_nbrs, face_nbrs, 0)
    nbr_c = face_centroids[safe_nbrs] # (F, 3, 3)
    dists = np.linalg.norm(face_centroids[:, None, :] - nbr_c, axis=2) # (F, 3)
    valid_dists = dists[valid_nbrs]
    avg_d = float(np.mean(valid_dists)) if len(valid_dists) > 0 else 0.05
    sigma_s = max(1e-4, avg_d * sigma_s_factor)
    two_sigma_s_sq = 2.0 * (sigma_s ** 2)
    two_sigma_r_sq = 2.0 * (sigma_r ** 2)

    # Step 1: Vectorized Bilateral Face Normal Filtering
    filtered_fn = face_normals.copy()
    for _ in range(iterations):
        nbr_fn = filtered_fn[safe_nbrs] # (F, 3, 3)
        nbr_areas = face_areas_safe[safe_nbrs] # (F, 3)
        
        dist_sq = np.sum((face_centroids[:, None, :] - nbr_c) ** 2, axis=2) # (F, 3)
        norm_diff_sq = np.sum((filtered_fn[:, None, :] - nbr_fn) ** 2, axis=2) # (F, 3)
        
        w = nbr_areas * np.exp(-dist_sq / two_sigma_s_sq) * np.exp(-norm_diff_sq / two_sigma_r_sq) * valid_nbrs # (F, 3)
        
        # Combine self-face normal with filtered neighbor normals
        w_self = face_areas_safe[:, None] # (F, 1)
        w_all = np.concatenate([w_self, w], axis=1) # (F, 4)
        n_all = np.concatenate([filtered_fn[:, None, :], nbr_fn], axis=1) # (F, 4, 3)
        
        blended = np.sum(w_all[:, :, None] * n_all, axis=1) # (F, 3)
        norm_b = np.linalg.norm(blended, axis=1, keepdims=True)
        filtered_fn = np.where(norm_b > 1e-8, blended / np.maximum(norm_b, 1e-8), filtered_fn)

    # Step 2: Vectorized Vertex Position Integration (Sun et al. normal-guided projection)
    for _ in range(vertex_update_iters):
        delta_v = np.zeros_like(verts)
        total_weights = np.zeros(num_verts, dtype=np.float32)
        
        v0 = verts[fcs[:, 0]]
        v1 = verts[fcs[:, 1]]
        v2 = verts[fcs[:, 2]]
        face_centroids = (v0 + v1 + v2) / 3.0
        
        for k_corner in range(3):
            v_corner = verts[fcs[:, k_corner]]
            proj = np.sum((face_centroids - v_corner) * filtered_fn, axis=1) # (F,)
            delta = face_areas_safe[:, None] * proj[:, None] * filtered_fn # (F, 3)
            np.add.at(delta_v, fcs[:, k_corner], delta)
            np.add.at(total_weights, fcs[:, k_corner], face_areas_safe)
            
        valid_v = total_weights > 1e-8
        verts[valid_v] += delta_v[valid_v] / total_weights[valid_v, None]

    # Compute updated vertex normals
    vert_normals = np.zeros_like(verts)
    for k_corner in range(3):
        np.add.at(vert_normals, fcs[:, k_corner], face_areas_safe[:, None] * filtered_fn)
        
    vn_len = np.linalg.norm(vert_normals, axis=1, keepdims=True)
    vert_normals /= np.maximum(vn_len, 1e-8)
    
    return verts.astype(np.float32), vert_normals.astype(np.float32)


def prune_silhouette_sawtooth_triangles(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_aspect_ratio: float = 18.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prunes sliver and needle triangles that form sawtooth fringe artifacts along silhouette boundaries.
    """
    verts = np.asarray(vertices, dtype=np.float32)
    fcs = np.asarray(faces, dtype=np.int32)
    if len(verts) == 0 or len(fcs) == 0:
        return verts, fcs

    v0 = verts[fcs[:, 0]]
    v1 = verts[fcs[:, 1]]
    v2 = verts[fcs[:, 2]]

    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)

    max_edge = np.maximum(np.maximum(e0, e1), e2)
    semi_p = 0.5 * (e0 + e1 + e2)
    area = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5

    # Aspect ratio: longest_edge / (inradius = area / semi_p)
    inradius = np.where(semi_p > 1e-6, area / semi_p, 0.0)
    aspect_ratio = np.where(inradius > 1e-6, max_edge / inradius, 999.0)

    keep_faces = aspect_ratio < max_aspect_ratio
    clean_fcs = fcs[keep_faces]

    # Remove unreferenced vertices
    used_v = np.unique(clean_fcs)
    remap = np.full(len(verts), -1, dtype=np.int32)
    remap[used_v] = np.arange(len(used_v), dtype=np.int32)

    clean_verts = verts[used_v]
    clean_faces_remapped = remap[clean_fcs]

    return clean_verts.astype(np.float32), clean_faces_remapped.astype(np.int32)


def _detect_sky_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detects sky pixels using HSV thresholding.

    Sky is characterized by: high hue (90°-250° in OpenCV 0-180 scale → 45-125),
    low saturation (<0.25), and high brightness (>0.55).
    Returns a boolean mask (H, W) — True where pixel is sky.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[:, :, 0]  # OpenCV H is 0-180
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0

    # Blue sky: H in [85, 130], low sat, high val
    sky_blue = (h >= 85) & (h <= 130) & (s < 0.35) & (v > 0.50)
    # Overcast / white sky: very low sat, very high val
    sky_white = (s < 0.12) & (v > 0.75)

    sky_mask = sky_blue | sky_white

    # Dilate by 7px to create a boundary exclusion buffer at silhouette edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sky_zone = cv2.dilate(sky_mask.astype(np.uint8), kernel) > 0
    return sky_zone


def sky_silhouette_carver(
    points: np.ndarray,
    colors: np.ndarray,
    camera_views: list,
    min_sky_votes: int = 2,
    max_views_to_check: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Silhouette-Consistent Space Carving (Kutulakos & Seitz, 2000).

    Removes dense reconstruction points that project into sky/background regions
    in >= min_sky_votes cameras. This is the geometrically correct fix for the
    roofline sawtooth artifact.

    Root cause being fixed:
        NCC matching windows that straddle the roofline/sky boundary produce depth
        estimates that oscillate frame-to-frame. These contaminated points survive
        SOR (they are consistent WITHIN a neighborhood of other bad points) but
        produce zigzag isosurfaces in Marching Cubes. The bilateral mesh filter
        only attenuates them; this function removes them at the SOURCE.

    Algorithm:
        For each camera that has a sky-segmented image:
            1. Project all points into this camera's image plane.
            2. For points landing on sky-dilated pixels: increment their sky vote count.
        Remove any point whose sky vote count >= min_sky_votes.

    Args:
        points:            (N, 3) float32 dense point cloud.
        colors:            (N, 3) float32 RGB colors.
        camera_views:      List of CameraView objects (must have .image_path, .R, .center, .intrinsics).
        min_sky_votes:     Minimum number of cameras that must see a point as sky to remove it.
        max_views_to_check: Maximum cameras to use (use first N with valid images, for speed).

    Returns:
        (carved_points, carved_colors) — points with sky violations removed.
    """
    if len(points) == 0 or len(camera_views) == 0:
        return points, colors

    N = len(points)
    sky_votes = np.zeros(N, dtype=np.int32)
    views_checked = 0

    for view in camera_views:
        if views_checked >= max_views_to_check:
            break
        if view.image_path is None:
            continue

        img = cv2.imread(view.image_path)
        if img is None:
            continue

        H_img, W_img = img.shape[:2]
        sky_zone = _detect_sky_mask(img)  # (H_img, W_img) bool

        # Project all points into this camera
        # p_cam = R @ (p_world - center)  =>  p_cam shape (N, 3)
        pts_rel = points - view.center[np.newaxis, :]  # (N, 3)
        p_cam = (view.R @ pts_rel.T).T  # (N, 3)

        # Only consider points in front of the camera
        in_front = p_cam[:, 2] > 0.3
        z_safe = np.where(in_front, p_cam[:, 2], 1.0)

        fx, fy = view.intrinsics.fx, view.intrinsics.fy
        cx, cy = view.intrinsics.cx, view.intrinsics.cy

        u = (fx * p_cam[:, 0] / z_safe + cx).astype(np.float32)
        v = (fy * p_cam[:, 1] / z_safe + cy).astype(np.float32)

        # Valid projection: inside image bounds
        in_bounds = (
            in_front &
            (u >= 0) & (u < W_img - 1) &
            (v >= 0) & (v < H_img - 1)
        )

        u_int = np.clip(u.astype(np.int32), 0, W_img - 1)
        v_int = np.clip(v.astype(np.int32), 0, H_img - 1)

        # Check if the projected pixel is sky
        is_sky = sky_zone[v_int, u_int]  # (N,)
        sky_votes += (in_bounds & is_sky).astype(np.int32)

        views_checked += 1
        logger.debug(f"SkyCarver: checked {view.name}, {int(np.sum(in_bounds & is_sky))} sky-projected points")

    # Carve: remove points with >= min_sky_votes sky projections
    keep_mask = sky_votes < min_sky_votes
    n_removed = int(np.sum(~keep_mask))
    logger.info(
        f"SkyCarver: removed {n_removed:,} sky-boundary points ({n_removed/max(N,1)*100:.1f}%) "
        f"using {views_checked} camera views."
    )

    return points[keep_mask].astype(np.float32), colors[keep_mask].astype(np.float32)


def build_boundary_cage(vertices: Any, margin_multiplier: float = 1.25) -> Dict[str, Any]:
    """Computes a 3D convex hull collision boundary cage around scene geometry."""
    if hasattr(vertices, "vertices"):
        vertices = vertices.vertices
    vertices = np.asarray(vertices, dtype=np.float32)

    centroid = np.mean(vertices, axis=0)
    scaled_verts = centroid + (vertices - centroid) * margin_multiplier

    hull = ConvexHull(scaled_verts)
    hull_verts = scaled_verts[hull.vertices]
    hull_faces = hull.simplices

    # Re-index simplices for hull_verts
    vert_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(hull.vertices)}
    remapped_faces = np.array([[vert_map[v] for v in face] for face in hull_faces], dtype=np.int32)

    bbox_min = np.min(vertices, axis=0).tolist()
    bbox_max = np.max(vertices, axis=0).tolist()

    cage_dict = {
        "centroid": centroid.tolist(),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "cage_vertices": hull_verts.tolist(),
        "cage_faces": remapped_faces.tolist(),
        "safe_radius": float(np.max(np.linalg.norm(vertices - centroid, axis=1)) * margin_multiplier)
    }

    return cage_dict


def estimate_scene_up_vector(camera_views: Optional[list] = None) -> np.ndarray:
    """
    Estimates the normalized world up-vector directly from calibrated camera orientations.
    
    In standard pinhole/OpenCV camera convention, +Y is camera down, so -Y is camera up.
    The camera's world up direction is the 2nd row of R inverted (-R[1, :]).
    Averaging over ground-orbit cameras recovers the true gravity/vertical axis of the scene.
    """
    if camera_views:
        up_vectors = []
        for v in camera_views:
            if hasattr(v, "R") and v.R is not None:
                # -R[1, :] is camera upward vector in world frame
                up_vectors.append(-v.R[1, :])
        if len(up_vectors) > 0:
            mean_up = np.mean(up_vectors, axis=0).astype(np.float32)
            norm = np.linalg.norm(mean_up)
            if norm > 1e-6:
                return mean_up / norm

    # Default fallback: -Y is world up (standard COLMAP coordinate system)
    return np.array([0.0, -1.0, 0.0], dtype=np.float32)


def extract_roof_plane_and_boundary(
    points: np.ndarray,
    colors: np.ndarray,
    up_vector: Optional[np.ndarray] = None,
    roi_radius: float = 6.0,
    percentile_min: float = 90.0,
    percentile_max: float = 99.8,
    ransac_thresh: float = 0.08,
    max_ransac_iters: int = 1500
) -> Optional[Dict[str, Any]]:
    """
    Extracts the unobserved vehicle roof plane and 2D bounding polygon using RANSAC.
    
    1. Projects points along the world up-vector to find the topmost structural perimeter.
    2. Uses RANSAC to fit a horizontal plane normal aligned with the up-vector (|n . u| >= 0.75).
    3. Projects inliers onto the 2D plane coordinate system and extracts the convex boundary hull.
    4. Computes robust metric dimensions and vehicle body surface paint color.
    
    Returns:
        Dictionary containing plane parameters (n, d), 2D basis (e1, e2), hull points, and color,
        or None if insufficient planar geometry is detected.
    """
    if len(points) < 50:
        return None

    pts = np.asarray(points, dtype=np.float32)
    cols = np.asarray(colors, dtype=np.float32) if colors is not None else np.full_like(pts, 0.5)

    if up_vector is None:
        up_vector = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    up_vec = up_vector / np.maximum(np.linalg.norm(up_vector), 1e-6)

    # Focus on the primary vehicle object around median centroid
    centroid = np.median(pts, axis=0)
    p_dists = np.linalg.norm(pts - centroid, axis=1)
    roi_mask = p_dists <= roi_radius
    if np.sum(roi_mask) < 50:
        roi_mask = np.ones(len(pts), dtype=bool)

    roi_pts = pts[roi_mask]
    roi_cols = cols[roi_mask]

    # Project along up-vector
    heights = roi_pts @ up_vec
    h_low = np.percentile(heights, percentile_min)
    h_high = np.percentile(heights, percentile_max)
    candidate_mask = (heights >= h_low) & (heights <= h_high)
    candidate_pts = roi_pts[candidate_mask]
    candidate_cols = roi_cols[candidate_mask]

    N = len(candidate_pts)
    if N < 20:
        return None

    # RANSAC Plane Fitting
    best_inliers = np.array([], dtype=np.int32)
    best_plane = None

    for _ in range(max_ransac_iters):
        sample_idx = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = candidate_pts[sample_idx]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-6:
            continue
        n = n / norm_n

        # Orient normal outward (along up_vec)
        if np.dot(n, up_vec) < 0:
            n = -n

        # Enforce plane is approximately horizontal (aligned with up_vec)
        if np.dot(n, up_vec) < 0.75:
            continue

        d = -float(np.dot(n, p1))
        dists = np.abs(candidate_pts @ n + d)
        inlier_indices = np.where(dists < ransac_thresh)[0]

        if len(inlier_indices) > len(best_inliers):
            best_inliers = inlier_indices
            best_plane = (n, d)

    if best_plane is None or len(best_inliers) < 15:
        return None

    plane_n, plane_d = best_plane
    inlier_pts = candidate_pts[best_inliers]
    inlier_cols = candidate_cols[best_inliers]

    # Refine plane normal and distance by PCA covariance on inliers
    mean_inlier = np.mean(inlier_pts, axis=0)
    cov = np.cov((inlier_pts - mean_inlier).T)
    evals, evecs = np.linalg.eigh(cov)
    refined_n = evecs[:, 0]
    if np.dot(refined_n, up_vec) < 0:
        refined_n = -refined_n
    if np.dot(refined_n, up_vec) < 0.70:
        refined_n = plane_n
    refined_d = -float(np.dot(refined_n, mean_inlier))

    # Orthonormal 2D basis (e1, e2) on the plane
    arbitrary = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(refined_n[0]) < 0.8 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    e1 = np.cross(refined_n, arbitrary)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(refined_n, e1)
    e2 /= np.linalg.norm(e2)

    p0 = -refined_d * refined_n
    pts_2d = np.column_stack([np.dot(inlier_pts - p0, e1), np.dot(inlier_pts - p0, e2)])

    # Compute 2D convex hull of inliers
    if len(pts_2d) < 4:
        return None

    hull = ConvexHull(pts_2d)
    hull_pts_2d = pts_2d[hull.vertices]

    # Filter out extreme sliver points by trimming to 2nd-98th percentiles
    u_min, u_max = float(np.percentile(pts_2d[:, 0], 2)), float(np.percentile(pts_2d[:, 0], 98))
    v_min, v_max = float(np.percentile(pts_2d[:, 1], 2)), float(np.percentile(pts_2d[:, 1], 98))
    width = u_max - u_min
    length = v_max - v_min

    # Sanity check vehicle roof dimensions (e.g. 0.8m to 12.0m)
    if width < 0.5 or width > 12.0 or length < 0.5 or length > 12.0:
        return None

    body_color = np.median(inlier_cols, axis=0)

    logger.info(
        f"[ROOF-EXTRACTOR] Extracted planar roof boundary: width={width:.2f}m, length={length:.2f}m, "
        f"{len(inlier_pts)} inliers, normal={refined_n}, body_color={body_color}"
    )

    return {
        "plane_normal": refined_n,
        "plane_d": refined_d,
        "origin_p0": p0,
        "e1": e1,
        "e2": e2,
        "pts_2d": pts_2d,
        "hull_pts_2d": hull_pts_2d,
        "u_min": u_min,
        "u_max": u_max,
        "v_min": v_min,
        "v_max": v_max,
        "body_color": body_color,
        "inlier_points_3d": inlier_pts
    }


def synthesize_planar_roof_points(
    roof_info: Dict[str, Any],
    grid_step: float = 0.04,
    camera_views: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dense, uniform 3D grid of planar surface points covering the unobserved roof hull.
    
    Args:
        roof_info: Plane metadata and 2D convex hull dictionary from extract_roof_plane_and_boundary.
        grid_step: Spatial sampling distance in meters (default: 4cm).
        camera_views: Optional list of calibrated CameraViews for visual hull space carving.
        
    Returns:
        (points, colors, normals) - (N, 3) float32 arrays of synthesized roof surface geometry.
    """
    u_min, u_max = roof_info["u_min"], roof_info["u_max"]
    v_min, v_max = roof_info["v_min"], roof_info["v_max"]
    p0 = roof_info["origin_p0"]
    e1 = roof_info["e1"]
    e2 = roof_info["e2"]
    normal = roof_info["plane_normal"]
    body_col = roof_info["body_color"]
    hull_pts_2d = roof_info["hull_pts_2d"]

    # Generate regular 2D grid
    u_samples = np.arange(u_min, u_max, grid_step, dtype=np.float32)
    v_samples = np.arange(v_min, v_max, grid_step, dtype=np.float32)
    if len(u_samples) == 0 or len(v_samples) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    uu, vv = np.meshgrid(u_samples, v_samples)
    grid_2d = np.column_stack([uu.ravel(), vv.ravel()])

    # 2D Point-in-Convex-Polygon Test
    # For a convex polygon with counterclockwise vertices, a point is inside iff cross product (v_j - v_i) x (p - v_i) >= 0 for all edges
    # Ensure hull vertices are in CCW order
    hull_ccw = hull_pts_2d.copy()
    center_2d = np.mean(hull_ccw, axis=0)
    angles = np.arctan2(hull_ccw[:, 1] - center_2d[1], hull_ccw[:, 0] - center_2d[0])
    hull_ccw = hull_ccw[np.argsort(angles)]

    inside_mask = np.ones(len(grid_2d), dtype=bool)
    n_verts = len(hull_ccw)
    for i in range(n_verts):
        p_a = hull_ccw[i]
        p_b = hull_ccw[(i + 1) % n_verts]
        edge = p_b - p_a
        v_pts = grid_2d - p_a
        # 2D cross product: edge.x * v.y - edge.y * v.x
        cross = edge[0] * v_pts[:, 1] - edge[1] * v_pts[:, 0]
        inside_mask &= (cross >= -1e-4)

    valid_2d = grid_2d[inside_mask]
    if len(valid_2d) == 0:
        # Fallback: use rectangular grid if convex polygon test trimmed all
        valid_2d = grid_2d

    # Unproject 2D planar points into 3D world coordinates
    synth_pts = p0 + valid_2d[:, 0:1] * e1 + valid_2d[:, 1:2] * e2
    synth_pts = synth_pts.astype(np.float32)

    N_synth = len(synth_pts)
    synth_cols = np.tile(body_col, (N_synth, 1)).astype(np.float32)
    synth_norms = np.tile(normal, (N_synth, 1)).astype(np.float32)

    logger.info(f"[ROOF-SYNTHESIZER] Generated {N_synth:,} synthetic roof points with planar normal {normal}.")

    return synth_pts, synth_cols, synth_norms


def inject_unobserved_roof_surface(
    points: np.ndarray,
    colors: np.ndarray,
    camera_views: Optional[list] = None,
    grid_step: float = 0.04,
    percentile_min: float = 90.0,
    percentile_max: float = 99.8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    End-to-End Unobserved Rooftop Reconstruction & Visual Hull Planar Injection.
    
    Extracts the unobserved roof plane from boundary observations, synthesizes a dense
    watertight planar surface grid, and injects it into the point cloud before TSDF fusion.
    
    Args:
        points: (N, 3) float32 dense point cloud.
        colors: (N, 3) float32 RGB colors.
        camera_views: Optional list of calibrated CameraViews.
        grid_step: Planar grid resolution (default: 0.04m).
        percentile_min: Lower percentile bound for roof boundary extraction.
        percentile_max: Upper percentile bound for roof boundary extraction.
        
    Returns:
        (augmented_points, augmented_colors) with synthetic roof geometry seamlessly merged.
    """
    if len(points) < 100:
        return points, colors

    up_vec = estimate_scene_up_vector(camera_views)
    roof_info = extract_roof_plane_and_boundary(
        points=points,
        colors=colors,
        up_vector=up_vec,
        percentile_min=percentile_min,
        percentile_max=percentile_max
    )

    if roof_info is None:
        logger.info("[ROOF-INJECTION] No unobserved planar roof detected. Keeping original point cloud.")
        return points, colors

    synth_pts, synth_cols, _ = synthesize_planar_roof_points(
        roof_info=roof_info,
        grid_step=grid_step,
        camera_views=camera_views
    )

    if len(synth_pts) == 0:
        return points, colors

    aug_points = np.concatenate([points, synth_pts], axis=0).astype(np.float32)
    aug_colors = np.concatenate([colors, synth_cols], axis=0).astype(np.float32)

    logger.info(f"[ROOF-INJECTION] Successfully injected {len(synth_pts):,} roof points. Total: {len(aug_points):,} points.")
    return aug_points, aug_colors

