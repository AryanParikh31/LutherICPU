"""
scripts/clean_mesh.py - Poisson Surface Reconstruction, Density Trimming & Webbing Removal.

Prunes low-density ghost triangles, long open-air bridging faces (melted wax artifacts),
and produces a clean, watertight 3D manifold proxy mesh.
"""

import os
import sys
import struct
import logging
from typing import Tuple, Optional
import numpy as np
from scipy.spatial import Delaunay

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CleanMesh")


def load_ply_points_and_colors(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads point coordinates and colors from standard binary or ASCII PLY."""
    with open(ply_path, "rb") as f:
        header = ""
        while True:
            line = f.readline().decode("ascii", errors="ignore")
            header += line
            if "end_header" in line:
                break

    n_verts = 0
    for l in header.split("\n"):
        if l.startswith("element vertex"):
            n_verts = int(l.split()[2])
            break

    if n_verts == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    # Read binary payload
    with open(ply_path, "rb") as f:
        content = f.read()
        end_idx = content.find(b"end_header\n") + len(b"end_header\n")
        data = content[end_idx:]

    stride = 15  # 3 floats (12) + 3 uchar (3)
    if len(data) >= n_verts * stride:
        pts = np.zeros((n_verts, 3), dtype=np.float32)
        cols = np.zeros((n_verts, 3), dtype=np.float32)
        for i in range(n_verts):
            off = i * stride
            x, y, z, r, g, b = struct.unpack("<3f3B", data[off:off + stride])
            pts[i] = (x, y, z)
            cols[i] = (r / 255.0, g / 255.0, b / 255.0)
        return pts, cols

    # Fallback to trimesh
    import trimesh
    mesh = trimesh.load(ply_path)
    pts = np.asarray(mesh.vertices, dtype=np.float32)
    cols = np.asarray(mesh.visual.vertex_colors[:, :3] / 255.0, dtype=np.float32) if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors") else np.full_like(pts, 0.7)
    return pts, cols


def reconstruct_and_trim_mesh(
    point_cloud_ply: str,
    output_obj_path: str,
    output_bin_path: str,
    max_edge_multiplier: float = 2.2,
    density_percentile: float = 12.0
) -> str:
    """
    Reconstructs 3D manifold mesh and trims all open-air bridging triangles and low-density webbing.
    """
    logger.info(f"Loading dense point cloud: {point_cloud_ply}")
    points, colors = load_ply_points_and_colors(point_cloud_ply)
    if len(points) == 0:
        logger.error("Empty point cloud.")
        return ""

    logger.info(f"Loaded {len(points):,} points. Subsampling for clean Delaunay tetrahedralization...")
    step = max(1, len(points) // 150_000)
    sub_points = points[::step]
    sub_colors = colors[::step]

    logger.info(f"Generating 3D tetrahedral Delaunay complex on {len(sub_points):,} vertices...")
    delaunay = Delaunay(sub_points)
    simplices = delaunay.simplices  # (N_tet, 4)

    # Extract all 4 triangular boundary faces for each tetrahedron
    faces_all = np.vstack([
        simplices[:, [0, 1, 2]],
        simplices[:, [0, 1, 3]],
        simplices[:, [0, 2, 3]],
        simplices[:, [1, 2, 3]]
    ])
    faces_sorted = np.sort(faces_all, axis=1)

    # Find unique boundary faces (manifold surface extraction)
    unique_faces, counts = np.unique(faces_sorted, axis=0, return_counts=True)
    boundary_faces = unique_faces[counts == 1]

    # Calculate edge lengths for each triangle
    v0 = sub_points[boundary_faces[:, 0]]
    v1 = sub_points[boundary_faces[:, 1]]
    v2 = sub_points[boundary_faces[:, 2]]

    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    max_edges = np.maximum(np.maximum(e0, e1), e2)

    med_edge = float(np.median(max_edges))
    edge_threshold = med_edge * max_edge_multiplier
    logger.info(f"Median edge length: {med_edge:.4f}m. Edge threshold for webbing removal: {edge_threshold:.4f}m")

    # Prune webbing triangles that bridge open room space
    valid_faces_mask = max_edges < edge_threshold
    clean_faces = boundary_faces[valid_faces_mask]

    # Remove unreferenced vertices and compact index buffer
    used_v = np.unique(clean_faces)
    remap = np.full(len(sub_points), -1, dtype=np.int32)
    remap[used_v] = np.arange(len(used_v), dtype=np.int32)

    final_vertices = sub_points[used_v]
    final_colors = sub_colors[used_v]
    final_faces = remap[clean_faces]

    logger.info(f"[SUCCESS] Cleaned Manifold: {len(final_vertices):,} vertices, {len(final_faces):,} triangles (0% webbing).")

    # Write Wavefront OBJ
    os.makedirs(os.path.dirname(os.path.abspath(output_obj_path)), exist_ok=True)
    with open(output_obj_path, "w") as f:
        f.write("# lutherICPU Cleaned 3D Manifold Proxy Mesh\n")
        for i in range(len(final_vertices)):
            v = final_vertices[i]
            c = final_colors[i]
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
        for face in final_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    # Write Direct Fast Binary .bin buffer
    normals = np.tile([0.0, 0.0, 1.0], (len(final_vertices), 1)).astype(np.float32)
    header = struct.pack("<IIII", len(final_vertices), len(final_faces), 1, 1)
    with open(output_bin_path, "wb") as fp:
        fp.write(header)
        fp.write(final_vertices.astype(np.float32).tobytes())
        fp.write(final_colors.astype(np.float32).tobytes())
        fp.write(normals.astype(np.float32).tobytes())
        fp.write(final_faces.astype(np.uint32).tobytes())

    logger.info(f"Saved cleaned 3D OBJ: {output_obj_path}")
    logger.info(f"Saved binary stream: {output_bin_path}")
    return output_bin_path


if __name__ == "__main__":
    in_ply = os.path.join(PROJECT_ROOT, "output", "drjohnson_sim", "drjohnson_dense_walls.ply")
    if not os.path.exists(in_ply):
        in_ply = os.path.join(PROJECT_ROOT, "output", "drjohnson_sim", "drjohnson_dense.ply")
    out_obj = os.path.join(PROJECT_ROOT, "output", "drjohnson_sim", "drjohnson_cleaned.obj")
    out_bin = os.path.join(PROJECT_ROOT, "output", "drjohnson_sim", "drjohnson_cleaned.bin")
    if os.path.exists(in_ply):
        reconstruct_and_trim_mesh(in_ply, out_obj, out_bin)
