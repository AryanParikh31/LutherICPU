"""lutherICPU UV Parameterization and Atlas Chart Packing.

Computes conformal and projection-based UV texture coordinates for 3D meshes
with chart padding to prevent texel bleeding.
"""
import math
import logging
from typing import Tuple, List, Dict
import numpy as np

from luther_core.types import SurfaceMesh
from luther_core.safe_memory import global_guardian

logger = logging.getLogger("lutherICPU.UVParameterizer")


def compute_box_projection_uvs(mesh: SurfaceMesh) -> np.ndarray:
    """Computes triplanar/box-projected UV coordinates for every triangle vertex.

    Returns:
        face_uvs: (F, 3, 2) float32 array in [0, 1]^2.
    """
    vertices = mesh.vertices
    faces = mesh.faces

    # Face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)
    fn_norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals /= np.maximum(fn_norm, 1e-8)

    # Dominant axis per face: 0=X (YZ plane), 1=Y (XZ plane), 2=Z (XY plane)
    dominant_axis = np.argmax(np.abs(face_normals), axis=1)

    # Scene bounding box
    min_b = np.min(vertices, axis=0)
    max_b = np.max(vertices, axis=0)
    span = np.maximum(max_b - min_b, 1e-4)

    face_uvs = np.zeros((len(faces), 3, 2), dtype=np.float32)

    for i in range(len(faces)):
        axis = dominant_axis[i]
        tri_v = vertices[faces[i]]  # (3, 3)

        if axis == 0:  # X-dominant -> project on YZ
            u = (tri_v[:, 1] - min_b[1]) / span[1]
            v = (tri_v[:, 2] - min_b[2]) / span[2]
        elif axis == 1:  # Y-dominant -> project on XZ
            u = (tri_v[:, 0] - min_b[0]) / span[0]
            v = (tri_v[:, 2] - min_b[2]) / span[2]
        else:  # Z-dominant -> project on XY
            u = (tri_v[:, 0] - min_b[0]) / span[0]
            v = (tri_v[:, 1] - min_b[1]) / span[1]

        face_uvs[i, :, 0] = np.clip(u, 0.0, 1.0)
        face_uvs[i, :, 1] = np.clip(v, 0.0, 1.0)

    return face_uvs


def pack_triangle_charts(
    mesh: SurfaceMesh, atlas_resolution: int = 2048, gutter_pixels: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Packs individual triangle charts into a non-overlapping 2D UV texture atlas.

    Returns:
        face_uvs: (F, 3, 2) float32 coordinates in [0, 1]^2.
        uv_per_vertex: (V, 2) float32 coordinates for indexed rendering.
    """
    global_guardian.check_safety("UV Chart Packing")

    num_faces = len(mesh.faces)
    num_verts = len(mesh.vertices)

    # Calculate grid layout for faces in atlas
    grid_side = int(math.ceil(math.sqrt(num_faces)))
    cell_size = atlas_resolution / grid_side

    face_uvs = np.zeros((num_faces, 3, 2), dtype=np.float32)
    uv_per_vertex = np.zeros((num_verts, 2), dtype=np.float32)
    vert_uv_counts = np.zeros(num_verts, dtype=np.int32)

    # Standard reference equilateral triangle inside [0, 1]^2 cell
    ref_tri = np.array([
        [0.1, 0.1],
        [0.9, 0.1],
        [0.5, 0.9]
    ], dtype=np.float32)

    pad_uv = gutter_pixels / atlas_resolution

    for f_idx in range(num_faces):
        grid_x = f_idx % grid_side
        grid_y = f_idx // grid_side

        u_offset = (grid_x * cell_size) / atlas_resolution + pad_uv
        v_offset = (grid_y * cell_size) / atlas_resolution + pad_uv
        scale = (cell_size / atlas_resolution) - 2.0 * pad_uv

        tri_uv = u_offset + ref_tri * max(scale, 1e-4)
        face_uvs[f_idx] = tri_uv

        # Update per-vertex UV (averaged)
        for vi in range(3):
            v_id = mesh.faces[f_idx, vi]
            uv_per_vertex[v_id] += tri_uv[vi]
            vert_uv_counts[v_id] += 1

    valid_mask = vert_uv_counts > 0
    uv_per_vertex[valid_mask] /= vert_uv_counts[valid_mask, np.newaxis]

    logger.info(f"Packed {num_faces} triangle charts into {atlas_resolution}x{atlas_resolution} UV atlas.")
    return face_uvs, uv_per_vertex
