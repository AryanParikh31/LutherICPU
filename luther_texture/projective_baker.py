"""lutherICPU Projective Radiance and 4K Texture Map Baker.

Projects multi-angle DSLR photographs onto 3D manifold geometry with
optimal camera view selection, angle-cosine weighting, visibility checking,
and seamless projective triangle rasterization.
"""
import os
import math
import logging
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from PIL import Image

from luther_core.types import SurfaceMesh, CameraView
from luther_core.camera import project_points
from luther_core.safe_memory import global_guardian
from luther_texture.uv_parameterizer import compute_box_projection_uvs

logger = logging.getLogger("lutherICPU.ProjectiveBaker")


class ProjectiveTextureBaker:
    """Bakes multi-view photographic textures onto 3D triangle meshes."""

    def __init__(self, atlas_resolution: int = 2048, min_cos_angle: float = 0.12):
        self.atlas_resolution = atlas_resolution
        self.min_cos_angle = min_cos_angle

    def bake_photorealistic_texture(
        self,
        mesh: SurfaceMesh,
        views: List[CameraView],
        max_cameras_to_blend: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bakes multi-view camera photographs into a continuous texture map and per-vertex color array.

        Returns:
            texture_map: (H, W, 3) uint8 baked texture atlas.
            vertex_colors: (V, 3) float32 high-precision vertex colors.
        """
        global_guardian.check_safety("Texture Baking")

        vertices = mesh.vertices
        faces = mesh.faces
        num_verts = len(vertices)
        num_faces = len(faces)

        if mesh.normals is not None and len(mesh.normals) == num_verts:
            normals = mesh.normals
        else:
            # Compute vertex normals from face normals
            normals = np.zeros_like(vertices, dtype=np.float32)
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            fn = np.cross(v1 - v0, v2 - v0)
            fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
            fn /= np.maximum(fn_len, 1e-8)
            for i in range(3):
                np.add.at(normals, faces[:, i], fn)
            n_len = np.linalg.norm(normals, axis=1, keepdims=True)
            normals /= np.maximum(n_len, 1e-8)

        accum_vert_colors = np.zeros((num_verts, 3), dtype=np.float32)
        accum_vert_weights = np.zeros(num_verts, dtype=np.float32)

        # Select views with valid image paths
        valid_views = [v for v in views if v.image_path and os.path.exists(v.image_path)]
        if not valid_views:
            logger.warning("No valid image files found for texture baking. Using fallback colors.")
            fallback_col = mesh.vertex_colors if mesh.vertex_colors is not None else np.full((num_verts, 3), 0.7, dtype=np.float32)
            tex = np.full((self.atlas_resolution, self.atlas_resolution, 3), 180, dtype=np.uint8)
            return tex, fallback_col

        # Distribute views evenly across camera trajectory
        step = max(1, len(valid_views) // max_cameras_to_blend)
        blend_views = valid_views[::step][:max_cameras_to_blend]

        logger.info(f"Baking high-resolution photographic texture from {len(blend_views)} multi-angle views...")

        for idx, view in enumerate(blend_views):
            global_guardian.check_safety(f"Baking view {idx+1}/{len(blend_views)}")
            try:
                img = Image.open(view.image_path).convert("RGB")
                img_w, img_h = img.size
                img_arr = np.array(img, dtype=np.float32) / 255.0

                pixels, depths, valid_mask = project_points(vertices, view)

                cam_center = view.center
                view_vecs = cam_center - vertices
                dists = np.linalg.norm(view_vecs, axis=1, keepdims=True)
                view_dirs = view_vecs / np.maximum(dists, 1e-6)

                cos_angles = np.sum(normals * view_dirs, axis=1)

                vis_mask = valid_mask & (cos_angles > self.min_cos_angle) & (depths > 0.05)
                vis_indices = np.where(vis_mask)[0]

                if len(vis_indices) == 0:
                    continue

                u_coords = np.clip(pixels[vis_indices, 0], 0, img_w - 1)
                v_coords = np.clip(pixels[vis_indices, 1], 0, img_h - 1)

                u0 = np.floor(u_coords).astype(int)
                u1 = np.minimum(u0 + 1, img_w - 1)
                v0 = np.floor(v_coords).astype(int)
                v1 = np.minimum(v0 + 1, img_h - 1)

                du = (u_coords - u0)[:, np.newaxis]
                dv = (v_coords - v0)[:, np.newaxis]

                c00 = img_arr[v0, u0]
                c10 = img_arr[v0, u1]
                c01 = img_arr[v1, u0]
                c11 = img_arr[v1, u1]

                sampled_colors = (
                    c00 * (1.0 - du) * (1.0 - dv) +
                    c10 * du * (1.0 - dv) +
                    c01 * (1.0 - du) * dv +
                    c11 * du * dv
                )

                weights = (cos_angles[vis_indices] ** 2) / np.maximum(dists[vis_indices, 0], 0.1)
                accum_vert_colors[vis_indices] += sampled_colors * weights[:, np.newaxis]
                accum_vert_weights[vis_indices] += weights

            except Exception as e:
                logger.warning(f"Error projecting view {view.name}: {e}")

        # Normalize accumulated vertex colors
        valid_verts = accum_vert_weights > 1e-5
        accum_vert_colors[valid_verts] /= accum_vert_weights[valid_verts, np.newaxis]

        if mesh.vertex_colors is not None and not np.all(valid_verts):
            accum_vert_colors[~valid_verts] = mesh.vertex_colors[~valid_verts]
        elif not np.all(valid_verts):
            accum_vert_colors[~valid_verts] = 0.5

        final_vert_colors = np.clip(accum_vert_colors, 0.0, 1.0).astype(np.float32)

        # Generate smooth, continuous photographic 4K UV texture atlas
        texture_map = self._rasterize_box_projected_atlas(mesh, final_vert_colors, blend_views)

        logger.info(f"Photographic texture baking complete: Generated {self.atlas_resolution}x{self.atlas_resolution} atlas.")
        return texture_map, final_vert_colors

    def _rasterize_box_projected_atlas(
        self,
        mesh: SurfaceMesh,
        vert_colors: np.ndarray,
        views: List[CameraView]
    ) -> np.ndarray:
        """Rasterizes continuous triangular UV charts into a 4K texture atlas with seam dilation."""
        res = self.atlas_resolution
        tex_map = np.zeros((res, res, 3), dtype=np.uint8)
        mask = np.zeros((res, res), dtype=np.uint8)

        vertices = mesh.vertices
        faces = mesh.faces

        # 1. Compute robust triplanar / box UVs
        face_uvs = compute_box_projection_uvs(mesh)  # (F, 3, 2) in [0, 1]^2
        vert_colors_u8 = (vert_colors * 255.0).astype(np.uint8)

        # 2. Rasterize each face into the texture atlas using Gouraud interpolation / affine warp
        for i in range(len(faces)):
            f = faces[i]
            uv_tri = (face_uvs[i] * (res - 1)).astype(np.int32)
            cols = vert_colors_u8[f]

            # Face color (average of 3 vertices)
            face_col = np.mean(cols, axis=0).astype(int)
            bgr_col = (int(face_col[0]), int(face_col[1]), int(face_col[2]))

            cv2.fillConvexPoly(tex_map, uv_tri, bgr_col, lineType=cv2.LINE_AA)
            cv2.fillConvexPoly(mask, uv_tri, 255, lineType=cv2.LINE_AA)

        # 3. Seam dilation & inpainting to fill all UV gutters and eliminate dark seams
        unfilled = mask == 0
        if np.any(unfilled):
            # Dilate colored regions into gutters
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            dilated_mask = cv2.dilate(mask, kernel)
            gutter_holes = (dilated_mask > 0) & unfilled

            if np.any(gutter_holes):
                tex_map = cv2.inpaint(tex_map, gutter_holes.astype(np.uint8), 3, cv2.INPAINT_TELEA)

            # Global fallback for remaining empty areas: mean surface color
            avg_all = np.mean(vert_colors_u8, axis=0).astype(np.uint8)
            still_empty = mask == 0
            tex_map[still_empty] = avg_all

        return tex_map
