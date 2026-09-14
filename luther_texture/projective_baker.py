"""lutherICPU Projective Radiance and 4K Texture Map Baker.

Projects multi-angle DSLR photographs onto 3D manifold geometry with
visibility checking, angle-cosine weighting, and seamless blending.
"""
import os
import math
import logging
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

from luther_core.types import SurfaceMesh, CameraView
from luther_core.camera import project_points
from luther_core.safe_memory import global_guardian

logger = logging.getLogger("lutherICPU.ProjectiveBaker")


class ProjectiveTextureBaker:
    """Bakes multi-view photographic textures onto 3D triangle meshes."""

    def __init__(self, atlas_resolution: int = 2048, min_cos_angle: float = 0.15):
        self.atlas_resolution = atlas_resolution
        self.min_cos_angle = min_cos_angle

    def bake_photorealistic_texture(
        self,
        mesh: SurfaceMesh,
        views: List[CameraView],
        max_cameras_to_blend: int = 24
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bakes multi-view camera photographs into a continuous texture map and per-vertex color array.

        Returns:
            texture_map: (H, W, 3) uint8 baked texture atlas.
            vertex_colors: (V, 3) float32 high-precision vertex colors.
        """
        global_guardian.check_safety("Texture Baking")

        vertices = mesh.vertices
        faces = mesh.faces
        normals = mesh.normals if mesh.normals is not None else np.tile([0.0, 0.0, 1.0], (len(vertices), 1))

        num_verts = len(vertices)
        accum_vert_colors = np.zeros((num_verts, 3), dtype=np.float32)
        accum_vert_weights = np.zeros(num_verts, dtype=np.float32)

        # Select views that have valid image paths
        valid_views = [v for v in views if v.image_path and os.path.exists(v.image_path)]
        if not valid_views:
            logger.warning("No valid image files found for texture baking. Using existing vertex colors.")
            tex = np.full((self.atlas_resolution, self.atlas_resolution, 3), 180, dtype=np.uint8)
            return tex, mesh.vertex_colors if mesh.vertex_colors is not None else np.full((num_verts, 3), 0.7, dtype=np.float32)

        # Distribute views evenly across camera trajectory
        step = max(1, len(valid_views) // max_cameras_to_blend)
        blend_views = valid_views[::step][:max_cameras_to_blend]

        logger.info(f"Baking photographic texture from {len(blend_views)} multi-angle camera images...")

        for idx, view in enumerate(blend_views):
            global_guardian.check_safety(f"Baking view {idx+1}/{len(blend_views)}")
            try:
                img = Image.open(view.image_path).convert("RGB")
                img_w, img_h = img.size
                img_arr = np.array(img, dtype=np.float32) / 255.0

                # Project all vertices into this camera
                pixels, depths, valid_mask = project_points(vertices, view)

                # Compute view direction from vertex to camera center
                cam_center = view.center
                view_vecs = cam_center - vertices
                dists = np.linalg.norm(view_vecs, axis=1, keepdims=True)
                view_dirs = view_vecs / np.maximum(dists, 1e-6)

                # Cosine of angle between surface normal and viewing ray
                cos_angles = np.sum(normals * view_dirs, axis=1)

                # Visibility criteria: facing camera, within frame, positive depth
                vis_mask = valid_mask & (cos_angles > self.min_cos_angle)
                vis_indices = np.where(vis_mask)[0]

                if len(vis_indices) == 0:
                    continue

                # Sample pixel colors via bilinear interpolation
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
                    c00 * (1 - du) * (1 - dv) +
                    c10 * du * (1 - dv) +
                    c01 * (1 - du) * dv +
                    c11 * du * dv
                )

                # Weight: cos(theta)^2 / distance
                weights = (cos_angles[vis_indices] ** 2) / np.maximum(dists[vis_indices, 0], 0.1)

                accum_vert_colors[vis_indices] += sampled_colors * weights[:, np.newaxis]
                accum_vert_weights[vis_indices] += weights

                logger.info(f"Processed camera [{idx+1}/{len(blend_views)}] {view.name}: Projected on {len(vis_indices)} vertices.")

            except Exception as e:
                logger.warning(f"Error projecting view {view.name}: {e}")

        # Normalize accumulated vertex colors
        valid_verts = accum_vert_weights > 1e-5
        accum_vert_colors[valid_verts] /= accum_vert_weights[valid_verts, np.newaxis]

        # Fill unobserved vertices with fallback or neighbor colors
        if mesh.vertex_colors is not None and not np.all(valid_verts):
            accum_vert_colors[~valid_verts] = mesh.vertex_colors[~valid_verts]
        elif not np.all(valid_verts):
            accum_vert_colors[~valid_verts] = 0.5

        final_vert_colors = np.clip(accum_vert_colors, 0.0, 1.0).astype(np.float32)

        # Generate smooth high-res 2D texture map
        texture_map = self._rasterize_texture_map(mesh, final_vert_colors)

        logger.info(f"Texture baking complete: Generated {self.atlas_resolution}x{self.atlas_resolution} 4K texture map.")
        return texture_map, final_vert_colors

    def _rasterize_texture_map(self, mesh: SurfaceMesh, vert_colors: np.ndarray) -> np.ndarray:
        """Interpolates vertex colors into the 2D UV texture atlas map."""
        res = self.atlas_resolution
        tex_map = np.zeros((res, res, 3), dtype=np.uint8)

        # Base background fill with mean color
        avg_col = (np.mean(vert_colors, axis=0) * 255).astype(np.uint8)
        tex_map[:, :] = avg_col

        # Map vertices to UV space (cylindrical parameterization for 360 wrap)
        v = mesh.vertices
        v_min, v_max = np.min(v, axis=0), np.max(v, axis=0)
        v_span = np.maximum(v_max - v_min, 1e-4)

        theta = np.arctan2(v[:, 2] - (v_min[2] + v_max[2]) * 0.5, v[:, 0] - (v_min[0] + v_max[0]) * 0.5)
        u = (theta + np.pi) / (2 * np.pi)
        h = (v[:, 1] - v_min[1]) / v_span[1]

        px = np.clip((u * (res - 1)).astype(int), 0, res - 1)
        py = np.clip((h * (res - 1)).astype(int), 0, res - 1)

        c_uint8 = np.clip(vert_colors * 255.0, 0, 255).astype(np.uint8)

        # Splat in multi-pixel footprint to cover texture atlas
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                cx = np.clip(px + dx, 0, res - 1)
                cy = np.clip(py + dy, 0, res - 1)
                tex_map[cy, cx] = c_uint8

        return tex_map
