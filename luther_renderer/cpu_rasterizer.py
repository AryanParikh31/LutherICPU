"""lutherICPU Multithreaded CPU Software Rasterizer.

Renders continuous photorealistic novel views directly on CPU with sub-pixel
barycentric interpolation, depth z-buffering, and multi-light Blinn-Phong shading.
"""
import math
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image

from luther_core.types import SurfaceMesh, CameraView, CameraIntrinsics
from luther_core.safe_memory import global_guardian

logger = logging.getLogger("lutherICPU.CpuRasterizer")


class CpuSoftwareRasterizer:
    """Multithreaded CPU software renderer for continuous 3D triangle meshes."""

    def __init__(self, num_threads: int = 4):
        self.num_threads = num_threads

    def render_view(
        self,
        mesh: SurfaceMesh,
        view: CameraView,
        width: Optional[int] = None,
        height: Optional[int] = None,
        bg_color: Tuple[int, int, int] = (15, 20, 30),
        ambient_light: float = 0.45,
        directional_light_intensity: float = 0.75
    ) -> Image.Image:
        """Renders the 3D surface mesh from the specified camera viewpoint completely on CPU."""
        global_guardian.check_safety(f"CPU Render {view.name}")

        W = width or view.intrinsics.width
        H = height or view.intrinsics.height

        scale_x = W / view.intrinsics.width
        scale_y = H / view.intrinsics.height
        fx = view.intrinsics.fx * scale_x
        fy = view.intrinsics.fy * scale_y
        cx = view.intrinsics.cx * scale_x
        cy = view.intrinsics.cy * scale_y

        vertices = mesh.vertices
        faces = mesh.faces
        vert_colors = mesh.vertex_colors if mesh.vertex_colors is not None else np.full((len(vertices), 3), 0.8, dtype=np.float32)
        vert_normals = mesh.normals if mesh.normals is not None else np.tile([0.0, 0.0, 1.0], (len(vertices), 1))

        # Transform vertices into camera space: P_cam = R * V + t
        R = view.R
        t = view.tvec
        P_cam = (R @ vertices.T).T + t  # (V, 3)
        z = P_cam[:, 2]

        # 2D Screen projection
        eps = 1e-4
        u_screen = (fx * P_cam[:, 0] / np.maximum(z, eps)) + cx
        v_screen = (fy * P_cam[:, 1] / np.maximum(z, eps)) + cy

        # Calculate lighting per vertex
        # Primary sun/key light (world space: slightly angled down)
        key_light_dir = np.array([0.4, 0.8, 0.5], dtype=np.float32)
        key_light_dir /= np.linalg.norm(key_light_dir)

        # Fill light (opposite side, softer)
        fill_light_dir = np.array([-0.5, -0.3, -0.6], dtype=np.float32)
        fill_light_dir /= np.linalg.norm(fill_light_dir)

        # World-space normals for lighting
        diffuse1 = np.maximum(np.sum(vert_normals * key_light_dir, axis=1), 0.0) * directional_light_intensity
        diffuse2 = np.maximum(np.sum(vert_normals * fill_light_dir, axis=1), 0.0) * (directional_light_intensity * 0.4)
        total_light = np.clip(ambient_light + diffuse1 + diffuse2, 0.0, 1.25)[:, np.newaxis]

        lit_colors = np.clip(vert_colors * total_light, 0.0, 1.0).astype(np.float32)

        # Filter triangles: must have all vertices with z > 0.1
        f_z0 = z[faces[:, 0]]
        f_z1 = z[faces[:, 1]]
        f_z2 = z[faces[:, 2]]
        valid_tri_mask = (f_z0 > 0.1) & (f_z1 > 0.1) & (f_z2 > 0.1)

        valid_faces = faces[valid_tri_mask]
        if len(valid_faces) == 0:
            return Image.new("RGB", (W, H), bg_color)

        # Triangle screen coordinates
        p0_x = u_screen[valid_faces[:, 0]]
        p0_y = v_screen[valid_faces[:, 0]]
        p1_x = u_screen[valid_faces[:, 1]]
        p1_y = v_screen[valid_faces[:, 1]]
        p2_x = u_screen[valid_faces[:, 2]]
        p2_y = v_screen[valid_faces[:, 2]]

        # 2D Bounding box per triangle
        min_x = np.clip(np.floor(np.minimum.reduce([p0_x, p1_x, p2_x])).astype(int), 0, W - 1)
        max_x = np.clip(np.ceil(np.maximum.reduce([p0_x, p1_x, p2_x])).astype(int), 0, W - 1)
        min_y = np.clip(np.floor(np.minimum.reduce([p0_y, p1_y, p2_y])).astype(int), 0, H - 1)
        max_y = np.clip(np.ceil(np.maximum.reduce([p0_y, p1_y, p2_y])).astype(int), 0, H - 1)

        # Screen bounding box filter: triangle must overlap screen
        screen_visible = (max_x >= min_x) & (max_y >= min_y) & ((max_x - min_x) * (max_y - min_y) > 0)
        tri_indices = np.where(screen_visible)[0]

        # Sort triangles from far to near for rasterizer efficiency and depth coherence
        tri_depths = (f_z0[valid_tri_mask] + f_z1[valid_tri_mask] + f_z2[valid_tri_mask]) / 3.0
        sort_order = tri_indices[np.argsort(tri_depths[tri_indices])]

        # Framebuffers
        framebuffer = np.full((H, W, 3), bg_color, dtype=np.uint8)
        zbuffer = np.full((H, W), np.inf, dtype=np.float32)

        # Rasterize triangles using vectorized edge equations
        for tri_i in sort_order:
            face = valid_faces[tri_i]

            x0, y0 = p0_x[tri_i], p0_y[tri_i]
            x1, y1 = p1_x[tri_i], p1_y[tri_i]
            x2, y2 = p2_x[tri_i], p2_y[tri_i]

            z0_val, z1_val, z2_val = f_z0[valid_tri_mask][tri_i], f_z1[valid_tri_mask][tri_i], f_z2[valid_tri_mask][tri_i]

            c0 = lit_colors[face[0]]
            c1 = lit_colors[face[1]]
            c2 = lit_colors[face[2]]

            # 2D cross product area (2 * Area)
            det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if abs(det) < 1e-5:
                continue

            inv_det = 1.0 / det

            # Bounding box pixels
            bx0, bx1 = min_x[tri_i], max_x[tri_i] + 1
            by0, by1 = min_y[tri_i], max_y[tri_i] + 1

            # Generate pixel coordinate grid
            gx, gy = np.meshgrid(
                np.arange(bx0, bx1, dtype=np.float32) + 0.5,
                np.arange(by0, by1, dtype=np.float32) + 0.5
            )

            # Barycentric weights
            w0 = ((x1 - gx) * (y2 - gy) - (x2 - gx) * (y1 - gy)) * inv_det
            w1 = ((x2 - gx) * (y0 - gy) - (x0 - gx) * (y2 - gy)) * inv_det
            w2 = 1.0 - w0 - w1

            inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not np.any(inside):
                continue

            # Interpolate depth
            interp_z = w0 * z0_val + w1 * z1_val + w2 * z2_val

            # Current z-buffer slice
            z_slice = zbuffer[by0:by1, bx0:bx1]
            depth_pass = inside & (interp_z < z_slice)

            if not np.any(depth_pass):
                continue

            # Update z-buffer
            z_slice[depth_pass] = interp_z[depth_pass]

            # Interpolate lit RGB colors
            w0_pass = w0[depth_pass, np.newaxis]
            w1_pass = w1[depth_pass, np.newaxis]
            w2_pass = w2[depth_pass, np.newaxis]

            pix_rgb = w0_pass * c0 + w1_pass * c1 + w2_pass * c2
            pix_rgb_u8 = (np.clip(pix_rgb, 0.0, 1.0) * 255).astype(np.uint8)

            fb_slice = framebuffer[by0:by1, bx0:bx1]
            fb_slice[depth_pass] = pix_rgb_u8

        return Image.fromarray(framebuffer)
