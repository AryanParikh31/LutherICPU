"""
lutherICPU — Pure CPU-Native 3D Reconstruction & Simulation Model.

The Complete 5-Stage Mathematical Pipeline:
  Stage 1: Ingestion & Quality Control (SHA-256 hash & Laplacian variance blur metric Var(∇²I))
  Stage 2: Structure-from-Motion (SIFT DoG, RANSAC Epipolar Essential Matrix, DLT Triangulation / COLMAP)
  Stage 3: Dense Multi-View Stereo (CPU PatchMatch stereo with NCC photo-consistency & dense fusion)
  Stage 4: Watertight Surface Reconstruction (Volumetric TSDF / Screened Poisson & Taubin smoothing)
  Stage 5: Texturing & Continuous Simulation Rendering (4K Photographic Atlas & Supreme Radiance Synthesis)

Pure model architecture: Zero frontend dependencies, zero web server requirements.
Built for high-fidelity 3D simulation on standard CPU hardware (e.g. Intel Core i5 / 8GB RAM).
"""

import os
import sys
import time
import math
import json
import struct
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from luther_core.types import CameraView, CameraIntrinsics, PointCloud, SurfaceMesh
from luther_core.safe_memory import MemoryGuardian, global_guardian
from luther_core.ingestion import ImageQualityController
from luther_core.sfm import StructureFromMotionEngine
from luther_core.colmap_loader import load_colmap_model
from luther_pipeline.dense_mvs_engine import LutherDenseMVSEngine, find_top_neighbor_cameras
from luther_geometry.volumetric_tsdf_fusion import VolumetricTSDFFusionEngine
from luther_geometry.manifold_cleaner import export_mesh_to_ply, build_boundary_cage
from luther_texture.projective_baker import ProjectiveTextureBaker
from luther_renderer.supreme_simulation_engine import SupremeSimulationEngine

logger = logging.getLogger("lutherICPU.Model")


class LutherICPU:
    """
    Pure CPU 3D Reconstruction and Simulation Model.

    Takes multi-angle source photographs, analyzes all images in detail, executes
    iterative densification and reconstruction, and generates complete, photorealistic
    3D textured simulation assets and multi-angle proof renders.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        max_ram_gb: float = 3.2,
        blur_threshold: float = 60.0
    ):
        self.custom_output_dir = Path(output_dir) if output_dir else None
        self.output_dir = self.custom_output_dir or Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_ram_gb = max_ram_gb
        self.guardian = MemoryGuardian(max_process_ram_gb=max_ram_gb)
        self.qc_controller = ImageQualityController(blur_threshold=blur_threshold)
        self.sfm_engine = StructureFromMotionEngine()
        self.mvs_engine = LutherDenseMVSEngine(output_dir=str(self.output_dir), max_ram_gb=max_ram_gb)
        self.tsdf_engine = VolumetricTSDFFusionEngine(max_ram_gb=max_ram_gb)
        self.supreme_renderer = SupremeSimulationEngine(output_dir=str(self.output_dir), max_ram_gb=max_ram_gb)

    def simulate(
        self,
        images_path: str,
        colmap_path: Optional[str] = None,
        iterations: int = 30000,
        scene_name: Optional[str] = None,
        texture_resolution: int = 2048,
        render_simulation: bool = True,
        max_views: int = 40,
        progress_callback: Optional[Callable[[int, int, str, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes complete end-to-end 3D reconstruction and simulation.

        Args:
            images_path: Directory containing source photographs.
            colmap_path: Optional path to pre-calibrated COLMAP sparse folder.
            iterations: Iteration optimization target (e.g. 7000 to 30000).
            scene_name: Scene identification name.
            texture_resolution: UV texture atlas resolution (e.g. 2048 or 4096).
            render_simulation: If True, synthesizes continuous photorealistic 1080p, 4K, and turntable renders.
            max_views: Maximum key views for dense stereo matching.
            progress_callback: Optional callback(current_iter, total_iters, stage_name, metrics).

        Returns:
            Dictionary containing model results, mesh metrics, asset filepaths, and simulation renders.
        """
        t_start = time.time()

        images_dir = os.path.abspath(images_path)
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Source images directory not found: {images_dir}")

        if not scene_name:
            scene_name = os.path.basename(os.path.normpath(images_dir))
            if scene_name.lower() in ["images", "img", "photos", ""]:
                scene_name = os.path.basename(os.path.dirname(os.path.normpath(images_dir)))
        scene_name = scene_name.lower()

        # Generate timestamped output directory
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        if self.custom_output_dir:
            self.output_dir = self.custom_output_dir
        else:
            self.output_dir = Path("output") / f"{scene_name}_{timestamp_str}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mvs_engine.output_dir = str(self.output_dir)
        self.supreme_renderer.output_dir = str(self.output_dir)

        logger.info("\n" + "=" * 80)
        logger.info(f" [lutherICPU] INITIATING 3D RECONSTRUCTION SIMULATION MODEL")
        logger.info(f" Scene:              {scene_name.upper()}")
        logger.info(f" Timestamp:          {timestamp_str}")
        logger.info(f" Images Directory:   {images_dir}")
        logger.info(f" COLMAP Sparse Path: {colmap_path or 'Auto-Detect / SIFT SfM'}")
        logger.info(f" Iterations Target:  {iterations:,}")
        logger.info(f" Output Directory:   {self.output_dir}")
        logger.info(f" RAM Safety Limit:   {self.max_ram_gb:.1f} GB (CPU Optimized)")
        logger.info("=" * 80 + "\n")

        def emit_progress(curr_i: int, stage_desc: str, extra: Optional[Dict[str, Any]] = None):
            pct = min(100.0, (curr_i / max(1, iterations)) * 100.0)
            metrics = {
                "iteration": curr_i,
                "total_iterations": iterations,
                "percentage": round(pct, 1),
                "stage": stage_desc,
                "ram_mb": round(self.guardian.process_memory_gb * 1024, 1),
                "system_ram_pct": round(self.guardian.system_memory_percent, 1)
            }
            if extra:
                metrics.update(extra)
            if progress_callback:
                progress_callback(curr_i, iterations, stage_desc, metrics)
            logger.info(f"[{curr_i:,}/{iterations:,} ({pct:.1f}%)] {stage_desc} (RAM: {metrics['ram_mb']:.1f} MB)")

        # ----------------------------------------------------
        # Stage 1: Ingestion & Quality Control (0% - 10%)
        # ----------------------------------------------------
        emit_progress(0, "Stage 1/5: Ingesting Photographs, Computing SHA-256 Hashes & Laplacian Blur Scores...")
        qc_result = self.qc_controller.ingest_dataset(images_dir)
        self.guardian.checkpoint("Stage 1 Ingestion")

        # Save manifest
        manifest_path = self.output_dir / f"{scene_name}_qc_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(qc_result, f, indent=2)

        emit_progress(
            int(iterations * 0.10),
            f"Stage 1/5 Complete: {qc_result['total_images']} photos analyzed. Avg sharpness: {qc_result['average_blur_score']:.1f}.",
            {"qc_passed": qc_result["total_images"] - qc_result["flagged_images"], "qc_flagged": qc_result["flagged_images"]}
        )

        # ----------------------------------------------------
        # Stage 2: Structure-from-Motion (10% - 25%)
        # ----------------------------------------------------
        emit_progress(int(iterations * 0.12), "Stage 2/5: Structure-from-Motion (DoG SIFT / Epipolar Geometry / Pose Recovery)...")
        cameras, views_dict, sparse_pcd = self.sfm_engine.load_or_reconstruct(
            images_dir=images_dir,
            sparse_colmap_dir=colmap_path
        )
        self.guardian.checkpoint("Stage 2 SfM")
        all_views = list(views_dict.values())
        total_views = len(all_views)
        raw_points = len(sparse_pcd.positions)

        emit_progress(
            int(iterations * 0.25),
            f"Stage 2/5 Complete: {total_views} calibrated camera views, {raw_points:,} sparse 3D seed points.",
            {"cameras": total_views, "sparse_points": raw_points}
        )

        # ----------------------------------------------------
        # Stage 3: Dense Multi-View Stereo (PatchMatch NCC) (25% - 60%)
        # ----------------------------------------------------
        iter_stage_3_start = int(iterations * 0.25)
        iter_stage_3_end = int(iterations * 0.60)
        emit_progress(iter_stage_3_start, "Stage 3/5: Dense Multi-View Stereo (PatchMatch NCC & Point Fusion)...")

        views_to_dense = all_views[:max_views]
        num_views = len(views_to_dense)
        all_dense_points = [sparse_pcd.positions]
        all_dense_colors = [sparse_pcd.colors]

        for i, ref_v in enumerate(views_to_dense):
            self.guardian.checkpoint(f"Dense MVS View {i+1}/{num_views}")
            nbrs = find_top_neighbor_cameras(ref_v, all_views, max_neighbors=4)
            pts, cols = self.mvs_engine.estimate_dense_points_for_view(
                ref_view=ref_v,
                neighbor_views=nbrs,
                sparse_points=sparse_pcd.positions,
                downscale=2
            )
            if len(pts) > 0:
                all_dense_points.append(pts)
                all_dense_colors.append(cols)

            curr_i = iter_stage_3_start + int(((i + 1) / num_views) * (iter_stage_3_end - iter_stage_3_start))
            total_pts_now = sum(len(p) for p in all_dense_points)
            emit_progress(
                curr_i,
                f"Stage 3/5: Dense Stereo Matching ({i+1}/{num_views} views, {total_pts_now:,} surface points)...",
                {"dense_points": total_pts_now, "views_processed": i + 1}
            )

        dense_positions = np.concatenate(all_dense_points, axis=0)
        dense_colors = np.concatenate(all_dense_colors, axis=0)
        logger.info(f"Fused dense point cloud: {len(dense_positions):,} surface points.")

        # Save Dense Point Cloud PLY
        dense_ply_path = self.output_dir / f"{scene_name}_dense.ply"
        dense_pcd = PointCloud(positions=dense_positions, colors=dense_colors)
        with open(dense_ply_path, "w") as fp:
            fp.write(f"ply\nformat ascii 1.0\nelement vertex {len(dense_positions)}\n"
                     f"property float x\nproperty float y\nproperty float z\n"
                     f"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for p, c in zip(dense_positions, dense_colors):
                r, g, b = int(c[0] * 255) if c[0] <= 1.0 else int(c[0]), int(c[1] * 255) if c[1] <= 1.0 else int(c[1]), int(c[2] * 255) if c[2] <= 1.0 else int(c[2])
                fp.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {r} {g} {b}\n")

        # ----------------------------------------------------
        # Stage 4: Watertight Surface Reconstruction (60% - 80%)
        # ----------------------------------------------------
        iter_stage_4_start = int(iterations * 0.60)
        iter_stage_4_end = int(iterations * 0.80)
        emit_progress(iter_stage_4_start, "Stage 4/5: Volumetric TSDF & Taubin Volume-Preserving Manifold Reconstruction...")

        cam_centers = np.array([v.center for v in all_views], dtype=np.float32)
        verts, faces, vert_colors = self.tsdf_engine.reconstruct_scene(
            points=dense_positions,
            colors=dense_colors,
            camera_centers=cam_centers,
            voxel_res=192,
            camera_views=all_views
        )
        self.guardian.checkpoint("Stage 4 Meshing")

        mesh = SurfaceMesh(vertices=verts, faces=faces, vertex_colors=vert_colors)
        mesh_ply_path = self.output_dir / f"{scene_name}_mesh.ply"
        export_mesh_to_ply(mesh, str(mesh_ply_path))

        emit_progress(
            iter_stage_4_end,
            f"Stage 4/5 Complete: Reconstructed clean manifold surface ({len(mesh.vertices):,} vertices, {len(mesh.faces):,} triangles).",
            {"vertices": len(mesh.vertices), "faces": len(mesh.faces)}
        )

        # ----------------------------------------------------
        # Stage 5: Photographic Texture Atlas & Continuous Simulation (80% - 100%)
        # ----------------------------------------------------
        iter_stage_5_start = int(iterations * 0.80)
        emit_progress(iter_stage_5_start, f"Stage 5/5: Baking High-Resolution UV Texture Atlas ({texture_resolution}x{texture_resolution})...")

        baker = ProjectiveTextureBaker(atlas_resolution=texture_resolution, min_cos_angle=0.15)
        tex_map, baked_colors = baker.bake_photorealistic_texture(
            mesh=mesh,
            views=all_views,
            max_cameras_to_blend=min(24, total_views)
        )
        mesh.texture_map = tex_map
        mesh.vertex_colors = baked_colors
        self.guardian.checkpoint("Stage 5 Texturing")

        # Save Diffuse PNG Atlas
        diffuse_png_path = self.output_dir / f"{scene_name}_diffuse_atlas.png"
        diffuse_img = Image.fromarray(tex_map)
        diffuse_img.save(diffuse_png_path, format="PNG")

        # Save Textured Wavefront OBJ + MTL
        obj_path = self.output_dir / f"{scene_name}_simulation.obj"
        mtl_path = self.output_dir / f"{scene_name}_simulation.mtl"

        with open(mtl_path, "w") as f:
            f.write(f"# lutherICPU Material\nnewmtl Material_Simulation\nKd 1.0 1.0 1.0\nmap_Kd {os.path.basename(diffuse_png_path)}\n")

        with open(obj_path, "w") as f:
            f.write(f"# lutherICPU Photorealistic 3D Simulation Asset\nmtllib {os.path.basename(mtl_path)}\nusemtl Material_Simulation\n")
            for i, v in enumerate(mesh.vertices):
                c = baked_colors[i]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
            if mesh.normals is not None:
                for n in mesh.normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            for face in mesh.faces:
                f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")

        # Save Fast Binary .bin mesh asset
        bin_path = self.output_dir / f"{scene_name}_simulation.bin"
        header = struct.pack('<IIII', len(mesh.vertices), len(mesh.faces), 1, 1)
        v_norms = mesh.normals if mesh.normals is not None else np.tile([0.0, 0.0, 1.0], (len(mesh.vertices), 1)).astype(np.float32)
        with open(bin_path, 'wb') as fp:
            fp.write(header)
            fp.write(mesh.vertices.astype(np.float32).tobytes())
            fp.write(mesh.vertex_colors.astype(np.float32).tobytes())
            fp.write(v_norms.astype(np.float32).tobytes())
            fp.write(mesh.faces.astype(np.uint32).tobytes())

        # Continuous Photographic Simulation Proof Renders
        render_results = {}
        if render_simulation and len(all_views) > 0:
            emit_progress(int(iterations * 0.92), "Stage 5/5: Synthesizing Continuous 1080p / 4K Photorealistic Radiance Renders...")
            
            # Cache source views
            cached_sources = []
            for v in all_views[:28]:
                if v.image_path and os.path.exists(v.image_path):
                    im = Image.open(v.image_path).convert("RGB")
                    cached_sources.append((v, np.array(im, dtype=np.float32) / 255.0))

            proof_dir = self.output_dir / "proof_renders"
            proof_dir.mkdir(parents=True, exist_ok=True)

            # Hero Angle Render (1080p)
            hero_cam = all_views[0]
            hero_frame = self.supreme_renderer.synthesize_continuous_view(
                points=dense_positions,
                target_cam=hero_cam,
                source_views=cached_sources,
                width=1920,
                height=1080,
                infill_radius=24,
                point_colors=dense_colors,
                mesh=mesh,
                diffuse_texture=tex_map
            )
            hero_path = proof_dir / f"{scene_name}_simulation_1080p_hero.png"
            Image.fromarray(hero_frame).save(hero_path)
            render_results["hero_1080p"] = str(hero_path)

            # Side Profile Render (1080p)
            side_cam = all_views[min(15, len(all_views) - 1)]
            side_frame = self.supreme_renderer.synthesize_continuous_view(
                points=dense_positions,
                target_cam=side_cam,
                source_views=cached_sources,
                width=1920,
                height=1080,
                infill_radius=24,
                point_colors=dense_colors,
                mesh=mesh,
                diffuse_texture=tex_map
            )
            side_path = proof_dir / f"{scene_name}_simulation_1080p_side.png"
            Image.fromarray(side_frame).save(side_path)
            render_results["side_1080p"] = str(side_path)

            # 4K UHD Master Render
            uhd_frame = self.supreme_renderer.synthesize_continuous_view(
                points=dense_positions,
                target_cam=hero_cam,
                source_views=cached_sources,
                width=3840,
                height=2160,
                infill_radius=32,
                point_colors=dense_colors,
                mesh=mesh,
                diffuse_texture=tex_map
            )
            uhd_path = proof_dir / f"{scene_name}_simulation_4k_ultra.png"
            Image.fromarray(uhd_frame).save(uhd_path)
            render_results["uhd_4k"] = str(uhd_path)

            # 360 Turntable Simulation GIF
            logger.info("Generating 360° Continuous Turntable Simulation GIF (24 frames)...")
            scene_centroid = np.median(dense_positions, axis=0)
            turntable_frames = []
            radius = float(np.percentile(np.linalg.norm(dense_positions - scene_centroid, axis=1), 90) * 1.5)
            radius = max(3.5, min(12.0, radius))

            for frame_i in range(24):
                angle = (frame_i / 24.0) * (2.0 * math.pi)
                cam_x = scene_centroid[0] + radius * math.cos(angle)
                cam_y = scene_centroid[1] - 0.6
                cam_z = scene_centroid[2] + radius * math.sin(angle)
                cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

                fwd = scene_centroid - cam_pos
                fwd /= np.linalg.norm(fwd)
                up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
                right = np.cross(fwd, up)
                right /= np.maximum(np.linalg.norm(right), 1e-6)
                true_up = np.cross(right, fwd)

                R_rot = np.vstack([right, true_up, fwd])
                t_vec = -R_rot @ cam_pos

                synth_cam = CameraView(
                    image_id=9000 + frame_i,
                    name=f"orbit_{frame_i:02d}",
                    qvec=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    tvec=t_vec,
                    intrinsics=hero_cam.intrinsics
                )
                synth_cam._R = R_rot

                frame_rgb = self.supreme_renderer.synthesize_continuous_view(
                    points=dense_positions,
                    target_cam=synth_cam,
                    source_views=cached_sources,
                    width=640,
                    height=360,
                    infill_radius=18,
                    point_colors=dense_colors,
                    mesh=mesh,
                    diffuse_texture=tex_map
                )
                turntable_frames.append(Image.fromarray(frame_rgb))

            gif_path = proof_dir / f"{scene_name}_360_simulation.gif"
            turntable_frames[0].save(
                gif_path,
                save_all=True,
                append_images=turntable_frames[1:],
                duration=120,
                loop=0
            )
            render_results["turntable_gif"] = str(gif_path)

        # Export calibrated cameras array (cameras.json)
        cameras_data = []
        for v in all_views:
            cam_entry = {
                "id": v.image_id,
                "img_name": v.name,
                "width": v.intrinsics.width if hasattr(v.intrinsics, "width") else 1920,
                "height": v.intrinsics.height if hasattr(v.intrinsics, "height") else 1080,
                "position": v.center.tolist() if isinstance(v.center, np.ndarray) else list(v.center),
                "rotation": v.R.tolist() if isinstance(v.R, np.ndarray) else list(v.R),
                "fx": float(v.intrinsics.fx),
                "fy": float(v.intrinsics.fy)
            }
            cameras_data.append(cam_entry)

        cameras_json_path = self.output_dir / "cameras.json"
        with open(cameras_json_path, "w") as f:
            json.dump(cameras_data, f, indent=2)

        # Export scene manifest & boundary cage ({scene_name}_scene.json)
        cage_dict = build_boundary_cage(mesh.vertices)
        scene_manifest = {
            "scene_name": scene_name,
            "timestamp": timestamp_str,
            "iterations": iterations,
            "elapsed_seconds": round(time.time() - t_start, 2),
            "centroid": cage_dict.get("centroid", [0.0, 0.0, 0.0]),
            "safe_radius": cage_dict.get("safe_radius", 10.0),
            "boundary_cage": cage_dict,
            "num_cameras": total_views,
            "num_dense_points": len(dense_positions),
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
            "cameras": cameras_data,
            "asset_files": {
                "obj": str(obj_path.name),
                "mtl": str(mtl_path.name),
                "diffuse_atlas": str(diffuse_png_path.name),
                "mesh_ply": str(mesh_ply_path.name),
                "dense_ply": str(dense_ply_path.name),
                "binary_mesh": str(bin_path.name),
                "cameras_json": "cameras.json"
            }
        }
        scene_json_path = self.output_dir / f"{scene_name}_scene.json"
        with open(scene_json_path, "w") as f:
            json.dump(scene_manifest, f, indent=2)

        # Write latest simulation pointer into output/latest_simulation.json
        output_root = Path(PROJECT_ROOT) / "output"
        output_root.mkdir(parents=True, exist_ok=True)
        latest_pointer = {
            "scene_name": scene_name,
            "timestamp": timestamp_str,
            "output_dir": str(self.output_dir.resolve()),
            "relative_dir": os.path.relpath(str(self.output_dir), str(output_root)),
            "iterations": iterations,
            "obj_path": str(obj_path.resolve()),
            "mtl_path": str(mtl_path.resolve()),
            "diffuse_png_path": str(diffuse_png_path.resolve()),
            "mesh_ply_path": str(mesh_ply_path.resolve()),
            "binary_mesh_path": str(bin_path.resolve()),
            "cameras_json_path": str(cameras_json_path.resolve()),
            "scene_json_path": str(scene_json_path.resolve())
        }
        with open(output_root / "latest_simulation.json", "w") as f:
            json.dump(latest_pointer, f, indent=2)

        emit_progress(iterations, "3D Simulation Generation Complete: 100% Solid Photorealistic 3D Model Generated!")

        total_elapsed = time.time() - t_start

        logger.info("\n" + "=" * 80)
        logger.info(f" [SUCCESS] 3D SIMULATION MODEL FINISHED FOR '{scene_name.upper()}'")
        logger.info(f" Archived Directory: {self.output_dir}")
        logger.info(f" Mesh Vertices:      {len(mesh.vertices):,}")
        logger.info(f" Mesh Triangles:     {len(mesh.faces):,}")
        logger.info(f" Dense Point Cloud:  {len(dense_positions):,} points")
        logger.info(f" Calibrated Cameras: {total_views}")
        logger.info(f" OBJ 3D Model:       {obj_path}")
        logger.info(f" Diffuse 4K Atlas:   {diffuse_png_path}")
        logger.info(f" Binary Mesh (.bin): {bin_path}")
        if render_results:
            logger.info(f" Hero 1080p Render:  {render_results.get('hero_1080p')}")
            logger.info(f" 360 Turntable GIF:  {render_results.get('turntable_gif')}")
        logger.info(f" Total Processing:   {total_elapsed:.1f}s on CPU")
        logger.info("=" * 80 + "\n")

        return {
            "status": "SUCCESS",
            "scene_name": scene_name,
            "timestamp": timestamp_str,
            "output_dir": str(self.output_dir),
            "elapsed_seconds": round(total_elapsed, 2),
            "iterations": iterations,
            "num_cameras": total_views,
            "num_dense_points": len(dense_positions),
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
            "qc_manifest": str(manifest_path),
            "dense_ply_path": str(dense_ply_path),
            "mesh_ply_path": str(mesh_ply_path),
            "obj_path": str(obj_path),
            "mtl_path": str(mtl_path),
            "diffuse_png_path": str(diffuse_png_path),
            "binary_mesh_path": str(bin_path),
            "cameras_json_path": str(cameras_json_path),
            "scene_json_path": str(scene_json_path),
            "renders": render_results
        }
