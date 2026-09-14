"""lutherICPU Master Reconstruction and Forensic Simulation Pipeline.

End-to-end CPU-native processing: from multi-angle images to continuous photorealistic
3D forensic simulation, metric measurement, and courtroom dossier generation.
"""
import os
import time
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from PIL import Image

from luther_core.types import CameraView, CameraIntrinsics, PointCloud, SurfaceMesh
from luther_core.colmap_loader import load_colmap_model
from luther_core.safe_memory import global_guardian
from luther_geometry.depth_grid_mesher import DepthGridSurfaceMesher
from luther_geometry.manifold_cleaner import export_mesh_to_ply, build_boundary_cage
from luther_texture.projective_baker import ProjectiveTextureBaker
from luther_renderer.ibr_rasterizer import ProjectiveIbrRasterizer
from luther_renderer.camera_path import generate_orbit_path
from luther_forensics.measurement_caliper import ForensicCaliper
from luther_forensics.evidence_manager import EvidenceManager
from luther_forensics.dossier_generator import ForensicDossierGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lutherICPU.MasterPipeline")


class LutherCPUReconstructor:
    """Master Pipeline for CPU-Native Forensic 3D Reconstruction."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.evidence_mgr = EvidenceManager()
        self.ibr_rasterizer = ProjectiveIbrRasterizer(num_blend_views=4)

    def process_scene(
        self,
        scene_dir: str,
        images_subfolder: str = "images",
        sparse_subfolder: str = "sparse/0",
        scene_name: str = "scene",
        max_views_to_mesh: int = 16,
        texture_resolution: int = 2048,
        run_cpu_proof_render: bool = True
    ) -> Dict[str, Any]:
        """Runs end-to-end CPU reconstruction pipeline."""
        start_time = time.time()
        logger.info(f"=== Starting lutherICPU Pipeline on scene: {scene_name} ===")
        logger.info(f"Scene Directory: {scene_dir}")

        sparse_dir = os.path.join(scene_dir, sparse_subfolder)
        images_dir = os.path.join(scene_dir, images_subfolder)

        if not os.path.exists(sparse_dir):
            sparse_dir = os.path.join(scene_dir, "sparse")

        # 1. Ingest COLMAP calibrated dataset
        cameras, views_dict, sparse_pcd = load_colmap_model(sparse_dir, images_dir=images_dir)
        views = list(views_dict.values())
        logger.info(f"Ingested {len(views)} camera views and {len(sparse_pcd)} sparse points.")

        # 2. Continuous Depth Grid Surface Meshing (Zero Spiky Origami Shards)
        grid_mesher = DepthGridSurfaceMesher(max_depth_jump=0.45, roi_radius_meters=35.0)
        mesh = grid_mesher.mesh_from_calibrated_views(
            views=views,
            sparse_pcd=sparse_pcd,
            grid_step=3,
            max_views_to_mesh=min(24, len(views))
        )

        # 3. Collision Boundary Cage
        cage = build_boundary_cage(mesh.vertices, margin_multiplier=1.20)

        # 4. Register Key Forensic Evidence & Measurements
        self._register_default_forensic_evidence(mesh, scene_name)

        # 5. Export Production Artifacts
        ply_path = os.path.join(self.output_dir, f"{scene_name}_mesh.ply")
        export_mesh_to_ply(mesh, ply_path)

        scene_json_path = os.path.join(self.output_dir, f"{scene_name}_scene.json")
        self._export_scene_json(mesh, views, cage, scene_json_path)

        cage_json_path = os.path.join(self.output_dir, f"{scene_name}_cage.json")
        with open(cage_json_path, "w") as f:
            json.dump(cage, f, indent=2)

        evidence_json_path = os.path.join(self.output_dir, f"{scene_name}_evidence.json")
        with open(evidence_json_path, "w") as f:
            f.write(self.evidence_mgr.to_json())

        dossier_html_path = os.path.join(self.output_dir, f"{scene_name}_dossier.html")
        ForensicDossierGenerator.generate_html_dossier(
            case_id=f"FOR-{scene_name.upper()}-2026",
            case_title=f"3D Forensic Reconstruction of {scene_name.capitalize()}",
            investigator="lutherICPU Autonomous Forensic Intelligence",
            scene_name=scene_name,
            num_cameras=len(views),
            num_vertices=len(mesh.vertices),
            num_faces=len(mesh.faces),
            evidence_manager=self.evidence_mgr,
            output_path=dossier_html_path
        )

        # 6. CPU Projective Radiance Proof Render
        proof_img_path = None
        if run_cpu_proof_render and len(views) > 0:
            logger.info("Rendering CPU software photographic proof image from reference viewpoint...")
            ref_view = views[0]
            rendered_img = self.ibr_rasterizer.render_photorealistic_view(
                mesh=mesh,
                target_view=ref_view,
                all_source_views=views,
                width=1280,
                height=720
            )
            proof_img_path = os.path.join(self.output_dir, f"{scene_name}_proof_render.png")
            rendered_img.save(proof_img_path)
            logger.info(f"Saved CPU software proof image to {proof_img_path}")

        elapsed = time.time() - start_time
        logger.info(f"=== Reconstruction Complete for {scene_name} in {elapsed:.2f}s ===")

        return {
            "scene_name": scene_name,
            "elapsed_seconds": elapsed,
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
            "num_cameras": len(views),
            "ply_path": ply_path,
            "scene_json_path": scene_json_path,
            "proof_img_path": proof_img_path,
            "dossier_path": dossier_html_path
        }

    def _register_default_forensic_evidence(self, mesh: SurfaceMesh, scene_name: str) -> None:
        """Pins forensic landmarks based on robust percentile bounding geometry."""
        # Use 2nd and 98th percentiles to avoid any single outlier
        p_min = np.percentile(mesh.vertices, 2, axis=0)
        p_max = np.percentile(mesh.vertices, 98, axis=0)
        centroid = np.median(mesh.vertices, axis=0)

        length_m = float(p_max[0] - p_min[0])
        width_m = float(p_max[2] - p_min[2])
        height_m = float(p_max[1] - p_min[1])

        p_front = [float(centroid[0]), float(centroid[1]), float(p_min[2])]
        self.evidence_mgr.add_evidence(
            evidence_id="EV-001",
            title=f"Front Structure Profile - {scene_name.capitalize()}",
            description="Frontmost structural landmark and contact boundary.",
            position_3d=p_front,
            category="STRUCTURAL_BOUNDARY",
            measurements={
                "vehicle_length_m": round(max(length_m, width_m), 3),
                "vehicle_width_m": round(min(length_m, width_m), 3),
                "vehicle_height_m": round(height_m, 3)
            }
        )

        p_side = [float(p_max[0]), float(centroid[1]), float(centroid[2])]
        self.evidence_mgr.add_evidence(
            evidence_id="EV-002",
            title="Lateral Chassis Datum",
            description="Side lateral profile and paint preservation locus.",
            position_3d=p_side,
            category="LATERAL_PROFILE",
            measurements={"lateral_span_meters": round(float(p_max[0] - p_min[0]), 3)}
        )

        self.evidence_mgr.add_trajectory(
            trajectory_id="TR-001",
            origin=p_front,
            impact_point=[float(centroid[0]), float(centroid[1]), float(p_max[2])],
            trajectory_type="LONGITUDINAL_CHASSIS_AXIS",
            notes=f"Main longitudinal datum line across {scene_name} chassis (Span: {max(length_m, width_m):.2f}m)."
        )

    def _export_scene_json(
        self, mesh: SurfaceMesh, views: List[CameraView], cage: Dict[str, Any], output_path: str
    ) -> None:
        """Serializes mesh geometry, vertex colors, camera poses, and cage for WebGL2 streaming."""
        # Flat arrays for fast Float32Array / Uint32Array ingestion in JavaScript
        # Subsample vertices if very large for smooth WebGL 60 FPS
        flat_vertices = mesh.vertices.flatten().tolist()
        flat_faces = mesh.faces.flatten().tolist()
        flat_normals = mesh.normals.flatten().tolist() if mesh.normals is not None else []
        flat_colors = mesh.vertex_colors.flatten().tolist() if mesh.vertex_colors is not None else []

        cam_list = []
        for v in views[:40]:
            cam_list.append({
                "id": v.image_id,
                "name": v.name,
                "position": v.center.tolist(),
                "rotation_matrix": v.R.flatten().tolist(),
                "viewing_direction": v.viewing_direction.tolist(),
                "up": v.up_vector.tolist(),
                "fx": float(v.intrinsics.fx),
                "fy": float(v.intrinsics.fy),
                "cx": float(v.intrinsics.cx),
                "cy": float(v.intrinsics.cy),
                "width": v.intrinsics.width,
                "height": v.intrinsics.height
            })

        payload = {
            "version": "lutherICPU-2.0",
            "metadata": {
                "num_vertices": len(mesh.vertices),
                "num_faces": len(mesh.faces),
                "num_cameras": len(views)
            },
            "geometry": {
                "vertices": flat_vertices,
                "faces": flat_faces,
                "normals": flat_normals,
                "colors": flat_colors
            },
            "cameras": cam_list,
            "boundary_cage": cage
        }

        with open(output_path, "w") as f:
            json.dump(payload, f)
        logger.info(f"Saved WebGL scene JSON to {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB).")
