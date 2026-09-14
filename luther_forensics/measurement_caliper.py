"""lutherICPU Forensic Measurement Caliper and Damage Analysis Suite.

Provides metric distance, crush depth, impact angle, area, and trajectory calculation
for courtroom forensic exhibits and insurance damage investigations.
"""
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class ForensicCaliper:
    """Forensic metric measurement and spatial analysis engine."""

    @staticmethod
    def measure_distance_3d(
        point_a: np.ndarray, point_b: np.ndarray, unit_scale_to_meters: float = 1.0
    ) -> Dict[str, float]:
        """Calculates 3D Euclidean distance, horizontal distance, and vertical elevation delta."""
        p_a = np.array(point_a, dtype=np.float32)
        p_b = np.array(point_b, dtype=np.float32)

        delta = p_b - p_a
        dist_3d = float(np.linalg.norm(delta) * unit_scale_to_meters)
        dist_horiz = float(np.linalg.norm(delta[[0, 2]]) * unit_scale_to_meters)  # X-Z plane
        dist_vert = float(abs(delta[1]) * unit_scale_to_meters)  # Y axis (height)

        return {
            "distance_meters": dist_3d,
            "distance_centimeters": dist_3d * 100.0,
            "distance_feet": dist_3d * 3.28084,
            "horizontal_meters": dist_horiz,
            "vertical_delta_meters": dist_vert,
            "delta_vector": delta.tolist()
        }

    @staticmethod
    def measure_impact_angle(
        origin_pt: np.ndarray, impact_pt: np.ndarray, surface_normal: np.ndarray
    ) -> Dict[str, float]:
        """Calculates incidence angle and trajectory vector relative to impact surface normal."""
        traj = np.array(impact_pt, dtype=np.float32) - np.array(origin_pt, dtype=np.float32)
        traj_norm = traj / np.maximum(np.linalg.norm(traj), 1e-6)

        n_norm = np.array(surface_normal, dtype=np.float32)
        n_norm = n_norm / np.maximum(np.linalg.norm(n_norm), 1e-6)

        cos_angle = float(np.clip(np.dot(-traj_norm, n_norm), -1.0, 1.0))
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)

        return {
            "incidence_angle_degrees": angle_deg,
            "glance_angle_degrees": 90.0 - angle_deg,
            "trajectory_unit_vector": traj_norm.tolist()
        }

    @staticmethod
    def calculate_crush_depth(
        damaged_points: np.ndarray, reference_plane_pt: np.ndarray, reference_plane_normal: np.ndarray
    ) -> Dict[str, float]:
        """Measures maximum and average deformation depth (intrusion) into vehicle chassis."""
        pts = np.array(damaged_points, dtype=np.float32)
        p0 = np.array(reference_plane_pt, dtype=np.float32)
        n = np.array(reference_plane_normal, dtype=np.float32)
        n = n / np.maximum(np.linalg.norm(n), 1e-6)

        # Distance of each point to the un-deformed reference plane
        diffs = pts - p0
        signed_depths = np.dot(diffs, n)

        max_intrusion = float(np.max(np.abs(signed_depths)))
        mean_intrusion = float(np.mean(np.abs(signed_depths)))

        return {
            "max_intrusion_depth_meters": max_intrusion,
            "max_intrusion_depth_inches": max_intrusion * 39.3701,
            "mean_intrusion_depth_meters": mean_intrusion,
            "sample_count": len(pts)
        }
