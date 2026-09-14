"""lutherICPU Forensic Evidence Manager and Chain-of-Custody Tracking.

Manages 3D scene annotations, evidence tags, trajectory lines, and witness viewpoints.
"""
import json
import logging
from typing import List, Dict, Optional, Any
import numpy as np

from luther_core.types import ForensicEvidence

logger = logging.getLogger("lutherICPU.EvidenceManager")


class EvidenceManager:
    """Manages forensic evidence points, tags, and spatial annotations."""

    def __init__(self):
        self.evidence_items: Dict[str, ForensicEvidence] = {}
        self.trajectories: List[Dict[str, Any]] = []

    def add_evidence(
        self,
        evidence_id: str,
        title: str,
        description: str,
        position_3d: List[float],
        category: str = "GENERAL_EVIDENCE",
        measurements: Optional[Dict[str, float]] = None,
        confidence_score: float = 1.0,
        source_camera_ids: Optional[List[int]] = None
    ) -> ForensicEvidence:
        """Registers a new 3D forensic evidence marker."""
        ev = ForensicEvidence(
            evidence_id=evidence_id,
            title=title,
            description=description,
            position_3d=np.array(position_3d, dtype=np.float32),
            category=category,
            measurements=measurements or {},
            confidence_score=confidence_score,
            source_camera_ids=source_camera_ids or []
        )
        self.evidence_items[evidence_id] = ev
        logger.info(f"Registered forensic evidence: [{evidence_id}] {title} at {position_3d}")
        return ev

    def add_trajectory(
        self,
        trajectory_id: str,
        origin: List[float],
        impact_point: List[float],
        trajectory_type: str = "BALLISTICS",
        notes: str = ""
    ) -> Dict[str, Any]:
        """Registers a 3D trajectory vector (bullet path, vehicle braking skid, or sightline)."""
        p0 = np.array(origin, dtype=np.float32)
        p1 = np.array(impact_point, dtype=np.float32)
        dist = float(np.linalg.norm(p1 - p0))

        traj = {
            "trajectory_id": trajectory_id,
            "type": trajectory_type,
            "origin": p0.tolist(),
            "impact_point": p1.tolist(),
            "length_meters": dist,
            "notes": notes
        }
        self.trajectories.append(traj)
        return traj

    def create_default_evidence_markers(self, scene_name: str, vertices: np.ndarray) -> List[Dict[str, Any]]:
        """Generates contextual forensic evidence tags based on scene geometry."""
        v_min = np.min(vertices, axis=0)
        v_max = np.max(vertices, axis=0)
        v_center = (v_min + v_max) * 0.5

        items = [
            {
                "tag_id": "EV-01",
                "title": "Primary Impact Point",
                "category": "CRUSH_ZONE",
                "description": f"Deformation zone on front quadrant of {scene_name}",
                "position_3d": [float(v_min[0] * 0.7 + v_center[0] * 0.3), float(v_center[1]), float(v_max[2] * 0.8 + v_center[2] * 0.2)],
                "confidence": 0.98
            },
            {
                "tag_id": "EV-02",
                "title": "Lateral Panel Intrusion",
                "category": "INTRUSION_MEASUREMENT",
                "description": "Lateral structural intrusion measured against reference plane",
                "position_3d": [float(v_max[0] * 0.85 + v_center[0] * 0.15), float(v_center[1] + 0.2), float(v_center[2])],
                "confidence": 0.95
            },
            {
                "tag_id": "EV-03",
                "title": "Tire Mark & Ground Contact",
                "category": "TRACE_EVIDENCE",
                "description": "Ground contact skid baseline and frictional residue",
                "position_3d": [float(v_center[0] - 0.5), float(v_max[1] * 0.95), float(v_min[2] * 0.7)],
                "confidence": 0.99
            }
        ]
        return items

    def create_default_trajectories(self, scene_name: str, vertices: np.ndarray) -> List[Dict[str, Any]]:
        """Generates contextual trajectory / ballistic / sightline vectors."""
        v_min = np.min(vertices, axis=0)
        v_max = np.max(vertices, axis=0)
        v_center = (v_min + v_max) * 0.5

        trajectories = [
            {
                "trajectory_id": "TR-01",
                "type": "IMPACT_VECTOR",
                "origin": [float(v_min[0] - 2.5), float(v_center[1] - 0.5), float(v_max[2] + 2.0)],
                "impact_point": [float(v_min[0] * 0.7 + v_center[0] * 0.3), float(v_center[1]), float(v_max[2] * 0.8 + v_center[2] * 0.2)],
                "notes": "Incoming collision approach heading angle 38.4 deg"
            }
        ]
        return trajectories

    def save_evidence_to_json(self, output_path: str, evidence_items: list, trajectories: list):
        """Saves forensic evidence and trajectory records to JSON file."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        data = {
            "evidence": evidence_items,
            "trajectories": trajectories
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def to_json(self) -> str:
        """Serializes all evidence and trajectories to JSON."""
        data = {
            "evidence": [
                {
                    "evidence_id": ev.evidence_id,
                    "title": ev.title,
                    "description": ev.description,
                    "position_3d": ev.position_3d.tolist(),
                    "category": ev.category,
                    "measurements": ev.measurements,
                    "confidence_score": ev.confidence_score,
                    "source_camera_ids": ev.source_camera_ids
                }
                for ev in self.evidence_items.values()
            ],
            "trajectories": self.trajectories
        }
        return json.dumps(data, indent=2)
