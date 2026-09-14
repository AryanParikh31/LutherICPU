"""Unit tests for lutherICPU Forensic Analysis, Measurement Caliper, and Dossier Generation."""
import os
import numpy as np
import pytest

from luther_forensics.measurement_caliper import ForensicCaliper
from luther_forensics.evidence_manager import EvidenceManager
from luther_forensics.dossier_generator import ForensicDossierGenerator


def test_metric_distance_calculation():
    """Verifies that 3D Euclidean caliper computes exact metric distances."""
    pA = [0.0, 0.0, 0.0]
    pB = [3.0, 4.0, 0.0]  # 3-4-5 right triangle

    meas = ForensicCaliper.measure_distance_3d(pA, pB, unit_scale_to_meters=1.0)
    assert pytest.approx(meas["distance_meters"], rel=1e-5) == 5.0
    assert pytest.approx(meas["distance_centimeters"], rel=1e-5) == 500.0


def test_crush_depth_calculation():
    """Verifies intrusion and deformation depth analysis against reference plane."""
    # Flat reference plane at Z = 0 with normal [0, 0, 1]
    ref_pt = [0.0, 0.0, 0.0]
    ref_norm = [0.0, 0.0, 1.0]

    # Damaged points dented inwards by up to 0.45 meters
    damaged_pts = np.array([
        [0.0, 0.0, -0.15],
        [0.5, 0.2, -0.45],
        [-0.3, -0.1, -0.10]
    ], dtype=np.float32)

    crush = ForensicCaliper.calculate_crush_depth(damaged_pts, ref_pt, ref_norm)
    assert pytest.approx(crush["max_intrusion_depth_meters"], rel=1e-5) == 0.45
    assert crush["sample_count"] == 3


def test_evidence_manager_and_dossier_generation(tmp_path):
    """Verifies that evidence pins are serialized and compiled into HTML dossier."""
    mgr = EvidenceManager()
    mgr.add_evidence(
        evidence_id="EV-001",
        title="Impact Crease",
        description="Crush intrusion on passenger door.",
        position_3d=[1.2, -0.5, 2.3],
        category="CRUSH_DAMAGE",
        measurements={"max_depth_m": 0.35}
    )

    mgr.add_trajectory(
        trajectory_id="TR-001",
        origin=[0.0, 0.0, 0.0],
        impact_point=[1.2, -0.5, 2.3],
        trajectory_type="IMPACT_VECTOR"
    )

    out_html = os.path.join(tmp_path, "test_dossier.html")
    ForensicDossierGenerator.generate_html_dossier(
        case_id="CASE-TEST-01",
        case_title="Test Accident Reconstruction",
        investigator="Agent Tester",
        scene_name="truck",
        num_cameras=251,
        num_vertices=25000,
        num_faces=48000,
        evidence_manager=mgr,
        output_path=out_html
    )

    assert os.path.exists(out_html)
    with open(out_html, "r", encoding="utf-8") as f:
        content = f.read()
        assert "EV-001" in content
        assert "Impact Crease" in content
        assert "TR-001" in content
