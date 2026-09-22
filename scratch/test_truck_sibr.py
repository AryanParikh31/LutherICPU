import os
import sys
import time
import json
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from luther_core.types import CameraView, CameraIntrinsics, PointCloud
from luther_renderer.supreme_simulation_engine import SupremeSimulationEngine

def load_ply(path):
    positions = []
    colors = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = True
        for line in f:
            if header:
                if line.strip() == "end_header":
                    header = False
                continue
            parts = line.strip().split()
            if len(parts) >= 6:
                positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
                colors.append([float(parts[3]) / 255.0, float(parts[4]) / 255.0, float(parts[5]) / 255.0])
    return np.array(positions, dtype=np.float32), np.array(colors, dtype=np.float32)

def test_sibr():
    ply_path = "output/truck_photos_dense.ply"
    if not os.path.exists(ply_path):
        print(f"File {ply_path} not found.")
        return

    print("Loading point cloud...")
    pts, cols = load_ply(ply_path)
    print(f"Loaded {len(pts):,} points.")

    cameras_path = "output/cameras.json"
    views = []
    if os.path.exists(cameras_path):
        with open(cameras_path, "r") as f:
            cams = json.load(f)
            for c in cams[:20]:
                intr = CameraIntrinsics(
                    width=c["width"], height=c["height"],
                    fx=c["fx"], fy=c["fy"],
                    cx=c.get("cx", c["width"] / 2.0),
                    cy=c.get("cy", c["height"] / 2.0),
                    model="PINHOLE"
                )
                R = np.array(c["rotation"], dtype=np.float32)
                pos = np.array(c["position"], dtype=np.float32)
                tvec = (-R @ pos.reshape(3, 1)).ravel().astype(np.float32)
                cam = CameraView(image_id=c["id"], name=c["img_name"], qvec=np.array([1, 0, 0, 0], dtype=np.float32), tvec=tvec, intrinsics=intr)
                cam._R = R
                img_path = os.path.join(r"F:\tandt_db\tandt\truck\images", c["img_name"])
                if not os.path.exists(img_path):
                    img_path = os.path.join(PROJECT_ROOT, "uploads", "truck_photos", "images", c["img_name"])
                if os.path.exists(img_path):
                    cam.image_path = img_path
                views.append(cam)

    print(f"Loaded {len(views)} calibrated cameras.")
    engine = LutherISIBREngine(output_dir="output/test_sibr_proofs")
    field = LutherIGaussianField(positions=pts, colors=cols)
    
    t0 = time.time()
    results = engine.synthesize_continuous_simulation(
        gaussian_field=field,
        camera_views=views,
        scene_name="truck_sibr_test",
        num_turntable_frames=12
    )
    print(f"Done synthesis in {time.time() - t0:.2f}s: {results}")

if __name__ == "__main__":
    test_sibr()
