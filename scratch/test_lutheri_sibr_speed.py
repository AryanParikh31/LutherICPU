import os
import sys
import time
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from luther_core.types import CameraView, CameraIntrinsics
from luther_renderer.supreme_simulation_engine import SupremeSimulationEngine

def benchmark():
    N = 100000
    print(f"Generating synthetic scene with {N:,} Gaussians...")
    np.random.seed(42)
    positions = np.random.uniform(-5.0, 5.0, size=(N, 3)).astype(np.float32)
    colors = np.random.uniform(0.1, 0.9, size=(N, 3)).astype(np.float32)
    
    t0 = time.time()
    field = LutherIGaussianField(positions, colors)
    print(f"Field init: {time.time() - t0:.3f}s")
    
    intrinsics = CameraIntrinsics(width=1920, height=1080, fx=1200.0, fy=1200.0, cx=960.0, cy=540.0, model="PINHOLE")
    R = np.eye(3, dtype=np.float32)
    tvec = np.array([0.0, 0.0, -8.0], dtype=np.float32)
    cam = CameraView(image_id=1, name="test", qvec=np.array([1, 0, 0, 0], dtype=np.float32), tvec=tvec, intrinsics=intrinsics)
    cam._R = R
    
    projector = LutherISIBRProjector()
    t0 = time.time()
    proj = projector.project(field, cam, 1920, 1080)
    print(f"Projected {proj['num_visible']:,} visible Gaussians in {time.time() - t0:.3f}s")
    
    rasterizer = LutherISIBRRasterizer(tile_size=16)
    t0 = time.time()
    img = rasterizer.render(proj, 1920, 1080)
    print(f"Rendered 1080p in {time.time() - t0:.3f}s")

if __name__ == "__main__":
    benchmark()
