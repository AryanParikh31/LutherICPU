import os
import sys
import time
import cv2
import numpy as np
from PIL import Image

def test_fast_raster():
    N = 200000
    H, W = 1080, 1920
    np.random.seed(42)
    
    centers = np.random.uniform([50, 50], [W - 50, H - 50], size=(N, 2)).astype(np.float32)
    depths = np.random.uniform(1.0, 20.0, size=N).astype(np.float32)
    colors = (np.random.uniform(0.1, 0.9, size=(N, 3)) * 255.0).astype(np.uint8)
    radii = np.random.randint(1, 4, size=N).astype(np.int32)
    
    # 1. Sort back to front or front to back
    t0 = time.time()
    order = np.argsort(-depths) # Back to front for painter's over-blend
    centers = centers[order]
    colors = colors[order]
    radii = radii[order]
    print(f"Sorting {N:,} points: {time.time() - t0:.3f}s")
    
    # Approach 1: OpenCV C++ batch circle drawing
    t0 = time.time()
    canvas = np.full((H, W, 3), (12, 14, 18), dtype=np.uint8)
    
    # Group by radius for maximum speed
    for r in np.unique(radii):
        mask = radii == r
        c_r = centers[mask].astype(np.int32)
        cols_r = colors[mask]
        for pt, col in zip(c_r, cols_r):
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), int(r), (int(col[0]), int(col[1]), int(col[2])), -1, lineType=cv2.LINE_AA)
            
    print(f"Rendered {N:,} anti-aliased Gaussian splats: {time.time() - t0:.3f}s")
    
    # Approach 2: Vectorized direct buffer splatting
    t0 = time.time()
    canvas2 = np.full((H, W, 3), (12, 14, 18), dtype=np.uint8)
    u = np.clip(np.round(centers[:, 0]).astype(np.int32), 0, W - 1)
    v = np.clip(np.round(centers[:, 1]).astype(np.int32), 0, H - 1)
    canvas2[v, u] = colors
    # Dilate slightly for solid coverage
    canvas2 = cv2.dilate(canvas2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    print(f"Rendered {N:,} vectorized splats: {time.time() - t0:.3f}s")

if __name__ == "__main__":
    test_fast_raster()
