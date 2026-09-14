import os
import sys
import numpy as np
import cv2
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from luther_core.colmap_loader import load_colmap_model

def analyze():
    colmap_path = r"uploads\truck_photos\sparse\0"
    images_path = r"F:\tandt_db\tandt\truck\images"
    
    cameras, views_dict, sparse_pcd = load_colmap_model(sparse_dir=colmap_path, images_dir=images_path)
    all_views = list(views_dict.values())
    print(f"Loaded {len(all_views)} cameras, {len(sparse_pcd)} points.")
    
    # Check point coordinates and colors
    pts = sparse_pcd.positions
    cols = sparse_pcd.colors
    print("Point range X:", pts[:, 0].min(), pts[:, 0].max())
    print("Point range Y:", pts[:, 1].min(), pts[:, 1].max())
    print("Point range Z:", pts[:, 2].min(), pts[:, 2].max())
    print("Color range:", cols.min(), cols.max())
    print("Any pure black points in sparse cloud?", np.sum(np.all(cols == 0, axis=1)))
    print("Dark points (<20):", np.sum(np.all(cols < 20, axis=1)))

if __name__ == "__main__":
    analyze()
