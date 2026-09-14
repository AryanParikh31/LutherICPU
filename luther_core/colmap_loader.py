"""lutherICPU Fast COLMAP Binary and Text Model Loader.

Parses cameras.bin, images.bin, and points3D.bin with high performance and memory safety.
"""
import os
import struct
import logging
import numpy as np
from typing import Dict, Tuple, List, Optional
from luther_core.types import CameraIntrinsics, CameraView, PointCloud

logger = logging.getLogger("lutherICPU.ColmapLoader")

# COLMAP camera model IDs
CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE"
}


def read_cameras_binary(path: str) -> Dict[int, CameraIntrinsics]:
    """Reads cameras.bin from a COLMAP reconstruction directory."""
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", fid.read(24))
            model_name = CAMERA_MODEL_NAMES.get(model_id, "PINHOLE")

            # Determine number of params
            if model_id in (0, 2):  # SIMPLE_PINHOLE, SIMPLE_RADIAL: f, cx, cy, (k)
                num_params = 3 if model_id == 0 else 4
            elif model_id in (1, 3):  # PINHOLE, RADIAL: fx, fy, cx, cy, (k1, k2)
                num_params = 4 if model_id == 1 else 6
            elif model_id == 4:  # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
                num_params = 8
            else:
                num_params = 4

            params = struct.unpack(f"<{num_params}d", fid.read(8 * num_params))

            if model_id in (0, 2):
                fx = fy = float(params[0])
                cx = float(params[1])
                cy = float(params[2])
            else:
                fx = float(params[0])
                fy = float(params[1])
                cx = float(params[2])
                cy = float(params[3])

            cameras[camera_id] = CameraIntrinsics(
                width=int(width),
                height=int(height),
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                model=model_name,
                params=np.array(params, dtype=np.float32)
            )
    return cameras


def read_images_binary(path: str, cameras: Dict[int, CameraIntrinsics], images_dir: Optional[str] = None) -> Dict[int, CameraView]:
    """Reads images.bin from a COLMAP reconstruction directory."""
    views = {}
    with open(path, "rb") as fid:
        num_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", fid.read(4))[0]
            qvec = np.array(struct.unpack("<4d", fid.read(32)), dtype=np.float32)
            tvec = np.array(struct.unpack("<3d", fid.read(24)), dtype=np.float32)
            camera_id = struct.unpack("<i", fid.read(4))[0]

            name_chars = []
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                name_chars.append(char.decode("utf-8", errors="ignore"))
            image_name = "".join(name_chars)

            num_points2d = struct.unpack("<Q", fid.read(8))[0]
            points2d_data = fid.read(24 * num_points2d)  # 2 double (xy) + 1 int64 (point3d_id)

            if num_points2d > 0:
                # Parse 2D point observations: each is (x: float64, y: float64, point3d_id: int64)
                raw_arr = np.frombuffer(points2d_data, dtype=[('x', '<f8'), ('y', '<f8'), ('id', '<i8')])
                xys = np.column_stack([raw_arr['x'], raw_arr['y']]).astype(np.float32)
                p3d_ids = raw_arr['id'].astype(np.int64)
            else:
                xys = np.empty((0, 2), dtype=np.float32)
                p3d_ids = np.array([], dtype=np.int64)

            intrinsics = cameras.get(camera_id, next(iter(cameras.values())))
            img_path = os.path.join(images_dir, image_name) if images_dir else None

            views[image_id] = CameraView(
                image_id=image_id,
                name=image_name,
                qvec=qvec,
                tvec=tvec,
                intrinsics=intrinsics,
                image_path=img_path,
                point3d_ids=p3d_ids,
                xys=xys
            )
    return views


def read_points3d_binary(path: str, max_points: Optional[int] = None) -> PointCloud:
    """Reads points3D.bin from a COLMAP reconstruction directory."""
    with open(path, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        read_count = min(num_points, max_points) if max_points else num_points

        positions = np.empty((read_count, 3), dtype=np.float32)
        colors = np.empty((read_count, 3), dtype=np.uint8)
        errors = np.empty((read_count,), dtype=np.float32)

        for i in range(read_count):
            _p_id = struct.unpack("<Q", fid.read(8))[0]
            xyz = struct.unpack("<3d", fid.read(24))
            rgb = struct.unpack("<3B", fid.read(3))
            error = struct.unpack("<d", fid.read(8))[0]
            track_len = struct.unpack("<Q", fid.read(8))[0]
            fid.seek(8 * track_len, os.SEEK_CUR)  # skip tracks for speed

            positions[i] = xyz
            colors[i] = rgb
            errors[i] = error

    return PointCloud(
        positions=positions,
        colors=colors,
        errors=errors
    )


def load_colmap_model(sparse_dir: str, images_dir: Optional[str] = None) -> Tuple[Dict[int, CameraIntrinsics], Dict[int, CameraView], PointCloud]:
    """Loads a complete COLMAP model (cameras, images, pointcloud) from binary files."""
    cam_bin = os.path.join(sparse_dir, "cameras.bin")
    img_bin = os.path.join(sparse_dir, "images.bin")
    pts_bin = os.path.join(sparse_dir, "points3D.bin")

    if not (os.path.exists(cam_bin) and os.path.exists(img_bin) and os.path.exists(pts_bin)):
        # Check subfolder '0' if present
        sub_dir = os.path.join(sparse_dir, "0")
        if os.path.exists(os.path.join(sub_dir, "cameras.bin")):
            cam_bin = os.path.join(sub_dir, "cameras.bin")
            img_bin = os.path.join(sub_dir, "images.bin")
            pts_bin = os.path.join(sub_dir, "points3D.bin")
        else:
            raise FileNotFoundError(f"Cannot find COLMAP binary files in {sparse_dir} or {sub_dir}")

    cameras = read_cameras_binary(cam_bin)
    views = read_images_binary(img_bin, cameras, images_dir=images_dir)
    pcd = read_points3d_binary(pts_bin)

    logger.info(f"Loaded COLMAP model: {len(cameras)} cameras, {len(views)} image poses, {len(pcd)} 3D points.")
    return cameras, views, pcd
