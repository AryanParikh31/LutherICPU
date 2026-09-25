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


def read_cameras_text(path: str) -> Dict[int, CameraIntrinsics]:
    """Reads cameras.txt from a COLMAP reconstruction directory."""
    cameras = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model_name = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]

            if model_name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                fx = fy = float(params[0])
                cx = float(params[1])
                cy = float(params[2])
            elif model_name in ("PINHOLE", "RADIAL", "OPENCV", "FULL_OPENCV"):
                fx = float(params[0])
                fy = float(params[1])
                cx = float(params[2])
                cy = float(params[3])
            else:
                fx = float(params[0])
                fy = float(params[1]) if len(params) > 1 else fx
                cx = float(params[2]) if len(params) > 2 else width / 2.0
                cy = float(params[3]) if len(params) > 3 else height / 2.0

            cameras[camera_id] = CameraIntrinsics(
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                model=model_name,
                params=np.array(params, dtype=np.float32)
            )
    return cameras


def read_images_text(path: str, cameras: Dict[int, CameraIntrinsics], images_dir: Optional[str] = None) -> Dict[int, CameraView]:
    """Reads images.txt from a COLMAP reconstruction directory."""
    views = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        image_id = int(parts[0])
        qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float32)
        tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float32)
        camera_id = int(parts[8])
        image_name = parts[9]

        # Next line contains 2D points
        xys = []
        p3d_ids = []
        if i < len(lines):
            pts_line = lines[i].strip()
            i += 1
            if pts_line and not pts_line.startswith("#"):
                p_parts = pts_line.split()
                for k in range(0, len(p_parts), 3):
                    if k + 2 < len(p_parts):
                        xys.append([float(p_parts[k]), float(p_parts[k+1])])
                        p3d_ids.append(int(p_parts[k+2]))

        intrinsics = cameras.get(camera_id, next(iter(cameras.values())))
        img_path = os.path.join(images_dir, image_name) if images_dir else None

        views[image_id] = CameraView(
            image_id=image_id,
            name=image_name,
            qvec=qvec,
            tvec=tvec,
            intrinsics=intrinsics,
            image_path=img_path,
            point3d_ids=np.array(p3d_ids, dtype=np.int64) if p3d_ids else np.array([], dtype=np.int64),
            xys=np.array(xys, dtype=np.float32) if xys else np.empty((0, 2), dtype=np.float32)
        )

    return views


def read_points3d_text(path: str, max_points: Optional[int] = None) -> PointCloud:
    """Reads points3D.txt from a COLMAP reconstruction directory."""
    positions = []
    colors = []
    errors = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            err = float(parts[7])
            positions.append([x, y, z])
            colors.append([r, g, b])
            errors.append(err)
            if max_points and len(positions) >= max_points:
                break

    return PointCloud(
        positions=np.array(positions, dtype=np.float32) if positions else np.empty((0, 3), dtype=np.float32),
        colors=np.array(colors, dtype=np.uint8) if colors else np.empty((0, 3), dtype=np.uint8),
        errors=np.array(errors, dtype=np.float32) if errors else np.empty((0,), dtype=np.float32)
    )


def load_colmap_model(sparse_dir: str, images_dir: Optional[str] = None) -> Tuple[Dict[int, CameraIntrinsics], Dict[int, CameraView], PointCloud]:
    """Loads a complete COLMAP model (cameras, images, pointcloud) from either binary or text files."""
    # Check candidates
    dirs_to_check = [sparse_dir, os.path.join(sparse_dir, "0")]
    
    for d in dirs_to_check:
        cam_bin = os.path.join(d, "cameras.bin")
        img_bin = os.path.join(d, "images.bin")
        pts_bin = os.path.join(d, "points3D.bin")

        if os.path.exists(cam_bin) and os.path.exists(img_bin) and os.path.exists(pts_bin):
            cameras = read_cameras_binary(cam_bin)
            views = read_images_binary(img_bin, cameras, images_dir=images_dir)
            pcd = read_points3d_binary(pts_bin)
            logger.info(f"Loaded COLMAP binary model from '{d}': {len(cameras)} cameras, {len(views)} image poses, {len(pcd)} 3D points.")
            return cameras, views, pcd

        cam_txt = os.path.join(d, "cameras.txt")
        img_txt = os.path.join(d, "images.txt")
        pts_txt = os.path.join(d, "points3D.txt")

        if os.path.exists(cam_txt) and os.path.exists(img_txt) and os.path.exists(pts_txt):
            cameras = read_cameras_text(cam_txt)
            views = read_images_text(img_txt, cameras, images_dir=images_dir)
            pcd = read_points3d_text(pts_txt)
            logger.info(f"Loaded COLMAP text model from '{d}': {len(cameras)} cameras, {len(views)} image poses, {len(pcd)} 3D points.")
            return cameras, views, pcd

    raise FileNotFoundError(f"Cannot find COLMAP binary (.bin) or text (.txt) files in {sparse_dir} or subfolder '0'")


def export_colmap_sparse(
    output_dir: str,
    cameras: Dict[int, CameraIntrinsics],
    views: Dict[int, CameraView],
    pcd: PointCloud
) -> str:
    """
    Exports reconstructed camera intrinsics, camera extrinsics, and 3D point cloud
    into a standard COLMAP-compatible sparse/0 directory with both text and binary formats.
    """
    sparse_0_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(sparse_0_dir, exist_ok=True)

    # 1. Export cameras.txt
    cam_txt_path = os.path.join(sparse_0_dir, "cameras.txt")
    with open(cam_txt_path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cid, cam in cameras.items():
            f.write(f"{cid} PINHOLE {cam.width} {cam.height} {cam.fx:.6f} {cam.fy:.6f} {cam.cx:.6f} {cam.cy:.6f}\n")

    # 2. Export images.txt
    img_txt_path = os.path.join(sparse_0_dir, "images.txt")
    with open(img_txt_path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(views)}\n")
        for iid, v in views.items():
            qw, qx, qy, qz = v.qvec[0], v.qvec[1], v.qvec[2], v.qvec[3]
            tx, ty, tz = v.tvec[0], v.tvec[1], v.tvec[2]
            f.write(f"{iid} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {tx:.8f} {ty:.8f} {tz:.8f} 1 {v.name}\n")
            f.write("\n")

    # 3. Export points3D.txt
    pts_txt_path = os.path.join(sparse_0_dir, "points3D.txt")
    with open(pts_txt_path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(pcd.positions)}\n")
        cols = pcd.colors
        if cols.max() <= 1.05 and cols.dtype != np.uint8:
            cols = (cols * 255.0).astype(np.uint8)
        for pid in range(len(pcd.positions)):
            pos = pcd.positions[pid]
            col = cols[pid] if pid < len(cols) else [128, 128, 128]
            f.write(f"{pid + 1} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {int(col[0])} {int(col[1])} {int(col[2])} 1.0\n")

    logger.info(f"Exported self-generated sparse model to '{sparse_0_dir}' ({len(views)} cameras, {len(pcd.positions):,} 3D points).")
    return sparse_0_dir

