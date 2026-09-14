"""Unit tests for lutherICPU CPU Software Rasterizer."""
import numpy as np
from PIL import Image

from luther_core.types import SurfaceMesh, CameraView, CameraIntrinsics
from luther_renderer.cpu_rasterizer import CpuSoftwareRasterizer


def test_cpu_rasterizer_renders_triangle():
    """Verifies that CPU software rasterizer renders lit geometry into an image buffer."""
    # A single front-facing equilateral triangle in front of camera
    verts = np.array([
        [-1.0, -1.0, 3.0],
        [1.0, -1.0, 3.0],
        [0.0, 1.0, 3.0]
    ], dtype=np.float32)

    faces = np.array([[0, 1, 2]], dtype=np.int32)
    normals = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    mesh = SurfaceMesh(vertices=verts, faces=faces, normals=normals, vertex_colors=colors)

    intrinsics = CameraIntrinsics(width=400, height=300, fx=300.0, fy=300.0, cx=200.0, cy=150.0)
    view = CameraView(
        image_id=1,
        name="test_view",
        qvec=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        tvec=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        intrinsics=intrinsics
    )

    rasterizer = CpuSoftwareRasterizer(num_threads=2)
    img = rasterizer.render_view(mesh, view, width=400, height=300, bg_color=(0, 0, 0))

    assert isinstance(img, Image.Image)
    assert img.size == (400, 300)

    # Check that center pixel is lit (not background black)
    arr = np.array(img)
    center_color = arr[150, 200]
    assert np.any(center_color > 0)
