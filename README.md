# LutherICPU 🚀
> **CPU-Native 3D Reconstruction & Photorealistic Simulation Model**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pure CPU](https://img.shields.io/badge/Hardware-Pure%20CPU%20(No%20CUDA%20Required)-green.svg)]()

**LutherICPU** is a high-performance, pure CPU-accelerated pipeline for photorealistic 3D scene reconstruction and simulation from multi-angle photography. Engineered with vectorized NumPy/SciPy kernels, LutherICPU delivers dense multi-view stereo, volumetric TSDF manifold reconstruction, and image-based radiance field rendering without requiring dedicated GPU or CUDA hardware.

---

## 🌟 Key Features

- ⚡ **Pure CPU Acceleration**: Fully vectorized SIMD-friendly computations in NumPy and SciPy.
- 📸 **End-to-End Reconstruction Pipeline**:
  - **Stage 1 (Ingestion & Forensics)**: SHA-256 data integrity hashing, image validation, Laplacian $\text{Var}(\nabla^2 I)$ blur detection, and camera EXIF extraction.
  - **Stage 2 (Structure-from-Motion)**: COLMAP parsing, DoG SIFT keypoint matching, Epipolar geometry (Essential Matrix & 8-point RANSAC), and DLT multi-view triangulation.
  - **Stage 3 (Dense Multi-View Stereo)**: Multi-scale PatchMatch NCC stereo estimation and spatial dense point cloud fusion.
  - **Stage 4 (Volumetric Manifold Reconstruction)**: Truncated Signed Distance Function (TSDF) voxel grids, Marching Cubes, Screened Poisson, and Taubin surface smoothing.
  - **Stage 5 (Radiance & Photorealistic Rendering)**: Continuous Image-Based Radiance (SIBR) rendering, projective homography warping, and adaptive view blending.
- 🖥️ **Interactive Web Studio & Viewer**: Built with FastAPI and a modern dark-mode browser frontend for model inspection, live rendering, and job monitoring.
- 📊 **Comprehensive Benchmarking & QC**: Automated quality control manifests, PSNR/SSIM evaluation, and test suites.

---

## 🏗️ Architecture

```
LutherICPU
├── luther/                  # Orchestrator & core pipeline stages
│   ├── ingestion/           # File ingestion & QC
│   ├── sfm/                 # Structure-from-Motion & keypoint matching
│   ├── geometry/            # PatchMatch MVS & dense fusion
│   ├── volumetric/          # TSDF & Poisson mesh reconstruction
│   └── texturing/           # Radiance texturing & IBR rendering
├── luther_core/             # Math primitives & vectorized spatial indexing
├── luther_renderer/         # CPU IBR rasterizer & projective warp engines
├── luther_forensics/        # Image quality analysis & blur assessment
├── luther_pipeline/         # High-level simulation model & iterative engine
├── luther_web/              # FastAPI server & interactive web GUI
├── benchmark/               # Scene benchmark runners & metrics
└── tests/                   # Pytest automated test suite
```

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AryanParikh31/LutherICPU.git
cd LutherICPU
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 🚀 Quick Start & Usage

### 1. Run 3D Reconstruction CLI
Run reconstruction from a directory of source photos:
```bash
# Full simulation pipeline
python run_luther.py --images "path/to/photos" --colmap "path/to/colmap/sparse/0" --iterations 30000

# Short syntax
python run_luther.py -i "path/to/photos" -n 30k
```

### 2. Launch Web GUI & Interactive Studio
Start the local FastAPI server and browser studio:
```bash
python run.py
```
Open your browser at `http://127.0.0.1:8000` to interactively upload photos, monitor 3D reconstruction progress, and view rendered angles.

### 3. Run Benchmark Scenes
```bash
python run_luther.py --scene truck
```

### 4. Run Automated Test Suite
```bash
pytest tests/ -v
# or via CLI launcher:
python run_luther.py --test
```

---

## 📋 Requirements

- **Python**: 3.10+
- **Core Libraries**:
  - `numpy >= 1.24.0`
  - `scipy >= 1.10.0`
  - `pillow >= 9.5.0`
  - `fastapi >= 0.100.0`
  - `uvicorn >= 0.22.0`
  - `psutil >= 5.9.0`
  - `pytest >= 7.3.0`

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
