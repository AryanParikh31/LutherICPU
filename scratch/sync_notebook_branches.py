import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Update pipeline_runner.ipynb
with open("pipeline_runner.ipynb", "r", encoding="utf-8") as f:
    nb1 = json.load(f)

# Update Cell 4 (pip install) to include scikit-image
cell_4_src = "".join(nb1["cells"][4]["source"])
if "scikit-image" not in cell_4_src:
    cell_4_src = cell_4_src.replace(
        "pip install -q plyfile tqdm scipy trimesh open3d opencv-python pillow psutil huggingface_hub",
        "pip install -q plyfile tqdm scipy trimesh open3d opencv-python pillow psutil huggingface_hub scikit-image"
    )
    nb1["cells"][4]["source"] = [l + "\n" for l in cell_4_src.split("\n")[:-1]] + [cell_4_src.split("\n")[-1]]

# Update Cell 6 (git clone / branch)
new_cell_6_lines = [
    "# Cell 3: Clone & Pull Model Code from GitHub\n",
    "import os, sys, shutil\n",
    "\n",
    "# Active branch configuration (explicitly checkout current active branch)\n",
    "BRANCH = \"LutherICPUv3\"  # @param {type:\"string\"}\n",
    "REPO_URL = \"https://github.com/AryanParikh31/LutherICPU.git\"\n",
    "TARGET_DIR = \"/content/LutherICPU\"\n",
    "\n",
    "if os.path.exists(TARGET_DIR):\n",
    "    print(f\"[*] Updating existing repository to branch '{BRANCH}'...\")\n",
    "    %cd /content/LutherICPU\n",
    "    !git fetch origin {BRANCH}\n",
    "    !git checkout -B {BRANCH} origin/{BRANCH}\n",
    "    !git pull origin {BRANCH}\n",
    "else:\n",
    "    print(f\"[*] Cloning branch '{BRANCH}' from {REPO_URL}...\")\n",
    "    !git clone --branch {BRANCH} --single-branch {REPO_URL} {TARGET_DIR}\n",
    "    %cd /content/LutherICPU\n",
    "\n",
    "if TARGET_DIR not in sys.path:\n",
    "    sys.path.insert(0, TARGET_DIR)\n",
    "\n",
    "# Verify module imports\n",
    "from luther_core.cuda_gaussian_trainer import CUDAGaussianTrainer, GaussianModel\n",
    "from luther_core.gaussian_splatting_engine import GaussianSplattingEngine\n",
    "from luther_core.ingestion import ImageQualityController\n",
    "print(f\"\\n[SUCCESS] LutherICPU model modules from branch '{BRANCH}' imported and validated successfully!\")\n"
]
nb1["cells"][6]["source"] = new_cell_6_lines

with open("pipeline_runner.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb1, f, indent=1)

print("[OK] pipeline_runner.ipynb updated.")

# Update LutherICPU_Colab_Trainer.ipynb
with open("LutherICPU_Colab_Trainer.ipynb", "r", encoding="utf-8") as f:
    nb2 = json.load(f)

new_cell_1_lines = [
    "# Step 1: Environment Setup & Verify NVIDIA T4 GPU Hardware\n",
    "!nvidia-smi\n",
    "!apt-get update -qq && apt-get install -y -qq colmap unzip wget ninja-build\n",
    "!pip install -q --upgrade pip\n",
    "!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121\n",
    "!pip install -q gsplat || pip install -q git+https://github.com/nerfstudio-project/gsplat.git || echo 'Using Native PyTorch Vectorized Differentiable CUDA Rasterizer'\n",
    "!pip install -q huggingface_hub plyfile tqdm scipy trimesh open3d onnxruntime opencv-python pillow scikit-image\n",
    "\n",
    "import torch\n",
    "print(\"=\" * 80)\n",
    "print(f\"[*] PyTorch Version:       {torch.__version__}\")\n",
    "print(f\"[*] CUDA Available:        {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"[*] GPU Device:            {torch.cuda.get_device_name(0)}\")\n",
    "    print(f\"[*] VRAM Total:            {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\")\n",
    "    print(f\"[*] Compute Capability:    {torch.cuda.get_device_capability(0)}\")\n",
    "print(\"=\" * 80)\n",
    "\n",
    "import os, shutil\n",
    "if os.path.exists(\"/content/LutherICPU\"):\n",
    "    shutil.rmtree(\"/content/LutherICPU\")\n",
    "\n",
    "# Active branch parameter (ensures clone from current branch rather than default main)\n",
    "BRANCH = \"LutherICPUv3\"  # @param {type:\"string\"}\n",
    "!git clone --branch {BRANCH} --single-branch https://github.com/AryanParikh31/LutherICPU.git /content/LutherICPU\n",
    "%cd /content/LutherICPU\n",
    "if os.path.exists(\"requirements.txt\"):\n",
    "    !pip install -q -r requirements.txt\n",
    "\n",
    "print(f\"\\n[SUCCESS] Native LutherICPU GPU environment initialized from branch '{BRANCH}' successfully!\")\n"
]
nb2["cells"][1]["source"] = new_cell_1_lines

with open("LutherICPU_Colab_Trainer.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=1)

print("[OK] LutherICPU_Colab_Trainer.ipynb updated.")
