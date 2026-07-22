#!/usr/bin/env bash
# Environment setup for the AnnoMI / MPSE pipeline on AutoDL.
# Base image assumed: PyTorch 2.1.0 / Python 3.10 / CUDA 12.1 (Ubuntu 22.04).
# The image already ships torch/torchvision/torchaudio matched to CUDA 12.1 — do NOT reinstall torch.
#
# Usage (on the server, GPU-less "无卡模式" is fine for install):
#   bash scripts/setup_server.sh
#
# Keeps big caches on the data disk (系统盘 is only 30GB).
set -e

# --- keep model/HF caches off the tiny system disk ---
export HF_HOME=/root/autodl-tmp/hf
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$HF_HOME"
grep -q 'HF_HOME' ~/.bashrc || cat >> ~/.bashrc <<'EOF'

# MPSE project: caches on data disk
export HF_HOME=/root/autodl-tmp/hf
export TRANSFORMERS_CACHE=$HF_HOME/transformers
EOF

# --- system deps ---
apt-get update -y && apt-get install -y ffmpeg

# --- python deps (torch is already in the image; pin the rest conservatively) ---
pip install --upgrade pip
pip install \
  "transformers==4.44.2" "accelerate==0.34.2" "peft==0.13.2" \
  "faster-whisper==1.0.3" \
  "mediapipe==0.10.14" \
  "librosa==0.10.2" "soundfile==0.12.1" \
  "opencv-python-headless==4.10.0.84" \
  "yt-dlp" \
  "numpy<2" "pandas" "pyyaml" "tqdm" "scipy"

# --- install this project (editable) so `import mpse_mvp` works ---
cd "$(dirname "$0")/.."
pip install -e .

echo
echo "=== versions ==="
python - <<'PY'
import torch, transformers, peft
print("torch       ", torch.__version__, "| cuda avail:", torch.cuda.is_available(),
      "| device:", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
print("transformers", transformers.__version__)
print("peft        ", peft.__version__)
for m in ("mediapipe", "librosa", "cv2", "faster_whisper", "yt_dlp", "scipy"):
    try:
        __import__(m); print(f"{m:14s} OK")
    except Exception as e:
        print(f"{m:14s} FAIL: {e}")
PY
echo
echo "setup done. Run: bash scripts/smoke_test.sh"
