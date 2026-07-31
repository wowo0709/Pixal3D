#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r "$(dirname "$0")/requirements.txt"
python -m pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless pandas open3d objaverse 'huggingface_hub[cli]' open_clip_torch
