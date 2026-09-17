#!/usr/bin/env bash
# One-shot setup on a rented GPU box (RunPod "PyTorch" template or any Ubuntu image with CUDA torch).
# Expects, in the current directory: opengen_code.zip (phase0 + phase6) and rasters.zip (the *_raster.png files).
#
#   bash cloud_setup.sh
#
# Leaves: ./gan_project/{phase0,phase6}, ./rasters/, ./cache/ (built), and prints the train command.
set -euo pipefail

echo "== packages"
pip install -q numpy pillow scipy 2>&1 | tail -1 || true
python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
command -v unzip >/dev/null || (apt-get update -qq && apt-get install -y -qq unzip)

echo "== code"
mkdir -p gan_project && unzip -q -o opengen_code.zip -d gan_project
ls gan_project

echo "== rasters"
mkdir -p rasters && unzip -q -o rasters.zip -d rasters
echo "$(ls rasters | grep -c _raster.png) rasters"

echo "== cache (about ten seconds per city)"
python gan_project/phase6/prepare.py --images rasters --out cache

cat <<'TXT'

== ready. train with (inside tmux, so the run survives a dropped ssh):

  tmux new -s train
  python gan_project/phase6/train.py --cache cache --out runs/run_cloud --batch 32 --no-grad-ckpt --steps 200000

  detach: Ctrl-b then d      reattach: tmux attach -t train      log: tail -f runs/run_cloud/log.txt
TXT
