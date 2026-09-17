# Training the inpainter on a rented GPU

The laptop's 4070 (8 GB, ~1.5 it/s with checkpointing) turns a 200k-step run into days. A
rented RTX 4090 (24 GB) runs batch 32 without checkpointing at roughly 5 it/s, so the same
run is about 11 hours and under ten dollars. Everything below is RunPod, but any provider
with an Ubuntu image, CUDA torch and ssh works the same way.

## 1. On the laptop: two zips

The second command is run from the repository root.

Rasters only, not the cache. The cache is 41 MB per city; the raster PNG is about a
megabyte, and the box rebuilds the cache in half an hour.

    Get-ChildItem C:\opengan\images_v2 -Recurse -Filter *_raster.png | Compress-Archive -DestinationPath C:\opengan\rasters.zip -Force

    Compress-Archive -Path .\phase0, .\phase6 -DestinationPath C:\opengan\opengen_code.zip -Force

The grid jsons are not needed; without them the cache assumes 3.125 m per pixel, which is
what osm_tiles writes at 128 px.

## 2. On RunPod

Deploy a pod: GPU RTX 4090, template "RunPod PyTorch" (any recent CUDA torch), and attach
a **network volume** of 30 GB mounted at `/workspace` so checkpoints survive the pod being
stopped or pre-empted. Enable ssh in the pod settings; the pod page shows the ssh command.

Upload from PowerShell (Windows has scp built in; the port and host come from the pod page):

    scp -P <port> C:\opengan\rasters.zip C:\opengan\opengen_code.zip .\phase6\cloud_setup.sh root@<host>:/workspace/

Then on the box:

    cd /workspace
    bash cloud_setup.sh

It installs the three Python packages the code needs beyond torch, unpacks both zips,
builds the cache, and prints the train command.

## 3. Train

    tmux new -s train
    python gan_project/phase6/train.py --cache cache --out runs/run_cloud --batch 32 --no-grad-ckpt --steps 200000

Detach with Ctrl-b then d; the run keeps going. `tail -f runs/run_cloud/log.txt` to watch.
The first log line prints it/s and VRAM; on a 24 GB card `--batch 48` is worth trying if
VRAM is under 12 GB at 32.

Warm-starting from the laptop run saves a few hours: upload `C:\opengan\inpaint\run01\latest.pt`
and add `--init /workspace/latest.pt`. Fresh optimiser, same weights.

## 4. Bring the model home

    scp -P <port> root@<host>:/workspace/runs/run_cloud/latest.pt C:\opengan\inpaint\run_cloud\latest.pt

The checkpoint carries its own height LUT and condition statistics, so `sample.py` and the
viewer need nothing else from the box. Stop the pod when done; the volume keeps the run
and costs cents per month.

## Cost

At RunPod's list prices (September 2026): 4090 $0.74/h, 5090 $0.99/h, A100 80 GB $1.59/h,
H100 $2.89 to 3.49/h, storage $0.07/GB/month. The 4090 is the best value for this model,
which is bandwidth-bound rather than compute-bound. A 200k-step run at batch 32 is about
11 hours on it, so $8 to 10; the 2x model or the 256 px variant two to four times that.
Interruptible ("spot") pods are cheaper and fine, since the run checkpoints every 10k steps
and `--resume` continues from the last one.
