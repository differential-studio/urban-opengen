"""
sample.py

Inpaint a region of a real city with a trained checkpoint, at any physical scale, on a
canvas of any size, with any subset of metric targets.

    # 800 m canvas from the Berlin cache, centre of the raster, rebuild the middle third at FAR 2.5
    python sample.py --ckpt C:\\opengan\\inpaint\\run01\\latest.pt --cache C:\\opengan\\cache_v2 --city Berlin ^
        --extent 800 --far 2.5 --out C:\\opengan\\inpaint\\out\\berlin_far25

    # a raster PNG straight from osm_tiles (encoded on the fly with the checkpoint's LUT)
    python sample.py --ckpt ... --raster C:\\opengan\\images_v2\\Tokyo\\Tokyo_raster.png --center 1600 1600 ^
        --extent 1200 --hole 0.25 0.25 0.75 0.75 --green 0.2 --street 0.15

    # zoomed out: one 128 px window covers 800 m, canvas 1600 m -> 256 px canvas, coarse fabric
    python sample.py --ckpt ... --cache ... --city Jakarta --extent 1600 --tile-m 800 --coverage 0.35

Region and scale
    --center Y X     raster pixel of the canvas centre (default: raster centre)
    --extent M       canvas side in metres
    --tile-m M       metres covered by one 128 px window: 400 is native, 800 is half resolution.
                     Must lie inside the range the model was trained on (see the checkpoint's args).
    The canvas is extent / tile_m * 128 working pixels and must be at least 128.

Hole
    --hole x0 y0 x1 y1   fractions of the canvas, default the middle 0.3..0.7
    --hole-png PATH      white = hole, resized to the canvas
    --all                the whole canvas (pure generation, context ignored)
    --keep-streets       cut the existing street band out of the hole: rebuild plots, keep the network

Targets (any subset; the rest is inferred from context)
    --coverage 0..1  --green 0..1  --street 0..1  --height M  --far F  --bsize M

Outputs <out>/candidate_<k>.png (RGB in the tile palette), <out>/candidate_<k>.npy (uint8
[4,H,W] fields), <out>/context.png (hole shown in grey), <out>/metrics.json (targets and
the achieved metrics of every candidate, so a wrong target is visible, not hidden).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    from PIL import Image
except ImportError as e:  # the system Python has neither torch nor scipy
    raise SystemExit(f"{e}\nrun with the training venv: C:\\opengan\\venv\\Scripts\\python.exe sample.py ...")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase0"))
from data import (raw_to_fields, hole_metrics, targets_to_cond, CondNorm, byte_to_metres_table,  # noqa: E402
                  fields_to_bytes, METRICS, NATIVE_M_PER_PX)
from unet import UNet  # noqa: E402
from diffusion import Schedule, sample, sample_canvas  # noqa: E402
from tile_codec import HeightLUT, channels_to_rgb  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def load_checkpoint(path: str, device):
    ck = torch.load(path, map_location=device)
    model = UNet(**ck["unet"]).to(device).eval()
    model.load_state_dict(ck["ema"])
    lut = HeightLUT(forward=ck["lut"]["forward"], inverse=ck["lut"]["inverse"], counts=ck["lut"]["counts"])
    norm = CondNorm(ck["cond_norm"]["mean"], ck["cond_norm"]["std"])
    return model, lut, norm, ck


def load_source(args, lut, empty_height: int) -> tuple[np.ndarray, float]:
    """uint8 [4, H, W] cache-layout fields and metres per raster pixel."""
    if args.cache and args.city:
        cache = Path(args.cache)
        meta = json.loads((cache / "cache.json").read_text())
        arr = np.load(cache / f"{args.city}.npy", mmap_mode="r")
        return arr, float(meta.get("res_m_per_px") or NATIVE_M_PER_PX)
    if args.raster or args.tile:
        from prepare import encode_raster
        rgb = np.asarray(Image.open(args.raster or args.tile).convert("RGB"))
        res = args.res_m or NATIVE_M_PER_PX
        return encode_raster(rgb, lut, empty_height), res
    sys.exit("give --cache and --city, or --raster, or --tile")


def crop_source(arr: np.ndarray, cy: int, cx: int, side: int) -> np.ndarray:
    """uint8 [4, side, side] centred on (cy, cx), padded with 'empty' where it runs off the raster."""
    _, H, W = arr.shape
    y0, x0 = cy - side // 2, cx - side // 2
    out = np.empty((4, side, side), np.uint8)
    out[[0, 2, 3]] = np.int8(-127).view(np.uint8)
    out[1] = 128
    ys0, xs0 = max(0, y0), max(0, x0)
    ys1, xs1 = min(H, y0 + side), min(W, x0 + side)
    if ys1 > ys0 and xs1 > xs0:
        out[:, ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = arr[:, ys0:ys1, xs0:xs1]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache"); ap.add_argument("--city")
    ap.add_argument("--raster"); ap.add_argument("--tile")
    ap.add_argument("--res-m", type=float, help="metres per pixel of --raster/--tile, default 3.125")
    ap.add_argument("--center", type=int, nargs=2, metavar=("Y", "X"))
    ap.add_argument("--extent", type=float, default=400.0, help="canvas side in metres")
    ap.add_argument("--tile-m", type=float, default=400.0, help="metres per 128 px window")
    ap.add_argument("--hole", type=float, nargs=4, default=[0.3, 0.3, 0.7, 0.7])
    ap.add_argument("--hole-png")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--keep-streets", action="store_true")
    for k in METRICS:
        ap.add_argument(f"--{k.replace('_m', '')}", type=float, dest=k)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--resample", type=int, default=0)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    model, lut, norm, ck = load_checkpoint(args.ckpt, dev)
    size = int(ck.get("size", 128))
    sched = Schedule(device=dev)
    byte_to_m = byte_to_metres_table(lut)
    lo, hi = ck["args"].get("scale_lo", 1.0), ck["args"].get("scale_hi", 1.0)

    arr, res_m = load_source(args, lut, int(ck.get("empty_height_byte", 128)))
    scale = args.tile_m / (size * res_m)
    if not (lo * 0.99 <= scale <= hi * 1.01):
        print(f"warning: tile-m {args.tile_m:.0f} is scale {scale:.2f}, model was trained for {lo}..{hi} "
              f"({lo * size * res_m:.0f}..{hi * size * res_m:.0f} m); results outside that are untested")
    canvas_px = int(round(args.extent / args.tile_m * size))
    canvas_px = max(size, (canvas_px + 7) // 8 * 8)
    side = int(round(canvas_px * scale))                       # raster pixels covered by the canvas
    cy, cx = args.center if args.center else (arr.shape[1] // 2, arr.shape[2] // 2)
    raw = torch.from_numpy(crop_source(arr, cy, cx, side)).unsqueeze(0).to(dev)
    x0 = raw_to_fields(raw, canvas_px)                          # [1,4,C,C]

    # mask
    if args.all:
        mask = torch.ones(1, 1, canvas_px, canvas_px, device=dev)
    elif args.hole_png:
        m = Image.open(args.hole_png).convert("L").resize((canvas_px, canvas_px), Image.NEAREST)
        mask = (torch.from_numpy(np.asarray(m)).float() / 255.0 > 0.5).float().view(1, 1, canvas_px, canvas_px).to(dev)
    else:
        x0f, y0f, x1f, y1f = args.hole
        mask = torch.zeros(1, 1, canvas_px, canvas_px, device=dev)
        mask[:, :, int(y0f * canvas_px):int(y1f * canvas_px), int(x0f * canvas_px):int(x1f * canvas_px)] = 1
    if args.keep_streets:
        mask = mask * (x0[:, 2:3] <= -1.5 / 8.0).float()

    targets = {k: getattr(args, k) for k in METRICS if getattr(args, k) is not None}
    cond = targets_to_cond(targets, norm, dev) if targets else None
    l2s = torch.full((1,), math.log2(scale), device=dev)
    ctx = x0 * (1 - mask)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ctx_rgb = channels_to_rgb(fields_to_bytes(x0)[0], lut)
    ctx_rgb[mask[0, 0].cpu().numpy() > 0.5] = (90, 90, 110)
    Image.fromarray(ctx_rgb).save(out / "context.png")
    print(f"canvas {canvas_px} px = {args.extent:.0f} m at {args.tile_m:.0f} m per window (scale {scale:.2f}), "
          f"hole {100 * float(mask.mean()):.1f}% of it, targets {targets or 'none'}")

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    results = []
    for k in range(args.n):
        if canvas_px == size:
            y = sample(model, sched, ctx, mask, cond, l2s, steps=args.steps, cfg=args.cfg,
                       resample=args.resample, generator=gen)
        else:
            y = sample_canvas(model, sched, ctx, mask, cond, l2s, steps=args.steps, cfg=args.cfg,
                              stride=args.stride, resample=args.resample, generator=gen,
                              progress=lambda i, n: print(f"  candidate {k} step {i}/{n}", end="\r"))
        got = hole_metrics(y, mask, scale, byte_to_m, res_m)[0]
        achieved = {m: round(float(v), 3) for m, v in zip(METRICS, got)}
        b = fields_to_bytes(y)[0]
        Image.fromarray(channels_to_rgb(b, lut)).save(out / f"candidate_{k}.png")
        np.save(out / f"candidate_{k}.npy", b)
        results.append({"candidate": k, "achieved": achieved})
        print(f"  candidate {k}: " + ", ".join(f"{m} {v}" for m, v in achieved.items()) + " " * 20)
    truth = {m: round(float(v), 3) for m, v in zip(METRICS, hole_metrics(x0, mask, scale, byte_to_m, res_m)[0])}
    (out / "metrics.json").write_text(json.dumps({
        "targets": targets, "truth_in_hole": truth, "candidates": results,
        "extent_m": args.extent, "tile_m": args.tile_m, "scale": scale, "canvas_px": canvas_px,
        "center": [cy, cx], "steps": args.steps, "cfg": args.cfg, "resample": args.resample}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
