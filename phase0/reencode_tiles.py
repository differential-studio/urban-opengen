"""
reencode_tiles.py

Phase 0b: build the 4-channel training dataset from the raw RGB tiles.

Two passes over the images:
  pass 1  histogram of building grey values across the whole corpus
          -> corpus-wide percentile height LUT (height_lut.json)
  pass 2  decode every usable tile to [footprint, height, street, green],
          optionally resize, store everything in one .npz, plus metrics

Usage:
    python reencode_tiles.py --images images --out dataset
    python reencode_tiles.py --images images --out dataset --size 128
    python reencode_tiles.py --images images --out dataset --size 128 --min-content-pct 5
    python reencode_tiles.py --images images --out dataset --size 128 --encoding sdf

Encodings (see tile_codec.py): "mask" stores hard 0/255 masks, "sdf" stores
signed distance fields with the edge at byte 128 and a cliff-free height
channel. Both decode with the same rule (byte >= 128), so everything
downstream (training previews, metrics, the viewer, fit_directions) is
encoding-agnostic. Use "sdf" for GAN training.

Outputs in --out:
    tiles.npz           x: uint8 [N, 4, S, S]   city: int16 [N]   idx: int16 [N]
    meta.csv            one row per kept tile with city, idx, grid row/col and metrics
    cities.json         city name <-> integer label
    height_lut.json     forward/inverse height mapping (needed to decode generated tiles)
    encoding.json       which encoding was used and with what parameters
    preview.png         a few tiles: original | 4 channels | round-trip decode
    stats.txt           what was kept and dropped

To read it back:
    d = np.load("dataset/tiles.npz")
    x = d["x"]            # [N, 4, S, S] uint8
    x = x.astype(np.float32) / 127.5 - 1.0   # [-1, 1] for a GAN
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from tile_codec import (
    HeightLUT, CHANNEL_NAMES, ENCODINGS, SDF_TRUNC_PX, HEIGHT_SIGMA_PX,
    load_rgb, split_classes, encode_tile, channels_to_rgb, tile_metrics,
)

TILE_RE = re.compile(r"^(?P<city>.+)_(?P<idx>\d+)\.png$", re.IGNORECASE)


def list_tiles(images: Path):
    for cd in sorted(p for p in images.iterdir() if p.is_dir()):
        for p in sorted(cd.glob("*.png")):
            m = TILE_RE.match(p.name)
            if m:
                yield cd.name, int(m.group("idx")), p


def make_preview(out: Path, samples: list[tuple[np.ndarray, np.ndarray]], lut: HeightLUT):
    """samples: list of (original_rgb, channels)"""
    if not samples:
        return
    S = 160
    cols = 1 + len(CHANNEL_NAMES) + 1
    gap = 4
    W = cols * (S + gap) + gap
    H = len(samples) * (S + gap) + gap + 18
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas)
    heads = ["original"] + list(CHANNEL_NAMES) + ["round-trip"]
    for c, t in enumerate(heads):
        d.text((gap + c * (S + gap) + 4, 3), t, fill=(0, 0, 0))
    for r, (rgb, ch) in enumerate(samples):
        y = 18 + gap + r * (S + gap)
        tiles = [rgb] + [np.stack([ch[i]] * 3, axis=-1) for i in range(4)] + [channels_to_rgb(ch, lut)]
        for c, t in enumerate(tiles):
            im = Image.fromarray(t).resize((S, S), Image.NEAREST)
            canvas.paste(im, (gap + c * (S + gap), y))
    canvas.save(out / "preview.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--size", type=int, default=0,
                    help="resize tiles to this square size (0 = keep native); nearest for mask, bilinear for sdf")
    ap.add_argument("--encoding", choices=ENCODINGS, default="mask",
                    help="mask = hard 0/255 masks; sdf = signed distance fields (recommended for training)")
    ap.add_argument("--sdf-range", type=float, default=SDF_TRUNC_PX,
                    help="sdf only: distance in native pixels at which the field saturates")
    ap.add_argument("--height-sigma", type=float, default=HEIGHT_SIGMA_PX,
                    help="sdf only: blur (native px) of the height fill outside footprints")
    ap.add_argument("--min-content-pct", type=float, default=5.0,
                    help="drop tiles with less than this percent non-background content")
    ap.add_argument("--grid", type=int, default=25)
    ap.add_argument("--tile-m", type=float, default=400.0, help="tile edge in metres, for the metric columns")
    ap.add_argument("--preview-n", type=int, default=6)
    args = ap.parse_args()

    images = Path(args.images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tiles = list(list_tiles(images))
    if not tiles:
        sys.exit(f"no tiles found under {images}")
    print(f"found {len(tiles)} tiles in {len({c for c, _, _ in tiles})} cities")

    # ---- pass 1: grey histogram + keep list -------------------------------
    grey_counts = np.zeros(256, dtype=np.int64)
    keep = []
    dropped = 0
    unreadable = []
    for i, (city, idx, p) in enumerate(tiles):
        try:
            rgb = load_rgb(p)
        except Exception as e:  # truncated or corrupt PNG
            unreadable.append(p)
            print(f"  skipping unreadable {p.name}: {e}", file=sys.stderr)
            continue
        footprint, grey, street, green, _ = split_classes(rgb)
        content_pct = 100.0 * (footprint | street | green).mean()
        if content_pct < args.min_content_pct:
            dropped += 1
            continue
        gv = grey[footprint]
        if gv.size:
            grey_counts += np.bincount(gv, minlength=256)
        keep.append((city, idx, p))
        if (i + 1) % 2000 == 0:
            print(f"  pass 1: {i + 1}/{len(tiles)}")

    lut = HeightLUT.from_counts(grey_counts)
    lut.save(out / "height_lut.json")
    # height byte for tiles that have no buildings at all (sdf encoding only)
    fwd = np.asarray(lut.forward, dtype=np.float64)
    empty_height = int(round((fwd * grey_counts).sum() / max(grey_counts.sum(), 1)))
    present = np.nonzero(grey_counts)[0]
    print(f"pass 1 done: keeping {len(keep)}, dropped {dropped} "
          f"(< {args.min_content_pct:g}% content). "
          f"grey levels present: {present.size}, range {present.min()}..{present.max()}")

    # ---- pass 2: encode --------------------------------------------------
    city_names = sorted({c for c, _, _ in keep})
    city_to_id = {c: i for i, c in enumerate(city_names)}
    (out / "cities.json").write_text(json.dumps({"names": city_names, "ids": city_to_id}, indent=1))

    first = load_rgb(keep[0][2])
    native = first.shape[0]
    S = args.size or native
    N = len(keep)
    x = np.zeros((N, 4, S, S), dtype=np.uint8)
    city_arr = np.zeros(N, dtype=np.int16)
    idx_arr = np.zeros(N, dtype=np.int16)
    meta_rows = []
    preview_samples = []
    preview_every = max(1, N // max(args.preview_n, 1))

    for n, (city, idx, p) in enumerate(keep):
        rgb = load_rgb(p)
        ch = encode_tile(rgb, lut, args.encoding, S, args.sdf_range, args.height_sigma, empty_height)
        x[n] = ch
        city_arr[n] = city_to_id[city]
        idx_arr[n] = idx
        m = tile_metrics(ch, lut, args.tile_m)
        r, c = divmod(idx, args.grid)
        meta_rows.append({"n": n, "city": city, "city_id": city_to_id[city], "idx": idx,
                          "row": r, "col": c, "file": p.name,
                          **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
        if n % preview_every == 0 and len(preview_samples) < args.preview_n:
            preview_samples.append((rgb, ch))
        if (n + 1) % 2000 == 0:
            print(f"  pass 2: {n + 1}/{N}")

    np.savez_compressed(out / "tiles.npz", x=x, city=city_arr, idx=idx_arr)
    (out / "encoding.json").write_text(json.dumps({
        "encoding": args.encoding, "native": native, "size": S,
        "sdf_range_px": args.sdf_range if args.encoding == "sdf" else None,
        "height_sigma_px": args.height_sigma if args.encoding == "sdf" else None,
        "empty_tile_height_byte": empty_height if args.encoding == "sdf" else None,
        "decode": "mask = byte >= 128; height read where footprint",
    }, indent=1))
    with (out / "meta.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        w.writeheader()
        w.writerows(meta_rows)
    make_preview(out, preview_samples, lut)

    # ---- round-trip check on a few native-size tiles ----------------------
    rt_ok = True
    if S == native:
        for city, idx, p in keep[:: max(1, N // 50)][:50]:
            rgb = load_rgb(p)
            back = channels_to_rgb(encode_tile(rgb, lut, args.encoding, 0, args.sdf_range, args.height_sigma, empty_height), lut)
            _, _, _, _, off = split_classes(rgb)
            if off.any():
                continue  # off-palette pixels cannot round-trip by design
            if not np.array_equal(rgb, back):
                rt_ok = False
                print(f"round-trip mismatch on {p.name}", file=sys.stderr)
                break

    stats = [
        f"tiles found:        {len(tiles)}",
        f"tiles kept:         {N}",
        f"tiles dropped:      {dropped}  (< {args.min_content_pct:g}% content)",
        f"tiles unreadable:   {len(unreadable)}" + (
            "  -> " + ", ".join(f"{p.parent.name}/{p.name}" for p in unreadable[:20])
            + (" ..." if len(unreadable) > 20 else "") if unreadable else ""),
        f"cities:             {len(city_names)}",
        f"native size:        {native}",
        f"output size:        {S}",
        f"encoding:           {args.encoding}" + (
            f"  (sdf range {args.sdf_range:g} px, height blur {args.height_sigma:g} px, "
            f"empty-tile height byte {empty_height})"
            if args.encoding == "sdf" else ""),
        f"channels:           {', '.join(CHANNEL_NAMES)}",
        f"grey levels used:   {present.size}  ({present.min()}..{present.max()})",
        f"height byte range:  {min(v for v in lut.forward if v)}..{max(lut.forward)}",
        f"lossless round-trip on palette tiles: {rt_ok if S == native else 'n/a (resized)'}",
        f"npz size:           {(out / 'tiles.npz').stat().st_size / 1e6:.1f} MB",
    ]
    (out / "stats.txt").write_text("\n".join(stats) + "\n", encoding="utf-8")
    print()
    print("\n".join(stats))
    print(f"\nwritten to {out.resolve()}")


if __name__ == "__main__":
    main()
