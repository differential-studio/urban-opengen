"""
prepare.py

Turn the city rasters from osm_tiles.py (one RGB PNG per city, 3200 x 3200 px at
3.125 m per pixel for a 10 km window) into the four-channel field cache the inpainter
trains on. One file per city, memory-mapped at training time, so a random crop at any
scale is a slice plus a GPU resize and the training loop never runs a distance transform.

    python prepare.py --images C:\\opengan\\images_v2 --out C:\\opengan\\cache_v2
    python prepare.py --images C:\\opengan\\images_v2 --out C:\\opengan\\cache_v2 --only Berlin,Tokyo
    python prepare.py --images C:\\opengan\\images_v2 --out C:\\opengan\\cache_v2 --drop Luanda,Papeete

Per city the cache holds uint8 [4, H, W]:

    0  building signed distance, int8 bits, quarter-pixel units, clipped at +-31 raster px
    1  height byte (rank-normalised through the corpus HeightLUT), filled outside footprints
    2  street signed distance, same units as 0
    3  green signed distance, same units as 0

Signed distances are positive inside the class, the zero crossing on the pixel edge,
exactly as tile_codec.signed_distance defines them, but kept untruncated to 31 px so the
crop code can truncate at 8 *working* pixels after resizing to whatever scale it picked.

The height LUT is rebuilt from the grey histogram of every raster in the set (first pass),
then applied (second pass). The LUT and the list of cities go next to the cache so a
checkpoint can always find the decoding it was trained with.

A raster is skipped with a warning when it is missing; nothing else is filtered here.
Use audit_osm.py to decide which cities to --drop.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
    from scipy import ndimage
except ImportError as e:  # the system Python has neither torch nor scipy
    raise SystemExit(f"{e}\nrun with the training venv: C:\\opengan\\venv\\Scripts\\python.exe prepare.py ...")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase0"))
from tile_codec import HeightLUT, split_classes, fill_height, HEIGHT_SIGMA_PX  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

SDF_CLIP_PX = 31.0        # raster pixels; 8 working px at the coarsest training scale (x2) is 16
SDF_QUANT = 4.0           # int8 stores quarter pixels
CACHE_VERSION = 1


def raster_path(images: Path, city: str) -> Path:
    """<images>/<City>/<City>_raster.png as osm_tiles writes it, or a flat <images>/<City>_raster.png."""
    nested = images / city / f"{city}_raster.png"
    return nested if nested.exists() else images / f"{city}_raster.png"


def find_cities(images: Path) -> list[str]:
    nested = {p.name for p in images.iterdir() if p.is_dir() and (p / f"{p.name}_raster.png").exists()}
    flat = {p.name[:-len("_raster.png")] for p in images.glob("*_raster.png")}
    return sorted(nested | flat)


def signed_distance_full(mask: np.ndarray, clip: float = SDF_CLIP_PX) -> np.ndarray:
    """Untruncated (well, clipped far out) signed distance in pixels, float32."""
    if mask.all():
        return np.full(mask.shape, clip, np.float32)
    if not mask.any():
        return np.full(mask.shape, -clip, np.float32)
    d_in = ndimage.distance_transform_edt(mask)
    d_out = ndimage.distance_transform_edt(~mask)
    s = np.where(mask, d_in - 0.5, -(d_out - 0.5)).astype(np.float32)
    return np.clip(s, -clip, clip)


def sdf_to_int8(s: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(s * SDF_QUANT), -127, 127).astype(np.int8)


def encode_raster(rgb: np.ndarray, lut: HeightLUT, empty_height: int) -> np.ndarray:
    """RGB [H, W, 3] -> uint8 [4, H, W] in the cache layout."""
    footprint, grey, street, green, _ = split_classes(rgb)
    height = fill_height(lut.apply(grey, footprint), footprint, HEIGHT_SIGMA_PX, empty_height)
    out = np.empty((4,) + footprint.shape, np.uint8)
    out[0] = sdf_to_int8(signed_distance_full(footprint)).view(np.uint8)
    out[1] = height
    out[2] = sdf_to_int8(signed_distance_full(street)).view(np.uint8)
    out[3] = sdf_to_int8(signed_distance_full(green)).view(np.uint8)
    return out


def grey_counts(rgb: np.ndarray) -> np.ndarray:
    footprint, grey, _, _, _ = split_classes(rgb)
    return np.bincount(grey[footprint].ravel(), minlength=256)[:256]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder with <City>/<City>_raster.png")
    ap.add_argument("--out", required=True, help="cache folder")
    ap.add_argument("--only", help="comma-separated city names")
    ap.add_argument("--drop", help="comma-separated city names to leave out")
    ap.add_argument("--lut", help="reuse an existing height_lut.json instead of rebuilding")
    a = ap.parse_args()

    images, out = Path(a.images), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cities = find_cities(images)
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        cities = [c for c in cities if c in want]
    if a.drop:
        drop = {s.strip() for s in a.drop.split(",")}
        cities = [c for c in cities if c not in drop]
    if not cities:
        sys.exit(f"no <City>/<City>_raster.png found under {images} (render with osm_tiles --raster)")
    print(f"{len(cities)} cities")

    # pass 1: height LUT from the whole corpus
    if a.lut:
        lut = HeightLUT.load(a.lut)
        print(f"height LUT from {a.lut}")
    else:
        counts = np.zeros(256, np.int64)
        t0 = time.time()
        for i, c in enumerate(cities, 1):
            counts += grey_counts(np.asarray(Image.open(raster_path(images, c)).convert("RGB")))
            print(f"  histogram {i}/{len(cities)} {c:20s} {time.time() - t0:5.0f}s", end="\r", flush=True)
        print()
        lut = HeightLUT.from_counts(counts)
        print(f"height LUT: {int((counts > 0).sum())} grey levels, mean byte {lut.mean_byte()}")
    lut.save(out / "height_lut.json")
    empty_height = lut.mean_byte()

    # pass 2: encode
    meta = {"version": CACHE_VERSION, "cities": [], "sdf_clip_px": SDF_CLIP_PX, "sdf_quant": SDF_QUANT,
            "empty_height_byte": empty_height, "layout": "uint8 [4,H,W]: fp_sdf(int8/4) height st_sdf gr_sdf",
            "res_m_per_px": None}
    t0 = time.time()
    for i, c in enumerate(cities, 1):
        dst = out / f"{c}.npy"
        grid = raster_path(images, c).with_name(f"{c}_grid.json")
        res = None
        if grid.exists():
            g = json.loads(grid.read_text())
            res = g.get("res_m_per_px")
            meta["res_m_per_px"] = meta["res_m_per_px"] or res
        if dst.exists():
            arr = np.load(dst, mmap_mode="r")
            meta["cities"].append({"city": c, "file": dst.name, "shape": list(arr.shape), "res_m_per_px": res})
            print(f"  [{i}/{len(cities)}] {c:20s} cached")
            continue
        rgb = np.asarray(Image.open(raster_path(images, c)).convert("RGB"))
        enc = encode_raster(rgb, lut, empty_height)
        np.save(dst, enc)
        fp = enc[0].view(np.int8) > 0
        meta["cities"].append({"city": c, "file": dst.name, "shape": list(enc.shape), "res_m_per_px": res,
                               "coverage_pct": round(100.0 * float(fp.mean()), 2)})
        print(f"  [{i}/{len(cities)}] {c:20s} {enc.shape[1]}x{enc.shape[2]}  cov {100 * fp.mean():5.1f}%  "
              f"{time.time() - t0:5.0f}s")
    (out / "cache.json").write_text(json.dumps(meta, indent=1))
    print(f"\ncache: {out}  ({len(meta['cities'])} cities, height_lut.json, cache.json)")


if __name__ == "__main__":
    main()
