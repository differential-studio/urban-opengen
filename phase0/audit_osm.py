"""
audit_osm.py

Read a rendered city folder (or all of them) and report what the tiles actually contain,
so the keep / drop / re-render decision is made on numbers instead of on the Overpass
feature counts, which are misleading: a city of large slab blocks returns few polygons
but covers plenty of ground, while a HOT-mapped city returns hundreds of thousands of
tiny sheds. Both look wrong in the render report and only one of them is.

    python audit_osm.py --root C:\\opengan\\images_v2
    python audit_osm.py --root C:\\opengan\\images_v2 --only Luanda,Beijing --sample 0

Per city it prints, over a sample of tiles:

    empty%      tiles with nothing at all
    nobld%      tiles with streets but no buildings, the signature of a street-only city
    bld%        share of pixels that are building footprint
    st% gr%     street and green share
    m2/bld      mean footprint, from this folder's coverage and the grid report's count
    meanH       mean building height in metres
    @7m%        share of footprint pixels sitting on the 7 m default, ie no height data
    greys       distinct height levels per tile, a proxy for how much the height map says

A city with high nobld% is under-mapped in OSM and will teach the model that its name
means empty. A city with @7m% near 100 has geometry but no heights. Neither is a bug in
the renderer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

RED = (255, 0, 0)
GREEN = (128, 255, 0)
GREY_MIN, GREY_MAX = 100.0, 255.0
H_MIN, H_MAX = 4.0, 100.0


def grey_to_m(g):
    return H_MIN + (g - GREY_MIN) * (H_MAX - H_MIN) / (GREY_MAX - GREY_MIN)


def m_to_grey(h):
    return int(round(GREY_MIN + (h - H_MIN) * (GREY_MAX - GREY_MIN) / (H_MAX - H_MIN)))


FALLBACK = m_to_grey(7.0)


def audit_city(folder: Path, sample: int, rng: np.random.RandomState):
    tiles = sorted(p for p in folder.glob("*.png") if not p.name.endswith("_raster.png"))
    if not tiles:
        return None
    if sample and len(tiles) > sample:
        tiles = [tiles[i] for i in sorted(rng.choice(len(tiles), sample, replace=False))]

    n_empty = n_nobld = 0
    bld = st = gr = []
    bld, st, gr, greys = [], [], [], []
    h_sum = h_n = fb_n = 0
    bld_px_total = 0
    px = None
    for p in tiles:
        a = np.asarray(Image.open(p).convert("RGB"))
        px = a.shape[0]
        s = (a == np.array(RED)).all(-1)
        g = (a == np.array(GREEN)).all(-1)
        b = (a[..., 0] == a[..., 1]) & (a[..., 1] == a[..., 2]) & (a[..., 0] > 0)
        if not (s.any() or g.any() or b.any()):
            n_empty += 1
        elif not b.any():
            n_nobld += 1
        bld.append(b.mean()); st.append(s.mean()); gr.append(g.mean())
        v = a[..., 0][b]
        bld_px_total += int(b.sum())
        if v.size:
            h_sum += float(grey_to_m(v.astype(np.float64)).sum()); h_n += v.size
            fb_n += int((v == FALLBACK).sum())
            greys.append(len(np.unique(v)))
    res_m2 = (400.0 / px) ** 2
    return {
        "tiles": len(tiles), "px": px,
        "empty_pct": 100.0 * n_empty / len(tiles),
        "nobld_pct": 100.0 * n_nobld / len(tiles),
        "bld_pct": 100.0 * float(np.mean(bld)),
        "st_pct": 100.0 * float(np.mean(st)),
        "gr_pct": 100.0 * float(np.mean(gr)),
        "mean_h_m": h_sum / h_n if h_n else 0.0,
        "fallback_pct": 100.0 * fb_n / h_n if h_n else 0.0,
        "greys": float(np.mean(greys)) if greys else 0.0,
        "bld_m2_sampled": bld_px_total * res_m2,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the images_v2 folder")
    ap.add_argument("--sample", type=int, default=80, help="tiles per city, 0 for all")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", help="comma-separated city names")
    ap.add_argument("--out", help="write the numbers as json")
    a = ap.parse_args()

    root = Path(a.root)
    report = {}
    rp = root / "render_report.json"
    if rp.exists():
        report = json.loads(rp.read_text())

    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        names = [n for n in names if n in want]

    rng = np.random.RandomState(a.seed)
    rows = {}
    print(f"{'city':22s} {'tiles':>5s} {'empty%':>7s} {'nobld%':>7s} {'bld%':>6s} {'st%':>6s} "
          f"{'gr%':>6s} {'m2/bld':>7s} {'meanH':>6s} {'@7m%':>6s} {'greys':>6s}")
    for name in names:
        r = audit_city(root / name, a.sample, rng)
        if r is None:
            continue
        # mean footprint needs the polygon count from the grid report, scaled to the sample
        n_bld = report.get(name, {}).get("buildings")
        n_all = report.get(name, {}).get("tiles") or r["tiles"]
        m2 = ""
        if n_bld:
            share = r["tiles"] / n_all
            est = n_bld * share
            if est >= 1:
                r["m2_per_building"] = r["bld_m2_sampled"] / est
                m2 = f"{r['m2_per_building']:7.0f}"
        rows[name] = r
        print(f"{name:22s} {r['tiles']:5d} {r['empty_pct']:7.1f} {r['nobld_pct']:7.1f} "
              f"{r['bld_pct']:6.2f} {r['st_pct']:6.2f} {r['gr_pct']:6.2f} {m2:>7s} "
              f"{r['mean_h_m']:6.1f} {r['fallback_pct']:6.1f} {r['greys']:6.1f}")

    flag = []
    for name, r in rows.items():
        why = []
        if r["empty_pct"] + r["nobld_pct"] > 40:
            why.append(f"{r['empty_pct'] + r['nobld_pct']:.0f}% of tiles have no buildings")
        if r["bld_pct"] < 5:
            why.append(f"only {r['bld_pct']:.1f}% building coverage")
        if r["fallback_pct"] > 85:
            why.append(f"{r['fallback_pct']:.0f}% of footprints have no height data")
        if why:
            flag.append((name, "; ".join(why)))
    if flag:
        print("\nunder-mapped, decide before rebuilding the dataset:")
        for name, why in flag:
            print(f"  {name:22s} {why}")
    else:
        print("\nnothing flagged")

    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
        print(f"\nnumbers: {a.out}")


if __name__ == "__main__":
    main()
