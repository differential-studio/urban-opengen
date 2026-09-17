"""
audit_tiles.py

Phase 0a: audit the raw tile dataset before touching any model.

Answers, per city and overall:
  * how many tiles are blank (all background) or near-blank
  * where in the 25 x 25 grid the blanks fall (were the windows well centred?)
  * the grey-value histogram of building pixels (the height encoding problem)
  * whether any pixels fall outside the four-colour palette (anti-aliasing check)
  * tile sizes and modes, in case anything is inconsistent

Usage:
    python audit_tiles.py --images images --out audit
    python audit_tiles.py --images images --out audit --near-blank-pct 5 --grid 25

Outputs in --out:
    tiles.csv           one row per tile
    cities.csv          one row per city
    summary.txt         the headline numbers
    grey_histogram.png  building grey-value distribution, linear and log
    blank_grids.png     one 25 x 25 map per city: blank / near-blank / usable
    city_blank_bars.png per-city usable fraction
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from tile_codec import load_rgb, split_classes

TILE_RE = re.compile(r"^(?P<city>.+)_(?P<idx>\d+)\.png$", re.IGNORECASE)


def scan_city(city_dir: Path, near_blank_pct: float):
    rows = []
    grey_counts = np.zeros(256, dtype=np.int64)
    for p in sorted(city_dir.glob("*.png")):
        m = TILE_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group("idx"))
        try:
            rgb = load_rgb(p)
        except Exception as e:  # truncated or corrupt PNG
            print(f"  unreadable {p.name}: {e}", file=sys.stderr)
            rows.append({"city": city_dir.name, "idx": idx, "file": p.name, "bytes": p.stat().st_size,
                         "width": 0, "height": 0, "status": "unreadable", "content_pct": 0.0,
                         "footprint_pct": 0.0, "street_pct": 0.0, "green_pct": 0.0, "off_palette_pct": 0.0,
                         "grey_min": -1, "grey_max": -1, "grey_levels": 0})
            continue
        h, w = rgb.shape[:2]
        footprint, grey, street, green, off = split_classes(rgb)
        n = h * w
        content = footprint | street | green
        content_pct = 100.0 * content.sum() / n
        if content_pct == 0.0:
            status = "blank"
        elif content_pct < near_blank_pct:
            status = "near_blank"
        else:
            status = "ok"

        gv = grey[footprint]
        if gv.size:
            grey_counts += np.bincount(gv, minlength=256)

        rows.append({
            "city": city_dir.name,
            "idx": idx,
            "file": p.name,
            "bytes": p.stat().st_size,
            "width": w,
            "height": h,
            "status": status,
            "content_pct": round(content_pct, 3),
            "footprint_pct": round(100.0 * footprint.sum() / n, 3),
            "street_pct": round(100.0 * street.sum() / n, 3),
            "green_pct": round(100.0 * green.sum() / n, 3),
            "off_palette_pct": round(100.0 * off.sum() / n, 4),
            "grey_min": int(gv.min()) if gv.size else -1,
            "grey_max": int(gv.max()) if gv.size else -1,
            "grey_levels": int(np.unique(gv).size) if gv.size else 0,
        })
    return rows, grey_counts


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_plots(out: Path, tiles: list[dict], cities: list[dict], grey_counts: np.ndarray, grid: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots", file=sys.stderr)
        return

    # 1. grey histogram
    xs = np.arange(256)
    present = grey_counts > 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, log in zip(axes, (False, True)):
        ax.bar(xs[present], grey_counts[present], width=1.0, color="#444")
        ax.set_xlim(95, 255)
        ax.set_xlabel("grey value of building pixel")
        ax.set_ylabel("pixel count")
        if log:
            ax.set_yscale("log")
            ax.set_title("same, log scale")
        else:
            ax.set_title("building grey values across the corpus")
    fig.tight_layout()
    fig.savefig(out / "grey_histogram.png", dpi=130)
    plt.close(fig)

    # 2. blank grids per city
    by_city = defaultdict(dict)
    for t in tiles:
        by_city[t["city"]][t["idx"]] = t["status"]
    names = sorted(by_city)
    ncol = 6
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.2, nrow * 2.4))
    axes = np.atleast_2d(axes)
    code = {"blank": 0, "near_blank": 1, "ok": 2, "unreadable": 3}
    cmap = matplotlib.colors.ListedColormap(["#111111", "#c98a00", "#2e8b57", "#c0392b"])
    for ax, name in zip(axes.flat, names):
        g = np.full((grid, grid), np.nan)
        for idx, st in by_city[name].items():
            r, c = divmod(idx, grid)
            if r < grid and c < grid:
                g[r, c] = code[st]
        ax.imshow(g, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        usable = sum(1 for s in by_city[name].values() if s == "ok")
        ax.set_title(f"{name}\n{usable}/{len(by_city[name])} usable", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.suptitle("Tile status by grid position   black = blank, amber = near-blank, green = usable, red = unreadable", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "blank_grids.png", dpi=130)
    plt.close(fig)

    # 3. per-city bars
    cities_sorted = sorted(cities, key=lambda c: c["usable_pct"])
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(cities_sorted))))
    ax.barh([c["city"] for c in cities_sorted], [c["usable_pct"] for c in cities_sorted], color="#2e8b57")
    ax.set_xlabel("usable tiles, percent")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(out / "city_blank_bars.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images", help="folder containing one sub-folder per city")
    ap.add_argument("--out", default="audit")
    ap.add_argument("--near-blank-pct", type=float, default=5.0,
                    help="tiles with less than this percent non-background content are flagged")
    ap.add_argument("--grid", type=int, default=25)
    args = ap.parse_args()

    images = Path(args.images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    city_dirs = sorted(p for p in images.iterdir() if p.is_dir())
    if not city_dirs:
        sys.exit(f"no city folders found under {images}")

    all_tiles = []
    city_rows = []
    grey_total = np.zeros(256, dtype=np.int64)
    for cd in city_dirs:
        rows, gc = scan_city(cd, args.near_blank_pct)
        if not rows:
            continue
        grey_total += gc
        all_tiles.extend(rows)
        n = len(rows)
        blank = sum(r["status"] == "blank" for r in rows)
        near = sum(r["status"] == "near_blank" for r in rows)
        bad = sum(r["status"] == "unreadable" for r in rows)
        ok = n - blank - near - bad
        sizes = {(r["width"], r["height"]) for r in rows if r["status"] != "unreadable"}
        city_rows.append({
            "city": cd.name,
            "tiles": n,
            "blank": blank,
            "near_blank": near,
            "usable": ok,
            "usable_pct": round(100.0 * ok / n, 1),
            "unreadable": bad,
            "sizes": ";".join(f"{w}x{h}" for w, h in sorted(sizes)),
            "mean_footprint_pct_usable": round(float(np.mean([r["footprint_pct"] for r in rows if r["status"] == "ok"]) if ok else 0.0), 2),
            "mean_street_pct_usable": round(float(np.mean([r["street_pct"] for r in rows if r["status"] == "ok"]) if ok else 0.0), 2),
            "mean_green_pct_usable": round(float(np.mean([r["green_pct"] for r in rows if r["status"] == "ok"]) if ok else 0.0), 2),
            "off_palette_tiles": sum(r["off_palette_pct"] > 0 for r in rows),
        })
        print(f"{cd.name:<16} tiles={n:4d}  blank={blank:4d}  near_blank={near:4d}  usable={ok:4d} ({100*ok/n:5.1f}%)")

    write_csv(out / "tiles.csv", all_tiles)
    write_csv(out / "cities.csv", city_rows)

    # summary
    N = len(all_tiles)
    blank = sum(t["status"] == "blank" for t in all_tiles)
    near = sum(t["status"] == "near_blank" for t in all_tiles)
    bad = sum(t["status"] == "unreadable" for t in all_tiles)
    ok = N - blank - near - bad
    present = np.nonzero(grey_total)[0]
    cum = np.cumsum(grey_total) / max(grey_total.sum(), 1)
    p01 = int(np.searchsorted(cum, 0.01))
    p50 = int(np.searchsorted(cum, 0.50))
    p99 = int(np.searchsorted(cum, 0.99))
    off_tiles = sum(t["off_palette_pct"] > 0 for t in all_tiles)
    sizes = defaultdict(int)
    for t in all_tiles:
        if t["status"] != "unreadable":
            sizes[(t["width"], t["height"])] += 1

    lines = [
        f"cities:                 {len(city_rows)}",
        f"tiles scanned:          {N}",
        f"blank (all background): {blank}  ({100*blank/N:.1f}%)",
        f"near-blank (<{args.near_blank_pct:g}%):     {near}  ({100*near/N:.1f}%)",
        f"unreadable (corrupt):   {bad}",
        f"usable:                 {ok}  ({100*ok/N:.1f}%)",
        f"tile sizes:             " + ", ".join(f"{w}x{h}: {c}" for (w, h), c in sorted(sizes.items())),
        "",
        f"building pixels:        {int(grey_total.sum())}",
        f"distinct grey levels:   {present.size}",
        f"grey range:             {int(present.min()) if present.size else -1} .. {int(present.max()) if present.size else -1}",
        f"grey 1st / 50th / 99th percentile: {p01} / {p50} / {p99}",
        f"  -> 98% of height information lives in {p99 - p01 + 1} grey levels out of 255",
        f"tiles with off-palette pixels: {off_tiles}",
    ]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print("\n".join(lines))

    make_plots(out, all_tiles, city_rows, grey_total, args.grid)
    print(f"\nwritten to {out.resolve()}")


if __name__ == "__main__":
    main()
