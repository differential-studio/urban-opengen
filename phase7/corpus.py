"""
corpus.py

Put a design space next to the real world.

`meta.csv` from phase 0 already holds the morphology of 30,519 real tiles from
60 cities, measured by the same code that measures a generated one. So an option
does not have to be read in a vacuum: it can be placed in the distribution of
real fabric, and it can be told which real place it most resembles.

Two things come out of that:

  rank      where each of an option's numbers falls among real tiles, as a
            percentile. "FAR at the 96th percentile, coverage at the 41st" says
            more to a planner than the raw pair, because it says how unusual the
            combination is.
  nearest   the real tiles closest to the option in morphology, standardised so
            no column dominates. An option that reads "nearest real fabric:
            Barcelona_411, Paris_287" is interpretable in a way that a latent
            coordinate never will be.

    python corpus.py meta.csv --space ../out/space_test    # annotate a run
    python corpus.py meta.csv --background corpus.csv      # the cloud to plot behind it

The match uses only what both sides measure: density, grain and height. The sun
and view columns have no counterpart in `meta.csv` yet, because computing them
over the whole corpus means running the analysis on all 30,519 tiles, which is
an overnight job on the machine that holds them, not something to slip into a
lookup. `python corpus.py meta.csv --sun <images folder>` does exactly that job
when you want it, and writes the extra columns alongside.

One caveat worth keeping in view: `meta.csv` measures the raster the renderer
drew, and the analysis measures the fitted polygons. `phase7/validate_fit.py`
sizes that gap at half a percentage point of built area, which is far below the
spacing between neighbouring tiles in this space, so the comparison holds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

# analysis column -> meta.csv column. Only the ones both sides really measure.
SHARED = {
    "far": "far_proxy",
    "coverage_pct": "coverage_pct",
    "green_pct": "green_pct",
    "street_pct": "street_pct",
    "mean_height_m": "mean_height_m",
    "max_height_m": "max_height_m",
    "building_count": "building_count",
    "mean_footprint_m2": "mean_footprint_m2",
    "gfa_m2": "gfa_m2",
}
# what decides "nearest": the shape of the fabric, not its absolute size
MATCH = ["far", "coverage_pct", "green_pct", "street_pct", "mean_height_m", "mean_footprint_m2"]


class Corpus:
    def __init__(self, meta_csv, match=None):
        rows = list(csv.DictReader(open(meta_csv, newline="", encoding="utf-8")))
        if not rows:
            raise SystemExit(f"{meta_csv} is empty")
        self.name = [r.get("file", r.get("n", "")).rsplit(".", 1)[0] for r in rows]
        self.city = [r.get("city", "") for r in rows]
        self.match = [m for m in (match or MATCH) if SHARED.get(m) in rows[0]]
        self.cols = [m for m in SHARED if SHARED[m] in rows[0]]
        self.X = np.array([[float(r[SHARED[m]] or "nan") for m in self.cols] for r in rows], np.float64)
        good = np.isfinite(self.X).all(1)
        self.X, self.name, self.city = self.X[good], [n for n, g in zip(self.name, good) if g], \
            [c for c, g in zip(self.city, good) if g]
        self.idx = {m: i for i, m in enumerate(self.cols)}
        self.mu = self.X.mean(0)
        self.sd = self.X.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        mi = [self.idx[m] for m in self.match]
        self.Z = (self.X[:, mi] - self.mu[mi]) / self.sd[mi]
        self.order = {m: np.sort(self.X[:, self.idx[m]]) for m in self.cols}
        print(f"corpus: {len(self.X)} tiles, {len(set(self.city))} cities, "
              f"{len(self.cols)} shared columns, matching on {', '.join(self.match)}")

    def rank(self, obj: dict) -> dict:
        """percentile of each shared value among real tiles"""
        out = {}
        for m in self.cols:
            v = obj.get(m)
            if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                out[m + "_pct_rank"] = None
                continue
            a = self.order[m]
            out[m + "_pct_rank"] = round(100.0 * float(np.searchsorted(a, float(v)) / len(a)), 1)
        return out

    def nearest(self, obj: dict, k: int = 3):
        """-> [(tile name, city, distance in standard deviations)]"""
        try:
            q = np.array([float(obj[m]) for m in self.match], np.float64)
        except (KeyError, TypeError, ValueError):
            return []
        mi = [self.idx[m] for m in self.match]
        z = (q - self.mu[mi]) / self.sd[mi]
        d = np.linalg.norm(self.Z - z, axis=1)
        o = np.argsort(d)[:k]
        return [(self.name[i], self.city[i], round(float(d[i]), 3)) for i in o]

    def background(self, path):
        """the corpus in the same column names a run's objectives.csv uses, to plot behind it"""
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "city"] + self.cols)
            for n, c, row in zip(self.name, self.city, self.X):
                w.writerow([n, c] + [round(float(v), 4) for v in row])
        print(f"{path}: {len(self.X)} real tiles as the background cloud")


def annotate(corpus: Corpus, space: Path, k=3, force=False):
    """
    add the rank and nearest columns to a run's objectives.csv, in place, keeping the original

    The corpus is whole 400 m tiles. A design space of infills is a plot of one or two
    hectares, and a plot's FAR, coverage and mean height are not on the same scale as a
    tile's: a 150 m courtyard block reads FAR 4 where the Eixample tile around it reads 2.
    So an infill run is refused unless you ask for it, and what you get then is a rough
    bearing, not a like-for-like comparison.
    """
    src = space / "objectives.csv"
    rows = list(csv.DictReader(open(src, newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"{src} has no rows")
    run = space / "run.json"
    if run.exists() and not force:
        try:
            info = json.loads(run.read_text(encoding="utf-8"))
            a, t = info.get("area_m2"), float(info.get("tile_m") or 0)
            if a and t and a < 0.5 * t * t:
                raise SystemExit(
                    f"{space.name} is an infill run: each option is {a / 1e4:.1f} ha of a "
                    f"{t:.0f} m tile, and the corpus is whole tiles. Pass --force to do it anyway.")
        except (ValueError, TypeError):
            pass
    out = []
    for r in rows:
        obj = {k2: (float(v) if v not in ("", None) else None) for k2, v in r.items() if k2 != "name"}
        near = corpus.nearest(obj, k)
        extra = corpus.rank(obj)
        extra["nearest_real"] = near[0][0] if near else ""
        extra["nearest_city"] = near[0][1] if near else ""
        extra["nearest_distance_sd"] = near[0][2] if near else ""
        extra["nearest_also"] = " ".join(n for n, _, _ in near[1:]) if len(near) > 1 else ""
        out.append({**r, **extra})
    cols = list(out[0])
    bak = space / "objectives_raw.csv"
    if not bak.exists():
        src.replace(bak)
    with open(src, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow({c: ("" if r.get(c) is None else r.get(c, "")) for c in cols})
    print(f"{src}: {len(out)} options, {len(cols)} columns (was {len(rows[0])}); original kept as {bak.name}")
    for r in out[:6]:
        print(f"  {r['name']}  FAR {r.get('far', ''):>6} ({r.get('far_pct_rank', '')}th pct)   "
              f"nearest real fabric: {r['nearest_real']} ({r['nearest_distance_sd']} sd)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("meta", help="phase 0 meta.csv")
    ap.add_argument("--space", default=None, help="a run folder: annotate its objectives.csv")
    ap.add_argument("--background", default=None, help="write the corpus as a CSV to plot behind a run")
    ap.add_argument("--k", type=int, default=3, help="how many nearest tiles to report")
    ap.add_argument("--force", action="store_true", help="annotate an infill run anyway (see annotate)")
    args = ap.parse_args()
    c = Corpus(args.meta)
    if args.background:
        c.background(args.background)
    if args.space:
        annotate(c, Path(args.space), args.k, args.force)
    if not args.background and not args.space:
        print("nothing to do: pass --space or --background")


if __name__ == "__main__":
    main()
