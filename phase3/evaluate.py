"""
evaluate.py

Phase 3: does the model actually match the corpus, and is it still improving?

Preview grids answer neither question. This samples tiles from one or more snapshots,
measures them with exactly the metrics `reencode_tiles.py` recorded for the real tiles,
and reports how far each generated distribution sits from the real one. Run it on two
snapshots to see whether the last N kimg bought anything.

    python evaluate.py --repo . --lut ..\\dataset128_sdf\\height_lut.json --meta ..\\dataset128_sdf\\meta.csv \\
        --snapshot runs_sdf\\<run>\\network-snapshot-001600.pkl runs_sdf\\<run>\\network-snapshot-002640.pkl \\
        --per-city 16 --out eval

Distance is the 1-Wasserstein distance between the generated and real samples for each
metric, divided by the real standard deviation, so it is comparable across metrics and
roughly reads as "how many standard deviations apart the two distributions are". Below
about 0.1 is a good match, above 0.5 means the model is producing a different kind of
city.

Sampling uses truncation 1.0 on purpose: truncation narrows the distribution and would
flatter the model.

    --per-city 16      tiles per city label (60 labels -> 960 tiles, a few minutes on CPU)
    --trunc 1.0        do not change this unless you know why
    --device cpu       cuda is faster but competes with a running training job
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parent / "phase0"):
    if (cand / "tile_codec.py").exists():
        sys.path.insert(0, str(cand))
        break
from tile_codec import tile_metrics, HeightLUT  # noqa: E402

KEYS = ["far_proxy", "coverage_pct", "green_pct", "street_pct", "mean_height_m", "max_height_m",
        "building_count", "mean_footprint_m2", "street_length_m", "intersections", "block_count"]


def _need(pkg, exc):
    sys.exit(f"\n{pkg} is not installed in this Python:\n    {sys.executable}\n\n"
             f"Run this with the environment you train in, by full path, for example:\n"
             f"    C:\\opengan\\venv\\Scripts\\python.exe <this script> ...\n\noriginal error: {exc}")


def sample(network, repo, lut, per_city, trunc, seed, device):
    sys.path.insert(0, repo)
    try:
        import torch
    except ImportError as e:
        _need("torch", e)
    import legacy
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    with open(network, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(dev).eval()
    rng = np.random.RandomState(seed)
    rows, t0 = [], time.time()
    n_lab = max(G.c_dim, 1)
    for c in range(n_lab):
        z = torch.from_numpy(rng.randn(per_city, G.z_dim).astype(np.float32)).to(dev)
        lab = torch.zeros(per_city, G.c_dim, device=dev)
        if G.c_dim > 0:
            lab[:, c] = 1
        with torch.no_grad():
            img = G(z, lab, truncation_psi=trunc, noise_mode="const")
        b = ((img.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()
        for i in range(per_city):
            m = tile_metrics(b[i], lut, 400.0)
            m["city_id"] = c
            rows.append(m)
        print(f"  {Path(network).name}: label {c + 1}/{n_lab}  {time.time() - t0:.0f}s", end="\r", flush=True)
    print(f"  {Path(network).name}: {len(rows)} tiles in {time.time() - t0:.0f}s" + " " * 20)
    return rows


def w1(a, b, n=200):
    qs = np.linspace(0.01, 0.99, n)
    return float(np.abs(np.quantile(a, qs) - np.quantile(b, qs)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--snapshot", nargs="+", required=True)
    ap.add_argument("--lut", required=True)
    ap.add_argument("--meta", required=True, help="meta.csv of the dataset it was trained on")
    ap.add_argument("--per-city", type=int, default=16)
    ap.add_argument("--trunc", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="eval")
    a = ap.parse_args()

    try:
        import pandas as pd
    except ImportError as e:
        _need("pandas", e)
    real = pd.read_csv(a.meta)
    lut = HeightLUT.load(a.lut)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    gens = {}
    for s in a.snapshot:
        name = Path(s).stem.replace("network-snapshot-", "")
        cache = out / f"metrics_{name}.json"
        if cache.exists():
            print(f"  {name}: reusing {cache}")
            gens[name] = pd.DataFrame(json.loads(cache.read_text()))
            continue
        rows = sample(s, a.repo, lut, a.per_city, a.trunc, a.seed, a.device)
        cache.write_text(json.dumps(rows))
        gens[name] = pd.DataFrame(rows)

    names = list(gens)
    w = max(9, max(len(n) for n in names) + 1)
    head = f"{'metric':22s} {'real':>10s} " + " ".join(f"{n:>{w}s}" for n in names) + "   | distance to real (real SDs)"
    print("\n" + head)
    print("-" * len(head))
    tot = {n: [] for n in names}
    report = {"real_median": {}, "generated_median": {n: {} for n in names}, "distance": {n: {} for n in names}}
    for m in KEYS:
        if m not in real.columns:
            continue
        r = real[m].values.astype(float)
        sd = r.std() or 1.0
        report["real_median"][m] = float(np.median(r))
        line = f"{m:22s} {np.median(r):10.2f} "
        ds = []
        for n in names:
            v = gens[n][m].values.astype(float)
            d = w1(r, v) / sd
            ds.append(d); tot[n].append(d)
            report["generated_median"][n][m] = float(np.median(v))
            report["distance"][n][m] = d
            line += f"{np.median(v):{w}.2f} "
        print(line + "   | " + "  ".join(f"{d:.3f}" for d in ds))
    print("-" * len(head))
    print(f"{'MEAN DISTANCE':22s} {'':10s} " + " ".join(f"{'':{w}s}" for _ in names) + "   | " +
          "  ".join(f"{np.mean(tot[n]):.3f}" for n in names))
    report["mean_distance"] = {n: float(np.mean(tot[n])) for n in names}

    # is the city conditioning working: does each label reproduce its own city's density?
    if "city_id" in real.columns:
        print("\nper-city conditioning (mean absolute error of median coverage, percentage points):")
        for n in names:
            errs = [abs(gens[n][gens[n].city_id == c]["coverage_pct"].median()
                        - real[real.city_id == c]["coverage_pct"].median())
                    for c in sorted(real.city_id.unique()) if (gens[n].city_id == c).any()]
            report.setdefault("city_coverage_mae", {})[n] = float(np.mean(errs))
            print(f"  {n:>{w}s}   {np.mean(errs):5.2f} pp")

    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"\nwritten to {out / 'report.json'}")
    if len(names) > 1:
        md = report["mean_distance"]
        best = min(names, key=lambda n: md[n])
        last = names[-1]
        line = "  ".join(f"{n} {md[n]:.3f}" for n in names)
        print(f"mean distance by snapshot: {line}")
        if best == last and md[last] < md[names[-2]] - 0.01:
            verdict = f"still improving, {last} is the best so far"
        elif best == last:
            verdict = f"{last} is best but only just: flat, more training of this model is not buying much"
        else:
            verdict = f"best snapshot is {best}, not the latest: use {best}"
        report["best"] = best
        print(verdict)


if __name__ == "__main__":
    main()
