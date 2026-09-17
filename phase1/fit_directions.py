"""
fit_directions.py

Phase 2: turn a trained generator into metric sliders.

  1. sample N latents, map to W, synthesise tiles
  2. measure every generated tile (coverage, street, green, height, FAR, counts)
  3. fit a linear direction in W for each metric, orthogonalise them
  4. verify: walk along each direction and check the measured metric moves monotonically

Usage (run from anywhere, needs the patched stylegan3 repo and phase0/tile_codec.py):
    python fit_directions.py --repo ../stylegan3 --network runs/.../network-snapshot-002000.pkl \
        --lut ../dataset128/height_lut.json --cities ../dataset128/cities.json --out directions --n 20000

Outputs in --out:
    samples.npz         w [N, 512] float32, c [N] int, metrics [N, K] float32, metric_names
    directions.npz      names, unit direction vectors [K, 512] (orthogonalised), fit R^2,
                        metric mean/std, w_avg, and the measured slope per unit alpha
    directions.csv      per metric: R^2, slope, monotonicity of the walk
    walk_<metric>.png   one base tile walked from -alpha to +alpha along that direction
    walk_curves.png     measured metric versus alpha for each direction
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parent / "phase0"):
    if (cand / "tile_codec.py").exists():
        sys.path.insert(0, str(cand))
        break
from tile_codec import HeightLUT, channels_to_rgb, tile_metrics, METRIC_INFO  # noqa: E402

# Default slider set, in Gram-Schmidt priority order: earlier metrics keep their
# direction exactly, later ones are made orthogonal to everything before them.
DEFAULT_METRICS = ["far_proxy", "coverage_pct", "green_pct", "street_pct", "mean_height_m",
                   "street_length_m", "intersections", "building_count", "mean_block_m2"]


def load_G(repo: Path, network: Path, device):
    sys.path.insert(0, str(repo))
    import torch
    import legacy  # noqa
    with open(network, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device).eval()
    return G


def to_uint8(img):
    """G output in [-1, 1] float -> uint8 [B, 4, H, W]"""
    return ((img.clamp(-1, 1) + 1) * 127.5).round().to("cpu").numpy().astype(np.uint8)


def one_hot(c_idx, dim, device):
    import torch
    c = torch.zeros(len(c_idx), dim, device=device)
    if dim > 0:
        c[torch.arange(len(c_idx)), torch.as_tensor(c_idx, device=device)] = 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to the patched stylegan3 clone")
    ap.add_argument("--network", required=True, help="network-snapshot-*.pkl")
    ap.add_argument("--lut", required=True, help="dataset/height_lut.json")
    ap.add_argument("--cities", default=None, help="dataset/cities.json (for labels)")
    ap.add_argument("--out", default="directions")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--trunc", type=float, default=1.0, help="truncation psi for sampling")
    ap.add_argument("--city", type=int, default=-1, help="fix a city label, -1 = random per sample")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--walk-steps", type=int, default=9)
    ap.add_argument("--walk-range", type=float, default=3.0, help="alpha range in units of W std")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile-m", type=float, default=400.0, help="tile edge in metres")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                    help="comma-separated metric keys from tile_codec.METRIC_INFO, in slider priority order")
    args = ap.parse_args()
    METRICS = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in METRICS if m not in METRIC_INFO]
    if unknown:
        sys.exit(f"unknown metrics: {unknown}. known: {list(METRIC_INFO)}")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lut = HeightLUT.load(args.lut)
    G = load_G(Path(args.repo), Path(args.network), device)
    c_dim = G.c_dim
    print(f"G: z_dim={G.z_dim} w_dim={G.w_dim} c_dim={c_dim} res={G.img_resolution} ch={G.img_channels} device={device}")

    # ---- 1 + 2: sample and measure -------------------------------------
    rng = np.random.RandomState(args.seed)
    W, C, M = [], [], []
    for start in range(0, args.n, args.batch):
        b = min(args.batch, args.n - start)
        z = torch.from_numpy(rng.randn(b, G.z_dim).astype(np.float32)).to(device)
        c_idx = (np.full(b, args.city) if args.city >= 0 else rng.randint(0, max(c_dim, 1), b))
        c = one_hot(c_idx, c_dim, device)
        with torch.no_grad():
            w = G.mapping(z, c, truncation_psi=args.trunc)     # [b, num_ws, w_dim]
            img = G.synthesis(w, noise_mode="const")
        ch = to_uint8(img)
        for i in range(b):
            m = tile_metrics(ch[i], lut, args.tile_m)
            M.append([m[k] for k in METRICS])
        W.append(w[:, 0, :].cpu().numpy())
        C.append(c_idx)
        if (start // args.batch) % 20 == 0:
            print(f"  sampled {start + b}/{args.n}")
    W = np.concatenate(W).astype(np.float32)
    C = np.concatenate(C).astype(np.int64)
    M = np.asarray(M, dtype=np.float32)
    np.savez_compressed(out / "samples.npz", w=W, c=C, metrics=M, metric_names=np.array(METRICS))

    # ---- 3: fit directions ---------------------------------------------
    if args.n < 10 * W.shape[1]:
        print(f"WARNING: {args.n} samples for a {W.shape[1]}-dim regression, R^2 will be inflated. Use --n 20000 for real directions.")
    w_mean = W.mean(0)
    w_std = W.std(0) + 1e-8
    Wn = (W - w_mean) / w_std                                  # standardised W
    m_mean = M.mean(0)
    m_std = M.std(0) + 1e-8
    Mn = (M - m_mean) / m_std

    XtX = Wn.T @ Wn + args.ridge * np.eye(Wn.shape[1], dtype=np.float32)
    coefs = np.linalg.solve(XtX, Wn.T @ Mn)                   # [512, K]
    pred = Wn @ coefs
    r2 = 1 - ((Mn - pred) ** 2).sum(0) / ((Mn - Mn.mean(0)) ** 2).sum(0)

    # directions in raw W units: standardised coef / w_std (chain rule), then unit norm
    raw = (coefs.T / w_std)                                     # [K, 512]
    dirs = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)
    # Jacobian d(metric)/d(w) in raw units: metric = m_mean + m_std * ((w - w_mean)/w_std @ coefs)
    jacobian = (raw * m_std[:, None]).astype(np.float32)         # [K, 512]

    # Gram-Schmidt in the order of METRICS so earlier sliders stay clean
    ortho = np.zeros_like(dirs)
    for k in range(len(METRICS)):
        v = dirs[k].copy()
        for j in range(k):
            v -= (v @ ortho[j]) * ortho[j]
        ortho[k] = v / (np.linalg.norm(v) + 1e-8)

    # ---- 4: verify by walking ------------------------------------------
    w_scale = float(np.linalg.norm(w_std))                     # typical W radius, one "alpha unit"
    alphas = np.linspace(-args.walk_range, args.walk_range, args.walk_steps)
    n_base = 64
    z = torch.from_numpy(rng.randn(n_base, G.z_dim).astype(np.float32)).to(device)
    c_idx = (np.full(n_base, args.city) if args.city >= 0 else rng.randint(0, max(c_dim, 1), n_base))
    with torch.no_grad():
        w_base = G.mapping(z, one_hot(c_idx, c_dim, device), truncation_psi=args.trunc)
    curves = np.zeros((len(METRICS), len(alphas)))
    strips = []
    rows = []
    for k, name in enumerate(METRICS):
        d = torch.from_numpy(ortho[k].astype(np.float32)).to(device)
        strip = []
        for a_i, a in enumerate(alphas):
            w = w_base.clone()
            w[:, :, :] = w_base + a * w_scale * d
            with torch.no_grad():
                ch = to_uint8(G.synthesis(w, noise_mode="const"))
            vals = [tile_metrics(ch[i], lut, args.tile_m)[name] for i in range(n_base)]
            curves[k, a_i] = float(np.mean(vals))
            strip.append(channels_to_rgb(ch[0], lut))
        strips.append(strip)
        diffs = np.diff(curves[k])
        mono = float(np.mean(np.sign(diffs) == np.sign(diffs.sum()))) if np.any(diffs) else 0.0
        slope = float((curves[k, -1] - curves[k, 0]) / (alphas[-1] - alphas[0]))
        rows.append({"metric": name, "r2": round(float(r2[k]), 4), "slope_per_alpha": round(slope, 4),
                     "monotonic_frac": round(mono, 3), "sample_mean": round(float(m_mean[k]), 4),
                     "sample_std": round(float(m_std[k]), 4)})
        print(f"  {name:<18} R2={r2[k]:.3f}  slope/alpha={slope:+.3f}  monotonic={mono:.2f}")

    np.savez(out / "directions.npz", names=np.array(METRICS), directions=ortho.astype(np.float32),
             directions_raw=dirs.astype(np.float32), jacobian=jacobian, r2=r2.astype(np.float32),
             metric_mean=m_mean, metric_std=m_std, w_mean=w_mean, w_std=w_std,
             w_scale=np.float32(w_scale), alphas=alphas.astype(np.float32), curves=curves.astype(np.float32),
             tile_m=np.float32(args.tile_m))
    with (out / "directions.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    # strips and curves
    from PIL import Image
    S = 128
    for k, name in enumerate(METRICS):
        canvas = Image.new("RGB", (S * len(alphas), S), "white")
        for i, rgb in enumerate(strips[k]):
            canvas.paste(Image.fromarray(rgb).resize((S, S), Image.NEAREST), (i * S, 0))
        canvas.save(out / f"walk_{name}.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(METRICS), figsize=(2.6 * len(METRICS), 2.6))
        for k, ax in enumerate(axes):
            ax.plot(alphas, curves[k], marker="o", ms=3)
            ax.set_title(f"{METRICS[k]}\nR2={r2[k]:.2f}", fontsize=8)
            ax.set_xlabel("alpha", fontsize=8); ax.tick_params(labelsize=7)
        fig.tight_layout(); fig.savefig(out / "walk_curves.png", dpi=120); plt.close(fig)
    except ImportError:
        pass
    print(f"written to {out.resolve()}")


if __name__ == "__main__":
    main()
