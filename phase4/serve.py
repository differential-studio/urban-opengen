"""
serve.py

Phase 4: local inference server for the Urban OpenGen viewer.

Loads a StyleGAN2 snapshot (any network-snapshot-*.pkl), decodes its 4-channel
output through the phase 0 height LUT, and serves index.html, the one app, which renders the
tile in Three.js and drives the latent space with sliders.

    from the repository root:
    python phase4\\serve.py --repo <stylegan3 clone> --lut assets\\height_lut.json --cities assets\\cities.json
    -> open http://127.0.0.1:5000

Optional:
    --network runs\\...\\network-snapshot-003000.pkl   load at start (else pick in the UI)
    --directions assets\\directions.npz               named metric sliders from fit_directions.py
    --runs runs                                       where to look for snapshots (default <repo>/runs)
    --port 5000
    --pca-samples 4000                                W samples for the PCA sliders

Sliders exposed by the page, all combined server-side into one w:
    seed A / seed B and a mix slider          w = lerp(w_A, w_B, mix)
    city A / city B and a blend slider        c = lerp(onehot_A, onehot_B, blend)
    truncation psi
    PCA 1..N of W (GANSpace-style)            w += alpha * std_i * pc_i
    metric directions if directions.npz       w += alpha * w_scale * dir_k
"""
from __future__ import annotations

import argparse
import base64
import math
import glob
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parent / "phase0", HERE.parent / "phase1"):
    if (cand / "tile_codec.py").exists():
        sys.path.insert(0, str(cand))
        break
from tile_codec import (  # noqa: E402
    HeightLUT, tile_metrics, METRIC_INFO, GREY_MIN, GREY_MAX, HEIGHT_M_MIN, HEIGHT_M_MAX, FLOOR_HEIGHT_M,
    load_rgb, encode_tile, split_classes,
)

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

app = Flask(__name__, static_folder=None)
LOCK = threading.Lock()
S: dict = {
    "repo": None, "runs": None, "device": None,
    "G": None, "pkl": None, "lut": None, "cities": [],
    "pca_comps": None, "pca_std": None, "w_mean": None, "w_scale": 1.0,
    "dir_names": [], "dir_vecs": None, "dir_scale": 1.0,
    "jacobian": None, "metric_mean": None, "metric_std": None, "dir_r2": None,
    "pca_n": 8, "pca_samples": 4000, "tile_m": 400.0,
    "target_iters": 4, "target_tol": 0.03,
    "images": None, "encoding": "sdf", "empty_height": 128,
    "inpaint_steps": 400, "context_m": 800.0,
    "dif_root": None, "osm_contact": "urban-opengen@example.com", "osm_cache": None,
}


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def _torch():
    import torch
    return torch


def list_snapshots():
    pats = [os.path.join(S["runs"], "**", "network-snapshot-*.pkl")]
    out = []
    for p in pats:
        out += glob.glob(p, recursive=True)
    out = sorted(set(out), key=lambda p: (os.path.dirname(p), p))
    return [os.path.relpath(p, S["runs"]) for p in out]


def one_hot(idx, dim, device):
    torch = _torch()
    c = torch.zeros(len(idx), dim, device=device)
    if dim > 0:
        c[torch.arange(len(idx)), torch.as_tensor(idx, device=device)] = 1
    return c


def compute_pca(G, device, n_samples, n_comps, seed=0):
    torch = _torch()
    rng = np.random.RandomState(seed)
    ws = []
    c_dim = G.c_dim
    for start in range(0, n_samples, 512):
        b = min(512, n_samples - start)
        z = torch.from_numpy(rng.randn(b, G.z_dim).astype(np.float32)).to(device)
        c_idx = rng.randint(0, max(c_dim, 1), b)
        with torch.no_grad():
            w = G.mapping(z, one_hot(c_idx, c_dim, device), truncation_psi=1.0)[:, 0, :]
        ws.append(w.cpu().numpy())
    W = np.concatenate(ws).astype(np.float64)
    mean = W.mean(0)
    X = W - mean
    _, sv, vt = np.linalg.svd(X, full_matrices=False)
    std = sv / np.sqrt(max(len(W) - 1, 1))
    k = min(n_comps, vt.shape[0])
    w_scale = float(np.linalg.norm(X.std(0)))
    return vt[:k].astype(np.float32), std[:k].astype(np.float32), mean.astype(np.float32), w_scale


def resolve_snapshot(p: str) -> str:
    """absolute path, or relative to the runs folder, or relative to the repo"""
    if os.path.isabs(p) and os.path.isfile(p):
        return p
    for base in (S["runs"], S["repo"], os.getcwd()):
        cand = os.path.join(base, p)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(p)


def load_model(rel_or_abs: str):
    torch = _torch()
    path = resolve_snapshot(rel_or_abs)
    import legacy  # from the stylegan3 repo
    t0 = time.time()
    with open(path, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(S["device"]).eval()
    comps, std, mean, w_scale = compute_pca(G, S["device"], S["pca_samples"], S["pca_n"])
    with LOCK:
        S.update(G=G, pkl=os.path.relpath(path, S["runs"]),
                 pca_comps=comps, pca_std=std, w_mean=mean, w_scale=w_scale)
    print(f"loaded {path}  z={G.z_dim} w={G.w_dim} c={G.c_dim} res={G.img_resolution} ch={G.img_channels}  "
          f"pca in {time.time() - t0:.1f}s")


def load_directions(path):
    d = np.load(path)
    names = [str(n) for n in d["names"]]
    S.update(dir_names=names, dir_vecs=d["directions"].astype(np.float32),
             dir_scale=float(d["w_scale"]) if "w_scale" in d else 1.0,
             jacobian=d["jacobian"].astype(np.float64) if "jacobian" in d else None,
             metric_mean=d["metric_mean"].astype(np.float64) if "metric_mean" in d else None,
             metric_std=d["metric_std"].astype(np.float64) if "metric_std" in d else None,
             dir_r2=[float(x) for x in d["r2"]] if "r2" in d else None)
    if "tile_m" in d and abs(float(d["tile_m"]) - S["tile_m"]) > 1e-6:
        print(f"note: directions were fitted with tile_m={float(d['tile_m'])}, server uses {S['tile_m']}")
    print(f"directions: {names}" + ("  (with jacobian: absolute targets enabled)" if S["jacobian"] is not None else ""))


def state_payload():
    G = S["G"]
    lut = S["lut"]
    return {
        "loaded": G is not None,
        "pkl": S["pkl"],
        "device": str(S["device"]),
        "z_dim": int(G.z_dim) if G else 0,
        "w_dim": int(G.w_dim) if G else 0,
        "c_dim": int(G.c_dim) if G else 0,
        "res": int(G.img_resolution) if G else 0,
        "channels": int(G.img_channels) if G else 0,
        "cities": S["cities"],
        "pca": {"n": int(len(S["pca_std"])) if S["pca_std"] is not None else 0,
                "std": [float(x) for x in S["pca_std"]] if S["pca_std"] is not None else []},
        "directions": {"names": S["dir_names"], "w_scale": S["dir_scale"],
                       "targets": S["jacobian"] is not None,
                       "mean": [float(x) for x in S["metric_mean"]] if S["metric_mean"] is not None else [],
                       "std": [float(x) for x in S["metric_std"]] if S["metric_std"] is not None else [],
                       "r2": S["dir_r2"] or []},
        "metric_info": {k: {"label": v[0], "unit": v[1], "decimals": v[2]} for k, v in METRIC_INFO.items()},
        "lut_inverse": lut.inverse if lut else None,
        "height_map": {"grey_min": GREY_MIN, "grey_max": GREY_MAX,
                       "h_min": HEIGHT_M_MIN, "h_max": HEIGHT_M_MAX, "floor_m": FLOOR_HEIGHT_M},
        "tile_m": S["tile_m"],
        "real_tiles": real_cities(),
        "inpaint_steps": S["inpaint_steps"],
        "context_m": S["context_m"],
        "dif": S["dif_root"] is not None,
        "osm": True,
    }


def real_cities():
    """{city: [tile indices]} of the raw tiles under --images, or None when not configured."""
    if not S["images"]:
        return None
    if S.get("_real_index") is None:
        idx = {}
        root = Path(S["images"])
        for cd in sorted(p for p in root.iterdir() if p.is_dir()):
            ids = []
            for p in cd.glob("*.png"):
                stem = p.stem
                if "_" in stem and stem.rsplit("_", 1)[1].isdigit() and p.stat().st_size > 600:   # > 600 bytes: not an empty tile
                    ids.append(int(stem.rsplit("_", 1)[1]))
            if ids:
                idx[cd.name] = sorted(ids)
        S["_real_index"] = idx
    return S["_real_index"]


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def gen_w(seed: int, c_vec, psi: float):
    torch = _torch()
    G = S["G"]
    z = torch.from_numpy(np.random.RandomState(int(seed) & 0x7FFFFFFF).randn(1, G.z_dim).astype(np.float32)).to(S["device"])
    with torch.no_grad():
        w = G.mapping(z, c_vec, truncation_psi=float(psi))          # [1, num_ws, w_dim]
    return w


def synth(w):
    torch = _torch()
    with torch.no_grad():
        img = S["G"].synthesis(w, noise_mode="const")                # [1, C, H, W] in [-1, 1]
    return ((img[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()


def measure(ch, tile_m=None):
    """The GAN readout. Lengths and areas scale with the tile, so the real extent matters:
    measuring a 1500 m context as if it were 400 m understates every length by 3.75x."""
    return tile_metrics(ch, S["lut"], float(tile_m or S["tile_m"])) if S["lut"] else {}


def solve_targets(w, targets: dict):
    """
    Move w so the measured metrics hit the requested absolute values.

    Uses the linear model from fit_directions.py as a fixed Jacobian and takes
    damped minimum-norm Newton steps, re-measuring the real tile after each one.
    Returns (w, channels, measured-for-targeted-metrics, iterations, converged).
    """
    torch = _torch()
    names = S["dir_names"]
    idx = [names.index(k) for k in targets if k in names and targets[k] is not None]
    if not idx or S["jacobian"] is None:
        ch = synth(w)
        return w, ch, {}, 0, True
    A = S["jacobian"][idx]                                          # [k, w_dim]
    t = np.array([float(targets[names[i]]) for i in idx])
    AAt = A @ A.T
    ridge = 1e-3 * np.trace(AAt) / len(idx) + 1e-12
    max_step = 2.0 * S["dir_scale"]
    tol_scale = np.maximum(np.abs(t), S["metric_std"][idx]) * S["target_tol"] + 1e-9

    def residual(ch):
        m = measure(ch)
        return t - np.array([m[names[i]] for i in idx]), m

    ch = synth(w)
    r, m = residual(ch)
    steps = 0
    while steps < S["target_iters"] and not np.all(np.abs(r) <= tol_scale):
        delta = A.T @ np.linalg.solve(AAt + ridge * np.eye(len(idx)), r)
        nrm = float(np.linalg.norm(delta))
        if nrm > max_step:
            delta *= max_step / nrm
        w = w + torch.from_numpy(delta.astype(np.float32)).to(S["device"])[None, None, :]
        ch = synth(w)
        r, m = residual(ch)
        steps += 1
    converged = bool(np.all(np.abs(r) <= tol_scale))
    achieved = {names[i]: m[names[i]] for i in idx}
    return w, ch, achieved, steps, converged


def generate(p: dict):
    torch = _torch()
    G = S["G"]
    dev = S["device"]
    c_dim = G.c_dim

    # class vector, blended between two cities
    c = torch.zeros(1, c_dim, device=dev)
    if c_dim > 0:
        a = int(p.get("city_a", 0)) % c_dim
        b = int(p.get("city_b", a)) % c_dim
        blend = float(p.get("city_mix", 0.0))
        c[0, a] += 1.0 - blend
        c[0, b] += blend

    psi = float(p.get("psi", 1.0))
    w_a = gen_w(p.get("seed_a", 0), c, psi)
    mix = float(p.get("mix", 0.0))
    if mix > 0:
        w_b = gen_w(p.get("seed_b", 1), c, psi)
        w = w_a * (1 - mix) + w_b * mix
    else:
        w = w_a

    # PCA offsets
    pca = p.get("pca") or []
    if S["pca_comps"] is not None and any(abs(float(a)) > 1e-6 for a in pca):
        off = np.zeros(G.w_dim, np.float32)
        for i, a in enumerate(pca[: len(S["pca_std"])]):
            off += float(a) * S["pca_std"][i] * S["pca_comps"][i]
        w = w + torch.from_numpy(off).to(dev)[None, None, :]

    # metric direction offsets
    dirs = p.get("dirs") or {}
    if S["dir_vecs"] is not None:
        off = np.zeros(G.w_dim, np.float32)
        for k, name in enumerate(S["dir_names"]):
            a = float(dirs.get(name, 0.0))
            if abs(a) > 1e-6:
                off += a * S["dir_scale"] * S["dir_vecs"][k]
        if np.any(off):
            w = w + torch.from_numpy(off).to(dev)[None, None, :]

    targets = p.get("targets") or {}
    w, ch, achieved, iters, converged = solve_targets(w, targets)
    return ch, w[0, 0].cpu().numpy(), achieved, iters, converged


# ---------------------------------------------------------------------------
# inpainting: masked projection into W+
# ---------------------------------------------------------------------------
#
# No extra training and no second model. The generator can only make tiles that look
# like the training set, so if we search its latent space for the tile that best matches
# the part of the image we are keeping, whatever it puts in the hole is by construction
# a plausible piece of city that continues the streets and blocks around it.
#
#   prior    the mean of W for the chosen city, which is where the search starts and
#            what it is pulled back toward, so the fill stays on the manifold
#   loss     squared error against the target, but only where the mask says "keep",
#            summed over four resolutions so the optimiser matches the big shapes
#            before the pixels, and weighted by a smooth halo around the hole so the
#            edges the fill has to meet count for more than the far corners of the tile.
#            Turning the far weight down ("context") lets the fit give up on the rest of
#            the tile and concentrate on the neighbourhood, which is what you want when
#            the model cannot represent the whole tile yet
#   space    W+ (one w per synthesis layer), which fits far better than a single w
#   output   the original tile outside the hole, the generated tile inside, blended
#            over a few pixels. Because the channels are signed distance fields the
#            blend happens on smooth fields and the decoded outline stays continuous,
#            which is not true of hard masks
#
# The class vector only enters through the prior, so "fill this hole in the style of
# Tokyo" is a one-line change from "fill it in the style of Barcelona".

CH_WEIGHTS = (1.0, 0.3, 1.0, 0.6)   # footprint, height, street, green
_W_AVG_CACHE: dict = {}


def class_vec(p: dict):
    """the [1, c_dim] class vector for a request, blended between two cities"""
    torch = _torch()
    c_dim = S["G"].c_dim
    c = torch.zeros(1, c_dim, device=S["device"])
    if c_dim > 0:
        a = int(p.get("city_a", 0)) % c_dim
        b = int(p.get("city_b", a)) % c_dim
        blend = float(p.get("city_mix", 0.0))
        c[0, a] += 1.0 - blend
        c[0, b] += blend
    return c


def class_w_avg(c_vec, n=512, seed=0):
    """(mean, spread) of W for this class: the projector's starting point and its prior"""
    torch = _torch()
    G = S["G"]
    key = tuple(np.round(c_vec.detach().cpu().numpy().ravel(), 4).tolist())
    if key in _W_AVG_CACHE:
        return _W_AVG_CACHE[key]
    rng = np.random.RandomState(seed)
    ws = []
    for start in range(0, n, 128):
        b = min(128, n - start)
        z = torch.from_numpy(rng.randn(b, G.z_dim).astype(np.float32)).to(S["device"])
        with torch.no_grad():
            ws.append(G.mapping(z, c_vec.repeat(b, 1), truncation_psi=1.0))
    W = torch.cat(ws)
    w_avg = W.mean(0, keepdim=True)                 # [1, num_ws, w_dim]
    w_std = float((W - w_avg).pow(2).sum() / len(W)) ** 0.5
    _W_AVG_CACHE[key] = (w_avg, w_std)
    return _W_AVG_CACHE[key]


def inpaint(target: np.ndarray, fill: np.ndarray, c_vec, steps: int, lr: float,
            w_reg: float, seam: float, feather: int, focus: int = 12,
            context: float = 1.0, report=None):
    """target: uint8 [4, H, W]; fill: uint8 [H, W], 1 where the model should invent."""
    torch = _torch()
    import torch.nn.functional as F
    G = S["G"]
    dev = S["device"]
    t = torch.from_numpy(target.astype(np.float32) / 127.5 - 1.0).to(dev)[None]
    f = torch.from_numpy(fill.astype(np.float32)).to(dev)[None, None]
    keep = 1.0 - f
    # context weight: a smooth halo around the hole, so the fit cares most about the
    # street and block edges the fill has to meet and is free further away
    r = max(int(focus), 1)
    near = F.avg_pool2d(F.avg_pool2d(f, 2 * r + 1, 1, r), 2 * r + 1, 1, r)
    near = near / near.max().clamp(min=1e-6)
    wmap = keep * (context + seam * near) * torch.tensor(CH_WEIGHTS, device=dev).view(1, 4, 1, 1)
    wt = wmap.sqrt()

    w_avg, w_std = class_w_avg(c_vec)
    w = w_avg.clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr, betas=(0.9, 0.999))
    best_loss, best_w = float("inf"), w_avg.detach().clone()
    hist = []
    for i in range(steps):
        # the schedule from NVIDIA's projector: noise on w early to get off the flat
        # starting point, learning rate ramped up briefly then down to a crawl
        p = i / max(steps - 1, 1)
        noise = w_std * 0.05 * max(0.0, 1.0 - p / 0.75) ** 2
        ramp = 0.5 - 0.5 * math.cos(math.pi * min(1.0, (1.0 - p) / 0.25))
        for g in opt.param_groups:
            g["lr"] = lr * ramp * min(1.0, p / 0.05 + 1e-6)
        ws = w + torch.randn_like(w) * noise
        img = G.synthesis(ws, noise_mode="const")
        err = (img - t) * wt
        fit = err.pow(2).mean()
        for k in (2, 4, 8):                                     # match shapes before pixels
            fit = fit + F.avg_pool2d(err, k).pow(2).mean()
        reg = (w - w_avg).pow(2).mean()
        total = fit + w_reg * reg
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        fl = float(fit.detach())
        if fl < best_loss:
            best_loss, best_w = fl, w.detach().clone()
        if i % max(steps // 20, 1) == 0 or i == steps - 1:
            hist.append(round(fl, 5))
            if report:
                report(i, fl)

    with torch.no_grad():
        img = G.synthesis(best_w, noise_mode="const")   # noqa: F841
        gen = ((img.clamp(-1, 1) + 1) * 127.5)
        a = f
        if feather > 0:                                         # smooth the seam
            r = int(feather)
            a = F.max_pool2d(a, 2 * r + 1, 1, r)
            a = F.avg_pool2d(a, 2 * r + 1, 1, r)
            a = F.avg_pool2d(a, 2 * r + 1, 1, r)
        orig = torch.from_numpy(target.astype(np.float32)).to(dev)[None]
        out = orig * (1 - a) + gen * a
        out = out.round().clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        raw = gen.round().clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    return out, raw, {"steps": steps, "loss": round(best_loss, 5), "loss_history": hist,
                      "filled_pct": round(100.0 * float(fill.mean()), 2)}


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    """The app: the models, the area you select, and the design space export."""
    return send_from_directory(str(HERE), "index.html")


@app.get("/lab")
def lab():
    """The same app. Kept so the /lab bookmark still works."""
    return send_from_directory(str(HERE), "index.html")


@app.get("/vendor/<path:name>")
def vendor(name):
    return send_from_directory(str(HERE / "vendor"), name)


@app.get("/regular.js")
def regular_js():
    # the mesh regulariser lives in its own module so phase 5 can reuse it
    return send_from_directory(str(HERE), "regular.js", mimetype="text/javascript")


@app.get("/api/state")
def api_state():
    return jsonify(state_payload())


@app.get("/api/snapshots")
def api_snapshots():
    return jsonify({"snapshots": list_snapshots()})


@app.post("/api/load")
def api_load():
    path = (request.get_json(force=True) or {}).get("path")
    if not path:
        return jsonify({"error": "path required"}), 400
    try:
        load_model(path)
    except Exception as e:  # noqa
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    return jsonify(state_payload())


@app.post("/api/generate")
def api_generate():
    if S["G"] is None:
        return jsonify({"error": "no model loaded"}), 409
    p = request.get_json(force=True) or {}
    t0 = time.time()
    with LOCK:
        ch, w, achieved, iters, converged = generate(p)
    t1 = time.time()
    m = measure(ch)
    C, H, W = ch.shape
    return jsonify({
        "C": int(C), "H": int(H), "W": int(W),
        "channels_b64": base64.b64encode(np.ascontiguousarray(ch).tobytes()).decode("ascii"),
        "metrics": {k: (round(float(v), 3) if isinstance(v, float) else int(v)) for k, v in m.items()},
        "achieved": {k: round(float(v), 3) for k, v in achieved.items()},
        "target_iters": int(iters), "target_converged": bool(converged),
        "w_norm": float(np.linalg.norm(w)),
        "ms_infer": round((t1 - t0) * 1000, 1),
    })


@app.get("/api/real")
def api_real():
    """A real tile from --images, encoded exactly like the training data, in the generate() payload shape."""
    if not S["images"]:
        return jsonify({"error": "server started without --images"}), 409
    city = request.args.get("city", "")
    idx = request.args.get("idx", "")
    p = Path(S["images"]) / city / f"{city}_{idx}.png"
    if not p.exists():
        return jsonify({"error": f"{p.name} not found"}), 404
    t0 = time.time()
    rgb = load_rgb(p)
    size = int(S["G"].img_resolution) if S["G"] is not None else 0
    ch = encode_tile(rgb, S["lut"], S["encoding"], size, 8.0, 1.5, S["empty_height"])
    m = measure(ch)
    C, H, W = ch.shape
    return jsonify({
        "C": int(C), "H": int(H), "W": int(W),
        "channels_b64": base64.b64encode(np.ascontiguousarray(ch).tobytes()).decode("ascii"),
        "metrics": {k: (round(float(v), 3) if isinstance(v, float) else int(v)) for k, v in m.items()},
        "achieved": {}, "target_iters": 0, "target_converged": True, "w_norm": 0.0,
        "ms_infer": round((time.time() - t0) * 1000, 1), "real": f"{city}_{idx}",
    })


@app.post("/api/inpaint")
def api_inpaint():
    """Erase the masked area and let the generator rebuild it from what is left."""
    if S["G"] is None:
        return jsonify({"error": "no model loaded"}), 409
    p = request.get_json(force=True) or {}
    res = int(S["G"].img_resolution)
    ch_n = int(S["G"].img_channels)
    try:
        target = np.frombuffer(base64.b64decode(p["channels_b64"]), np.uint8).reshape(ch_n, res, res).copy()
        fill = np.frombuffer(base64.b64decode(p["mask_b64"]), np.uint8).reshape(res, res)
    except Exception as e:
        return jsonify({"error": f"bad tile or mask: {e}"}), 400
    fill = (fill > 0).astype(np.uint8)
    if not fill.any():
        return jsonify({"error": "nothing masked"}), 400

    steps = max(10, min(int(p.get("steps", S["inpaint_steps"])), 2000))
    t0 = time.time()
    with LOCK:
        out, raw, info = inpaint(
            target, fill, class_vec(p), steps,
            float(p.get("lr", 0.1)), float(p.get("w_reg", 0.0)),
            float(p.get("seam", 3.0)), int(p.get("feather", 2)), int(p.get("focus", 12)),
            float(p.get("context", 1.0)))
    m = measure(out)
    info["ms"] = round((time.time() - t0) * 1000)
    body = {
        "C": ch_n, "H": res, "W": res,
        "channels_b64": base64.b64encode(np.ascontiguousarray(out).tobytes()).decode("ascii"),
        "raw_b64": base64.b64encode(np.ascontiguousarray(raw).tobytes()).decode("ascii"),
        "metrics": {k: (round(float(v), 3) if isinstance(v, float) else int(v)) for k, v in m.items()},
        "achieved": {}, "target_iters": 0, "target_converged": True, "w_norm": 0.0,
        "ms_infer": info["ms"], "inpaint": info,
    }
    return jsonify(body)


# ---------------------------------------------------------------------------
# vector geometry (vectorize.py): the Vector geometry mode and the GeoJSON export
# ---------------------------------------------------------------------------

VEC_OPTION_KEYS = {
    "axis", "min_building_m2", "min_hole_m2", "split_necks", "neck_ratio", "hstep_m", "chamfer_min_m",
    "line_sep_m", "line_min_m", "max_gap_m", "open_px", "simplify_px",
    "spur_m", "isolated_m", "merge_m", "street_eps_px", "street_snap_deg", "street_snap45",
    "width_min_m", "width_max_m", "min_green_m2", "green_eps_px", "face_m2",
}


def _vectorize_request(p: dict):
    import vectorize
    H = int(p["H"]); W = int(p["W"]); C = int(p.get("C", 4))
    ch = np.frombuffer(base64.b64decode(p["channels_b64"]), np.uint8).reshape(C, H, W).copy()
    tile_m = float(p.get("tile_m") or S["tile_m"])
    opts = {k: v for k, v in (p.get("options") or {}).items() if k in VEC_OPTION_KEYS}
    t0 = time.time()
    res = vectorize.vectorize(ch, S["lut"], tile_m, **opts)
    res["ms"] = round((time.time() - t0) * 1000)
    if p.get("roundtrip", True):
        rt = vectorize.roundtrip(ch, res, None, tile_m)
        res["roundtrip"] = {k: round(v, 3) for k, v in rt.items() if k.startswith("iou")}
    return res


@app.post("/api/vectorize")
def api_vectorize():
    """Fit clean polygons, street centrelines with widths, plates, blocks and greens to a tile."""
    p = request.get_json(force=True) or {}
    try:
        return jsonify(_vectorize_request(p))
    except ImportError as e:
        return jsonify({"error": f"vectorize needs shapely (pip install shapely): {e}"}), 500
    except Exception as e:  # noqa
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


def _export_request(p: dict):
    """the fit plus what the page shows around it: real OSM geometry (`real`) outside the rebuilt `rect`"""
    p["roundtrip"] = False
    res = _vectorize_request(p)
    real = p.get("real") or None
    rect = p.get("rect") or None
    if rect is not None:
        rect = [float(v) for v in rect]
    name = (p.get("name") or "tile").replace("/", "_").replace("\\", "_")
    return res, real, rect, name


@app.post("/api/vectorize/geojson")
def api_vectorize_geojson():
    """The same result as GeoJSON in local metres (x east, y north, origin at the tile centre), as a download."""
    import vectorize
    p = request.get_json(force=True) or {}
    try:
        res, real, rect, name = _export_request(p)
        body = json.dumps(vectorize.to_geojson(res, real, rect))
    except Exception as e:  # noqa
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    return app.response_class(body, mimetype="application/geo+json",
                              headers={"Content-Disposition": f'attachment; filename="{name}.geojson"'})


@app.post("/api/vectorize/3dm")
def api_vectorize_3dm():
    """A Rhino .3dm: building solids with courtyards, outlines, streets with widths, plates, blocks, greens, on layers, in metres."""
    import vectorize
    p = request.get_json(force=True) or {}
    try:
        res, real, rect, name = _export_request(p)
        data = vectorize.to_3dm(res, real, rect, p.get("anchor") or None, name)
    except ImportError as e:
        return jsonify({"error": f"the 3dm export needs the rhino3dm package (pip install rhino3dm): {e}"}), 500
    except Exception as e:  # noqa
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    return app.response_class(data, mimetype="application/octet-stream",
                              headers={"Content-Disposition": f'attachment; filename="{name}.3dm"'})


# ---------------------------------------------------------------------------
# the design space export (export_space.py)
# ---------------------------------------------------------------------------
#
# The page sends the controls it is showing plus a range and a step count for the
# ones to sweep, and the canvas and painted area it is working on. Everything else
# happens on a background thread here: generate, fit, measure, write. The result is
# the folder the multiobjective viewer reads, plus a zip of it for the browser.


def export_generate(engine: str, body: dict, canvas: dict) -> np.ndarray:
    """one option's tile, from whichever model the viewer has loaded"""
    if engine == "dif":
        import dif
        if dif.D["model"] is None:
            raise RuntimeError("no inpainter loaded")
        ch, H, W = canvas_get(canvas)
        mask = np.frombuffer(base64.b64decode(canvas["mask_b64"]), np.uint8).reshape(H, W)
        targets = {k: (None if v is None else float(v)) for k, v in (body.get("targets") or {}).items()}
        with LOCK:
            out, _ = dif.fill(
                ch, mask, float(canvas.get("tile_m") or S["tile_m"]), targets,
                max(4, min(int(body.get("steps", 40)), 250)), float(body.get("cfg", 2.0)),
                max(0, min(int(body.get("resample", 0)), 4)), int(body.get("seed", 0)),
                bool(body.get("keep_streets")), int(body.get("stride", 64)),
                guide_strength=max(0.0, min(float(body.get("guide", 0.0)), 2.0)),
                best_of=max(1, min(int(body.get("best_of", 1)), 32)))
        return out
    if S["G"] is None:
        raise RuntimeError("no model loaded")
    with LOCK:
        ch, _w, _a, _i, _c = generate(body)
    return ch


@app.post("/api/export/start")
def api_export_start():
    """Walk the grid and write one Rhino model per option, a context model and the two CSVs."""
    import export_space
    p = request.get_json(force=True) or {}
    try:
        p.setdefault("tile_m", S["tile_m"])
        p["vectorize"] = {k: v for k, v in (p.get("vectorize") or {}).items() if k in VEC_OPTION_KEYS}
        root = Path(S["runs"] or ".")
        return jsonify(export_space.start(p, export_generate, S["lut"], root))
    except Exception as e:  # noqa
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 400


@app.get("/api/export/progress")
def api_export_progress():
    import export_space
    return jsonify(export_space.progress())


@app.post("/api/export/cancel")
def api_export_cancel():
    import export_space
    return jsonify(export_space.cancel())


@app.get("/api/export/zip")
def api_export_zip():
    """The finished folder as one download."""
    import export_space
    j = export_space.progress()
    z = Path(j.get("zip") or "")
    if not z.is_file():
        return jsonify({"error": "nothing to download yet"}), 404
    return send_from_directory(str(z.parent), z.name, as_attachment=True)


# ---------------------------------------------------------------------------
# the diffusion inpainter and the OSM loader (dif.py)
# ---------------------------------------------------------------------------

@app.get("/api/dif/state")
def api_dif_state():
    import dif
    return jsonify(dif.state())


@app.get("/api/dif/progress")
def api_dif_progress():
    import dif
    return jsonify(dif.PROGRESS)


@app.post("/api/dif/cancel")
def api_dif_cancel():
    """Stop whatever long thing is running: today that is the OpenStreetMap download."""
    import dif
    dif.CANCEL["flag"] = True
    return jsonify({"ok": True})


@app.post("/api/dif/load")
def api_dif_load():
    import dif
    path = (request.get_json(force=True) or {}).get("path")
    if not path:
        return jsonify({"error": "path required"}), 400
    try:
        with LOCK:
            return jsonify(dif.load_inpainter(path, S["device"]))
    except Exception as e:  # noqa
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.post("/api/dif/fill")
def api_dif_fill():
    """Rebuild the masked part of whatever the viewer is showing with the diffusion model."""
    import dif
    if dif.D["model"] is None:
        return jsonify({"error": "no inpainter loaded"}), 409
    p = request.get_json(force=True) or {}
    try:
        ch, H, W = canvas_get(p)
        mask = np.frombuffer(base64.b64decode(p["mask_b64"]), np.uint8).reshape(H, W)
    except Exception as e:
        return jsonify({"error": f"bad tile or mask: {e}"}), 400
    targets = {k: (None if v is None else float(v)) for k, v in (p.get("targets") or {}).items()}
    try:
        with LOCK:
            out, info = dif.fill(
                ch, mask, float(p.get("tile_m") or S["tile_m"]), targets,
                max(4, min(int(p.get("steps", 40)), 250)), float(p.get("cfg", 2.0)),
                max(0, min(int(p.get("resample", 0)), 4)), int(p.get("seed", 0)),
                bool(p.get("keep_streets")), int(p.get("stride", 64)),
                guide_strength=max(0.0, min(float(p.get("guide", 0.0)), 2.0)),
                best_of=max(1, min(int(p.get("best_of", 1)), 32)))
    except Exception as e:  # noqa
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    m = measure(out, float(p.get("tile_m") or S["tile_m"]))
    return jsonify({
        "C": 4, "H": H, "W": W, "canvas_id": canvas_put(out),
        "channels_b64": base64.b64encode(np.ascontiguousarray(out).tobytes()).decode("ascii"),
        "metrics": {k: (round(float(v), 3) if isinstance(v, float) else int(v)) for k, v in m.items()},
        "achieved": {}, "target_iters": 0, "target_converged": True, "w_norm": 0.0,
        "ms_infer": info["ms"], "dif": info, "tile_m": float(p.get("tile_m") or S["tile_m"]),
    })


@app.post("/api/dif/probe")
def api_dif_probe():
    """Two diagnostics on the same mask: is the condition wired, and what can the model do
    here when nothing is asked of it."""
    import dif
    if dif.D["model"] is None:
        return jsonify({"error": "no inpainter loaded"}), 409
    p = request.get_json(force=True) or {}
    try:
        ch, H, W = canvas_get(p)
        mask = np.frombuffer(base64.b64decode(p["mask_b64"]), np.uint8).reshape(H, W)
    except Exception as e:
        return jsonify({"error": f"bad tile or mask: {e}"}), 400
    targets = {k: (None if v is None else float(v)) for k, v in (p.get("targets") or {}).items()}
    try:
        with LOCK:
            out = dif.probe(
                ch, mask, float(p.get("tile_m") or S["tile_m"]), targets,
                max(4, min(int(p.get("steps", 30)), 120)), float(p.get("cfg", 2.0)),
                max(2, min(int(p.get("n", 12)), 32)), int(p.get("seed", 0)),
                bool(p.get("keep_streets")), p.get("metric"))
    except Exception as e:  # noqa
        dif.PROGRESS.update(what="", done=0, total=0)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    return jsonify(out)


# The working context is a megabyte and a half of channels. The browser holds it for the 3D
# view, but it has no business posting it back on every selection change, so the server keeps
# the last few canvases it produced and the browser refers to one by id.
CANVAS = {}
CANVAS_SEQ = [0]


def canvas_put(ch: np.ndarray) -> str:
    CANVAS_SEQ[0] += 1
    cid = f"c{CANVAS_SEQ[0]}"
    CANVAS[cid] = np.ascontiguousarray(ch)
    for k in sorted(CANVAS, key=lambda k: int(k[1:]))[:-6]:      # keep the last six
        CANVAS.pop(k, None)
    return cid


def canvas_get(p):
    """The canvas this request is about: by id if the browser has one, else the bytes it sent."""
    cid = p.get("canvas_id")
    if cid and cid in CANVAS:
        ch = CANVAS[cid]
        return ch.copy(), int(ch.shape[1]), int(ch.shape[2])
    H, W = int(p["H"]), int(p["W"])
    ch = np.frombuffer(base64.b64decode(p["channels_b64"]), np.uint8).reshape(4, H, W).copy()
    return ch, H, W


@app.post("/api/dif/measure")
def api_dif_measure():
    """The six metrics of whatever is selected. This is what the sliders rest at."""
    import dif
    if dif.D["model"] is None:
        return jsonify({"error": "no inpainter loaded"}), 409
    p = request.get_json(force=True) or {}
    try:
        ch, H, W = canvas_get(p)
        mask = (np.frombuffer(base64.b64decode(p["mask_b64"]), np.uint8).reshape(H, W)
                if p.get("mask_b64") else None)
    except Exception as e:
        return jsonify({"error": f"bad tile or mask: {e}"}), 400
    tile_m = float(p.get("tile_m") or S["tile_m"])
    try:
        with LOCK:
            out = {"metrics": dif.measure(ch, tile_m, mask)}
            if mask is not None and (mask > 0).any():
                res_m = tile_m / H
                try:
                    wx, wy, wpx, scale = dif.choose_window(mask, res_m)
                    ys, xs = np.nonzero(mask > 0)
                    # The window always sits tight around the selection by design, so its own
                    # margin is not news. What is worth saying is when the window could not be
                    # centred because it ran into the edge of what was loaded: compare where it
                    # wanted to be against where it ended up.
                    want_x = int(round((xs.min() + xs.max() + 1) / 2.0 - wpx / 2.0))
                    want_y = int(round((ys.min() + ys.max() + 1) / 2.0 - wpx / 2.0))
                    shift = max(abs(wx - want_x), abs(wy - want_y))
                    out["window"] = {"x": wx, "y": wy, "px": wpx, "scale": round(scale, 3),
                                     "m": round(wpx * res_m),
                                     "clamped_m": round(shift * res_m) if shift else 0}
                except ValueError as e:
                    out["window_error"] = str(e)
                ys, xs = np.nonzero(mask > 0)
                out["selection"] = {
                    "m": round(float(max(np.ptp(ys), np.ptp(xs)) + 1) * res_m),
                    "ha": round(float((mask > 0).sum()) * res_m * res_m / 1e4, 2)}
    except Exception as e:  # noqa
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    return jsonify(out)


@app.get("/api/osm/geocode")
def api_osm_geocode():
    import dif
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"error": "q required"}), 400
    try:
        return jsonify({"hits": dif.geocode(q)})
    except Exception as e:  # noqa
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 502


@app.post("/api/osm/fetch")
def api_osm_fetch():
    """OpenStreetMap around a point, rendered and encoded exactly like the training data."""
    import dif
    p = request.get_json(force=True) or {}
    try:
        lat, lon = float(p["lat"]), float(p["lon"])
    except Exception as e:
        return jsonify({"error": f"lat and lon required: {e}"}), 400
    # a working context, not a model window. You look at real city at the corpus' own
    # resolution and select a plot inside it; dif.choose_window crops the one window the
    # model reads around that selection when you press Fill.
    extent = float(p.get("extent_m") or S["context_m"])
    extent = max(600.0, min(extent, 4000.0))
    res_m = float(dif.D["res_m"] or 3.125)
    try:
        rgb, info, vec = dif.fetch_canvas(lat, lon, extent, res_m, S["osm_contact"],
                                          Path(S["osm_cache"]) if S["osm_cache"] else None)
    except Exception as e:  # noqa
        dif.PROGRESS.update(what="", done=0, total=0, error=str(e))
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 502
    extent = float(info["extent_m"])
    ch = encode_tile(rgb, S["lut"], "sdf", 0, 8.0, 1.5, S["empty_height"])
    m = measure(ch, extent)
    # the GAN readout and the inpainter's six metrics are different quantities measured
    # differently; the diffusion sliders need the second set, so send both.
    try:
        dm = dif.measure(ch, float(extent))
    except Exception:  # noqa - a missing checkpoint must not cost us the fetch
        dm = {}
    return jsonify({
        "C": int(ch.shape[0]), "H": int(ch.shape[1]), "W": int(ch.shape[2]),
        "canvas_id": canvas_put(ch),
        "channels_b64": base64.b64encode(np.ascontiguousarray(ch).tobytes()).decode("ascii"),
        "metrics": {k: (round(float(v), 3) if isinstance(v, float) else int(v)) for k, v in m.items()},
        "dif_metrics": dm,
        "achieved": {}, "target_iters": 0, "target_converged": True, "w_norm": 0.0,
        "ms_infer": 0, "osm": info, "tile_m": float(extent), "vector": vec,
    })


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="path to the patched stylegan3 clone")
    ap.add_argument("--runs", default=None, help="folder to scan for snapshots (default <repo>/runs)")
    ap.add_argument("--network", default=None, help="snapshot to load at start")
    ap.add_argument("--lut", required=True, help="assets/height_lut.json, or the one written next to your dataset")
    ap.add_argument("--cities", default=None, help="assets/cities.json, or the one written next to your dataset")
    ap.add_argument("--directions", default=None, help="assets/directions.npz, or your own from fit_directions.py")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--pca", type=int, default=8, help="number of PCA sliders")
    ap.add_argument("--pca-samples", type=int, default=4000)
    ap.add_argument("--tile-m", type=float, default=400.0, help="tile edge in metres, for the 3D scale and metrics")
    ap.add_argument("--target-iters", type=int, default=4, help="max Newton steps when solving for absolute targets")
    ap.add_argument("--target-tol", type=float, default=0.03, help="relative tolerance for a target to count as hit")
    ap.add_argument("--images", default=None,
                    help="raw tile folder (images_all) to offer real OSM tiles as references in the viewer")
    ap.add_argument("--encoding", default=None, help="mask or sdf for --images (default: read dataset/encoding.json next to --lut, else sdf)")
    ap.add_argument("--inpaint-steps", type=int, default=400,
                    help="default optimisation steps for the inpaint panel (a 4070 does ~50/s at 128 px)")
    ap.add_argument("--context-m", type=float, default=800.0,
                    help="metres of OpenStreetMap to load around the address (default 800, which is "
                         "the largest window the model reads, so the load is one window)")
    ap.add_argument("--inpaint", default=None,
                    help="phase 6 checkpoint, or a folder of them, to enable the diffusion inpainter panel")
    ap.add_argument("--osm-contact", default="urban-opengen@example.com",
                    help="email for the OpenStreetMap User-Agent, required by their policy when loading an address")
    ap.add_argument("--osm-cache", default=None, help="folder for cached Overpass responses (default <phase4>/osm_cache)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    torch = _torch()
    S["repo"] = str(Path(args.repo).resolve())
    S["runs"] = str(Path(args.runs).resolve()) if args.runs else os.path.join(S["repo"], "runs")
    S["device"] = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    S["lut"] = HeightLUT.load(args.lut)
    S["pca_n"] = args.pca
    S["pca_samples"] = args.pca_samples
    S["tile_m"] = args.tile_m
    S["target_iters"] = args.target_iters
    S["target_tol"] = args.target_tol
    S["inpaint_steps"] = args.inpaint_steps
    S["context_m"] = float(args.context_m)
    sys.path.insert(0, S["repo"])
    if args.cities:
        S["cities"] = json.loads(Path(args.cities).read_text())["names"]
    if args.images:
        S["images"] = str(Path(args.images).resolve())
        enc_json = Path(args.lut).parent / "encoding.json"
        if enc_json.exists():
            enc = json.loads(enc_json.read_text())
            S["encoding"] = enc.get("encoding", "sdf")
            S["empty_height"] = int(enc.get("empty_tile_height_byte") or 128)
        if args.encoding:
            S["encoding"] = args.encoding
        print(f"real tiles from {S['images']} ({S['encoding']} encoding)")
    import dif
    S["osm_contact"] = args.osm_contact
    S["osm_cache"] = str(Path(args.osm_cache).resolve()) if args.osm_cache else str(HERE / "osm_cache")
    dif.D["contact"] = S["osm_contact"]
    dif.D["cache"] = S["osm_cache"]
    dif.D["host_lut"] = S["lut"]
    if args.inpaint:
        S["dif_root"] = str(Path(args.inpaint).resolve())
        dif.D["root"] = S["dif_root"]
        dif.D["device"] = S["device"]
        cks = dif.list_checkpoints(S["dif_root"])
        print(f"inpainter checkpoints under {S['dif_root']}: {len(cks)}")
        # the diffusion panel is the app, so it should be usable the moment the page opens:
        # open the newest checkpoint here rather than making the first click a Load. Two things
        # can go wrong when a run is still going, and both are ordinary rather than exceptional.
        # The newest file may be a latest.pt that train.py is in the middle of writing, which
        # reads as a truncated archive; and a checkpoint may simply be broken. So skip anything
        # touched in the last half minute and fall through to the next one instead of giving up.
        fresh = 30.0
        for c in cks:
            age = time.time() - float(c.get("mtime") or 0)
            if age < fresh and len(cks) > 1:
                print(f"skipping {c['name']}: written {age:.0f} s ago, probably still being saved")
                continue
            try:
                dif.load_inpainter(c["path"], S["device"])
                break
            except Exception as e:  # noqa - a bad checkpoint must not stop the server
                print(f"could not open {c['name']}: {type(e).__name__}: {e}")
    if args.directions:
        load_directions(args.directions)
    if args.network:
        load_model(args.network)
    else:
        snaps = list_snapshots()
        print(f"{len(snaps)} snapshots under {S['runs']}; pick one in the UI")

    print(f"device {S['device']}  ->  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
