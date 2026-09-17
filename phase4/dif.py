"""
dif.py

Phase 4: the diffusion inpainter and the OpenStreetMap loader behind the viewer's
Inpainter panel. Kept out of serve.py because neither needs the GAN: the model here is a
phase 6 checkpoint, and the canvas it works on can come from the GAN, from a real tile, or
straight from OSM at any address.

Two halves:

  the model    load_inpainter() opens a phase 6 checkpoint (ckpt_*.pt / latest.pt), fill()
               takes the tile the viewer is showing plus a painted mask and rebuilds the
               masked part, conditioned on whichever of the six metrics the user asked for.
               Canvases larger than the model's window are stitched with MultiDiffusion,
               the same path sample.py uses.

  the place    geocode() resolves an address with Nominatim, fetch_canvas() pulls the OSM
               features around it with the phase 0 rasteriser and renders them at the
               training conventions, so what the model sees from a real address is encoded
               exactly like what it was trained on. Progress is published in PROGRESS so
               the page can show it while Overpass takes its time.

Nothing here touches the GAN state in serve.py, and nothing is imported at module load:
torch, requests and shapely are only needed once the panel is actually used.
"""
from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent / "phase0", HERE.parent / "phase6"):
    if cand.exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

# the six the inpainter conditions on, in the order the checkpoint stores them
METRIC_UI = {
    "coverage": {"label": "coverage", "unit": "%", "scale": 100, "min": 0, "max": 80, "step": 1, "dec": 0},
    "green": {"label": "green", "unit": "%", "scale": 100, "min": 0, "max": 60, "step": 1, "dec": 0},
    "street": {"label": "street", "unit": "%", "scale": 100, "min": 0, "max": 40, "step": 1, "dec": 0},
    "height_m": {"label": "mean height", "unit": "m", "scale": 1, "min": 3, "max": 120, "step": 1, "dec": 0},
    "far": {"label": "FAR", "unit": "", "scale": 1, "min": 0, "max": 8, "step": 0.05, "dec": 2},
    "bsize_m": {"label": "block size", "unit": "m", "scale": 1, "min": 8, "max": 120, "step": 1, "dec": 0},
}

D: dict = {
    "path": None, "model": None, "lut": None, "norm": None, "ck": None, "host_lut": None,
    "to_model": None, "to_host": None,
    "device": None, "size": 128, "scale_lo": 1.0, "scale_hi": 1.0, "step": 0, "params": 0.0,
    "res_m": 3.125, "root": None, "contact": "urban-opengen@example.com", "cache": None,
}
PROGRESS = {"what": "", "done": 0, "total": 0, "t0": 0.0, "error": ""}
# Stop, for the parts of a request that run long enough to be worth stopping. A download from
# Overpass can take a minute on a busy day and there was no way to call it off: the page could
# stop waiting, but the server carried on and the next request queued behind it.
CANCEL = {"flag": False}


def cancelled():
    return bool(CANCEL["flag"])
_LOCK = threading.Lock()


def _torch():
    import torch
    return torch


def say(what, done=0, total=0):
    PROGRESS.update(what=what, done=done, total=total, error="")
    if not PROGRESS["t0"]:
        PROGRESS["t0"] = time.time()
    print(f"  {what} {done}/{total}" if total else f"  {what}", flush=True)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

def list_checkpoints(root):
    """[{path, name, step_hint}] for every *.pt under --inpaint, newest first."""
    if not root:
        return []
    root = Path(root)
    if root.is_file():
        return [{"path": str(root), "name": root.name}]
    out = []
    for p in sorted(root.rglob("*.pt")):
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            rel = p.name
        out.append({"path": str(p), "name": rel, "mtime": p.stat().st_mtime})
    out.sort(key=lambda d: (-d.get("mtime", 0),))
    return out


def load_inpainter(path: str, device):
    """Open a phase 6 checkpoint and keep its EMA weights ready to sample from."""
    torch = _torch()
    from sample import load_checkpoint
    model, lut, norm, ck = load_checkpoint(str(path), device)
    model.grad_ckpt = False
    D.update(path=str(path), model=model, lut=lut, norm=norm, ck=ck, device=device,
             size=int(ck.get("size", 128)),
             scale_lo=float(ck["args"].get("scale_lo", 1.0)), scale_hi=float(ck["args"].get("scale_hi", 1.0)),
             step=int(ck.get("step", 0)), params=round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
             res_m=float(ck.get("res_m_per_px", 3.125)))
    height_bridge()
    print(f"inpainter: {Path(path).name}, {D['params']} M params, step {D['step']}, "
          f"{D['size']} px windows at scale {D['scale_lo']}..{D['scale_hi']} "
          f"({D['scale_lo'] * D['size'] * D['res_m']:.0f}..{D['scale_hi'] * D['size'] * D['res_m']:.0f} m)")
    return state()


def state():
    size, res = D["size"], D["res_m"]
    return {
        "loaded": D["model"] is not None,
        "path": D["path"], "name": Path(D["path"]).name if D["path"] else None,
        "step": D["step"], "params": D["params"], "size": size,
        "window_m": [round(D["scale_lo"] * size * res), round(D["scale_hi"] * size * res)],
        "res_m": res, "metrics": list(METRIC_UI), "metric_ui": METRIC_UI,
        "checkpoints": list_checkpoints(D["root"]),
        "osm": bool(D["contact"]),
    }


def height_bridge():
    """
    The viewer's tiles are encoded with the GAN dataset's height LUT; the checkpoint carries
    the LUT of the corpus it was trained on. Both are rank-normalised, so the same byte can
    mean a different number of metres in each. These two tables convert the height channel
    between them through the grey value they agree on, so nothing is measured with the wrong
    ruler. Identical LUTs give identity tables and the conversion is skipped.
    """
    host, model = D.get("host_lut"), D.get("lut")
    if host is None or model is None or list(host.inverse) == list(model.inverse):
        D["to_model"] = D["to_host"] = None
        return
    hi = np.asarray(host.inverse, np.int32)                  # host byte -> grey
    mi = np.asarray(model.inverse, np.int32)                 # model byte -> grey
    # nearest byte in the other table by the grey it decodes to: forward[] has holes where a
    # grey never occurred in that corpus, so going through it would collapse those to zero
    D["to_model"] = np.abs(mi[None, :] - hi[:, None]).argmin(1).astype(np.uint8)
    D["to_host"] = np.abs(hi[None, :] - mi[:, None]).argmin(1).astype(np.uint8)
    print("  height LUTs differ between the viewer and the checkpoint: converting the height channel both ways")


def bytes_to_fields(ch: np.ndarray, device):
    """uint8 [4,H,W] in the tile_codec sdf byte layout -> float [1,4,H,W] in [-1,1].

    The exact inverse of phase 6's fields_to_bytes, so a tile can go out to the viewer as
    bytes and come back without drifting."""
    torch = _torch()
    x = np.empty(ch.shape, np.float32)
    x[[0, 2, 3]] = (ch[[0, 2, 3]].astype(np.float32) - 128.0) / 127.0
    h = ch[1] if D["to_model"] is None else D["to_model"][ch[1]]
    x[1] = h.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(np.clip(x, -1, 1)).unsqueeze(0).to(device)


# metrics the guide can push on: the ones that are a smooth function of the fields. Block
# size is a perimeter ratio and its gradient is mostly noise, so it is left to the condition.
GUIDABLE = ("coverage", "green", "street", "height_m", "far")


def make_guide(targets: dict, mask, scale: float, byte_to_m, strength: float,
               from_frac: float = 0.15, to_frac: float = 0.92):
    """
    A guide for the samplers: measure the model's estimate of the finished tile, and push it
    toward the numbers that were asked for.

    Classifier-free guidance leans on the network having learned to use the condition. This
    does not: it differentiates the metric itself and takes a step in the direction that
    improves it, so a request moves the result whether or not the network was listening.

    The step is a Newton step on the squared error, `-(L / |G|^2) G`, which is the move that
    would close the whole gap if the metric were linear in the fields. `strength` is then the
    fraction of the remaining gap to close per step, so it means the same thing on every site
    and at every step, it shrinks to nothing once the target is met instead of oscillating
    around it, and a metric the fields cannot reach at all (asking for green where the
    gradient has saturated) is held back by the per-step cap rather than blowing up. It fades
    out near the end so the last steps are free to tidy the geometry rather than fight for a
    decimal place.

    Returns None when there is nothing to guide, so the caller can pass it straight through.
    """
    torch = _torch()
    from data import METRICS, soft_hole_metrics
    idx = [i for i, k in enumerate(METRICS)
           if k in GUIDABLE and targets.get(k) is not None]
    if not idx or strength <= 0:
        return None
    dev = D["device"]
    tgt = torch.tensor([float(targets[METRICS[i]]) for i in idx], device=dev)
    span = D["norm"].std.to(dev)[idx].clamp(min=1e-3)     # the model's own units, so the six compare
    keep = mask
    lr = min(1.0, float(strength))
    CAP = 0.2                                             # most a pixel may move in one step

    def guide(x0_hat, i, n):
        f = i / max(1, n - 1)
        if f < from_frac or f > to_frac:
            return x0_hat
        fade = min(1.0, (to_frac - f) / 0.25)             # ease out over the last quarter
        with torch.enable_grad(), torch.autocast(device_type=dev.type, enabled=False):
            z = x0_hat.detach().float().requires_grad_(True)
            got = soft_hole_metrics(z, keep, scale, byte_to_m, D["res_m"])[:, idx]
            per = (((got - tgt) / span) ** 2).sum(1)      # one loss per tile in the batch
            g, = torch.autograd.grad(per.sum(), z)
        g = g * keep
        gg = (g * g).flatten(1).sum(1).view(-1, 1, 1, 1).clamp(min=1e-12)
        step = (per.detach().view(-1, 1, 1, 1) / gg) * g  # closes the gap to first order
        return x0_hat - (lr * fade * step).clamp(-CAP, CAP) * keep

    return guide


def _score(got, asked: dict, norm):
    """
    Distance from a measurement to what was asked for, in the model's own units.

    Every metric is divided by the standard deviation the checkpoint carries for it, so a
    tenth of a FAR and a percent of green count the same, and asking for two things at once
    does not let the easy one drown the hard one.
    """
    torch = _torch()
    from data import METRICS
    idx = [i for i, k in enumerate(METRICS) if asked.get(k) is not None]
    if not idx:
        return torch.zeros(got.shape[0], device=got.device)
    tgt = torch.tensor([float(asked[METRICS[i]]) for i in idx], device=got.device)
    span = norm.std.to(got.device)[idx].clamp(min=1e-3)
    return (((got[:, idx] - tgt) / span) ** 2).sum(1).sqrt()


def choose_window(mask: np.ndarray, res_m: float, margin_frac: float = 0.22):
    """
    One model window around whatever is selected, chosen by the app.

    The canvas is now a large piece of city at the corpus resolution, and the model reads a
    single 128 px window somewhere inside it. This picks that window: the smallest the model
    was trained for that holds the selection with real fabric around it on all four sides,
    and never larger than the biggest it was trained for. If the selection does not fit in
    the biggest one, we say so rather than splitting it or stitching windows together.

    -> (x0, y0, wpx, scale) in canvas pixels, or raises ValueError
    """
    ys, xs = np.nonzero(mask > 0)
    if not len(xs):
        raise ValueError("nothing selected")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    size = D["size"]
    lo = max(1, int(round(size * D["scale_lo"])))
    hi = max(lo, int(round(size * D["scale_hi"])))
    need = int(math.ceil(max(bw, bh) * (1.0 + 2.0 * margin_frac)))
    if need > hi:
        span = max(bw, bh) * res_m
        raise ValueError(
            f"the selection is {span:.0f} m across and the model's largest window is "
            f"{hi * res_m:.0f} m, which leaves no room for context around it. "
            f"Select something under about {hi * res_m * (1 - 2 * margin_frac) / (1 + 0):.0f} m "
            f"across, or split it into plots and fill them one at a time.")
    wpx = int(min(hi, max(lo, need)))
    cx, cy = (x0 + x1 + 1) / 2.0, (y0 + y1 + 1) / 2.0
    H = mask.shape[0]
    if wpx > H:
        raise ValueError(f"the loaded area is {H * res_m:.0f} m, smaller than the "
                         f"{wpx * res_m:.0f} m window this selection needs. Load a larger area.")
    wx = int(round(cx - wpx / 2.0)); wy = int(round(cy - wpx / 2.0))
    wx = max(0, min(wx, mask.shape[1] - wpx))
    wy = max(0, min(wy, H - wpx))
    return wx, wy, wpx, (wpx * res_m) / (size * D["res_m"])


def fill(ch: np.ndarray, mask: np.ndarray, tile_m: float, targets: dict, steps: int, cfg: float,
         resample: int, seed: int, keep_streets: bool, stride: int = 64,
         guide_strength: float = 0.0, best_of: int = 1):
    """
    ch     uint8 [4,H,W] the tile the viewer is showing
    mask   uint8 [H,W], non-zero where the model has to generate
    tile_m metres across the whole canvas
    -> (uint8 [4,H,W] filled, info dict)

    Three ways to make a request bite, and they compose:
      cfg            classifier-free guidance, the network's own use of the condition
      guide_strength metric guidance, the sampler pushed by the measurement itself
      best_of        N seeds sampled and the one closest to the request kept
    """
    torch = _torch()
    from data import hole_metrics, targets_to_cond, METRICS, fields_to_bytes, byte_to_metres_table
    from diffusion import Schedule, sample
    import torch.nn.functional as F
    if D["model"] is None:
        raise RuntimeError("no inpainter loaded")
    dev = D["device"]
    size = D["size"]
    H, W = ch.shape[1:]
    if H != W:
        raise ValueError("the canvas has to be square")
    res_m = tile_m / H                                   # metres per canvas pixel
    x0 = bytes_to_fields(ch, dev)
    m = torch.from_numpy((mask > 0).astype(np.float32)).view(1, 1, H, W).to(dev)
    if keep_streets:
        m = m * (x0[:, 2:3] <= -1.5 / 8.0).float()       # leave the existing street band alone
    if not float(m.sum()):
        raise ValueError("nothing masked")
    byte_to_m = byte_to_metres_table(D["lut"])
    # the selection is measured at the canvas' own resolution, not the model's, so the
    # numbers under the sliders mean the same thing whatever window the app picked
    cscale = res_m / D["res_m"]
    truth = hole_metrics(x0, m, cscale, byte_to_m, D["res_m"])[0]
    asked = {k: v for k, v in (targets or {}).items() if v is not None}
    cond = targets_to_cond(asked, D["norm"], dev) if asked else None
    # One window, chosen around the selection. The canvas can be a kilometre and a half of
    # city; the model still reads a single tile of its own size, with the selection inside it
    # and real fabric on all four sides. Everything outside the window is untouched by
    # construction, and everything outside the mask is put back byte for byte below.
    wx, wy, wpx, scale = choose_window(mask, res_m)
    l2s = torch.full((1,), math.log2(max(scale, 1e-6)), device=dev)
    mw = m[:, :, wy:wy + wpx, wx:wx + wpx]               # the mask, at the canvas' resolution
    wctx = (x0 * (1 - m))[:, :, wy:wy + wpx, wx:wx + wpx]
    wm = mw
    if wpx != size:                                      # the window, at the model's resolution
        wctx = F.interpolate(wctx, size=(size, size), mode="bilinear", align_corners=False)
        wm = (F.interpolate(mw, size=(size, size), mode="area") > 0).float()
        wctx = wctx * (1 - wm)
    sched = Schedule(device=dev)
    amp = dev.type == "cuda"
    guide = make_guide(asked, wm, scale, byte_to_m, guide_strength)
    N = max(1, min(int(best_of), 32))
    t0 = time.time()

    def one(sd):
        """One sample. Each candidate gets its own seed so it can be reproduced on its own:
        setting `best of` back to 1 and typing that seed gives exactly this fill again."""
        gen = torch.Generator(device=dev).manual_seed(int(sd))
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            y = sample(D["model"], sched, wctx, wm, cond, l2s, steps=steps, cfg=cfg,
                       resample=resample, generator=gen, guide=guide).float()
            if wpx != size:                              # back to the canvas' own pixels
                y = F.interpolate(y, size=(wpx, wpx), mode="bilinear", align_corners=False)
            full = x0.clone()
            full[:, :, wy:wy + wpx, wx:wx + wpx] = (
                full[:, :, wy:wy + wpx, wx:wx + wpx] * (1 - mw) + y * mw)
            return full

    seeds = [int(seed) + k * 7919 for k in range(N)]
    outs = []
    for k, sd in enumerate(seeds):
        say(f"sampling {N} seed{'' if N == 1 else 's'}" if N > 1 else f"inpainting {H} px", k, N)
        outs.append(one(sd))
    ys = torch.cat(outs)

    got_all = hole_metrics(ys, m.expand(ys.shape[0], -1, -1, -1), cscale, byte_to_m, D["res_m"])
    pick = int(_score(got_all, asked, D["norm"]).argmin()) if (N > 1 and asked) else 0
    y = ys[pick:pick + 1]
    got = got_all[pick]
    out = fields_to_bytes(y)[0]
    if D["to_host"] is not None:
        out[1] = D["to_host"][out[1]]
    PROGRESS.update(what="", done=0, total=0)
    info = {
        "ms": round((time.time() - t0) * 1000),
        "canvas_px": int(H), "scale": round(float(scale), 3), "window_m": round(wpx * res_m),
        "window_px": [int(wx), int(wy), int(wpx)],
        "in_range": bool(D["scale_lo"] * 0.99 <= scale <= D["scale_hi"] * 1.01),
        "selection_m": round(float(max(np.ptp(np.nonzero(mask > 0)[0]), np.ptp(np.nonzero(mask > 0)[1])) + 1) * res_m),
        "selection_ha": round(float((mask > 0).sum()) * res_m * res_m / 10000.0, 2),
        "masked_pct": round(100 * float(mw.mean()), 1),   # of the window the model read
        "truth": {k: round(float(v), 4) for k, v in zip(METRICS, truth)},
        "achieved": {k: round(float(v), 4) for k, v in zip(METRICS, got)},
        "targets": {k: round(float(v), 4) for k, v in asked.items()},
        "steps": steps, "cfg": cfg, "resample": resample, "seed": seeds[pick],
        "guide": round(float(guide_strength), 3), "guided": [k for k in asked if k in GUIDABLE] if guide else [],
        "best_of": int(ys.shape[0]), "picked": pick,
        "candidates": [{"seed": seeds[j], "metrics": {k: round(float(v), 4) for k, v in zip(METRICS, got_all[j])}}
                       for j in range(ys.shape[0])] if N > 1 else [],
    }
    return out, info


def probe(ch: np.ndarray, mask: np.ndarray, tile_m: float, targets: dict, steps: int, cfg: float,
          n: int, seed: int, keep_streets: bool, metric: str | None = None):
    """
    Two questions the sliders cannot answer on their own, in one pass.

    `wired`   is the condition reaching the network at all? The same seed and the same
              context are sampled twice, once with one metric pinned low and once pinned
              high. If the two fills come back identical the condition is not being used,
              and no amount of guidance strength will change that: it is a wiring problem.

    `spread`  what can the model do here without being asked? `n` seeds are sampled with
              every metric withheld and all six measured on each. That is the natural range
              of this context. A target outside it is not something the model has ever
              produced in a place like this, so a slider that fails to reach it is the model
              being honest rather than deaf.
    """
    torch = _torch()
    from data import hole_metrics, targets_to_cond, METRICS, byte_to_metres_table
    from diffusion import Schedule, sample
    import torch.nn.functional as F
    if D["model"] is None:
        raise RuntimeError("no inpainter loaded")
    dev, size = D["device"], D["size"]
    H, W = ch.shape[1:]
    if H != W:
        raise ValueError("the canvas has to be square")
    res_m = tile_m / H
    x0 = bytes_to_fields(ch, dev)
    mfull = torch.from_numpy((mask > 0).astype(np.float32)).view(1, 1, H, W).to(dev)
    if keep_streets:
        mfull = mfull * (x0[:, 2:3] <= -1.5 / 8.0).float()
    if not float(mfull.sum()):
        raise ValueError("nothing masked")
    byte_to_m = byte_to_metres_table(D["lut"])
    # the same single window the fill would use, so the diagnosis is about the fill you get
    wx, wy, wpx, scale = choose_window(mask, res_m)
    m = mfull[:, :, wy:wy + wpx, wx:wx + wpx]
    ctx = (x0 * (1 - mfull))[:, :, wy:wy + wpx, wx:wx + wpx]
    if wpx != size:
        ctx = F.interpolate(ctx, size=(size, size), mode="bilinear", align_corners=False)
        m = (F.interpolate(m, size=(size, size), mode="area") > 0).float()
        ctx = ctx * (1 - m)
    sched = Schedule(device=dev)
    amp = dev.type == "cuda"
    l2s = torch.full((1,), math.log2(max(scale, 1e-6)), device=dev)
    N = max(2, min(int(n), 32))
    t0 = time.time()

    def run(cond, batch, sd):
        gen = torch.Generator(device=dev).manual_seed(int(sd))
        b = lambda t: None if t is None else t.expand(batch, *t.shape[1:])
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            return sample(D["model"], sched, b(ctx), b(m), b(cond), l2s.expand(batch),
                          steps=steps, cfg=cfg, generator=gen).float()

    # ---- is the condition doing anything?
    metric = metric if metric in METRIC_UI else next((k for k in (targets or {}) if targets[k] is not None), "far")
    ui = METRIC_UI[metric]
    lo_v, hi_v = float(ui["min"]) / ui["scale"], float(ui["max"]) / ui["scale"]
    say(f"asking for {metric} at both ends", 0, 2)
    lo = run(targets_to_cond({metric: lo_v}, D["norm"], dev), 1, seed)
    say(f"asking for {metric} at both ends", 1, 2)
    hi = run(targets_to_cond({metric: hi_v}, D["norm"], dev), 1, seed)
    mlo = hole_metrics(lo, m, scale, byte_to_m, D["res_m"])[0]
    mhi = hole_metrics(hi, m, scale, byte_to_m, D["res_m"])[0]
    i_m = METRICS.index(metric)
    diff_px = float(((lo - hi).abs() * m).sum() / m.sum().clamp(min=1) / 4)
    moved = float(abs(mhi[i_m] - mlo[i_m]))

    # ---- what does it do when nothing is asked?
    say(f"sampling {N} seeds with nothing asked for", 0, N)
    free = run(None, N, seed + 101)
    vals = hole_metrics(free, m.expand(N, -1, -1, -1), scale, byte_to_m, D["res_m"])
    q = lambda v, f: float(v.sort().values[min(len(v) - 1, max(0, int(round(f * (len(v) - 1)))))])
    spread = {k: {"min": round(q(vals[:, i], 0), 4), "p25": round(q(vals[:, i], .25), 4),
                  "median": round(q(vals[:, i], .5), 4), "p75": round(q(vals[:, i], .75), 4),
                  "max": round(q(vals[:, i], 1), 4), "mean": round(float(vals[:, i].mean()), 4)}
              for i, k in enumerate(METRICS)}
    PROGRESS.update(what="", done=0, total=0)
    span = float(D["norm"].std[i_m])
    return {
        "ms": round((time.time() - t0) * 1000), "n": N, "steps": steps, "cfg": cfg,
        "masked_pct": round(100 * float(m.mean()), 1),
        "wired": {
            "metric": metric, "label": ui["label"], "unit": ui["unit"], "scale": ui["scale"],
            "asked_low": round(lo_v, 4), "asked_high": round(hi_v, 4),
            "got_low": round(float(mlo[i_m]), 4), "got_high": round(float(mhi[i_m]), 4),
            "moved": round(moved, 4), "moved_sigma": round(moved / max(span, 1e-6), 3),
            "field_diff": round(diff_px, 5),
            "verdict": ("dead" if diff_px < 1e-5 else "weak" if moved / max(span, 1e-6) < 0.25 else "live"),
        },
        "spread": spread,
        "targets": {k: round(float(v), 4) for k, v in (targets or {}).items() if v is not None},
    }


def measure(ch: np.ndarray, tile_m: float, mask: np.ndarray | None = None):
    """
    The six inpainter metrics, over the selection if there is one and over the whole canvas
    if there is not. This is what the sliders rest at: what the thing you chose measures
    right now. Always computed at the canvas' own resolution, so it does not move when the
    app picks a different window.
    """
    torch = _torch()
    from data import hole_metrics, METRICS, byte_to_metres_table
    if D["model"] is None:
        return {}
    H = ch.shape[1]
    scale = (tile_m / H) / D["res_m"]
    x0 = bytes_to_fields(ch, D["device"])
    if mask is None:
        m = torch.ones(1, 1, H, ch.shape[2], device=D["device"])
    else:
        m = torch.from_numpy((mask > 0).astype(np.float32)).view(1, 1, H, ch.shape[2]).to(D["device"])
        if not float(m.sum()):
            return {}
    v = hole_metrics(x0, m, scale, byte_to_metres_table(D["lut"]), D["res_m"])[0]
    return {k: round(float(x), 4) for k, x in zip(METRICS, v)}


# ---------------------------------------------------------------------------
# the place
# ---------------------------------------------------------------------------

def geocode(query: str, limit: int = 5):
    """Nominatim search -> [{lat, lon, name}]. Their policy wants a real contact in the UA."""
    import requests
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params={"q": query, "format": "json", "limit": limit},
                     headers={"User-Agent": f"urban-opengen ({D['contact']})"}, timeout=30)
    r.raise_for_status()
    return [{"lat": float(h["lat"]), "lon": float(h["lon"]), "name": h.get("display_name", "")} for h in r.json()]


def vector_geometry(feats, size_m: float, simplify: float = 0.5, max_parts: int = 4000):
    """
    The OSM features as geometry, in local metres, x east, y north, centred on the window.

    The raster is what the model reads; this is what the viewer draws. Buildings keep their
    real outline and their real height instead of being read back off a 3.125 m grid, the
    blocks are the true complement of the street bands (so kerbs land where they land), and
    the street centrelines come from the ways themselves rather than from scanning pixels.
    """
    from shapely.geometry import box as shbox
    from shapely.ops import unary_union
    half = size_m / 2
    win = shbox(-half, -half, half, half)

    def rings(geom, min_area=3.0):
        out = []
        if geom.is_empty:
            return out
        gs = [geom] if geom.geom_type == "Polygon" else [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
        for g in gs:
            if g.is_empty or g.area < min_area:
                continue
            g = g.simplify(simplify)
            if g.is_empty or len(g.exterior.coords) < 4:
                continue
            out.append({
                "ring": [[round(x, 2), round(y, 2)] for x, y in g.exterior.coords[:-1]],
                "holes": [[[round(x, 2), round(y, 2)] for x, y in i.coords[:-1]] for i in g.interiors if i.length > 8],
            })
        return out

    buildings = []
    for poly, h in feats["buildings"]:
        for r in rings(poly.intersection(win), 6.0):
            buildings.append({**r, "h": round(float(h), 1)})
        if len(buildings) >= max_parts:
            break
    bands = [ln.buffer(w / 2, cap_style=2, join_style=1) for ln, w in feats["streets"] if w > 0]
    bands += [g for g, w in feats["streets"] if w == 0 and g.geom_type in ("Polygon", "MultiPolygon")]
    street = unary_union(bands).intersection(win) if bands else None
    blocks = rings(win.difference(street), 20.0) if street is not None and not street.is_empty else rings(win)
    greens = []
    for g in feats["greens"]:
        greens += rings(g.intersection(win), 20.0)
    lines = []
    for ln, w in feats["streets"]:
        if w <= 0 or ln.geom_type not in ("LineString", "MultiLineString"):
            continue
        g = ln.intersection(win)
        for part in ([g] if g.geom_type == "LineString" else list(getattr(g, "geoms", []))):
            if part.geom_type != "LineString" or part.length < 12:
                continue
            lines.append({"pts": [[round(x, 1), round(y, 1)] for x, y in part.simplify(1.0).coords], "w": round(float(w), 1)})
    return {"buildings": buildings, "blocks": blocks, "greens": greens, "centrelines": lines}


def fetch_canvas(lat: float, lon: float, extent_m: float, res_m: float, contact: str, cache: Path | None):
    """
    Render the OSM around (lat, lon) exactly as the training data is rendered.
    -> (rgb uint8 [px, px, 3], info dict)

    The canvas is a large working context, a kilometre and a half by default, at the corpus'
    own 3.125 m per pixel. It is not a model window and is not meant to be: you look at real
    city, select a plot or a block somewhere inside it, and the app crops the one window the
    model reads around that selection (see choose_window). Rendering at the fixed corpus
    resolution is what makes that crop free: a 400 m window is exactly 128 canvas pixels, an
    800 m one is 256 and is halved to 128, which is inside the scale range the model was
    trained on. Nothing is stitched and nothing outside the selection is touched.
    """
    import osm_tiles as O
    size_m = float(extent_m)
    res_m = float(res_m or 3.125)                            # fixed metres per pixel
    work = max(64, int(round(size_m / res_m)))               # the canvas, in pixels
    size_m = work * res_m                                    # snap the extent to whole pixels
    conv = {**O.CONVENTIONS, "px": 128}
    half = size_m / 2 + 150.0
    dlat = 1 / 111320.0
    dlon = 1 / (111320.0 * math.cos(math.radians(lat)))
    # One query for the whole context wherever Overpass will take it. Pre-splitting a 1.5 km
    # load into four 900 m boxes cost four slot waits, four heavy queries and three sleeps,
    # which is where ten minutes went; a single 1.8 km box is one of those. fetch_chunk still
    # splits itself into four when the server says the query was too heavy, so the dense
    # cities adapt instead of every city paying for them.
    n = max(1, math.ceil(2 * half / 2600.0))
    step = 2 * half / n
    elements, seen = [], set()
    PROGRESS["t0"] = time.time()
    CANCEL["flag"] = False
    for i in range(n):
        for j in range(n):
            if cancelled():
                PROGRESS.update(what="", done=0, total=0, t0=0.0)
                raise RuntimeError("cancelled")
            say(f"reading OpenStreetMap around {lat:.4f}, {lon:.4f}"
                + (f", box {i * n + j + 1} of {n * n}" if n > 1 else ""), i * n + j, n * n)
            y0, x0 = -half + i * step, -half + j * step
            data = O.fetch_chunk(lat + y0 * dlat, lon + x0 * dlon,
                                 lat + (y0 + step) * dlat, lon + (x0 + step) * dlon,
                                 conv, contact, cache)
            for el in data.get("elements", []):
                k = (el["type"], el["id"])
                if k not in seen:
                    seen.add(k)
                    elements.append(el)
            if n * n > 1:
                time.sleep(1.0)
    if not elements:
        raise RuntimeError("OpenStreetMap returned nothing for this place. Overpass was "
                           "probably busy; try again in a moment.")
    say(f"drawing {len(elements)} OpenStreetMap elements", n * n, n * n)
    frame = O.Frame(lat, lon)
    feats = O.build_features(elements, frame, conv)
    rgb, _ = O.render_window(feats, 0.0, 0.0, size_m, res_m, conv)
    say("tracing the geometry", n * n, n * n)  # noqa: kept as its own phase for the progress line
    vec = vector_geometry(feats, size_m)
    # how many buildings carried a real height in OSM. Anything with neither a `height` tag
    # nor `building:levels` lands on conv["default_height_m"], a flat 7 m, and in many cities
    # that is most of them. Worth knowing before blaming the model for a flat skyline.
    dflt = float(conv["default_height_m"])
    tagged = sum(1 for _, h in feats["buildings"] if abs(float(h) - dflt) > 1e-6)
    info = {"lat": lat, "lon": lon, "extent_m": round(size_m), "px": int(rgb.shape[0]), "res_m_per_px": res_m,
            "heights": {"tagged": tagged, "total": len(feats["buildings"]), "default_m": dflt},
            "scale": round(res_m / D["res_m"], 3),
            "window_m": [round(D["scale_lo"] * D["size"] * D["res_m"]),
                         round(D["scale_hi"] * D["size"] * D["res_m"])] if D["model"] is not None else None,
            "buildings": len(feats["buildings"]), "streets": len(feats["streets"]), "greens": len(feats["greens"]),
            "elements": len(elements), "seconds": round(time.time() - PROGRESS["t0"], 1),
            "parts": {k: len(v) for k, v in vec.items()}}
    PROGRESS.update(what="", done=0, total=0, t0=0.0)
    return rgb, info, vec
