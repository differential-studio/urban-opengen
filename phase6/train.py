"""
train.py

Train the conditional diffusion inpainter on the city cache from prepare.py.

    python train.py --cache C:\\opengan\\cache_v2 --out C:\\opengan\\inpaint\\run01
    python train.py --cache C:\\opengan\\cache_v2 --out C:\\opengan\\inpaint\\smoke --tiny --steps 3000
    python train.py --cache C:\\opengan\\cache_v2 --out C:\\opengan\\inpaint\\run01 --resume
    python train.py --cache C:\\opengan\\cache_v2 --out C:\\opengan\\inpaint\\run02 --init C:\\opengan\\inpaint\\run01\\latest.pt --steps 40000

Every batch: random crops at a random physical scale (log-uniform between --scale-lo and
--scale-hi times 400 m), a random hole per crop, the metrics of the hole from the truth,
metrics withheld at random so any subset works at inference. Loss on v with min-SNR
weighting. EMA weights are what get sampled and saved as the model.

Writes to --out:
    latest.pt              everything needed to resume (and to sample)
    ckpt_<step>.pt         every --ckpt-every steps
    preview_<step>.png     context | fill | truth at scale 1 and at --scale-hi, EMA weights
    log.txt
    metrics.jsonl          one record per preview: training loss, the preview target error,
                           the conditioning response, and the held-out validation loss

Three things are measured, and they answer different questions. The preview target error
asks the model for the true metrics of the hole it just cut, so it says whether the fill is
realistic, not whether a request is obeyed. The conditioning response asks for a metric well
above and well below the corpus mean from the same noise and reports how much of that
request arrives, which is the number that decides whether the targets are controls at all.
The validation loss runs on cities held out of training entirely, with the timesteps and the
noise frozen, and is the only one of the three that can show overfitting.

Budget on one consumer GPU at 128 px, batch 16, default model (23M params): a few steps
per second with AMP, so 100k steps is an overnight run. --tiny proves the pipeline in a
few minutes. Every log line carries the peak VRAM: on Windows a run that exceeds the card
does not crash, the driver spills into system RAM and the speed drops 100x, so if it/s
is far below 1, lower --batch first.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase0"))
import data as data_mod  # noqa: E402
from data import (CityCache, CropSampler, CondNorm, random_masks, hole_metrics, train_flags,  # noqa: E402
                  cond_with_flags, fields_to_bytes, soft_hole_metrics, METRICS, N_COND,
                  MASK_KINDS, MASK_MARGIN, MASK_QUARTERS)
from unet import UNet, count_params  # noqa: E402
from diffusion import Schedule, training_loss, sample  # noqa: E402
from tile_codec import channels_to_rgb  # noqa: E402


def _need():
    try:
        import torch  # noqa
        import scipy  # noqa
    except ImportError as e:
        sys.exit(f"{e}\nrun with the training venv: C:\\opengan\\venv\\Scripts\\python.exe train.py ...")


def build_model(args, tiny: bool) -> UNet:
    if tiny:
        return UNet(base=32, mults=(1, 2, 2), blocks=1, attn_levels=(2,), dropout=0.0)
    return UNet(base=args.base, mults=tuple(args.mults), blocks=args.blocks, attn_levels=tuple(args.attn),
                dropout=args.dropout)


@torch.no_grad()
def ema_update(ema, model, decay):
    pe = [p for p in ema.parameters()]
    pm = [p.detach() for p in model.parameters()]
    torch._foreach_mul_(pe, decay)
    torch._foreach_add_(pe, pm, alpha=1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def render_rgb(fields: torch.Tensor, lut) -> list[np.ndarray]:
    return [channels_to_rgb(b, lut) for b in fields_to_bytes(fields)]


def preview_png(rows: list[list[np.ndarray]], path: Path, up: int = 2):
    h, w = rows[0][0].shape[:2]
    pad = 4
    W = len(rows[0]) * (w * up + pad) + pad
    H = len(rows) * (h * up + pad) + pad
    canvas = Image.new("RGB", (W, H), (40, 40, 40))
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            im = Image.fromarray(img).resize((w * up, h * up), Image.NEAREST)
            canvas.paste(im, (pad + c * (w * up + pad), pad + r * (h * up + pad)))
    canvas.save(path)


def context_rgb(x0: torch.Tensor, mask: torch.Tensor, lut) -> list[np.ndarray]:
    imgs = render_rgb(x0, lut)
    m = mask[:, 0].cpu().numpy() > 0.5
    out = []
    for img, mm in zip(imgs, m):
        img = img.copy()
        img[mm] = (90, 90, 110)
        out.append(img)
    return out


@torch.no_grad()
def make_preview(ema, sched, sampler, norm, cache, step, out: Path, scales, n=6, steps=30, cfg=2.0,
                 mask_seed: int = 4242):
    """
    The same crops with the same holes at every preview, so the pictures and the numbers
    can be compared across the run. The holes used to be drawn from the training generator,
    which meant every preview cut different holes: the target error then moved more with
    mask luck than with the model, and a good step could read worse than a bad one.

    What this measures is worth being clear about. The condition handed to the model is the
    true metrics of the hole that was just cut, so the request always agrees with the fabric
    around it. A model that ignored the condition entirely would still score well here,
    because the context predicts the metrics. Read it as "is the fill realistic and does it
    land where the neighbourhood implies", not as "is the request obeyed": that is what
    conditioning_response below is for.
    """
    rows = []
    report = []
    for sc in scales:
        x0, scale = sampler.fixed(n, sc)
        mgen = torch.Generator(device=x0.device).manual_seed(mask_seed + int(round(sc * 1000)))
        mask = random_masks(x0, mgen)
        raw = hole_metrics(x0, mask, scale, cache.byte_to_m, cache.res_m)
        cond = cond_with_flags(norm(raw), torch.ones_like(raw))
        l2s = torch.full((n,), math.log2(scale), device=x0.device)
        ctx = x0 * (1 - mask)
        y = sample(ema, sched, ctx, mask, cond, l2s, steps=steps, cfg=cfg, generator=mgen)
        got = hole_metrics(y, mask, scale, cache.byte_to_m, cache.res_m)
        err = (got - raw).abs().mean(0)
        report.append({"scale": scale, "mean_abs_err": {k: round(float(v), 3) for k, v in zip(METRICS, err)}})
        a, b, c = context_rgb(x0, mask, cache.lut), render_rgb(y, cache.lut), render_rgb(x0, cache.lut)
        for i in range(n):
            rows.append([a[i], b[i], c[i]])
    preview_png(rows, out / f"preview_{step:07d}.png")
    return report


# ---------------------------------------------------------------------------
# the two numbers the run could not answer before: does it generalise, and is the
# condition an instruction or a decoration
# ---------------------------------------------------------------------------

def split_cities(cache, k: int, named: str | None, seed: int = 1234):
    """
    (train indices, validation indices). The held-out cities are drawn once from a fixed
    seed so a resumed run splits the same way, and they are spread through the city list
    rather than taken off the end, which on a list ordered by continent would hold out one
    region and nothing else.
    """
    n = len(cache)
    if named:
        want = [w.strip().lower() for w in named.split(",") if w.strip()]
        val = [i for i, c in enumerate(cache.cities) if c.lower() in want]
        missing = [w for w in want if not any(cache.cities[i].lower() == w for i in val)]
        if missing:
            sys.exit(f"--holdout-cities: not in this cache: {', '.join(missing)}")
    else:
        k = max(0, min(int(k), max(0, n - 4)))
        val = sorted(np.random.RandomState(seed).choice(n, k, replace=False).tolist()) if k else []
    if val and len(val) > 0.2 * n:
        print(f"  WARNING: holding out {len(val)} of {n} cities. That is a large slice of the corpus to "
              f"train without; lower --holdout unless the cache is meant to be this small.", flush=True)
    train = [i for i in range(n) if i not in set(val)]
    return train, val


def make_valset(sampler, sched, cache, norm, gen, dev, batches: int, batch: int, scales, seed: int = 99):
    """
    A frozen validation set: the same crops, the same holes, the same timesteps and the
    same noise, every time it is evaluated. Diffusion loss swings far more with which
    timestep was drawn than with the state of the model, so a validation number that
    redraws them is mostly noise. Cheap to hold: a few batches of fields on the GPU.
    """
    items = []
    g = torch.Generator(device=dev).manual_seed(seed)
    for b in range(batches):
        sc = scales[b % len(scales)]
        x0, scale = sampler.fixed(batch, sc, seed=seed + 17 * b)
        mask = random_masks(x0, g)
        raw = hole_metrics(x0, mask, scale, cache.byte_to_m, cache.res_m)
        flags = train_flags(x0.shape[0], g, dev, 0.3, 0.15)      # the same mix of given and withheld as training
        cond = cond_with_flags(norm(raw), flags)
        t = torch.randint(0, sched.T, (x0.shape[0],), device=dev, generator=g)
        noise = torch.randn(x0.shape, device=dev, generator=g)
        items.append((x0, mask, cond, float(scale), t, noise))
    return items


@torch.no_grad()
def val_loss(net, sched, items, ctx_weight, dev, amp_dtype, use_amp):
    tot = 0.0
    for x0, mask, cond, scale, t, noise in items:
        l2s = torch.full((x0.shape[0],), math.log2(scale), device=dev)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
            tot += float(training_loss(net, sched, x0, mask, cond, l2s, ctx_weight, t=t, noise=noise))
    return tot / max(1, len(items))


@torch.no_grad()
def conditioning_response(ema, sched, sampler, norm, cache, dev, metrics, n=3, steps=24, cfg=2.0,
                          scale=1.0, delta=1.5, seed=777, max_batch=24, mask=None):
    """
    How much of a request actually arrives.

    The same holes are filled twice for each metric, once asking for `delta` sigma below
    the corpus mean and once `delta` sigma above, with every other metric withheld. The
    number reported is how far the fills actually moved apart, as a fraction of how far
    they were asked to: 1.0 means the request was delivered exactly, 0 means the network
    ignored it and produced whatever the context implied both times.

    This is the number that predicts whether the sliders in the viewer will do anything,
    and it is the one the target error above cannot see, because there the request is
    always what the context already implies.

    Every variant starts from the same noise on the same holes, so the only difference
    between the low and the high fill is the request itself. Without that the probe would
    mostly measure which seed each side happened to get. The batch is split into chunks of
    whole variants so a wide probe cannot run the card out of memory.
    """
    idx = [METRICS.index(m) for m in metrics if m in METRICS]
    if not idx:
        return []
    x0, sc = sampler.fixed(n, scale, seed=seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    if mask is None:
        mask = random_masks(x0, g)
    elif mask.shape[0] != x0.shape[0]:
        mask = mask[:1].expand(x0.shape[0], -1, -1, -1).contiguous()
    ctx = x0 * (1 - mask)
    mean, std = norm.mean.to(dev), norm.std.to(dev)
    conds, wants = [], []
    for i in idx:
        for sgn in (-1.0, 1.0):
            raw = torch.zeros(1, N_COND, device=dev)
            flags = torch.zeros(1, N_COND, device=dev)
            v = float((mean[i] + sgn * delta * std[i]).clamp(min=0.0))
            raw[0, i] = v
            flags[0, i] = 1.0
            conds.append(cond_with_flags(norm(raw), flags))
            wants.append(v)
    V = len(conds)
    per = max(1, min(V, max(1, max_batch // max(1, n))))         # whole variants per chunk
    xT_one = torch.randn(x0.shape, device=dev, generator=g)      # the same noise for every variant
    outs = []
    for a in range(0, V, per):
        cs = conds[a:a + per]
        k = len(cs)
        gc = torch.Generator(device=dev).manual_seed(seed + 1)    # same stream in every chunk
        y = sample(ema, sched, ctx.repeat(k, 1, 1, 1), mask.repeat(k, 1, 1, 1),
                   torch.cat([c.expand(n, -1) for c in cs]),
                   torch.full((k * n,), math.log2(sc), device=dev),
                   steps=steps, cfg=cfg, generator=gc, x_T=xT_one.repeat(k, 1, 1, 1))
        outs.append(hole_metrics(y, mask.repeat(k, 1, 1, 1), sc, cache.byte_to_m, cache.res_m))
    got = torch.cat(outs)
    out = []
    for k, i in enumerate(idx):
        lo = float(got[(2 * k) * n:(2 * k + 1) * n, i].mean())
        hi = float(got[(2 * k + 1) * n:(2 * k + 2) * n, i].mean())
        asked = wants[2 * k + 1] - wants[2 * k]
        out.append({"metric": METRICS[i], "asked_low": round(wants[2 * k], 3), "asked_high": round(wants[2 * k + 1], 3),
                    "got_low": round(lo, 3), "got_high": round(hi, 3),
                    "delivered": round((hi - lo) / asked, 3) if abs(asked) > 1e-9 else 0.0})
    return out


def main():
    _need()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16, help="16 fits 8 GB; raise it only if vram in the log stays well under the card")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--scale-lo", type=float, default=0.75, help="smallest window, x 400 m")
    ap.add_argument("--scale-hi", type=float, default=2.0, help="largest window, x 400 m; equal to lo disables")
    ap.add_argument("--p-each", type=float, default=0.3, help="chance each metric is withheld")
    ap.add_argument("--p-all", type=float, default=0.15, help="chance all metrics are withheld")
    ap.add_argument("--ctx-weight", type=float, default=0.1)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--mults", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--attn", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--tiny", action="store_true", help="small model for a smoke test")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="store activations instead of recomputing them; needs a card with 16 GB or more")
    ap.add_argument("--mask-margin", type=int, default=None, metavar="PX",
                    help="working pixels of context kept on every side of every mask "
                         f"(default {MASK_MARGIN}); the app always frames a selection with fabric "
                         "around it, so training does too")
    ap.add_argument("--mask-probs", help="override the mask mix, comma separated in the order "
                                         + ",".join(MASK_KINDS))
    ap.add_argument("--mask-quarters", type=int, default=None, metavar="N",
                    help=f"most quarters one selection may walk to (default {MASK_QUARTERS}); a "
                         "ceiling, not a quota, since a walk that runs out of room stops early")
    ap.add_argument("--holdout", type=int, default=4,
                    help="cities kept out of training entirely, for the validation loss (0 disables)")
    ap.add_argument("--holdout-cities", help="name them instead of drawing them, comma separated")
    ap.add_argument("--val-every", type=int, default=2000, help="steps between validation passes")
    ap.add_argument("--val-batches", type=int, default=6, help="frozen batches in the validation set")
    ap.add_argument("--metric-weight", type=float, default=0.0,
                    help="weight on the metric-consistency term: the model's own estimate of the "
                         "finished tile is measured and pushed toward what was asked for. 0 is off. "
                         "Start at 0.05 and watch the two numbers in the step line: weight x metric "
                         "should stay within a few times loss, or the geometry pays for the numbers")
    ap.add_argument("--metric-warmup", type=int, default=2000,
                    help="steps over which the metric term ramps in, so it does not fight the model "
                         "learning to denoise at all")
    ap.add_argument("--ask-frac", type=float, default=0.0,
                    help="fraction of each batch handed a request the surroundings do not imply, "
                         "which is what stops the condition being redundant. 0 is off, 0.3 is a "
                         "good starting point; needs --metric-weight to do anything")
    ap.add_argument("--ask-delta", type=float, default=2.0,
                    help="how far those requests are moved, in corpus standard deviations")
    ap.add_argument("--ask-min-frac", type=float, default=0.08,
                    help="smallest hole, as a fraction of the window, that is allowed to be given a "
                         "request it cannot read off the surroundings. Below this the context "
                         "determines the answer and the request is unanswerable")
    ap.add_argument("--ask-hole-w", type=float, default=0.15,
                    help="weight kept on the denoising loss inside the hole for asked samples, "
                         "where the true tile is no longer the right answer")
    ap.add_argument("--response-metrics", default="far,coverage,green,street",
                    help="metrics the conditioning probe pushes on, comma separated; empty disables it")
    ap.add_argument("--response-delta", type=float, default=1.5,
                    help="how far from the corpus mean the probe asks, in standard deviations, each way")
    ap.add_argument("--response-n", type=int, default=3, help="holes per variant in the probe")
    ap.add_argument("--response-batch", type=int, default=16, help="largest batch the probe will put on the card")
    ap.add_argument("--response-steps", type=int, default=24, help="DDIM steps in the probe")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--preview-every", type=int, default=2000)
    ap.add_argument("--first-preview", type=int, default=500, help="an early preview so a broken run is visible in minutes")
    ap.add_argument("--workers", type=int, default=2, help="crop threads feeding the GPU")
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", help="warm-start model and EMA weights from this checkpoint, fresh optimiser and step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True                 # autotune conv kernels for the fixed shapes
    log = open(out / "log.txt", "a")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n"); log.flush()

    if args.mask_margin is not None:
        data_mod.MASK_MARGIN = max(0, args.mask_margin)
    if args.mask_quarters is not None:
        data_mod.MASK_QUARTERS = max(1, args.mask_quarters)
    if args.mask_probs:
        pr = [float(v) for v in args.mask_probs.split(",")]
        if len(pr) != len(MASK_KINDS):
            sys.exit(f"--mask-probs needs {len(MASK_KINDS)} numbers, one per kind: {','.join(MASK_KINDS)}")
        data_mod.MASK_PROBS = tuple(v / sum(pr) for v in pr)

    cache = CityCache(args.cache)
    train_idx, val_idx = split_cities(cache, args.holdout, args.holdout_cities)
    sampler = CropSampler(cache, args.size, args.scale_lo, args.scale_hi, seed=args.seed, device=dev,
                          indices=train_idx)
    sampler.start(args.batch, args.workers)
    say(f"cache: {len(cache)} cities at {cache.res_m} m/px, working size {args.size}, "
        f"scale {args.scale_lo}..{args.scale_hi} ({args.scale_lo * 400:.0f}..{args.scale_hi * 400:.0f} m per window)")
    say("masks: " + ", ".join(f"{k} {p:.0%}" for k, p in zip(MASK_KINDS, data_mod.MASK_PROBS))
        + f"  ·  {data_mod.MASK_MARGIN} px of context kept on every side"
        + f"  ·  up to {data_mod.MASK_QUARTERS} quarters per selection")
    val_sampler = None
    if val_idx:
        val_sampler = CropSampler(cache, args.size, args.scale_lo, args.scale_hi, seed=args.seed + 5,
                                  device=dev, indices=val_idx)
        say(f"held out of training: {', '.join(cache.cities[i] for i in val_idx)}  "
            f"({len(train_idx)} cities left to train on)")
    else:
        say("no held-out cities: there will be no validation loss, only the training one")

    model = build_model(args, args.tiny).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    model.grad_ckpt = not args.no_grad_ckpt
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    sched = Schedule(device=dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1)
    use_amp = dev.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    gpu = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    vram_total = torch.cuda.get_device_properties(dev).total_memory / 2**30 if dev.type == "cuda" else 0
    say(f"model: {count_params(model) / 1e6:.1f} M params, amp {use_amp} ({amp_dtype if use_amp else 'fp32'}), "
        f"grad checkpointing {'on' if model.grad_ckpt else 'off'}, device {gpu}" + (f" {vram_total:.1f} GB" if vram_total else ""))
    if dev.type != "cuda":
        say("WARNING: not on a GPU. torch.cuda.is_available() is False in this Python; training will be ~100x slower")

    step = 0
    latest = out / "latest.pt"
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location=dev)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step = ck["step"]
        norm = CondNorm(ck["cond_norm"]["mean"], ck["cond_norm"]["std"])
        say(f"resumed at step {step}")
        was = ck.get("holdout")
        now = [cache.cities[i] for i in val_idx]
        if was is not None and sorted(was) != sorted(now):
            say(f"  WARNING: this run holds out {now or 'nothing'} but the checkpoint held out {was or 'nothing'}. "
                f"The validation loss is not comparable with the earlier part of the run, and cities the model "
                f"already trained on are now in the validation set.")
    else:
        if args.init:
            ck = torch.load(args.init, map_location=dev)
            model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
            say(f"weights from {args.init} (step {ck['step']}), optimiser and step fresh")
        say("estimating condition statistics ...")
        norm = CondNorm.estimate(sampler, cache.byte_to_m)
        say("  " + ", ".join(f"{k} {m:.3g}±{s:.3g}" for k, m, s in zip(METRICS, norm.mean.tolist(), norm.std.tolist())))

    def save(path: Path):
        torch.save({
            "model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
            "args": vars(args), "unet": ema.cfg, "cond_norm": norm.state(), "metrics": list(METRICS),
            "lut": {"forward": cache.lut.forward, "inverse": cache.lut.inverse, "counts": cache.lut.counts},
            "res_m_per_px": cache.res_m, "empty_height_byte": cache.empty_height, "size": args.size,
            "holdout": [cache.cities[i] for i in val_idx],
        }, path)

    val_items = None
    if val_sampler is not None:
        scales_v = [1.0] if args.scale_hi <= args.scale_lo else [1.0, args.scale_hi]
        val_items = make_valset(val_sampler, sched, cache, norm, gen, dev, args.val_batches, args.batch, scales_v)
        say(f"validation: {args.val_batches} frozen batches of {args.batch} from the held-out cities, "
            f"fixed holes, timesteps and noise")
    resp_metrics = [m.strip() for m in (args.response_metrics or "").split(",") if m.strip()]
    hist = open(out / "metrics.jsonl", "a")

    def record(rec):
        hist.write(json.dumps(rec) + "\n"); hist.flush()

    t0 = time.time()
    step0 = step
    loss_acc, n_acc = torch.zeros((), device=dev), 0
    mloss_acc, mn_acc = torch.zeros((), device=dev), 0
    mloss_smooth = None
    loss_smooth = None                                    # slow EMA of the training loss, for metrics.jsonl
    tt = {"data": 0.0, "prep": 0.0, "gpu": 0.0}

    def clock(profile):
        if profile and dev.type == "cuda":
            torch.cuda.synchronize(dev)
        return time.perf_counter()

    while step < args.steps:
        profile = step - step0 < 200            # sync-timed breakdown for the first 200 steps only
        c0 = clock(profile)
        x0, scale = sampler.next()
        c1 = clock(profile)
        B = x0.shape[0]
        mask = random_masks(x0, gen)
        raw = hole_metrics(x0, mask, scale, cache.byte_to_m, cache.res_m)
        flags = train_flags(B, gen, dev, args.p_each, args.p_all)
        want, hole_w = raw, None
        if args.ask_frac > 0:
            # A fraction of the batch is handed numbers the fabric around the hole does NOT
            # imply. This is the whole point: while the condition always agrees with the
            # context, it carries no information the context does not already have, and the
            # network correctly learns to ignore it. Here it cannot: the only way to satisfy
            # the request is to use it. Inside the hole the true tile stops being the right
            # answer for those samples, so its weight drops; the surroundings keep theirs.
            # Only ask where there is room to answer. The mask sweep on run02 showed delivery
            # scaling with hole size, from 0.01 at a 56 m hole to 0.16 at a 347 m one: where the
            # surroundings determine the fill there is nothing a number can change, and a request
            # there is unanswerable. Training on those is noise at best, and at worst teaches the
            # model that requests are to be ignored, which is the habit being unlearned.
            big = (mask.mean((1, 2, 3), keepdim=False) >= args.ask_min_frac).float().view(B, 1)
            ask = (torch.rand(B, 1, device=dev, generator=gen) < args.ask_frac).float() * big
            k = 1 + (torch.rand(B, 1, device=dev, generator=gen) < 0.4).float()      # one metric, sometimes two
            pick = (torch.rand(B, N_COND, device=dev, generator=gen).argsort(1) < k).float()
            jit = (torch.rand(B, N_COND, device=dev, generator=gen) * 2 - 1) * args.ask_delta
            asked = (raw + jit * norm.std.to(dev)).clamp(min=0.0)
            flags = flags * (1 - ask) + pick * ask
            want = raw * (1 - ask) + asked * ask
            hole_w = 1.0 - ask[:, 0] * (1.0 - args.ask_hole_w)
        cond = cond_with_flags(norm(want), flags)
        l2s = torch.full((B,), math.log2(scale), device=dev)
        c2 = clock(profile)

        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
            parts = {} if args.metric_weight > 0 else None
            loss = training_loss(model, sched, x0, mask, cond, l2s, args.ctx_weight,
                                 hole_w=hole_w, out=parts)
            mloss = None
            if parts is not None and float(flags.sum()) > 0:
                # Measure what the model says the finished tile is and penalise the distance
                # to what was asked for, in units of the corpus spread so the six are
                # comparable. Only the metrics that were actually given count, and the term
                # is trusted in proportion to how much signal is left in x_t: at high noise
                # the estimate is mostly noise and so is its measurement.
                # in float32: these are sums over sixteen thousand pixels and bfloat16 costs a
                # few tenths of a percent on them, measured. Small, but it is free to remove
                # and this number is the one being optimised against
                got = soft_hole_metrics(parts["x0_hat"].float(), mask, scale, cache.byte_to_m, cache.res_m)
                r = (got - want) / norm.std.to(dev).clamp(min=1e-3)
                wt = sched.alphas_cumprod[parts["t"]].view(-1, 1).to(r.dtype)
                d = F.smooth_l1_loss(r, torch.zeros_like(r), reduction="none")
                mloss = (d * flags * wt).sum() / (flags * wt).sum().clamp(min=1e-3)
                mw = args.metric_weight * min(1.0, (step + 1) / max(1, args.metric_warmup))
                loss = loss + mw * mloss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        ema_update(ema, model, args.ema if step > args.warmup else 0.99)
        c3 = clock(profile)
        if profile:
            tt["data"] += c1 - c0; tt["prep"] += c2 - c1; tt["gpu"] += c3 - c2
        step += 1
        loss_acc += loss.detach(); n_acc += 1
        if mloss is not None:
            mloss_acc += mloss.detach(); mn_acc += 1
        lv = float(loss.detach())
        loss_smooth = lv if loss_smooth is None else 0.99 * loss_smooth + 0.01 * lv

        if step % args.log_every == 0 or step - step0 == 10:
            el = time.time() - t0
            vram = f"  vram {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GB" if dev.type == "cuda" else ""
            prof = ""
            if step - step0 <= 200 and sum(tt.values()) > 0:
                tot = sum(tt.values())
                prof = ("  [" + " ".join(f"{k} {v / tot * 100:.0f}%" for k, v in tt.items()) +
                        f", {tot / (step - step0):.2f} s/step]")
            mtxt = f"  metric {mloss_acc.item() / mn_acc:.3f}" if mn_acc else ""
            if mn_acc:
                mv = mloss_acc.item() / mn_acc
                mloss_smooth = mv if mloss_smooth is None else 0.9 * mloss_smooth + 0.1 * mv
            say(f"step {step:7d}  loss {loss_acc.item() / n_acc:.4f}{mtxt}  scale {scale:.2f}  lr {lr:.1e}  "
                f"{(step - step0) / max(el, 1e-6):.2f} it/s{vram}  {el / 60:.1f} min{prof}")
            if dev.type == "cuda" and vram_total and torch.cuda.max_memory_allocated(dev) / 2**30 > 0.92 * vram_total:
                say("  WARNING: at the VRAM limit; on Windows the driver spills to system RAM and everything crawls. Lower --batch.")
            loss_acc, n_acc = torch.zeros((), device=dev), 0
            mloss_acc, mn_acc = torch.zeros((), device=dev), 0
        if step % args.preview_every == 0 or step == args.steps or step == args.first_preview:
            # The weights first, then the diagnostics. The preview and the probe are the
            # heaviest moments on the card, and they are exactly where a tight run falls over:
            # sampling batches on top of the model, its EMA copy and the optimiser state. The
            # diagnostics are expendable and the weights are not, so they are written before
            # anything risky runs, and a failure in either costs the numbers rather than the run.
            save(latest)
            scales = [1.0] if args.scale_hi <= args.scale_lo else [1.0, args.scale_hi]
            rep, resp = [], []
            try:
                rep = make_preview(ema, sched, sampler, norm, cache, step, out, scales)
                for r in rep:
                    say(f"  preview scale {r['scale']:.2f}: target error " +
                        ", ".join(f"{k} {v}" for k, v in r["mean_abs_err"].items()))
                if resp_metrics:
                    resp = conditioning_response(ema, sched, sampler, norm, cache, dev, resp_metrics,
                                                 n=args.response_n, steps=args.response_steps,
                                                 delta=args.response_delta, max_batch=args.response_batch)
                    if resp:
                        say(f"  condition delivered (1.0 = the request arrives in full, 0 = ignored): "
                            + ", ".join(f"{r['metric']} {r['delivered']:.2f}" for r in resp))
            except torch.cuda.OutOfMemoryError as e:  # noqa
                say(f"  diagnostics skipped, out of memory: {str(e)[:90]}. "
                    f"Lower --response-batch or --response-n; training carries on.")
                torch.cuda.empty_cache()
            if dev.type == "cuda":
                torch.cuda.empty_cache()                 # hand the probe's peak back before training resumes
            record({"step": step, "train_loss": round(loss_smooth, 6),
                    **({"metric_loss": round(mloss_smooth, 4)} if mloss_smooth is not None else {}),
                    "preview": rep, "response": resp})
        if val_items and (step % args.val_every == 0 or step == args.steps or step == args.first_preview):
            vl = val_loss(ema, sched, val_items, args.ctx_weight, dev, amp_dtype, use_amp)
            say(f"  val loss {vl:.4f}  (EMA weights, {len(val_items)} frozen batches from "
                f"{len(val_idx)} cities the model has never seen)")
            record({"step": step, "val_loss": round(vl, 6)})
        if step % args.ckpt_every == 0:
            save(out / f"ckpt_{step:07d}.pt")
    save(latest)
    say(f"done, {step} steps in {(time.time() - t0) / 3600:.2f} h -> {latest}")


if __name__ == "__main__":
    main()
