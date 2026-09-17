"""
diffusion.py

The noise schedule, the training loss and the samplers.

Training: cosine schedule, v-prediction, min-SNR-5 weighting. The loss is taken over the
whole tile, with the hole weighted 1 and the known region `ctx_weight` (0.1 by default):
the known region is trivial to copy from the context input but keeping a little loss on
it teaches the network that the two halves of a seam are the same field.

Sampling: DDIM (eta 0) with the known region re-injected at every step (the RePaint
replacement), optional RePaint resampling for seam harmony, and classifier free
guidance on the metric condition. `sample_canvas` runs the same model over overlapping
windows on a canvas of any size and averages the predictions where windows overlap
(MultiDiffusion), which is how a 1.2 km site is filled by a model that only ever saw
128 px.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class Schedule:
    def __init__(self, T: int = 1000, s: float = 0.008, device="cpu"):
        self.T = T
        t = torch.linspace(0, T, T + 1, dtype=torch.float64)
        f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        ab = (f / f[0]).clamp(1e-5, 1.0)
        self.alphas_cumprod = ab[1:].float().to(device)          # index t = 0..T-1
        self.snr = (self.alphas_cumprod / (1 - self.alphas_cumprod))

    def to(self, device):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.snr = self.snr.to(device)
        return self

    def coefs(self, t: torch.Tensor):
        ab = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return ab.sqrt(), (1 - ab).sqrt()

    def q_sample(self, x0, t, noise):
        a, b = self.coefs(t)
        return a * x0 + b * noise

    def v_target(self, x0, noise, t):
        a, b = self.coefs(t)
        return a * noise - b * x0

    def from_v(self, v, x_t, t):
        """v-prediction -> (x0_hat, eps_hat)"""
        a, b = self.coefs(t)
        x0 = a * x_t - b * v
        eps = b * x_t + a * v
        return x0, eps

    def min_snr_weight(self, t: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
        snr = self.snr[t]
        return (snr.clamp(max=gamma) / (snr + 1)).view(-1, 1, 1, 1)


def model_input(x_t, x0_ctx, mask):
    return torch.cat([x_t, x0_ctx * (1 - mask), mask], 1)


def training_loss(model, sched: Schedule, x0, mask, cond, log2scale, ctx_weight: float = 0.1,
                  t=None, noise=None, hole_w=None, out=None):
    """
    The v-prediction loss on one batch.

    `t` and `noise` are drawn here during training. Pass them in to hold them fixed, which
    is what makes a validation number comparable between steps: the loss varies far more
    with which timestep was drawn than with how good the model is, so a validation set has
    to freeze the timesteps and the noise along with the crops.

    `hole_w` [B] scales the weight of the loss *inside* the hole, per sample. It exists for
    the samples that are handed a condition the fabric does not imply: there the true tile
    is no longer the right answer inside the hole, so reproducing it pixel for pixel is the
    wrong thing to ask, while the context term still has to hold. The surroundings keep
    full weight either way.

    `out`, if given, is filled with the model's own estimate of the finished tile and the
    timesteps it was drawn at, so the caller can measure that estimate and put a second
    term on it without a second forward pass.
    """
    B = x0.shape[0]
    t = torch.randint(0, sched.T, (B,), device=x0.device) if t is None else t
    noise = torch.randn_like(x0) if noise is None else noise
    x_t = sched.q_sample(x0, t, noise)
    v = model(model_input(x_t, x0, mask), t, cond, log2scale)
    target = sched.v_target(x0, noise, t)
    hw = 1.0 if hole_w is None else hole_w.view(-1, 1, 1, 1).to(mask.dtype)
    w = mask * hw + (1 - mask) * ctx_weight
    per = ((v - target) ** 2 * w).mean((1, 2, 3)) / w.mean((1, 2, 3)).clamp(min=1e-3)
    if out is not None:
        out["x0_hat"] = sched.from_v(v, x_t, t)[0]
        out["t"] = t
    return (per * sched.min_snr_weight(t)[:, 0, 0, 0]).mean()


def _guided(model, x_in, t, cond, log2scale, cfg: float):
    if cfg == 1.0 or cond is None:
        return model(x_in, t, cond, log2scale)
    uncond = torch.zeros_like(cond)
    v = model(torch.cat([x_in, x_in]), torch.cat([t, t]), torch.cat([cond, uncond]), torch.cat([log2scale, log2scale]))
    vc, vu = v.chunk(2)
    return vu + cfg * (vc - vu)


def ddim_timesteps(T: int, steps: int):
    ts = torch.linspace(T - 1, 0, steps).round().long().tolist()
    return ts


@torch.no_grad()
def sample(model, sched: Schedule, x0_ctx, mask, cond, log2scale, steps: int = 50, cfg: float = 2.0,
           resample: int = 0, generator=None, x_T=None, on_step=None, guide=None):
    """
    Inpaint one batch of tiles.
      x0_ctx   [B,4,S,S] known fields (contents inside the hole are ignored)
      mask     [B,1,S,S] 1 where to generate
      cond     [B,12] or None
      log2scale[B]
      on_step  optional callback (i, x0_hat, x_t) after every denoising step, both already
               composited with the context: x0_hat is the model's estimate of the finished
               tile, x_t the state it is actually in, which starts as noise. For films and
               debugging; neither is used by the sampler itself
      guide    optional guide(x0_hat, i, n) -> x0_hat. Unlike on_step this one is in the
               loop: whatever it returns is what the step is taken from. A guide that
               measures x0_hat and pushes it toward a requested number turns the condition
               from a hint the network may ignore into a constraint on the trajectory
    Returns [B,4,S,S] with the known region copied back exactly.
    """
    B = x0_ctx.shape[0]
    dev = x0_ctx.device
    x = torch.randn(x0_ctx.shape, device=dev, generator=generator) if x_T is None else x_T
    ts = ddim_timesteps(sched.T, steps)
    for i, ti in enumerate(ts):
        t = torch.full((B,), ti, device=dev, dtype=torch.long)
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        for r in range(resample + 1):
            v = _guided(model, model_input(x, x0_ctx, mask), t, cond, log2scale, cfg)
            x0_hat, eps = sched.from_v(v, x, t)
            x0_hat = x0_hat.clamp(-1, 1)
            if guide is not None:
                x0_hat = guide(x0_hat, i, len(ts)).clamp(-1, 1)
            if t_prev >= 0:
                ab_prev = sched.alphas_cumprod[t_prev]
                x_prev = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * eps
                known = sched.q_sample(x0_ctx, torch.full_like(t, t_prev), torch.randn(x.shape, device=dev, generator=generator))
            else:
                x_prev = x0_hat
                known = x0_ctx
            x_prev = mask * x_prev + (1 - mask) * known
            if on_step is not None and r == resample:
                on_step(i, mask * x0_hat + (1 - mask) * x0_ctx, mask * x_prev + (1 - mask) * x0_ctx)
            if r < resample and t_prev >= 0:
                # RePaint: go back up to t and predict again, so the fill re-harmonises with the seam
                ab_t, ab_p = sched.alphas_cumprod[ti], sched.alphas_cumprod[t_prev]
                ratio = ab_t / ab_p
                x = ratio.sqrt() * x_prev + (1 - ratio).sqrt() * torch.randn(x.shape, device=dev, generator=generator)
            else:
                x = x_prev
    return mask * x + (1 - mask) * x0_ctx


def window_weight(size: int, device) -> torch.Tensor:
    w = torch.hann_window(size, periodic=False, device=device).clamp(min=1e-2)
    return (w.view(size, 1) * w.view(1, size)).view(1, 1, size, size)


def window_origins(H: int, W: int, win: int, stride: int):
    ys = list(range(0, max(H - win, 0) + 1, stride))
    xs = list(range(0, max(W - win, 0) + 1, stride))
    if ys[-1] != H - win:
        ys.append(H - win)
    if xs[-1] != W - win:
        xs.append(W - win)
    return [(y, x) for y in ys for x in xs]


@torch.no_grad()
def sample_canvas(model, sched: Schedule, x0_ctx, mask, cond, log2scale, steps: int = 50, cfg: float = 2.0,
                  window: int = 128, stride: int = 64, batch_windows: int = 16, resample: int = 0,
                  generator=None, progress=None, guide=None):
    """
    MultiDiffusion over a canvas larger than one window.
      x0_ctx [1,4,H,W], mask [1,1,H,W], cond [1,12] or None, log2scale [1]
    Windows that contain no hole are skipped; the rest share one global noisy canvas and
    their v-predictions are blended with a Hann weight where they overlap.
    """
    _, _, H, W = x0_ctx.shape
    assert H >= window and W >= window, "canvas smaller than the window; use sample()"
    dev = x0_ctx.device
    origins = [(y, x) for (y, x) in window_origins(H, W, window, stride)
               if mask[0, 0, y:y + window, x:x + window].sum() > 0]
    wgt = window_weight(window, dev)
    x = torch.randn(x0_ctx.shape, device=dev, generator=generator)
    ts = ddim_timesteps(sched.T, steps)
    cond_b = None if cond is None else cond.expand(batch_windows, -1)
    l2s_b = log2scale.expand(batch_windows)

    for i, ti in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        for r in range(resample + 1):
            v_acc = torch.zeros_like(x)
            w_acc = torch.zeros(1, 1, H, W, device=dev)
            for s in range(0, len(origins), batch_windows):
                chunk = origins[s:s + batch_windows]
                xs = torch.cat([x[:, :, y:y + window, x_:x_ + window] for (y, x_) in chunk])
                cs = torch.cat([x0_ctx[:, :, y:y + window, x_:x_ + window] for (y, x_) in chunk])
                ms = torch.cat([mask[:, :, y:y + window, x_:x_ + window] for (y, x_) in chunk])
                n = xs.shape[0]
                t = torch.full((n,), ti, device=dev, dtype=torch.long)
                v = _guided(model, model_input(xs, cs, ms), t, None if cond_b is None else cond_b[:n], l2s_b[:n], cfg)
                for j, (y, x_) in enumerate(chunk):
                    v_acc[:, :, y:y + window, x_:x_ + window] += v[j:j + 1] * wgt
                    w_acc[:, :, y:y + window, x_:x_ + window] += wgt
            v_all = v_acc / w_acc.clamp(min=1e-6)
            t_full = torch.full((1,), ti, device=dev, dtype=torch.long)
            x0_hat, eps = sched.from_v(v_all, x, t_full)
            x0_hat = x0_hat.clamp(-1, 1)
            if guide is not None:                       # on the whole canvas, not per window:
                x0_hat = guide(x0_hat, i, len(ts)).clamp(-1, 1)   # the metric is a canvas-wide number
            if t_prev >= 0:
                ab_prev = sched.alphas_cumprod[t_prev]
                x_prev = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * eps
                known = sched.q_sample(x0_ctx, torch.full((1,), t_prev, device=dev, dtype=torch.long),
                                       torch.randn(x.shape, device=dev, generator=generator))
            else:
                x_prev, known = x0_hat, x0_ctx
            x_prev = mask * x_prev + (1 - mask) * known
            if r < resample and t_prev >= 0:
                ratio = sched.alphas_cumprod[ti] / sched.alphas_cumprod[t_prev]
                x = ratio.sqrt() * x_prev + (1 - ratio).sqrt() * torch.randn(x.shape, device=dev, generator=generator)
            else:
                x = x_prev
        if progress:
            progress(i + 1, len(ts))
    return mask * x + (1 - mask) * x0_ctx
