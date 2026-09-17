"""
data.py

Everything between the city cache and the model: scale-aware random crops, the hole
masks, and the metrics of the hole that the model is conditioned on. All of it runs on
the GPU on whole batches; the CPU only slices uint8 windows out of memory-mapped
arrays.

Scale
-----
A training sample is a square window of `side` raster pixels resampled to `size`
working pixels (128). With rasters at 3.125 m per pixel, side = 128 is the native
400 m tile, side = 256 is an 800 m window seen at half resolution, side = 96 is a 300 m
window seen at 1.33x. `scale` = side / size = metres per working pixel divided by the
native 3.125, and log2(scale) is given to the model as a condition, so one model learns
the fabric at every scale in the range and can be asked for any of them at inference.

Signed distances are stored in raster pixels and converted to working pixels before the
resize (multiply by size / side), then truncated at SDF_TRUNC working pixels and mapped
to [-1, 1]. The band the network sees is therefore always 8 pixels wide whatever the
scale, which is what makes the fields look alike across scales and lets the scale token
do its job.

Metrics of the hole
-------------------
All six are physical and scale-free, so a target means the same thing at every scale:

    coverage   building footprint share of the hole, 0..1
    green      green share, 0..1
    street     street share, 0..1 (streets are 8 m wide, so this is length density too)
    height_m   mean building height in metres over the footprint pixels in the hole
    far        floor area ratio of the hole, footprint x floors (3 m each) / hole area
    bsize_m    equivalent building side in metres, 4 x footprint area / footprint perimeter

Each condition can be given or withheld independently (a flag per metric), so at
inference the user may set only FAR, or only green and street, and the model fills the
rest from context. Withholding everything is the unconditional model that classifier
free guidance needs.
"""
from __future__ import annotations

import json
import math
import sys
import threading
import queue
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase0"))
from tile_codec import HeightLUT, grey_to_metres, FLOOR_HEIGHT_M  # noqa: E402

SDF_TRUNC = 8.0            # working pixels
SDF_QUANT = 4.0            # cache stores quarter raster pixels in int8
NATIVE_M_PER_PX = 3.125
METRICS = ("coverage", "green", "street", "height_m", "far", "bsize_m")
N_COND = len(METRICS)


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

class CityCache:
    """Memory-mapped per-city field arrays plus the decoding tables."""

    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        meta = json.loads((self.dir / "cache.json").read_text())
        self.meta = meta
        self.cities = [c["city"] for c in meta["cities"]]
        self.arrays = [np.load(self.dir / c["file"], mmap_mode="r") for c in meta["cities"]]
        self.res_m = float(meta.get("res_m_per_px") or NATIVE_M_PER_PX)
        self.lut = HeightLUT.load(self.dir / "height_lut.json")
        self.empty_height = int(meta.get("empty_height_byte", self.lut.mean_byte()))
        self.byte_to_m = byte_to_metres_table(self.lut)

    def __len__(self):
        return len(self.arrays)


def byte_to_metres_table(lut: HeightLUT) -> torch.Tensor:
    """height byte (0..255) -> metres, through the LUT inverse and the grey convention."""
    inv = np.asarray(lut.inverse, dtype=np.float64)
    return torch.from_numpy(grey_to_metres(inv).astype(np.float32))


# ---------------------------------------------------------------------------
# crops
# ---------------------------------------------------------------------------

def side_for_scale(scale: float, size: int) -> int:
    return max(8, int(round(scale * size)))


def sample_scale(rng: np.random.RandomState, lo: float, hi: float) -> float:
    """Log-uniform in [lo, hi]; lo == hi disables scale augmentation."""
    if hi <= lo:
        return float(lo)
    return float(2 ** rng.uniform(math.log2(lo), math.log2(hi)))


def crop_raw(cache: CityCache, n: int, side: int, rng: np.random.RandomState,
             indices: list[int] | None = None) -> np.ndarray:
    """n random uint8 windows [n, 4, side, side], cities sampled uniformly.

    `indices` restricts the draw to those cities, which is how the held-out set is kept
    out of training: the validation cities are simply never in the list the trainer draws
    from."""
    out = np.empty((n, 4, side, side), np.uint8)
    pool = indices if indices else list(range(len(cache)))
    for i in range(n):
        arr = cache.arrays[pool[rng.randint(len(pool))]]
        H, W = arr.shape[1:]
        y = rng.randint(0, H - side + 1)
        x = rng.randint(0, W - side + 1)
        out[i] = arr[:, y:y + side, x:x + side]
    return out


def raw_to_fields(raw: torch.Tensor, size: int, trunc: float = SDF_TRUNC) -> torch.Tensor:
    """
    uint8 [B, 4, side, side] on any device -> float [B, 4, size, size] in [-1, 1].
    Channels 0, 2, 3 are signed distances, 1 is the height byte.
    """
    side = raw.shape[-1]
    k = size / side                                  # raster px -> working px
    sdf = raw[:, [0, 2, 3]].view(torch.int8).float() / SDF_QUANT * k
    h = raw[:, 1:2].float() / 127.5 - 1.0
    x = torch.cat([sdf[:, :1], h, sdf[:, 1:]], 1)
    if side != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=(side > size))
    x[:, [0, 2, 3]] = (x[:, [0, 2, 3]].clamp(-trunc, trunc) / trunc)
    return x.clamp(-1, 1)


def content_fraction(raw: torch.Tensor) -> torch.Tensor:
    s = raw[:, [0, 2, 3]].view(torch.int8)
    return (s > 0).any(1).float().mean((1, 2))


class CropSampler:
    """
    Produces training batches: fields x0 [B,4,size,size] in [-1,1] on the device, and the
    scale of the batch. One scale per batch keeps the resize a single call. Crops with
    less than `min_content` of any class are rejected, the rest of the raster is water,
    countryside or unmapped, and there is nothing to learn from an empty hole in an
    empty context.
    """

    def __init__(self, cache: CityCache, size: int = 128, scale_lo: float = 0.75, scale_hi: float = 2.0,
                 min_content: float = 0.02, seed: int = 0, device="cuda", prefetch: int = 3,
                 indices: list[int] | None = None):
        self.cache, self.size = cache, size
        self.scale_lo, self.scale_hi = scale_lo, scale_hi
        self.min_content = min_content
        self.indices = list(indices) if indices else None   # None = every city in the cache
        self.rng = np.random.RandomState(seed)
        self.device = torch.device(device)
        self.q: queue.Queue = queue.Queue(maxsize=prefetch)
        self._batch = None
        self._thread = None

    def start(self, batch: int, workers: int = 2):
        self._batch = batch
        self._stop = False
        self._threads = []
        for i in range(max(1, workers)):
            rng = np.random.RandomState(self.rng.randint(1 << 30))
            t = threading.Thread(target=self._worker, args=(rng,), daemon=True)
            t.start()
            self._threads.append(t)

    def _worker(self, rng):
        while not self._stop:
            scale = sample_scale(rng, self.scale_lo, self.scale_hi)
            side = side_for_scale(scale, self.size)
            n = int(self._batch * 1.5) + 2
            raw = crop_raw(self.cache, n, side, rng, self.indices)
            self.q.put((raw, scale))

    def next(self):
        """(x0 [B,4,S,S] float on device, scale float)"""
        raw, scale = self.q.get()
        raw = torch.from_numpy(raw).to(self.device, non_blocking=True)
        keep = content_fraction(raw) >= self.min_content
        idx = torch.nonzero(keep).flatten()
        if idx.numel() < self._batch:                      # pad with whatever there is
            fill = torch.arange(raw.shape[0], device=raw.device)
            idx = torch.cat([idx, fill])[: self._batch]
        raw = raw[idx[: self._batch]]
        return raw_to_fields(raw, self.size), scale

    def fixed(self, n: int, scale: float, seed: int = 123):
        """A repeatable batch for previews."""
        rng = np.random.RandomState(seed)
        side = side_for_scale(scale, self.size)
        got = []
        while sum(g.shape[0] for g in got) < n:
            raw = torch.from_numpy(crop_raw(self.cache, n * 2, side, rng, self.indices)).to(self.device)
            keep = content_fraction(raw) >= max(self.min_content, 0.15)
            got.append(raw[keep])
        raw = torch.cat(got)[:n]
        return raw_to_fields(raw, self.size), scale


# ---------------------------------------------------------------------------
# masks: 1 where the model has to generate
# ---------------------------------------------------------------------------

# The app never hands the model a hole that runs off the edge of the window: whatever is
# selected, the window is rendered around it, so there is always context on all four sides.
# So there is no `full` and no `half` here, and every kind is kept MASK_MARGIN working
# pixels clear of the border. The hole may be most of the tile, it may come close to the
# edge, but the ring of real fabric around it is always there, because at inference it
# always is.
MASK_KINDS = ("rect", "multirect", "brush", "plots", "qpart", "quarters", "qgrown")
MASK_PROBS = (0.17, 0.08, 0.16, 0.09, 0.12, 0.22, 0.16)
MASK_MARGIN = 6           # working px of context kept on every side, at any size
MASK_QUARTERS = 6         # most quarters one selection may walk to; a ceiling, not a quota


def _interior(m: torch.Tensor, margin: int) -> torch.Tensor:
    """Clip a mask to the window minus `margin` px on every side."""
    if margin <= 0:
        return m
    keep = torch.zeros_like(m)
    keep[:, :, margin:-margin, margin:-margin] = 1.0
    return m * keep


def _flood(seed: torch.Tensor, allow: torch.Tensor, reach: int, step: int = 2) -> torch.Tensor:
    """
    Grow `seed` through `allow` until it has travelled `reach` pixels: the connected
    component of `allow` that the seed sits in.

    This is how a quarter is found. A quarter is a connected piece of ground the street
    network encloses, which is a connected component of "not street", which is a flood
    fill, which on a GPU is a max pool in a loop. No labelling pass, no trip to the CPU.
    `step` pixels per pass, so a wider kernel means fewer passes; keep it well under the
    width of a street band or the fill will leak across one between re-maskings.
    """
    m = seed
    k = 2 * step + 1
    for _ in range(max(1, reach // step)):
        m = F.max_pool2d(m, k, stride=1, padding=step) * allow
    return m


def _shift(m: torch.Tensor, d: torch.Tensor, b: int) -> torch.Tensor:
    """Move each sample's mask `b` px in its own direction d (0 down, 1 up, 2 right, 3 left)."""
    size = m.shape[-1]
    down = F.pad(m, (0, 0, b, 0))[:, :, :size, :]
    up = F.pad(m, (0, 0, 0, b))[:, :, b:, :]
    right = F.pad(m, (b, 0, 0, 0))[:, :, :, :size]
    left = F.pad(m, (0, b, 0, 0))[:, :, :, b:]
    return torch.where(d == 0, down, torch.where(d == 1, up, torch.where(d == 2, right, left)))


def _seed_in(allow: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """One random pixel per sample, drawn uniformly from where `allow` is set."""
    n, _, size, _ = allow.shape
    w = allow.view(n, -1) + 1e-6                      # never all zero, so multinomial is safe
    idx = torch.multinomial(w, 1, generator=gen)
    seed = torch.zeros(n, size * size, device=allow.device)
    seed.scatter_(1, idx, 1.0)
    return seed.view(n, 1, size, size)


def _quarters(x0: torch.Tensor, gen: torch.Generator, margin: int, street_band: torch.Tensor,
              k_max: int = MASK_QUARTERS, reach: int = 56, bridge: int = 7) -> torch.Tensor:
    """
    One to `k_max` whole quarters, the way a person picks them: a seed block somewhere in
    the middle of the window, then its neighbours, each reached by stepping over the street
    from what is already selected.

    The walk is done once per batch and the grown variant is derived from it, because the
    walk is the expensive part and each sample only ever uses one kind of mask. `bridge`
    has to clear a street band to reach the next quarter, and is also what stops a selection
    from hopping a motorway: those are wide enough that it cannot.

    A walk that runs out of room, at the window edge or against a wide road, simply stops
    adding: the step lands on street, the flood finds nothing, and the union is unchanged.
    So `k_max` is a ceiling rather than a quota, and raising it widens the distribution of
    selection sizes instead of forcing every mask to be large.
    """
    n, _, size, _ = x0.shape
    dev = x0.device
    free = 1.0 - street_band                                   # everything that is not street
    inner = torch.zeros_like(free)
    inner[:, :, margin:-margin, margin:-margin] = 1.0
    m = _flood(_seed_in(free * inner, gen), free, reach)
    k = torch.randint(1, k_max + 1, (n, 1, 1, 1), generator=gen, device=dev).float()
    for j in range(2, k_max + 1):
        d = torch.randint(0, 4, (n, 1, 1, 1), generator=gen, device=dev)
        nxt = _flood(_shift(m, d, bridge) * free, free, reach)
        m = torch.where(k >= j, torch.maximum(m, nxt), m)
    return m


def _grow_into_street(m: torch.Tensor, street_band: torch.Tensor, grow: int) -> torch.Tensor:
    """
    Push a selection out into the street around it, so neighbouring quarters join and the
    street between them is inside the hole too. Kept inside (selection or street) so it
    cannot reach into a quarter that was not picked.
    """
    return _flood(m, torch.maximum(m, street_band), grow, step=1)


def _rects(n: int, size: int, k: int, device, gen: torch.Generator, lo=0.2, hi=0.8,
           margin: int = 0) -> torch.Tensor:
    ys = torch.arange(size, device=device).view(1, 1, size, 1).float()
    xs = torch.arange(size, device=device).view(1, 1, 1, size).float()
    m = torch.zeros(n, 1, size, size, device=device)
    span = size - 2 * margin                          # rectangles are drawn inside the margin,
    for _ in range(k):                                # not drawn and then clipped against it
        w = torch.empty(n, device=device).uniform_(lo, hi, generator=gen) * span
        h = torch.empty(n, device=device).uniform_(lo, hi, generator=gen) * span
        x0 = margin + torch.rand(n, device=device, generator=gen) * (span - w)
        y0 = margin + torch.rand(n, device=device, generator=gen) * (span - h)
        r = ((xs >= x0.view(n, 1, 1, 1)) & (xs < (x0 + w).view(n, 1, 1, 1)) &
             (ys >= y0.view(n, 1, 1, 1)) & (ys < (y0 + h).view(n, 1, 1, 1)))
        m = torch.maximum(m, r.float())
    return m


def _brush(n: int, size: int, device, gen: torch.Generator, strokes=(1, 4), radius=(0.04, 0.12),
           steps=(6, 16)) -> torch.Tensor:
    ys = torch.arange(size, device=device).view(1, size, 1).float()
    xs = torch.arange(size, device=device).view(1, 1, size).float()
    m = torch.zeros(n, 1, size, size, device=device)
    ns = int(torch.randint(strokes[0], strokes[1] + 1, (1,), generator=gen, device=device))
    for _ in range(ns):
        r = torch.empty(n, device=device).uniform_(radius[0], radius[1], generator=gen) * size
        p = torch.rand(n, 2, device=device, generator=gen) * size
        k = int(torch.randint(steps[0], steps[1] + 1, (1,), generator=gen, device=device))
        ang = torch.rand(n, device=device, generator=gen) * 2 * math.pi
        for _ in range(k):
            ang = ang + torch.randn(n, device=device, generator=gen) * 0.6
            step = r * 0.9
            p = p + torch.stack([torch.cos(ang), torch.sin(ang)], 1) * step.view(n, 1)
            p = p.clamp(0, size - 1)
            d2 = (ys - p[:, 1].view(n, 1, 1)) ** 2 + (xs - p[:, 0].view(n, 1, 1)) ** 2
            m[:, 0] = torch.maximum(m[:, 0], (d2 <= (r.view(n, 1, 1) ** 2)).float())
    return m


def random_masks(x0: torch.Tensor, gen: torch.Generator, probs=None,
                 margin: int | None = None, grow_px: int = 4) -> torch.Tensor:
    """
    [B,1,S,S] float masks, one kind per sample drawn from MASK_PROBS.

    Seven kinds, in three families, chosen to match what the viewer actually sends:

      rect, multirect, brush   a drawn selection, any shape, streets and all
      plots                    a drawn selection with the street band cut out of it, which
                               is the viewer's "keep the existing streets" checkbox
      qpart                    part of one quarter: a single plot inside a block
      quarters                 whole quarters, streets left alone
      qgrown                   whole quarters pushed out into the street between them, so
                               the street is inside the hole and has to be rebuilt too

    The three quarter kinds are the majority because they are the majority of what gets
    asked for, and because they are where the metric condition has the most to say: with
    the block outline given by the mask boundary, what is left undetermined is how much is
    built, how tall and how green, which is exactly what the six numbers control. A
    rectangle cutting across a street grid spends most of the model's effort on putting the
    streets back, and the numbers are a small residual on top of that.

    Every kind is kept `margin` working pixels clear of the border. It may come close, and
    it may be most of the tile, but the ring of real fabric around it is always there.
    """
    n, _, size, _ = x0.shape
    dev = x0.device
    probs = MASK_PROBS if probs is None else probs      # resolved here, not at import, so the
    margin = MASK_MARGIN if margin is None else margin  # trainer's flags reach every caller
    kind = torch.multinomial(torch.tensor(probs, device=dev), n, replacement=True, generator=gen)
    street_band = (x0[:, 2:3] > -1.5 / SDF_TRUNC).float()          # street plus a 1.5 px buffer
    rect = _rects(n, size, 1, dev, gen, lo=0.2, hi=0.95, margin=margin)
    multi = _rects(n, size, 3, dev, gen, lo=0.15, hi=0.5, margin=margin)
    brush = _interior(_brush(n, size, dev, gen), margin)
    plots = _rects(n, size, 1, dev, gen, lo=0.4, hi=0.95, margin=margin) * (1.0 - street_band)
    walk = _quarters(x0, gen, margin, street_band, k_max=MASK_QUARTERS)   # one walk, three kinds
    quarters = _interior(walk, margin)
    qgrown = _interior(_grow_into_street(walk, street_band, grow_px), margin)
    qpart = quarters * _rects(n, size, 1, dev, gen, lo=0.3, hi=0.9, margin=margin)
    stack = torch.stack([rect, multi, brush, plots, qpart, quarters, qgrown], 0)   # [7,B,1,S,S]
    m = stack[kind, torch.arange(n, device=dev)]
    # a crop with no real street network floods to the whole tile, and an unlucky quarter
    # pick can come back empty; either way fall back to the plain rectangle for that sample
    frac = m.mean((1, 2, 3), keepdim=True)
    bad = (frac > 0.92) | (frac < 0.004)
    return torch.where(bad, rect, m)


# ---------------------------------------------------------------------------
# metrics of the hole
# ---------------------------------------------------------------------------

def hole_metrics(x0: torch.Tensor, mask: torch.Tensor, scale: float, byte_to_m: torch.Tensor,
                 res_m: float = NATIVE_M_PER_PX) -> torch.Tensor:
    """
    [B,4,S,S] truth in [-1,1], [B,1,S,S] mask -> [B,6] raw metrics in physical units.
    Works on generated tiles too (that is how achieved metrics are reported).
    """
    fp = x0[:, 0] >= 0          # >= 0 matches the byte decoder (128 is inside); an exact 2x
    st = x0[:, 2] >= 0          # downsample puts boundary pixels at exactly 0
    gr = x0[:, 3] >= 0
    m = mask[:, 0] > 0.5
    area = m.sum((1, 2)).float().clamp(min=1)
    hb = ((x0[:, 1] + 1) * 127.5).round().clamp(0, 255).long()
    metres = byte_to_m.to(x0.device)[hb]
    fpm = fp & m
    nfp = fpm.sum((1, 2)).float()
    cov = nfp / area
    green = (gr & m).sum((1, 2)).float() / area
    street = (st & m).sum((1, 2)).float() / area
    h_sum = (metres * fpm).sum((1, 2))
    height = torch.where(nfp > 0, h_sum / nfp.clamp(min=1), torch.zeros_like(h_sum))
    far = h_sum / FLOOR_HEIGHT_M / area
    eroded = -F.max_pool2d(-fp.float().unsqueeze(1), 3, 1, 1)[:, 0]
    boundary = fp & (eroded < 0.5) & m
    per = boundary.sum((1, 2)).float()
    mpp = scale * res_m
    bsize = torch.where(per > 0, 4.0 * nfp * mpp / per.clamp(min=1), torch.zeros_like(per))
    return torch.stack([cov, green, street, height, far, bsize], 1)


def soft_hole_metrics(x0: torch.Tensor, mask: torch.Tensor, scale: float, byte_to_m: torch.Tensor,
                      res_m: float = NATIVE_M_PER_PX, sharp: float = 12.0) -> torch.Tensor:
    """
    hole_metrics, but differentiable: the same six numbers, with every hard decision
    replaced by a smooth one.

    The class channels are signed distance fields with the boundary at 0, so `inside` is a
    sigmoid of the field rather than a threshold, and the height is read out of the LUT by
    linear interpolation rather than by rounding to a byte. The numbers agree with
    hole_metrics to within a percent or so on a clean tile and drift on a noisy one, which
    is exactly where the gradient is wanted. Used to steer sampling toward a requested
    metric; never use it to report an achieved value, hole_metrics is the measurement.
    """
    fp = torch.sigmoid(x0[:, 0] * sharp)
    st = torch.sigmoid(x0[:, 2] * sharp)
    gr = torch.sigmoid(x0[:, 3] * sharp)
    m = (mask[:, 0] > 0.5).to(x0.dtype)
    area = m.sum((1, 2)).clamp(min=1.0)
    hb = ((x0[:, 1] + 1) * 127.5).clamp(0, 255)
    tab = byte_to_m.to(device=x0.device, dtype=x0.dtype)
    i0 = hb.detach().floor().long().clamp(0, 254)
    metres = tab[i0] + (hb - i0.to(x0.dtype)) * (tab[i0 + 1] - tab[i0])
    fpm = fp * m
    nfp = fpm.sum((1, 2))
    cov = nfp / area
    green = (gr * m).sum((1, 2)) / area
    street = (st * m).sum((1, 2)) / area
    h_sum = (metres * fpm).sum((1, 2))
    height = h_sum / nfp.clamp(min=1e-3)
    far = h_sum / FLOOR_HEIGHT_M / area
    eroded = -F.max_pool2d(-fp.unsqueeze(1), 3, 1, 1)[:, 0]
    per = ((fp - eroded).clamp(min=0) * m).sum((1, 2))
    bsize = 4.0 * nfp * (scale * res_m) / per.clamp(min=1e-3)
    return torch.stack([cov, green, street, height, far, bsize], 1)


class CondNorm:
    """Mean / std of the raw metrics, so the model sees O(1) inputs. Saved in checkpoints."""

    def __init__(self, mean=None, std=None):
        self.mean = torch.tensor(mean if mean is not None else [0.0] * N_COND)
        self.std = torch.tensor(std if std is not None else [1.0] * N_COND)

    @classmethod
    def estimate(cls, sampler: "CropSampler", byte_to_m, batches: int = 20, batch: int = 64, seed: int = 5):
        gen = torch.Generator(device=sampler.device).manual_seed(seed)
        vals = []
        for _ in range(batches):
            x0, scale = sampler.next()
            m = random_masks(x0, gen)
            r = hole_metrics(x0, m, scale, byte_to_m, sampler.cache.res_m)
            vals.append(r[m[:, 0].mean((1, 2)) > 0.01])
        v = torch.cat(vals).cpu()
        return cls(v.mean(0).tolist(), (v.std(0) + 1e-3).tolist())

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw - self.mean.to(raw.device)) / self.std.to(raw.device)

    def state(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "names": list(METRICS)}


def cond_with_flags(norm_vals: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    """[B,6] normalised metrics and [B,6] 0/1 flags -> [B,12] model condition."""
    return torch.cat([norm_vals * flags, flags], 1)


def train_flags(n: int, gen: torch.Generator, device, p_each: float = 0.3, p_all: float = 0.15) -> torch.Tensor:
    """Per-metric dropout so any subset can be requested; p_all is the fully unconditional draw."""
    f = (torch.rand(n, N_COND, device=device, generator=gen) >= p_each).float()
    keep_all = (torch.rand(n, 1, device=device, generator=gen) >= p_all).float()
    return f * keep_all


def targets_to_cond(targets: dict, norm: CondNorm, device) -> torch.Tensor:
    """{"far": 2.0, "green": 0.15} -> [1,12] condition; missing metrics are withheld."""
    raw = torch.zeros(1, N_COND)
    flags = torch.zeros(1, N_COND)
    for i, k in enumerate(METRICS):
        if k in targets and targets[k] is not None:
            raw[0, i] = float(targets[k])
            flags[0, i] = 1.0
    return cond_with_flags(norm(raw), flags).to(device)


# ---------------------------------------------------------------------------
# decode for previews / export
# ---------------------------------------------------------------------------

def fields_to_bytes(x: torch.Tensor) -> np.ndarray:
    """
    [B,4,S,S] in [-1,1] -> uint8 [B,4,S,S] in the tile_codec sdf byte layout: distance
    channels 128 + 127 x (so byte >= 128 is inside, exactly as tile_codec decodes), the
    height channel (x + 1) * 127.5 (the inverse of raw_to_fields).
    """
    x = x.clamp(-1, 1)
    b = torch.empty_like(x)
    s = (128.0 + 127.0 * x[:, [0, 2, 3]]).round()
    # keep the round trip exact: x >= 0 always lands on a byte >= 128, x < 0 on <= 127
    b[:, [0, 2, 3]] = torch.where(x[:, [0, 2, 3]] >= 0, s.clamp(min=128), s.clamp(max=127))
    b[:, 1] = ((x[:, 1] + 1.0) * 127.5).round()
    return b.clamp(0, 255).to(torch.uint8).cpu().numpy()
