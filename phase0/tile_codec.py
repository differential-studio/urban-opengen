"""
tile_codec.py

Shared definitions for the Urban OpenGen tile format.

RGB palette used by the Grasshopper renderer:
    background / open ground   (0, 0, 0)
    street                     (255, 0, 0)
    greenery                   (128, 255, 0)
    building                   (v, v, v)   grey, v encodes height (observed 105..241)

4-channel training format (all uint8, shape [4, H, W]). Two encodings share
the same decode rule (mask = byte >= 128, height read only where footprint):

  "mask"  0  footprint   0 or 255
          1  height      0 outside footprint, HEIGHT_MIN..255 inside (corpus-wide percentile rank)
          2  street      0 or 255
          3  green       0 or 255

  "sdf"   0  footprint   truncated signed distance to the footprint edge, 128 = edge,
                         >128 inside, <128 outside, saturating SDF_TRUNC_PX away
          1  height      as above inside the footprint; outside, the nearest
                         building's height, blurred, so the field has no cliffs
          2  street      signed distance, as channel 0
          3  green       signed distance, as channel 0

The "sdf" encoding exists because hard 0/255 masks are a give-away for the
GAN discriminator: a generator's output is never exactly binary, so the
discriminator can tell real from fake by that alone, no matter how strong the
augmentation. Smooth fields remove that shortcut, survive resampling without
aliasing, and threshold back to clean masks with sub-pixel edges.

Nothing here depends on torch, so the same module can be used by the
audit, the re-encoder, the metrics script, and the export step for the UI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (128, 255, 0)

# Renderer's height encoding, confirmed against the Grasshopper definition:
# grey 100 -> 4 m, grey 255 -> 100 m, linear in between.
GREY_MIN, GREY_MAX = 100.0, 255.0
HEIGHT_M_MIN, HEIGHT_M_MAX = 4.0, 100.0
FLOOR_HEIGHT_M = 3.0


def grey_to_metres(grey) -> np.ndarray:
    g = np.clip(np.asarray(grey, dtype=np.float64), GREY_MIN, GREY_MAX)
    return HEIGHT_M_MIN + (g - GREY_MIN) * (HEIGHT_M_MAX - HEIGHT_M_MIN) / (GREY_MAX - GREY_MIN)


def metres_to_grey(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    g = GREY_MIN + (m - HEIGHT_M_MIN) * (GREY_MAX - GREY_MIN) / (HEIGHT_M_MAX - HEIGHT_M_MIN)
    return np.clip(np.round(g), GREY_MIN, GREY_MAX)

# Height channel value assigned to the lowest building. Keeping it well above 0
# means "shortest building" and "no building" are never confused in channel 1.
HEIGHT_MIN = 51  # 0.2 * 255

CHANNEL_NAMES = ("footprint", "height", "street", "green")


# ---------------------------------------------------------------------------
# Class masks from RGB
# ---------------------------------------------------------------------------

def load_rgb(path: str | Path) -> np.ndarray:
    """Load a tile as an [H, W, 3] uint8 RGB array. Alpha is discarded."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def split_classes(rgb: np.ndarray):
    """
    Return (footprint, grey_value, street, green, off_palette) from an RGB tile.

    footprint, street, green, off_palette : bool [H, W]
    grey_value                            : uint8 [H, W], raw grey where footprint else 0
    """
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)

    black = (r == 0) & (g == 0) & (b == 0)
    street = (r == 255) & (g == 0) & (b == 0)
    green = (r == 128) & (g == 255) & (b == 0)
    footprint = (r == g) & (g == b) & (r > 0)
    off_palette = ~(black | street | green | footprint)

    grey_value = np.where(footprint, r, 0).astype(np.uint8)
    return footprint, grey_value, street, green, off_palette


# ---------------------------------------------------------------------------
# Height look-up table (corpus-wide percentile transform)
# ---------------------------------------------------------------------------

def _inverse_from_forward(forward: np.ndarray) -> np.ndarray:
    """
    Height byte -> grey. Only ever applied inside the footprint mask (invert() zeroes
    the rest), so a byte below HEIGHT_MIN, which a generator can produce, decodes to
    the shortest building rather than to grey 0 (which would read as -58 m).
    """
    forward = np.asarray(forward, dtype=np.int64)
    present = np.nonzero(forward > 0)[0]
    if present.size == 0:
        return np.zeros(256, dtype=np.int64)
    vals = forward[present]
    inverse = np.zeros(256, dtype=np.int64)
    for h in range(256):
        inverse[h] = present[0] if h < HEIGHT_MIN else present[np.argmin(np.abs(vals - h))]
    return inverse


@dataclass
class HeightLUT:
    """
    Maps raw grey (0..255) to a rank-normalised height byte and back.

    forward[g]  : uint8 height value for grey g (0 for greys that never occur as a building)
    inverse[h]  : grey value whose percentile is nearest to height byte h
    counts[g]   : how many building pixels had grey g in the corpus (for the record)
    """
    forward: list[int]
    inverse: list[int]
    counts: list[int]

    @classmethod
    def from_counts(cls, counts: np.ndarray, min_step: int = 4) -> "HeightLUT":
        """
        Histogram-equalised mapping with a guaranteed minimum spacing.

        Pure percentile equalisation gives the common heights lots of room but
        collapses rare tall buildings onto the same byte, which loses them.
        So: place each present grey level at its mid-rank percentile within
        [HEIGHT_MIN, 255], then enforce at least `step` bytes between
        consecutive levels with a forward pass (push up) and a backward pass
        (pull down from 255). `step` is `min_step` when there is room, and
        shrinks towards 1 as the number of distinct levels approaches the
        available byte range. The result is always injective, so the round
        trip is lossless.
        """
        counts = np.asarray(counts, dtype=np.int64)
        assert counts.shape == (256,)
        total = counts.sum()
        if total == 0:
            raise ValueError("no building pixels found, cannot build height LUT")

        present_greys = np.nonzero(counts > 0)[0]
        k = len(present_greys)
        span = 255 - HEIGHT_MIN
        if k - 1 > span:
            raise ValueError(f"{k} distinct grey levels do not fit in {span + 1} height bytes; lower HEIGHT_MIN")
        step = max(1, min(min_step, span // max(k - 1, 1)))

        cdf = np.cumsum(counts) / total
        cdf_prev = np.concatenate([[0.0], cdf[:-1]])
        mid = ((cdf_prev + cdf) / 2.0)[present_greys]          # mid-rank per present level

        vals = np.round(HEIGHT_MIN + mid * span).astype(np.int64)
        vals[0] = HEIGHT_MIN
        if k > 1:
            vals[-1] = 255
        for i in range(1, k):                                   # forward: at least step above previous
            vals[i] = max(vals[i], vals[i - 1] + step)
        vals[-1] = min(vals[-1], 255)
        for i in range(k - 2, -1, -1):                          # backward: at least step below next
            vals[i] = min(vals[i], vals[i + 1] - step)
        assert vals[0] >= HEIGHT_MIN and vals[-1] <= 255, "height LUT out of range"
        assert len(set(vals.tolist())) == k, "height LUT is not injective"

        forward = np.zeros(256, dtype=np.int64)
        forward[present_greys] = vals

        return cls(forward=forward.tolist(), inverse=_inverse_from_forward(forward).tolist(),
                   counts=counts.tolist())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "HeightLUT":
        d = json.loads(Path(path).read_text())
        # The inverse is derived; rebuild it so files written by older versions of this
        # module pick up the current decoding rules.
        d["inverse"] = _inverse_from_forward(np.asarray(d["forward"])).tolist()
        return cls(**d)

    def apply(self, grey_value: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        fwd = np.asarray(self.forward, dtype=np.uint8)
        out = fwd[grey_value]
        out[~footprint] = 0
        return out

    def invert(self, height: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        inv = np.asarray(self.inverse, dtype=np.uint8)
        out = inv[height]
        out[~footprint] = 0
        return out

    def mean_byte(self) -> int:
        """Corpus-mean height byte (pixel-weighted); 128 if the counts are missing."""
        fwd = np.asarray(self.forward, dtype=np.float64)
        cnt = np.asarray(self.counts, dtype=np.float64)
        return int(round((fwd * cnt).sum() / cnt.sum())) if cnt.sum() > 0 else 128


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

def rgb_to_channels(rgb: np.ndarray, lut: HeightLUT) -> np.ndarray:
    """RGB [H, W, 3] -> uint8 [4, H, W]."""
    footprint, grey_value, street, green, _ = split_classes(rgb)
    height = lut.apply(grey_value, footprint)
    return np.stack([
        footprint.astype(np.uint8) * 255,
        height,
        street.astype(np.uint8) * 255,
        green.astype(np.uint8) * 255,
    ])


ENCODINGS = ("mask", "sdf")
SDF_TRUNC_PX = 8.0      # native pixels (4 m each) at which the signed distance saturates
HEIGHT_SIGMA_PX = 1.5   # blur of the height fill outside footprints, native pixels


def signed_distance(mask: np.ndarray, trunc: float = SDF_TRUNC_PX) -> np.ndarray:
    """
    Truncated signed distance field of a boolean mask, in pixels. Positive
    inside, negative outside, the zero crossing on the pixel edge: a boundary
    pixel inside reads +0.5, its outside neighbour -0.5, so `sdf > 0` recovers
    the mask exactly.
    """
    from scipy import ndimage
    if mask.all():
        return np.full(mask.shape, trunc, dtype=np.float32)
    if not mask.any():
        return np.full(mask.shape, -trunc, dtype=np.float32)
    d_in = ndimage.distance_transform_edt(mask)
    d_out = ndimage.distance_transform_edt(~mask)
    s = np.where(mask, d_in - 0.5, -(d_out - 0.5)).astype(np.float32)
    return np.clip(s, -trunc, trunc)


def sdf_to_bytes(s: np.ndarray, trunc: float = SDF_TRUNC_PX) -> np.ndarray:
    """[-trunc, trunc] -> uint8 with 128 on the boundary. Inside lands >= 136."""
    return np.clip(np.rint(128.0 + s / trunc * 127.0), 0, 255).astype(np.uint8)


def fill_height(height: np.ndarray, footprint: np.ndarray, sigma: float = HEIGHT_SIGMA_PX,
                empty_value: int = 128) -> np.ndarray:
    """
    Height field without cliffs: exact height inside every footprint, and
    outside, the height of the nearest building blurred by `sigma`. Decoding
    only ever reads the channel where the footprint is set, so the interior
    stays lossless. A tile with no buildings at all becomes a flat field at
    `empty_value` (the re-encoder passes the corpus mean height byte), so it
    looks like any other far-from-buildings region rather than an all-zero
    tile that occurs nowhere else in the data.
    """
    from scipy import ndimage
    if not footprint.any():
        return np.full_like(height, np.uint8(empty_value))
    iy, ix = ndimage.distance_transform_edt(~footprint, return_distances=False, return_indices=True)
    filled = height[iy, ix].astype(np.float32)
    if sigma > 0:
        filled = ndimage.gaussian_filter(filled, sigma, mode="nearest")
    out = np.where(footprint, height.astype(np.float32), filled)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _resize_field(f: np.ndarray, size: int) -> np.ndarray:
    """Antialiased bilinear resize of a float32 field."""
    if f.shape[0] == size and f.shape[1] == size:
        return f
    im = Image.fromarray(np.ascontiguousarray(f, dtype=np.float32), mode="F")
    return np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32)


def encode_tile(rgb: np.ndarray, lut: HeightLUT, encoding: str = "mask", size: int = 0,
                sdf_trunc: float = SDF_TRUNC_PX, height_sigma: float = HEIGHT_SIGMA_PX,
                empty_height: int | None = None) -> np.ndarray:
    """
    RGB [H, W, 3] -> uint8 [4, S, S] in the chosen encoding, resized to `size`
    (0 keeps the native size). "mask" resizes with nearest-neighbour so the
    masks stay crisp; "sdf" computes the fields at native resolution and
    resamples them bilinearly, which is what a distance field wants.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")
    if encoding == "mask":
        ch = rgb_to_channels(rgb, lut)
        if size and ch.shape[1] != size:
            ch = np.stack([np.asarray(Image.fromarray(c).resize((size, size), Image.NEAREST)) for c in ch])
        return ch

    footprint, grey_value, street, green, _ = split_classes(rgb)
    if empty_height is None:
        empty_height = lut.mean_byte()
    height = fill_height(lut.apply(grey_value, footprint), footprint, height_sigma, empty_height)
    S = size or rgb.shape[0]
    fields = [
        signed_distance(footprint, sdf_trunc),
        height.astype(np.float32),
        signed_distance(street, sdf_trunc),
        signed_distance(green, sdf_trunc),
    ]
    fields = [_resize_field(f, S) for f in fields]
    return np.stack([
        sdf_to_bytes(fields[0], sdf_trunc),
        np.clip(np.rint(fields[1]), 0, 255).astype(np.uint8),
        sdf_to_bytes(fields[2], sdf_trunc),
        sdf_to_bytes(fields[3], sdf_trunc),
    ])


def channels_to_rgb(ch: np.ndarray, lut: HeightLUT, threshold: int = 128) -> np.ndarray:
    """
    uint8 [4, H, W] -> RGB [H, W, 3] in the original palette.

    Draw order matches the renderer: buildings, then green, then streets on top.
    Works on generated output too, since masks are thresholded.
    """
    footprint = ch[0] >= threshold
    street = ch[2] >= threshold
    green = ch[3] >= threshold
    grey = lut.invert(ch[1], footprint)

    out = np.zeros(ch.shape[1:] + (3,), dtype=np.uint8)
    out[footprint] = np.stack([grey[footprint]] * 3, axis=-1)
    out[green] = GREEN
    out[street] = RED
    return out


# ---------------------------------------------------------------------------
# Per-tile metrics (works on real and generated tiles alike)
# ---------------------------------------------------------------------------

def _label(mask: np.ndarray, eight: bool = False):
    """Connected components. eight=True counts diagonal touches as connected."""
    from scipy import ndimage
    structure = np.ones((3, 3), dtype=bool) if eight else None
    labels, n = ndimage.label(mask, structure=structure)
    return labels, int(n)


def skeletonize(mask: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """
    Zhang-Suen thinning, vectorised. Reduces a street mask to one-pixel-wide
    centrelines. Pure numpy so it has no dependency beyond what is already used.
    """
    img = mask.astype(np.uint8).copy()
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            p2 = P[:-2, 1:-1]; p3 = P[:-2, 2:]; p4 = P[1:-1, 2:]; p5 = P[2:, 2:]
            p6 = P[2:, 1:-1]; p7 = P[2:, :-2]; p8 = P[1:-1, :-2]; p9 = P[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(n.astype(np.int16) for n in nb)
            seq = nb + [p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                cond = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                cond = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            remove = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & cond
            if remove.any():
                img[remove] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def _skeleton_stats(skel: np.ndarray, cell_m: float):
    """centreline length (m), junction count, dead-end count from a skeleton"""
    if not skel.any():
        return 0.0, 0, 0
    P = np.pad(skel.astype(np.uint8), 1)
    n4 = P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:]
    nd = P[:-2, :-2] + P[:-2, 2:] + P[2:, :-2] + P[2:, 2:]
    n8 = n4 + nd
    # each edge counted from both ends, so halve; diagonal steps are sqrt(2) long
    length = float(((n4 * 1.0 + nd * np.sqrt(2.0)) * skel).sum() / 2.0 * cell_m)
    junction_px = skel & (n8 >= 3)
    _, junctions = _label(junction_px, eight=True)        # thinning leaves small clusters at junctions
    dead_ends = int((skel & (n8 == 1)).sum())
    return length, junctions, dead_ends


def tile_metrics(ch: np.ndarray, lut: HeightLUT | None = None, tile_m: float = 400.0,
                 threshold: int = 128) -> dict:
    """
    Morphology metrics from a 4-channel tile.

    tile_m is the tile edge in metres (400 for the 10 km / 25 grid); cell size
    follows from the pixel count. Height metrics go through the LUT inverse to
    raw grey and then grey_to_metres(). Every value is a plain float or int.
    """
    footprint = ch[0] >= threshold
    street = ch[2] >= threshold
    green = ch[3] >= threshold
    H, W = footprint.shape
    n = footprint.size
    cell_m = tile_m / W
    cell_m2 = cell_m * cell_m
    tile_m2 = tile_m * tile_m

    m = {
        "coverage_pct": 100.0 * footprint.sum() / n,     # ground coverage ratio
        "street_pct": 100.0 * street.sum() / n,
        "green_pct": 100.0 * green.sum() / n,
        "open_pct": 100.0 * (~(footprint | street | green)).sum() / n,
        "footprint_area_m2": float(footprint.sum() * cell_m2),
        "street_area_m2": float(street.sum() * cell_m2),
        "green_area_m2": float(green.sum() * cell_m2),
    }

    if lut is not None:
        grey = lut.invert(ch[1], footprint).astype(np.float64)
    else:
        grey = ch[1].astype(np.float64)

    if footprint.any():
        h = grey_to_metres(grey[footprint])
        m["mean_height_m"] = float(h.mean())
        m["max_height_m"] = float(h.max())
        m["far_proxy"] = float((h / FLOOR_HEIGHT_M).sum() / n)     # gross floor area / site area
        m["gfa_m2"] = float((h / FLOOR_HEIGHT_M).sum() * cell_m2)
    else:
        m["mean_height_m"] = 0.0
        m["max_height_m"] = 0.0
        m["far_proxy"] = 0.0
        m["gfa_m2"] = 0.0

    _, nb = _label(footprint, eight=False)                # corner-touching = separate buildings
    m["building_count"] = nb
    m["mean_footprint_m2"] = float(m["footprint_area_m2"] / nb) if nb else 0.0

    _, ns = _label(street, eight=True)                    # diagonal streets stay connected
    m["street_components"] = ns                          # 1 means a fully connected network
    length, junctions, dead_ends = _skeleton_stats(skeletonize(street), cell_m)
    m["street_length_m"] = length
    m["street_length_km_per_km2"] = float(length / 1000.0 / (tile_m2 / 1e6))
    m["intersections"] = junctions
    m["dead_ends"] = dead_ends

    # blocks: land not taken by streets, 4-connected so a diagonal street still splits
    labels, nblk = _label(~street, eight=False)
    m["block_count"] = nblk
    m["mean_block_m2"] = float((~street).sum() * cell_m2 / nblk) if nblk else 0.0
    return {k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in m.items()}


# Display names and units for UIs; order is a sensible default for sliders.
METRIC_INFO = {
    "far_proxy":           ("FAR",                 "",     2),
    "coverage_pct":        ("coverage",            "%",    1),
    "green_pct":           ("green",               "%",    1),
    "street_pct":          ("street",              "%",    1),
    "mean_height_m":       ("mean height",         "m",    1),
    "max_height_m":        ("max height",          "m",    1),
    "street_length_m":     ("street length",       "m",    0),
    "intersections":       ("intersections",       "",     0),
    "dead_ends":           ("dead ends",           "",     0),
    "building_count":      ("buildings",           "",     0),
    "mean_footprint_m2":   ("mean footprint",      "m2",   0),
    "block_count":         ("blocks",              "",     0),
    "mean_block_m2":       ("mean block",          "m2",   0),
    "footprint_area_m2":   ("footprint area",      "m2",   0),
    "street_area_m2":      ("street area",         "m2",   0),
    "green_area_m2":       ("green area",          "m2",   0),
    "gfa_m2":              ("gross floor area",    "m2",   0),
    "street_components":   ("street parts",        "",     0),
    "open_pct":            ("open ground",         "%",    1),
    "street_length_km_per_km2": ("street density", "km/km2", 1),
}
