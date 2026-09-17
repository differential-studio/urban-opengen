"""
analysis.py

Direct sun and view analysis for an Urban OpenGen option, plus the density
numbers, computed on the same geometry the viewer draws and the 3dm export
writes.

This is for comparing rough pre-concept schemes against each other. It is not a
daylight report: there is no diffuse light, no reflection, no glazing and no
material. What it does compute is exact for the massing it is given, which is
what makes two options comparable.

Three simplifications, stated once here and repeated in the output:

  * nothing outside the tile casts a shadow or blocks a view, so points near the
    edge are optimistic. Set `margin_m` to exclude a border from the statistics
  * the massing is the extruded footprints, so a pitched roof, a setback or a
    balcony does not exist
  * facade results are sampled every few metres at a few levels, which is a
    block-scale reading, not a room-scale one

What comes out
--------------
form   FAR, ground coverage, open space ratio, mean and max height, facade area
       and facade-to-floor ratio, green and street share, building count and
       mean footprint. All from the polygons, so they describe the model you can
       download rather than the raster the model drew.
sun    Direct sun hours on open ground and on the facades, for one or more days,
       from the true solar geometry at the tile's latitude. A point sees the sun
       when nothing between it and the sun rises above the sun ray.
view   From every facade sample point, the share of the outward hemisphere taken
       by sky, by other buildings, by greenery and by street or open ground; the
       share taken by buildings within 25 m, which is the overlooking or privacy
       reading; and the mean distance a view ray travels before it hits
       something, which is a reading of enclosure.

How the sun part is fast
------------------------
For one sun position, `horizon_field` returns, for every cell, the lowest height
from which the sun is visible there. A point at (x, y, z) sees the sun when
z > M(x, y), so the same field answers the question for the ground (z = 0) and
for every facade sample above it. M is the running maximum along the ray toward
the sun of the obstruction height minus the height the ray has climbed, and that
maximum is computed by doubling: the maximum over 2k steps is the maximum over k
steps combined with the same field shifted k steps and lowered by what the ray
climbs in k steps. So 128 steps of marching cost 8 array shifts, not 128, and a
whole day of sun positions takes a few hundred milliseconds.

Only the view rays are marched one by one, batched over all rays at once.

Dependencies: numpy, scipy, shapely, pillow. No torch, no solar library.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

NEG = -1.0e9
FLOOR_M = 3.0

DEFAULTS = dict(
    px_m=2.0,               # analysis raster resolution in metres; 2 m is finer than the tile's 3.1 m
    margin_m=0.0,           # exclude this much of the tile border from the ground statistics
    # sun
    days=("equinox", "winter"),
    sun_step_min=30,        # sun is sampled every this many minutes of the day
    sun_steps=128,          # march resolution of the horizon field (doubling, so cost is log2 of this)
    sun_threshold_h=2.0,    # "share of open ground above N hours"
    # facades
    facade_spacing_m=10.0,  # sample spacing along every facade
    facade_levels=2,        # sample heights per facade, spread over its height
    facade_offset_m=1.0,    # how far the sample sits off the wall
    facade_min_h=3.0,       # facades shorter than this are ignored
    # view
    view_az=8,              # ray directions across the facade, over the full outward half
    view_el=8,              # ray directions up and down; forced even, see view_directions
    view_max_m=250.0,
    view_step_m=2.0,
    overlook_m=25.0,        # a building surface closer than this counts as overlooking
)

DECLINATION = {"equinox": 0.0, "winter": -23.44, "summer": 23.44}


def _opts(kw):
    o = dict(DEFAULTS)
    o.update({k: v for k, v in kw.items() if v is not None})
    return o


# ---------------------------------------------------------------------------
# the analysis raster: height in metres and a class per cell, from the polygons
# ---------------------------------------------------------------------------

CLS_OPEN, CLS_STREET, CLS_GREEN, CLS_BUILDING = 0, 1, 2, 3


def rasterize_parts(parts: dict, px_m: float, supersample: int = 3):
    """
    -> (H metres [R, C], cls [R, C], (x0, y1)) with column east and row south.

    Heights are the maximum of the polygons covering a cell, so a stepped
    building keeps its taller part. Classes are painted open, street, green,
    building, in that order, so a building always wins.
    """
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = parts["tile"].bounds
    C = max(1, int(round((x1 - x0) / px_m)))
    R = max(1, int(round((y1 - y0) / px_m)))
    S = supersample

    def to_px(xy):
        return [(float((x - x0) / px_m * S), float((y1 - y) / px_m * S)) for x, y in xy]

    def paint(items, value_of, mode, blocks_max=False):
        im = Image.new(mode, (C * S, R * S), 0)
        dr = ImageDraw.Draw(im)
        for it in items:
            g, v = it if isinstance(it, tuple) else (it, 1)
            for poly in ([g] if isinstance(g, Polygon) else list(getattr(g, "geoms", []))):
                if poly.is_empty:
                    continue
                dr.polygon(to_px(poly.exterior.coords), fill=value_of(v))
                for h in poly.interiors:
                    dr.polygon(to_px(h.coords), fill=0)
        a = np.asarray(im)
        a = a.reshape(R, S, C, S)
        return a.max((1, 3)) if blocks_max else (a.mean((1, 3)) >= 0.5)

    H = paint([(g, h) for g, h in parts["buildings"]], lambda v: float(v), "F", blocks_max=True).astype(np.float32)
    built = paint([g for g, _ in parts["buildings"]], lambda v: 1, "1")
    street = paint(parts["plates"], lambda v: 1, "1")
    green = paint(parts["greens"], lambda v: 1, "1")
    cls = np.full((R, C), CLS_OPEN, np.uint8)
    cls[street] = CLS_STREET
    cls[green] = CLS_GREEN
    cls[built] = CLS_BUILDING
    H[~built] = 0.0
    return H, cls, (x0, y1)


def rasterize_area(area, tile, px_m: float, shape) -> np.ndarray:
    """one polygon as a cell mask on the analysis grid"""
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = tile.bounds
    R, C = shape
    im = Image.new("1", (C, R), 0)
    dr = ImageDraw.Draw(im)
    for poly in ([area] if isinstance(area, Polygon) else list(getattr(area, "geoms", []))):
        dr.polygon([((x - x0) / px_m, (y1 - y) / px_m) for x, y in poly.exterior.coords], fill=1)
    return np.asarray(im, bool)


# ---------------------------------------------------------------------------
# sun
# ---------------------------------------------------------------------------

def solar_positions(lat_deg: float, decl_deg: float, step_min: int = 30):
    """
    -> (altitude deg [n], azimuth deg from north clockwise [n], hours each [n])

    Sun positions through one day at this latitude and declination, in true
    solar time, which is all that "hours of direct sun" needs: longitude only
    decides what a clock would say. The weights sum to the astronomical day
    length, so an unobstructed point returns exactly that.
    """
    phi = math.radians(lat_deg)
    dec = math.radians(decl_deg)
    c = -math.tan(phi) * math.tan(dec)
    if c >= 1.0:
        return np.zeros(0), np.zeros(0), np.zeros(0)          # polar night
    omega_s = math.pi if c <= -1.0 else math.acos(c)          # sunset hour angle
    day_h = 2.0 * math.degrees(omega_s) / 15.0
    step = math.radians(15.0 * step_min / 60.0)
    n = max(1, int(round(2 * omega_s / step)))
    omega = -omega_s + (np.arange(n) + 0.5) * (2 * omega_s / n)
    sin_alt = math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * np.cos(omega)
    alt = np.degrees(np.arcsin(np.clip(sin_alt, -1, 1)))
    az = np.degrees(np.arctan2(np.sin(omega),
                               np.cos(omega) * math.sin(phi) - math.tan(dec) * math.cos(phi))) + 180.0
    w = np.full(n, day_h / n)
    keep = alt > 0.0
    return alt[keep], az[keep] % 360.0, w[keep]


def horizon_field(H: np.ndarray, alt_deg: float, az_deg: float, px_m: float,
                  steps: int = 128, reach_m: float | None = None) -> np.ndarray:
    """
    Lowest height, per cell, from which this sun position is visible, in metres.

    A point at (x, y, z) sees the sun when z > M(x, y). Costs log2(steps) shifts.
    Outside the raster there is no obstruction, which is the optimistic edge
    assumption this module is honest about.
    """
    R, C = H.shape
    alt = math.radians(max(alt_deg, 0.25))
    az = math.radians(az_deg)
    if reach_m is None:
        reach_m = min(float(H.max()) / math.tan(alt) + px_m, math.hypot(R, C) * px_m)
    reach_m = max(reach_m, px_m)
    s = max(px_m, reach_m / steps)
    dc = math.sin(az) * s / px_m            # column grows east
    dr = -math.cos(az) * s / px_m           # row grows south
    dz = math.tan(alt) * s
    rr, cc = np.mgrid[0:R, 0:C].astype(np.float32)

    def shifted(A, k):
        """A sampled k steps along the ray toward the sun"""
        return ndimage.map_coordinates(A, [rr + k * dr, cc + k * dc], order=1,
                                       mode="constant", cval=NEG)

    M = shifted(H, 1) - dz                                   # max over one step
    k = 1
    while k < steps:
        M = np.maximum(M, shifted(M, k) - k * dz)            # max over 2k steps
        k *= 2
    return M


# ---------------------------------------------------------------------------
# facade sample points
# ---------------------------------------------------------------------------

def facade_points(parts: dict, o: dict, px_m: float | None = None):
    """
    -> dict of arrays: xy [n, 2] metres (offset off the wall), z [n], normal [n, 2],
       area [n] (the facade area each point stands for), height [n] (its building's).

    Courtyard walls are included; both rings give an outward normal of (dy, -dx)
    once the polygon is oriented, because a hole runs the other way round.
    """
    xs, ys, zs, nx, ny, ar, bh = [], [], [], [], [], [], []
    # a polygon edge that falls mid-cell makes the raster building up to half a cell fatter than it
    # is, so a sample 1 m off the wall can sit on the raster boundary and be shadowed, or seen, by
    # its own building. Standing one and a half cells clear costs a little realism and removes that.
    off = max(o["facade_offset_m"], 1.5 * (px_m if px_m else o["px_m"]))
    sp = o["facade_spacing_m"]
    L = max(1, int(o["facade_levels"]))
    for g, h in parts["buildings"]:
        if h < o["facade_min_h"]:
            continue
        try:
            g = orient(g, 1.0)
        except Exception:
            continue
        for ring in [g.exterior] + list(g.interiors):
            p = np.asarray(ring.coords[:-1], float)
            n = len(p)
            if n < 3:
                continue
            for i in range(n):
                a, b = p[i], p[(i + 1) % n]
                d = b - a
                seg = math.hypot(*d)
                if seg < 1e-6:
                    continue
                u = d / seg
                nrm = np.array([u[1], -u[0]])           # outward for an oriented ring
                m = max(1, int(round(seg / sp)))
                for j in range(m):
                    t = (j + 0.5) / m
                    q = a + d * t + nrm * off
                    for k in range(L):
                        xs.append(q[0]); ys.append(q[1])
                        zs.append(h * (k + 0.5) / L)
                        nx.append(nrm[0]); ny.append(nrm[1])
                        ar.append(seg / m * h / L)
                        bh.append(h)
    if not xs:
        return None
    return dict(xy=np.stack([xs, ys], 1), z=np.asarray(zs), normal=np.stack([nx, ny], 1),
                area=np.asarray(ar), height=np.asarray(bh))


# ---------------------------------------------------------------------------
# ray marching, for the view analysis
# ---------------------------------------------------------------------------

def march(H, cls, origin_xy, origin_z, dirs, px_m, anchor, max_m, step_m):
    """
    Cast rays and report what each one hits first.

    -> (kind [n] in CLS_*, or -1 for nothing, distance [n] metres)

    A ray is inside solid when its height drops below the height field, which is
    zero on open ground, so a downward ray hits the ground and an upward ray hits
    a building, with one test. Rays that leave the raster or run out of distance
    report nothing, which the caller reads as sky above the horizon and as
    unknown ground below it.
    """
    x0, y1 = anchor
    R, C = H.shape
    n = len(origin_z)
    kind = np.full(n, -1, np.int8)
    dist = np.full(n, np.inf)
    alive = np.arange(n)
    K = max(1, int(math.ceil(max_m / step_m)))
    for k in range(1, K + 1):
        if alive.size == 0:
            break
        t = k * step_m
        px = origin_xy[alive, 0] + dirs[alive, 0] * t
        py = origin_xy[alive, 1] + dirs[alive, 1] * t
        pz = origin_z[alive] + dirs[alive, 2] * t
        col = (px - x0) / px_m - 0.5
        row = (y1 - py) / px_m - 0.5
        inside = (col >= -0.5) & (col <= C - 0.5) & (row >= -0.5) & (row <= R - 0.5)
        if not inside.any():
            alive = alive[np.zeros(alive.size, bool)]
            continue
        idx = alive[inside]
        h = ndimage.map_coordinates(H, [row[inside], col[inside]], order=1, mode="nearest")
        hit = pz[inside] <= h
        if hit.any():
            hid = idx[hit]
            r = np.clip(np.rint(row[inside][hit]).astype(int), 0, R - 1)
            c = np.clip(np.rint(col[inside][hit]).astype(int), 0, C - 1)
            kind[hid] = np.where(h[hit] > 0.5, CLS_BUILDING, cls[r, c])
            dist[hid] = t
        alive = idx[~hit]
    return kind, dist


def view_directions(normal_xy, o):
    """
    Rays over the outward hemisphere of each facade point, every one carrying the
    same solid angle.

    The solid angle element is cos(elevation) dElevation dAzimuth, so sampling
    the midpoints of equal bands of sin(elevation) and of azimuth samples the
    hemisphere uniformly and needs no weights. Exactly half the rays point above
    the horizon, so an unobstructed facade reads 50 percent sky, which is what
    makes the number mean something.
    """
    na, ne = int(o["view_az"]), int(o["view_el"])
    ne += ne % 2          # an odd count puts one band exactly on the horizon, which is half sky
    az = (np.arange(na) + 0.5) / na * math.pi - math.pi / 2          # midpoints over -90 .. 90
    el = np.arcsin((np.arange(ne) + 0.5) / ne * 2.0 - 1.0)           # midpoints over -1 .. 1 in sin
    A, E = np.meshgrid(az, el, indexing="ij")
    A = A.ravel(); E = E.ravel()
    w = np.full(A.size, 1.0 / A.size)
    ca, sa = np.cos(A), np.sin(A)
    n = normal_xy[:, None, :]                                   # [p, 1, 2]
    hx = n[..., 0] * ca - n[..., 1] * sa                        # rotate the normal in plane
    hy = n[..., 0] * sa + n[..., 1] * ca
    ce, se = np.cos(E), np.sin(E)
    d = np.stack([hx * ce, hy * ce, np.broadcast_to(se, hx.shape)], -1)   # [p, r, 3]
    return d, w


# ---------------------------------------------------------------------------
# form metrics, straight off the polygons
# ---------------------------------------------------------------------------

def form_metrics(parts: dict, area=None) -> dict:
    site = (area if area is not None else parts["tile"]).area
    blds = parts["buildings"] if area is None else [
        (g, h) for g, h in parts["buildings"] if area.contains(g.representative_point())]
    foot = sum(g.area for g, _ in blds)
    gfa = sum(g.area * max(1, round(h / FLOOR_M)) for g, h in blds)
    fac = 0.0
    for g, h in blds:
        per = g.exterior.length + sum(i.length for i in g.interiors)
        fac += per * h
    hs = np.asarray([h for _, h in blds]) if blds else np.zeros(0)
    ws = np.asarray([g.area for g, _ in blds]) if blds else np.zeros(0)
    # the fit splits a building at its height steps and at narrow necks, so the polygons are parts,
    # not buildings. Counting the pieces of their union counts buildings, which is what the raster
    # metrics in meta.csv count and what anyone reading the number expects.
    merged = unary_union([g for g, _ in blds]) if blds else None
    whole = [] if merged is None or merged.is_empty else (
        [merged] if merged.geom_type == "Polygon" else list(getattr(merged, "geoms", [])))
    clip = (lambda g: g if area is None else g.intersection(area))  # noqa: E731
    green = clip(unary_union(parts["greens"])).area if parts["greens"] else 0.0
    street = clip(unary_union(parts["plates"])).area if parts["plates"] else 0.0
    return {
        "site_area_m2": round(site, 1),
        "far": round(gfa / site, 3) if site else 0.0,
        "gfa_m2": round(gfa, 1),
        "coverage_pct": round(100.0 * foot / site, 2) if site else 0.0,
        "open_space_ratio": round((site - foot) / gfa, 3) if gfa else 0.0,
        "green_pct": round(100.0 * green / site, 2) if site else 0.0,
        "street_pct": round(100.0 * street / site, 2) if site else 0.0,
        "mean_height_m": round(float((hs * ws).sum() / ws.sum()), 2) if ws.sum() else 0.0,
        "max_height_m": round(float(hs.max()), 2) if len(hs) else 0.0,
        "building_count": len(whole),
        "mean_footprint_m2": round(foot / len(whole), 1) if whole else 0.0,
        "part_count": len(blds),
        "facade_area_m2": round(fac, 1),
        "facade_to_floor": round(fac / gfa, 3) if gfa else 0.0,
    }


# ---------------------------------------------------------------------------
# the whole thing
# ---------------------------------------------------------------------------

def analyse(parts: dict, lat: float = 52.0, lon: float | None = None, maps: bool = False,
            area=None, **kw) -> dict:
    """
    parts: the dict from vectorize.parts_m (shapely geometry in metres).
    lat:   latitude of the tile, which is all the sun geometry needs.
    maps:  also return the ground sun-hour rasters, for drawing.
    area:  report the numbers for this polygon only, while everything in `parts`
           still casts shadows and blocks views. This is what an infill needs:
           the plot is what is being judged, the city around it is what shades it.
    """
    o = _opts(kw)
    out = dict(form_metrics(parts, area))
    out["latitude"] = round(float(lat), 4)
    if lon is not None:
        out["longitude"] = round(float(lon), 4)

    H, cls, anchor = rasterize_parts(parts, o["px_m"])
    R, C = H.shape
    built = cls == CLS_BUILDING
    openg = ~built
    if area is not None:
        openg &= rasterize_area(area, parts["tile"], o["px_m"], (R, C))
    if o["margin_m"] > 0:
        m = int(round(o["margin_m"] / o["px_m"]))
        inner = np.zeros_like(openg)
        inner[m:R - m, m:C - m] = True
        openg = openg & inner
    fp = facade_points(parts, o, o["px_m"])
    if fp is not None and area is not None:
        from shapely.geometry import MultiPoint
        keep = np.array([area.contains(p) for p in MultiPoint(fp["xy"]).geoms], bool)
        fp = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in fp.items()} if keep.any() else None
    out["facade_samples"] = 0 if fp is None else len(fp["z"])

    ground_maps = {}
    point_sun = {}
    for day in o["days"]:
        decl = DECLINATION[day] if day in DECLINATION else float(day)
        if day == "winter":
            decl = -23.44 if lat >= 0 else 23.44
        elif day == "summer":
            decl = 23.44 if lat >= 0 else -23.44
        alt, az, w = solar_positions(lat, decl, o["sun_step_min"])
        gh = np.zeros((R, C), np.float32)
        fh = np.zeros(out["facade_samples"], np.float32)
        for a, z, hrs in zip(alt, az, w):
            M = horizon_field(H, a, z, o["px_m"], o["sun_steps"])
            gh += np.where(M <= 0.05, hrs, 0.0).astype(np.float32)
            if fp is not None:
                col = (fp["xy"][:, 0] - anchor[0]) / o["px_m"] - 0.5
                row = (anchor[1] - fp["xy"][:, 1]) / o["px_m"] - 0.5
                mv = ndimage.map_coordinates(M, [row, col], order=1, mode="nearest")
                sd = math.sin(math.radians(z)) * fp["normal"][:, 0] + math.cos(math.radians(z)) * fp["normal"][:, 1]
                fh += np.where((fp["z"] > mv) & (sd > 0.0), hrs, 0.0).astype(np.float32)
        tag = day if isinstance(day, str) else f"decl{day}"
        day_h = float(w.sum())
        out[f"sun_{tag}_daylength_h"] = round(day_h, 2)
        # every key is always present, so a batch of options is a rectangular table. A tile with
        # no street or no buildings gets an empty cell rather than a zero, which would be a lie.
        thr = o["sun_threshold_h"]
        g = gh[openg] if openg.any() else None
        st = openg & (cls == CLS_STREET)
        a_w = fp["area"] if fp is not None and len(fh) else None
        out[f"sun_{tag}_ground_h"] = round(float(g.mean()), 2) if g is not None else None
        out[f"sun_{tag}_ground_above_{thr:g}h_pct"] = round(100.0 * float((g >= thr).mean()), 1) if g is not None else None
        out[f"sun_{tag}_street_h"] = round(float(gh[st].mean()), 2) if st.any() else None
        out[f"sun_{tag}_facade_h"] = round(float((fh * a_w).sum() / a_w.sum()), 2) if a_w is not None else None
        out[f"sun_{tag}_facade_above_{thr:g}h_pct"] = round(100.0 * float(a_w[fh >= thr].sum() / a_w.sum()), 1) if a_w is not None else None
        if maps:
            ground_maps[tag] = gh
            point_sun[tag] = fh

    VIEW_KEYS = ("view_sky_pct", "view_building_pct", "view_green_pct", "view_street_pct",
                 "view_open_pct", "view_overlooked_pct", "view_obstruction_m")
    for k in VIEW_KEYS:
        out[k] = None
    if fp is not None:
        d, w = view_directions(fp["normal"], o)
        P, Rn, _ = d.shape
        ox = np.repeat(fp["xy"], Rn, axis=0)
        oz = np.repeat(fp["z"], Rn)
        dd = d.reshape(-1, 3)
        kind, dist = march(H, cls, ox, oz, dd, o["px_m"], anchor, o["view_max_m"], o["view_step_m"])
        kind = kind.reshape(P, Rn)
        dist = dist.reshape(P, Rn)
        up = dd[:, 2].reshape(P, Rn) >= 0
        sky = (kind < 0) & up
        beyond = (kind < 0) & ~up
        aw = fp["area"][:, None] * w[None, :]
        tot = aw.sum()
        share = lambda m: round(100.0 * float((aw * m).sum() / tot), 1)  # noqa: E731
        per_point = {"sky": sky.mean(1), "building": (kind == CLS_BUILDING).mean(1),
                     "green": (kind == CLS_GREEN).mean(1),
                     "overlooked": ((kind == CLS_BUILDING) & (dist <= o["overlook_m"])).mean(1)}
        out["view_sky_pct"] = share(sky)
        out["view_building_pct"] = share(kind == CLS_BUILDING)
        out["view_green_pct"] = share(kind == CLS_GREEN)
        out["view_street_pct"] = share(kind == CLS_STREET)
        out["view_open_pct"] = share((kind == CLS_OPEN) | beyond)
        out["view_overlooked_pct"] = share((kind == CLS_BUILDING) & (dist <= o["overlook_m"]))
        blocked = np.isfinite(dist)
        if blocked.any():
            bw = aw * blocked
            out["view_obstruction_m"] = round(float((bw * np.where(blocked, dist, 0.0)).sum() / bw.sum()), 1)
        else:
            out["view_obstruction_m"] = float(o["view_max_m"])
    out["_note"] = "no context outside the tile; massing only; facade values are block-scale"
    if maps:
        out["_maps"] = {"height": H, "class": cls, "anchor": anchor, "px_m": o["px_m"], "ground_sun": ground_maps}
        if fp is not None:
            out["_points"] = {**fp, "sun": point_sun, "view": per_point if "per_point" in dir() else None}
    return out


# ---------------------------------------------------------------------------
# self-test: cases with an answer known in advance
# ---------------------------------------------------------------------------

def _box_parts(tile_m=400.0, size=40.0, h=30.0, centre=(0.0, 0.0)):
    half = tile_m / 2
    t = box(-half, -half, half, half)
    b = box(centre[0] - size / 2, centre[1] - size / 2, centre[0] + size / 2, centre[1] + size / 2)
    return {"tile": t, "buildings": [(b, h)], "streets": [], "plates": [], "blocks": [t.difference(b)], "greens": []}


def _tower_parts(spacing=45.0, tile_m=400.0, w=30.0, h=40.0):
    half = tile_m / 2
    t = box(-half, -half, half, half)
    bs = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = sx * spacing / 2, sy * spacing / 2
            bs.append((box(cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2), h))
    return {"tile": t, "buildings": bs, "streets": [], "plates": [],
            "blocks": [t.difference(unary_union([b for b, _ in bs]))], "greens": []}


def _brute_force_sun(H, anchor, px_m, xy, z, alt_deg, az_deg, step_m=1.0, max_m=700.0):
    """the same question asked the slow, obvious way: walk toward the sun and see what is in the way"""
    x0, y1 = anchor
    R, C = H.shape
    dx, dy = math.sin(math.radians(az_deg)), math.cos(math.radians(az_deg))
    dz = math.tan(math.radians(max(alt_deg, 0.25)))
    for t in np.arange(step_m, max_m, step_m):
        px, py, pz = xy[0] + dx * t, xy[1] + dy * t, z + dz * t
        c = (px - x0) / px_m - 0.5
        r = (y1 - py) / px_m - 0.5
        if c < -0.5 or r < -0.5 or c > C - 0.5 or r > R - 0.5:
            return True
        if ndimage.map_coordinates(H, [[r], [c]], order=1, mode="constant", cval=0.0)[0] > pz:
            return False
    return True


def facade_sun_by_orientation(parts, lat, decl, o=None):
    """mean direct sun hours per cardinal facade orientation, for the self-test"""
    o = o or _opts({})
    H, cls, anchor = rasterize_parts(parts, o["px_m"])
    fp = facade_points(parts, o, o["px_m"])
    alt, az, w = solar_positions(lat, decl, o["sun_step_min"])
    tot = np.zeros(len(fp["z"]))
    for a, z, hrs in zip(alt, az, w):
        M = horizon_field(H, a, z, o["px_m"], o["sun_steps"])
        col = (fp["xy"][:, 0] - anchor[0]) / o["px_m"] - 0.5
        row = (anchor[1] - fp["xy"][:, 1]) / o["px_m"] - 0.5
        mv = ndimage.map_coordinates(M, [row, col], order=1, mode="nearest")
        sd = math.sin(math.radians(z)) * fp["normal"][:, 0] + math.cos(math.radians(z)) * fp["normal"][:, 1]
        tot += np.where((fp["z"] > mv) & (sd > 0.0), hrs, 0.0)
    out = {}
    for name, n in (("east", (1, 0)), ("north", (0, 1)), ("west", (-1, 0)), ("south", (0, -1))):
        sel = (np.abs(fp["normal"][:, 0] - n[0]) < 1e-6) & (np.abs(fp["normal"][:, 1] - n[1]) < 1e-6)
        out[name] = float(tot[sel].mean()) if sel.any() else 0.0
    out["_all"] = float(tot.mean())
    out["_day_h"] = float(w.sum())
    return out


def selftest():
    ok = [True]

    def check(name, got, want, tol):
        good = abs(got - want) <= tol
        ok[0] = ok[0] and good
        print(f"  {'pass' if good else 'FAIL'}  {name}: {got:.2f} (want {want:.2f} +- {tol:g})")

    def check_lt(name, a, b):
        ok[0] = ok[0] and (a < b)
        print(f"  {'pass' if a < b else 'FAIL'}  {name}: {a:.2f} < {b:.2f}")

    print("1. day length against acos(-tan(lat) tan(decl)), which is what the weights must sum to.")
    print("   Geometric sunrise, sun centre on a flat horizon: a published table adds the solar disc")
    print("   and refraction, eight or nine minutes at these latitudes, which is not usable direct sun.")
    for lat, day, want in [(41.39, "equinox", 12.00), (41.39, "winter", 9.00), (59.91, "winter", 5.55),
                           (1.35, "equinox", 12.00), (-33.45, "winter", 9.78), (52.23, "summer", 16.54)]:
        decl = DECLINATION[day]
        if day == "winter":
            decl = -23.44 if lat >= 0 else 23.44
        if day == "summer":
            decl = 23.44 if lat >= 0 else -23.44
        _, _, w = solar_positions(lat, decl, 10)
        check(f"lat {lat:+.2f} {day}", float(w.sum()), want, 0.05)

    print("\n2. the horizon field against a brute-force march, on four towers that do shade each other")
    p = _tower_parts(45.0)
    o = _opts({})
    H, cls, anchor = rasterize_parts(p, o["px_m"])
    fp = facade_points(p, o, o["px_m"])
    alt, az, _ = solar_positions(41.39, -23.44, 60)
    rng = np.random.RandomState(0)
    pick = rng.choice(len(fp["z"]), 12, replace=False)
    agree = total = 0
    for a, z in zip(alt, az):
        M = horizon_field(H, a, z, o["px_m"], o["sun_steps"])
        col = (fp["xy"][pick, 0] - anchor[0]) / o["px_m"] - 0.5
        row = (anchor[1] - fp["xy"][pick, 1]) / o["px_m"] - 0.5
        mv = ndimage.map_coordinates(M, [row, col], order=1, mode="nearest")
        for j, i in enumerate(pick):
            sd = math.sin(math.radians(z)) * fp["normal"][i, 0] + math.cos(math.radians(z)) * fp["normal"][i, 1]
            if sd <= 0:
                continue                                  # sun behind the wall, nothing to compare
            fast = bool(fp["z"][i] > mv[j])
            slow = _brute_force_sun(H, anchor, o["px_m"], fp["xy"][i], fp["z"][i], a, z)
            total += 1
            agree += int(fast == slow)
    good = agree >= 0.95 * total
    ok[0] = ok[0] and good
    print(f"  {'pass' if good else 'FAIL'}  {agree} of {total} facade samples agree with the slow march"
          f" (a ray grazing a raster edge is genuinely ambiguous, so a few may differ)")

    print("\n3. an isolated block, Barcelona, winter solstice: the sun is south of east all day, so")
    print("   the south wall should see the whole day, the north wall none, east and west half each")
    r = facade_sun_by_orientation(_box_parts(size=40.0, h=30.0), 41.39, -23.44)
    day = r["_day_h"]
    check("south wall (h)", r["south"], day, 0.25)
    check("north wall (h)", r["north"], 0.0, 0.05)
    check("east wall (h)", r["east"], day / 2, 0.4)
    check("west wall (h)", r["west"], day / 2, 0.4)

    print("\n4. the noon shadow of a 30 m block reaches h / tan(noon altitude) to the north")
    noon_alt = 90 - abs(41.39 - (-23.44))
    want = 30.0 / math.tan(math.radians(noon_alt))
    pb = _box_parts(size=40.0, h=30.0)
    Hb, clsb, anb = rasterize_parts(pb, 1.0)
    M = horizon_field(Hb, noon_alt, 180.0, 1.0, 256)
    col = M.shape[1] // 2
    rows = np.nonzero(M[:, col] > 0.05)[0]                # shadowed cells in the column through the block
    north_edge = np.nonzero(Hb[:, col] > 0)[0].min()      # the block's north face
    check("noon shadow reach north (m)", float(north_edge - rows.min()), want, 2.5)

    print("\n5. views from an isolated block: half the outward hemisphere is above the horizon")
    v = analyse(_box_parts(size=40.0, h=30.0), lat=41.39, days=("winter",))
    check("sky share (%)", v["view_sky_pct"], 50.0, 1.0)
    check("building share (%)", v["view_building_pct"], 0.0, 0.5)
    check("overlooked share (%)", v["view_overlooked_pct"], 0.0, 0.5)

    print("\n6. an empty tile at the equinox: every open cell gets the whole day")
    empty = {"tile": box(-200, -200, 200, 200), "buildings": [], "streets": [],
             "plates": [], "blocks": [box(-200, -200, 200, 200)], "greens": []}
    e = analyse(empty, lat=41.39, days=("equinox",), px_m=4.0)
    check("open ground sun (h)", e["sun_equinox_ground_h"], 12.0, 0.05)

    print("\n7. four towers, same buildings, moved closer: every reading should get worse")
    tight = analyse(_tower_parts(45.0), lat=41.39, days=("winter",))
    loose = analyse(_tower_parts(140.0), lat=41.39, days=("winter",))
    for k in ("view_sky_pct", "view_building_pct", "view_overlooked_pct", "view_obstruction_m",
              "sun_winter_facade_h", "sun_winter_ground_h", "sun_winter_facade_above_2h_pct"):
        print(f"   {k:28s} 45 m apart {tight[k]:7.2f}   140 m apart {loose[k]:7.2f}")
    check_lt("tight sees less sky", tight["view_sky_pct"], loose["view_sky_pct"])
    check_lt("tight is more overlooked", loose["view_overlooked_pct"], tight["view_overlooked_pct"])
    check_lt("tight sees less far", tight["view_obstruction_m"], loose["view_obstruction_m"])
    check_lt("tight has less wall above 2 h", tight["sun_winter_facade_above_2h_pct"], loose["sun_winter_facade_above_2h_pct"])
    check_lt("tight gets less sun on the walls", tight["sun_winter_facade_h"], loose["sun_winter_facade_h"])

    print("\n" + ("all checks passed" if ok[0] else "SOME CHECKS FAILED"))
    return ok[0]


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(0 if selftest() else 1)
    print(__doc__)
