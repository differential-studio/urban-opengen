"""
osm_tiles.py

Phase 0c: render Urban OpenGen tiles straight from OpenStreetMap, in Python, for any
place on earth. Replaces the Grasshopper renderer and makes the pipeline reproducible:
the same code that builds the training set later feeds the app when a user types an
address.

Output is the palette encoding the rest of the project reads (tile_codec.py):
    black                    open ground
    grey 100..255            building, grey = height, 4 m .. 100 m linear
    red   (255, 0, 0)        street
    green (128, 255, 0)      greenery
drawn in the order green, buildings, streets (later wins), exactly like the existing
tiles, so reencode_tiles.py accepts the output unchanged.

Conventions (the things the Grasshopper tiles never wrote down), all in CONVENTIONS
below and saved next to the output as conventions.json:
    street width      8 m for every class (measured on the Grasshopper tiles: 2 px at
                      4 m/px for every road), or per-class widths with --class-widths
    street classes    motorway .. living_street, pedestrian, and their _link variants;
                      service roads, footways, paths, cycleways, tracks excluded
    building height   `height` tag, else `building:levels` x 3 m, else 7 m
    green             parks, gardens, pitches, grass, forest, meadow, cemeteries, ...

Modes:

  grid      the training layout: a square window around a centre, cut into a grid of
            400 m tiles, named <City>_<idx>.png with idx = row * grid + col, row 0 at
            the north, col 0 at the west. Writes <City>_grid.json with the georeference.

      python osm_tiles.py grid --city Sao_Paulo --center -23.5505,-46.6333 --out ..\\images_all
      python osm_tiles.py grid --city Tokyo --address Tokyo --px 128 --raster --out ..\\images_v2
      python osm_tiles.py grid --city Barcelona --address "Barcelona, Spain" --out ..\\images_v2

  canvas    one image of any size around an address or a point, for the app:

      python osm_tiles.py canvas --address "Carrer de Mallorca 401, Barcelona" --size-m 1200 --out site.png

  compare   render the window of an existing Grasshopper city and put old and new tiles
            side by side, with per-class IoU, to check the conventions:

      python osm_tiles.py compare --city Barcelona --center 41.3874,2.1686 --images ..\\images_all --idx 312 183 88

Data comes from the public Overpass API in cached 2.5 km chunks (--cache), so a city is
fetched once (a few minutes, be polite) and re-rendered for free. Geocoding uses
Nominatim. Both need a User-Agent, set --contact to your email.

Needs: requests, shapely, pyproj, numpy, Pillow  (pip install requests shapely pyproj)
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tile_codec import RED, GREEN, GREY_MIN, GREY_MAX, HEIGHT_M_MIN, HEIGHT_M_MAX, load_rgb, split_classes  # noqa: E402

# ---------------------------------------------------------------------------
# conventions
# ---------------------------------------------------------------------------

CONVENTIONS = {
    "tile_m": 400.0,
    "px": 100,                       # pixels per tile edge; 100 = 4 m/px like the Grasshopper tiles, 128 = 3.125 m/px
    "supersample": 2,                # rasterise at this multiple, then majority-downsample
    "street_width_m": 8.0,           # every class, unless class_widths is on
    "class_widths": False,
    "street_class_widths_m": {       # used when class_widths is on (total width, metres)
        "motorway": 24, "motorway_link": 12, "trunk": 20, "trunk_link": 10,
        "primary": 16, "primary_link": 10, "secondary": 13, "secondary_link": 8,
        "tertiary": 10, "tertiary_link": 7, "residential": 8, "unclassified": 8,
        "living_street": 6, "pedestrian": 6,
    },
    "street_classes": [
        "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
        "secondary", "secondary_link", "tertiary", "tertiary_link", "residential",
        "unclassified", "living_street", "pedestrian",
    ],
    "green_tags": {
        "landuse": ["grass", "forest", "meadow", "recreation_ground", "village_green", "cemetery",
                    "orchard", "allotments", "greenfield", "flowerbed"],
        "leisure": ["park", "garden", "pitch", "playground", "nature_reserve", "golf_course",
                    "dog_park", "common"],
        "natural": ["wood", "grassland", "scrub", "heath"],
    },
    "level_height_m": 3.0,
    "default_height_m": 7.0,         # buildings with neither `height` nor `building:levels`
    "draw_order": ["green", "building", "street"],
}


def metres_to_grey(h: float) -> int:
    h = min(max(float(h), HEIGHT_M_MIN), HEIGHT_M_MAX)
    return int(round(GREY_MIN + (h - HEIGHT_M_MIN) * (GREY_MAX - GREY_MIN) / (HEIGHT_M_MAX - HEIGHT_M_MIN)))


_NUM = re.compile(r"[-+]?\d*\.?\d+")


def parse_height(tags: dict, conv: dict) -> float:
    """metres from OSM tags, with the documented fallbacks"""
    for key in ("height", "building:height"):
        v = tags.get(key)
        if v:
            m = _NUM.search(v.replace(",", "."))
            if m:
                h = float(m.group())
                if "ft" in v or "'" in v:
                    h *= 0.3048
                if h > 0:
                    return h
    lv = tags.get("building:levels")
    if lv:
        m = _NUM.search(lv.replace(",", "."))
        if m and float(m.group()) > 0:
            return float(m.group()) * conv["level_height_m"]
    return conv["default_height_m"]


def is_green(tags: dict, conv: dict) -> bool:
    for k, vals in conv["green_tags"].items():
        if tags.get(k) in vals:
            return True
    return False


# ---------------------------------------------------------------------------
# fetching (Overpass + Nominatim)
# ---------------------------------------------------------------------------

OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.osm.ch/api/interpreter",
            "https://overpass.openstreetmap.ru/api/interpreter"]
NOMINATIM = "https://nominatim.openstreetmap.org/search"


def wait_for_slot(url, contact, cap=180.0):
    """
    Overpass hands out a small number of query slots per client. /api/status says how many
    are free and, when none are, how many seconds until one is. Asking and waiting is far
    better behaved than retrying blind, which is what gets a client throttled harder.
    Returns the seconds waited (0 when a slot was free or the status could not be read).
    """
    import requests
    try:
        r = requests.get(url.replace("/interpreter", "/status"),
                         headers={"User-Agent": f"urban-opengen ({contact})"}, timeout=30)
        txt = r.text
    except Exception:
        return 0.0
    if re.search(r"\d+ slots available now", txt) or "slots available now" in txt:
        return 0.0
    waits = [float(m) for m in re.findall(r"in (\d+) seconds", txt)]
    if not waits:
        return 0.0
    w = min(max(min(waits) + 2.0, 2.0), cap)
    print(f"\n  no free slot on {url.split('/')[2]}, waiting {w:.0f} s", flush=True)
    time.sleep(w)
    return w


def geocode(query: str, contact: str):
    import requests
    r = requests.get(NOMINATIM, params={"q": query, "format": "json", "limit": 1},
                     headers={"User-Agent": f"urban-opengen ({contact})"}, timeout=60)
    r.raise_for_status()
    hits = r.json()
    if not hits:
        sys.exit(f"nothing found for '{query}'")
    return float(hits[0]["lat"]), float(hits[0]["lon"]), hits[0].get("display_name", "")


def overpass_query(south, west, north, east, conv):
    bb = f"({south:.6f},{west:.6f},{north:.6f},{east:.6f})"
    g = conv["green_tags"]
    parts = [f'nwr["building"]{bb};', f'way["highway"]{bb};']
    for k, vals in g.items():
        rx = "^(" + "|".join(vals) + ")$"
        parts.append(f'nwr["{k}"~"{rx}"]{bb};')
    return "[out:json][timeout:300][maxsize:1073741824];(" + "".join(parts) + ");out geom;"


class OverpassBusy(Exception):
    pass


def _overpass_call(south, west, north, east, conv, contact, cache: Path | None, attempts=10):
    """
    One box, cached on disk as gzip json. Retries patiently across mirrors, asking each
    for a free slot before firing, because the public servers throttle impatient clients.
    Raises OverpassBusy only when a server says the query itself was too heavy, which is
    the one case where splitting the box actually helps.
    """
    import requests
    key = f"{south:.5f}_{west:.5f}_{north:.5f}_{east:.5f}.json.gz"
    if cache:
        p = cache / key
        if p.exists():
            with gzip.open(p, "rt", encoding="utf-8") as f:
                return json.load(f)
    q = overpass_query(south, west, north, east, conv)
    last = "no attempt made"
    for attempt in range(attempts):
        url = OVERPASS[attempt % len(OVERPASS)]
        wait_for_slot(url, contact)
        try:
            r = requests.post(url, data={"data": q}, headers={"User-Agent": f"urban-opengen ({contact})"}, timeout=300)
        except requests.RequestException as e:
            last = f"{type(e).__name__} from {url.split('/')[2]}"
        else:
            if r.status_code == 200:
                try:
                    data = r.json()
                except ValueError:
                    last = f"unparseable response from {url.split('/')[2]}"
                    data = None
                if data is not None:
                    remark = str(data.get("remark", ""))
                    if "timed out" in remark or "out of memory" in remark:
                        raise OverpassBusy(f"query too heavy: {remark.strip()[:120]}")
                    # An empty box is almost always a server having a bad day rather than a
                    # genuinely empty piece of world, and a cached empty box is a hole in the
                    # map that never heals. Return it, but do not write it down.
                    if not data.get("elements"):
                        print(f"\n  empty response for {south:.5f},{west:.5f} from "
                              f"{url.split('/')[2]}, not cached", flush=True)
                        return data
                    if cache:
                        cache.mkdir(parents=True, exist_ok=True)
                        with gzip.open(cache / key, "wt", encoding="utf-8") as f:
                            json.dump(data, f)
                    return data
            elif r.status_code in (429, 504):
                last = f"{r.status_code} from {url.split('/')[2]}"     # busy or queued, not too heavy
            else:
                last = f"{r.status_code} from {url.split('/')[2]}"
        back = min(15 * (attempt + 1), 120)
        print(f"\n  {last}, retrying in {back} s ({attempt + 1}/{attempts})", flush=True)
        time.sleep(back)
    raise OverpassBusy(last)


def fetch_chunk(south, west, north, east, conv, contact, cache: Path | None, depth: int = 0):
    """
    Fetch a box, cached on disk as gzip json. When the servers time out (dense city
    centres are heavy) the box is split into four and each quarter is fetched instead,
    down to three levels (an eighth of the original edge).
    """
    try:
        return _overpass_call(south, west, north, east, conv, contact, cache)
    except OverpassBusy as e:
        if depth >= 2:
            sys.exit(f"Overpass failed for {south:.5f},{west:.5f}: {e}")
        mlat, mlon = (south + north) / 2, (west + east) / 2
        print(f"\n  busy ({e}), splitting a chunk in four", flush=True)
        merged, seen = {"elements": []}, set()
        for (s_, w_, n_, e_) in [(south, west, mlat, mlon), (south, mlon, mlat, east), (mlat, west, north, mlon), (mlat, mlon, north, east)]:
            part = fetch_chunk(s_, w_, n_, e_, conv, contact, cache, depth + 1)
            for el in part.get("elements", []):
                k = (el["type"], el["id"])
                if k not in seen:
                    seen.add(k)
                    merged["elements"].append(el)
            time.sleep(1.0)
        return merged


def fetch_window(lat, lon, size_m, conv, contact, cache, chunk_m=1500.0, margin_m=150.0, verbose=True):
    """all elements within a square window (plus margin), fetched in chunks; returns a list of elements"""
    half = size_m / 2 + margin_m
    dlat = 1 / 111320.0
    dlon = 1 / (111320.0 * math.cos(math.radians(lat)))
    n = max(1, math.ceil(2 * half / chunk_m))
    step = 2 * half / n
    elements, seen = [], set()
    for i in range(n):
        for j in range(n):
            y0 = -half + i * step
            x0 = -half + j * step
            s, w = lat + y0 * dlat, lon + x0 * dlon
            nn, e = lat + (y0 + step) * dlat, lon + (x0 + step) * dlon
            if verbose:
                print(f"  chunk {i * n + j + 1}/{n * n}", end="\r", flush=True)
            data = fetch_chunk(s, w, nn, e, conv, contact, cache)
            for el in data.get("elements", []):
                k = (el["type"], el["id"])
                if k not in seen:
                    seen.add(k)
                    elements.append(el)
            time.sleep(2.0)
    if verbose:
        print(f"  {len(elements)} elements" + " " * 20)
    return elements


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

class Frame:
    """local metric frame: transverse Mercator centred on the window, x east, y north, metres"""

    def __init__(self, lat, lon):
        from pyproj import Transformer
        self.lat, self.lon = lat, lon
        self.t = Transformer.from_crs("EPSG:4326", f"+proj=tmerc +lat_0={lat} +lon_0={lon} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m", always_xy=True)

    def xy(self, coords):
        lons = np.fromiter((c["lon"] for c in coords), float, len(coords))
        lats = np.fromiter((c["lat"] for c in coords), float, len(coords))
        x, y = self.t.transform(lons, lats)
        return np.stack([x, y], -1)


def build_features(elements, frame: Frame, conv):
    """-> dict with lists of shapely geometries: buildings [(poly, height)], streets [(line, width)], greens [poly]"""
    from shapely.geometry import LineString, Polygon, MultiPolygon
    from shapely.ops import linemerge, polygonize, unary_union
    buildings, streets, greens = [], [], []
    classes = set(conv["street_classes"])
    for el in elements:
        tags = el.get("tags", {}) or {}
        if el["type"] == "way":
            geom = el.get("geometry")
            if not geom or len(geom) < 2:
                continue
            pts = frame.xy(geom)
            closed = len(pts) >= 4 and np.allclose(pts[0], pts[-1])
            if "building" in tags and closed and tags["building"] != "no":
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 1:
                    buildings.append((poly, parse_height(tags, conv)))
                elif not poly.is_valid:
                    poly = poly.buffer(0)
                    if not poly.is_empty:
                        buildings.append((poly, parse_height(tags, conv)))
            if "highway" in tags and tags["highway"] in classes:
                if tags.get("area") == "yes" and closed:
                    streets.append((Polygon(pts).buffer(0), 0.0))
                else:
                    w = conv["street_width_m"]
                    if conv["class_widths"]:
                        w = conv["street_class_widths_m"].get(tags["highway"], w)
                    streets.append((LineString(pts), w))
            if is_green(tags, conv) and closed:
                poly = Polygon(pts).buffer(0)
                if not poly.is_empty:
                    greens.append(poly)
        elif el["type"] == "relation":
            members = el.get("members", [])
            outers, inners = [], []
            for m in members:
                g = m.get("geometry")
                if m.get("type") != "way" or not g or len(g) < 2:
                    continue
                (inners if m.get("role") == "inner" else outers).append(LineString(frame.xy(g)))
            if not outers:
                continue
            try:
                outer_polys = list(polygonize(unary_union(linemerge(unary_union(outers)) if len(outers) > 1 else outers[0])))
                inner_polys = list(polygonize(unary_union(linemerge(unary_union(inners)) if len(inners) > 1 else inners[0]))) if inners else []
            except Exception:
                continue
            if not outer_polys:
                continue
            geom = unary_union(outer_polys)
            if inner_polys:
                geom = geom.difference(unary_union(inner_polys))
            if geom.is_empty:
                continue
            if "building" in tags and tags["building"] != "no":
                buildings.append((geom, parse_height(tags, conv)))
            elif is_green(tags, conv):
                greens.append(geom)
    return {"buildings": buildings, "streets": streets, "greens": greens}


# ---------------------------------------------------------------------------
# rasterising
# ---------------------------------------------------------------------------

def _draw_poly(draw: ImageDraw.ImageDraw, geom, fill, to_px):
    """fill a (multi)polygon with holes: outer in `fill`, holes back to 0"""
    from shapely.geometry import MultiPolygon, Polygon
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for p in polys:
        if not isinstance(p, Polygon) or p.is_empty:
            continue
        ext = [tuple(q) for q in to_px(np.asarray(p.exterior.coords))]
        if len(ext) >= 3:
            draw.polygon(ext, fill=fill)
        for ring in p.interiors:
            hole = [tuple(q) for q in to_px(np.asarray(ring.coords))]
            if len(hole) >= 3:
                draw.polygon(hole, fill=0)


def render_window(features, cx, cy, size_m, res_m, conv):
    """
    Rasterise features (local metres) into a square window of `size_m` centred on (cx, cy)
    at `res_m` per pixel. Returns (rgb uint8 [N, N, 3], masks dict) with N = size_m / res_m.
    """
    ss = int(conv["supersample"])
    N = int(round(size_m / res_m))
    M = N * ss
    r = res_m / ss
    x0, y1 = cx - size_m / 2, cy + size_m / 2     # top-left corner in metres

    def to_px(xy):
        return np.stack([(xy[:, 0] - x0) / r, (y1 - xy[:, 1]) / r], -1)

    # class masks at supersampled resolution, and a height image for buildings
    st_img = Image.new("L", (M, M), 0); gr_img = Image.new("L", (M, M), 0)
    fp_img = Image.new("L", (M, M), 0); h_img = Image.new("L", (M, M), 0)
    d_st, d_gr, d_fp, d_h = ImageDraw.Draw(st_img), ImageDraw.Draw(gr_img), ImageDraw.Draw(fp_img), ImageDraw.Draw(h_img)

    for poly in features["greens"]:
        _draw_poly(d_gr, poly, 255, to_px)
    # shorter buildings first, so overlapping building:part outlines keep the taller value
    for poly, h in sorted(features["buildings"], key=lambda t: t[1]):
        _draw_poly(d_fp, poly, 255, to_px)
        _draw_poly(d_h, poly, metres_to_grey(h), to_px)
    from shapely.ops import unary_union
    lines = [ln.buffer(w / 2, cap_style=2, join_style=1) for ln, w in features["streets"] if w > 0]
    areas = [g for g, w in features["streets"] if w == 0]
    if lines or areas:
        _draw_poly(d_st, unary_union(lines + areas), 255, to_px)

    def down(img):
        a = np.asarray(img, dtype=np.float32).reshape(N, ss, N, ss)
        return a.mean(axis=(1, 3))

    st = down(st_img) >= 127.5
    gr = down(gr_img) >= 127.5
    fp = down(fp_img) >= 127.5
    # height: mean grey over the building sub-pixels only
    hs = np.asarray(h_img, dtype=np.float32).reshape(N, ss, N, ss)
    fs = (np.asarray(fp_img, dtype=np.float32).reshape(N, ss, N, ss) > 0)
    hmean = (hs * fs).sum(axis=(1, 3)) / np.maximum(fs.sum(axis=(1, 3)), 1)
    grey = np.clip(np.round(hmean), GREY_MIN, GREY_MAX).astype(np.uint8)

    rgb = np.zeros((N, N, 3), np.uint8)
    for layer in conv["draw_order"]:
        if layer == "green":
            rgb[gr] = GREEN
        elif layer == "building":
            rgb[fp] = np.stack([grey, grey, grey], -1)[fp]
        elif layer == "street":
            rgb[st] = RED
    return rgb, {"street": st, "green": gr, "footprint": fp}


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def resolve_center(args):
    if args.center:
        lat, lon = (float(v) for v in args.center.split(","))
        return lat, lon, args.center
    if args.address:
        lat, lon, name = geocode(args.address, args.contact)
        print(f"  {args.address} -> {lat:.5f}, {lon:.5f}  ({name})")
        return lat, lon, name
    sys.exit("give --center lat,lon or --address")


def load_conventions(args):
    conv = json.loads(json.dumps(CONVENTIONS))
    if args.conventions:
        conv.update(json.loads(Path(args.conventions).read_text()))
    if getattr(args, "px", None):
        conv["px"] = args.px
    if getattr(args, "class_widths", False):
        conv["class_widths"] = True
    if getattr(args, "street_width", None):
        conv["street_width_m"] = args.street_width
    if getattr(args, "default_height", None):
        conv["default_height_m"] = args.default_height
    return conv


def get_features(lat, lon, size_m, conv, args):
    cache = Path(args.cache) if args.cache else None
    t0 = time.time()
    print(f"fetching {size_m / 1000:.1f} km window around {lat:.5f}, {lon:.5f}")
    els = fetch_window(lat, lon, size_m, conv, args.contact, cache)
    frame = Frame(lat, lon)
    feats = build_features(els, frame, conv)
    print(f"  {len(feats['buildings'])} buildings, {len(feats['streets'])} street segments, {len(feats['greens'])} green areas  ({time.time() - t0:.0f} s)")
    return feats


def mode_grid(args):
    from pyproj import Transformer
    conv = load_conventions(args)
    lat, lon, name = resolve_center(args)
    tile_m, grid, px = conv["tile_m"], args.grid, conv["px"]
    size_m = tile_m * grid
    res_m = tile_m / px
    feats = get_features(lat, lon, size_m, conv, args)
    t0 = time.time()
    rgb, _ = render_window(feats, 0.0, 0.0, size_m, res_m, conv)
    out = Path(args.out) / args.city
    out.mkdir(parents=True, exist_ok=True)
    n_empty = 0
    for row in range(grid):
        for col in range(grid):
            tile = rgb[row * px:(row + 1) * px, col * px:(col + 1) * px]
            if not tile.any():
                n_empty += 1
            Image.fromarray(tile).save(out / f"{args.city}_{row * grid + col}.png")
    # lat/lon of every tile centre, so a tile can be located without the arithmetic
    inv = Transformer.from_crs(f"+proj=tmerc +lat_0={lat} +lon_0={lon} +k=1 +ellps=WGS84 +units=m", "EPSG:4326", always_xy=True)
    tiles = []
    for row in range(grid):
        for col in range(grid):
            tx = -size_m / 2 + (col + 0.5) * tile_m
            ty = size_m / 2 - (row + 0.5) * tile_m
            tlon, tlat = inv.transform(tx, ty)
            tiles.append({"idx": row * grid + col, "row": row, "col": col, "lat": round(tlat, 6), "lon": round(tlon, 6)})
    if getattr(args, "raster", False):
        Image.fromarray(rgb).save(out / f"{args.city}_raster.png")
    meta = {
        "city": args.city, "center": {"lat": lat, "lon": lon, "query": name},
        "raster": f"{args.city}_raster.png" if getattr(args, "raster", False) else None,
        "tiles": tiles,
        "window_m": size_m, "tile_m": tile_m, "grid": grid, "px": px, "res_m_per_px": res_m,
        "index": "row * grid + col, row 0 north, col 0 west",
        "crs": f"+proj=tmerc +lat_0={lat} +lon_0={lon} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m",
        "tile_origin_m": "x = -window/2 + col * tile, y_top = window/2 - row * tile",
        "conventions": conv, "source": "OpenStreetMap via Overpass, ODbL",
    }
    (out / f"{args.city}_grid.json").write_text(json.dumps(meta, indent=1))
    (Path(args.out) / "conventions.json").write_text(json.dumps(conv, indent=1))
    print(f"wrote {grid * grid} tiles ({n_empty} empty) to {out}  ({time.time() - t0:.0f} s render)")
    return {"city": args.city, "tiles": grid * grid, "empty": n_empty, "px": px,
            "buildings": len(feats["buildings"]), "streets": len(feats["streets"]),
            "greens": len(feats["greens"]), "lat": lat, "lon": lon}


def mode_canvas(args):
    conv = load_conventions(args)
    lat, lon, name = resolve_center(args)
    res_m = conv["tile_m"] / conv["px"]
    size_m = args.size_m
    feats = get_features(lat, lon, size_m + 100, conv, args)
    rgb, _ = render_window(feats, 0.0, 0.0, size_m, res_m, conv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out)
    meta = {"center": {"lat": lat, "lon": lon, "query": name}, "size_m": size_m, "px": rgb.shape[0],
            "res_m_per_px": res_m, "crs": f"+proj=tmerc +lat_0={lat} +lon_0={lon} +k=1 +ellps=WGS84 +units=m",
            "conventions": conv}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} ({rgb.shape[0]} px, {size_m} m)")


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def mode_compare(args):
    """render the Grasshopper window, then old | new for the requested tiles with IoU per class"""
    conv = load_conventions(args)
    lat, lon, name = resolve_center(args)
    tile_m, grid, px = conv["tile_m"], args.grid, conv["px"]
    size_m = tile_m * grid
    res_m = tile_m / px
    feats = get_features(lat, lon, size_m + 2 * args.search_m, conv, args)
    big_m = size_m + 2 * args.search_m
    rgb, masks = render_window(feats, 0.0, 0.0, big_m, res_m, conv)
    off = int(round(args.search_m / res_m))       # pixel offset of the nominal window inside the big raster
    old_tiles = {}
    for idx in args.idx:
        p = Path(args.images) / args.city / f"{args.city}_{idx}.png"
        if not p.exists():
            print(f"missing {p}")
            continue
        old_tiles[idx] = load_rgb(p)
    if not old_tiles:
        sys.exit("no existing tiles found")

    # optional calibration: shift the window to best match the old street masks
    dx = dy = 0
    if args.search_m > 0:
        best = (-1, 0, 0)
        step = max(1, int(round(args.search_step_m / res_m)))
        olds = {i: split_classes(t)[2] for i, t in old_tiles.items()}
        for sy in range(-off, off + 1, step):
            for sx in range(-off, off + 1, step):
                score = 0.0
                for idx, st_old in olds.items():
                    r, c = divmod(idx, grid)
                    y0 = off + r * px + sy; x0 = off + c * px + sx
                    st_new = masks["street"][y0:y0 + px, x0:x0 + px]
                    if st_new.shape != st_old.shape:
                        continue
                    score += iou(st_new, st_old)
                if score > best[0]:
                    best = (score, sx, sy)
        _, dx, dy = best
        print(f"best window shift: {dx * res_m:+.0f} m east, {-dy * res_m:+.0f} m north  (street IoU sum {best[0]:.2f} over {len(olds)} tiles)")
        if dx or dy:
            # the shift is in the local frame; report it as a corrected centre too
            from pyproj import Transformer
            t = Transformer.from_crs(f"+proj=tmerc +lat_0={lat} +lon_0={lon} +k=1 +ellps=WGS84 +units=m", "EPSG:4326", always_xy=True)
            clon, clat = t.transform(dx * res_m, -dy * res_m)
            print(f"  corrected --center {clat:.6f},{clon:.6f}")

    strips = []
    for idx, old in old_tiles.items():
        r, c = divmod(idx, grid)
        y0 = off + r * px + dy; x0 = off + c * px + dx
        new = rgb[y0:y0 + px, x0:x0 + px]
        fo, _, so, go, _ = split_classes(old)
        fn, _, sn, gn, _ = split_classes(new)
        print(f"tile {idx:4d}  IoU street {iou(sn, so):.2f}  building {iou(fn, fo):.2f}  green {iou(gn, go):.2f}   "
              f"street% old {100 * so.mean():.1f} new {100 * sn.mean():.1f}   fp% old {100 * fo.mean():.1f} new {100 * fn.mean():.1f}")
        strips.append(np.concatenate([old, np.full((px, 4, 3), 60, np.uint8), new], 1))
    sheet = np.concatenate([np.concatenate([s, np.full((4, s.shape[1], 3), 60, np.uint8)], 0) for s in strips], 0)
    out = Path(args.out)
    Image.fromarray(sheet).resize((sheet.shape[1] * 3, sheet.shape[0] * 3), Image.NEAREST).save(out)
    print(f"wrote {out}  (left: existing tile, right: rendered from OSM)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--center", help="lat,lon of the window centre")
        p.add_argument("--address", help="free-text place or address, geocoded with Nominatim")
        p.add_argument("--contact", default="urban-opengen@example.com", help="email for the OSM User-Agent (required by their policy)")
        p.add_argument("--cache", default="osm_cache", help="folder for cached Overpass responses")
        p.add_argument("--conventions", help="json file overriding CONVENTIONS keys")
        p.add_argument("--px", type=int, help="pixels per 400 m tile (100 = Grasshopper, 128 = training size)")
        p.add_argument("--class-widths", action="store_true", help="per-class street widths instead of one width")
        p.add_argument("--street-width", type=float, help="street width in metres when not using class widths")
        p.add_argument("--default-height", type=float, help="height for buildings without height or levels tags")

    g = sub.add_parser("grid", help="render a city window as a grid of 400 m tiles")
    common(g)
    g.add_argument("--city", required=True, help="folder and file prefix, e.g. Sao_Paulo")
    g.add_argument("--grid", type=int, default=25)
    g.add_argument("--out", required=True, help="images root; tiles go to <out>/<city>/")
    g.add_argument("--raster", action="store_true",
                   help="also write the whole window as one <City>_raster.png, for training on random crops")

    c = sub.add_parser("canvas", help="render one image of any size around a point")
    common(c)
    c.add_argument("--size-m", type=float, default=1200.0)
    c.add_argument("--out", required=True, help="output png (a .json sidecar is written next to it)")

    k = sub.add_parser("compare", help="old Grasshopper tiles next to freshly rendered ones")
    common(k)
    k.add_argument("--city", required=True)
    k.add_argument("--images", required=True, help="existing images root (images_all)")
    k.add_argument("--idx", type=int, nargs="+", required=True, help="tile indices to compare")
    k.add_argument("--grid", type=int, default=25)
    k.add_argument("--search-m", type=float, default=0.0, help="search this many metres around the centre for the best alignment")
    k.add_argument("--search-step-m", type=float, default=20.0)
    k.add_argument("--out", default="compare.png")

    args = ap.parse_args()
    {"grid": mode_grid, "canvas": mode_canvas, "compare": mode_compare}[args.mode](args)


if __name__ == "__main__":
    main()
