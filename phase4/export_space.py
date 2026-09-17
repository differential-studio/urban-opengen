"""
export_space.py

The design space export behind the viewer's Export button.

The viewer sends the controls it is showing, with a range and a number of steps for
the ones to sweep. This walks the full factorial grid, generates each combination
with whatever model is loaded, fits geometry to it, measures it and writes the
folder the multiobjective viewer reads:

    001.3dm  002.3dm  ...     one Rhino model per option
    context.3dm               the surroundings, written once so they are not reloaded
    parameters.csv            first column the option name, one column per swept slider
    objectives.csv            first column the option name, one column per measurement
    run.json                  the spec, the axes and the timings
    thumbs/001.png            a small plan of each option

On the diffusion tab the option model holds only what the model put inside the
painted area, and the measurements describe that piece: the surrounding city is
still there while the sun and view rays are cast, it is just not what is reported.
On the GAN tab the option is the whole tile and the context is only its boundary.

The run happens on a background thread so the page stays alive; it reports progress
and can be stopped.
"""
from __future__ import annotations

import base64
import csv
import itertools
import json
import re
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _c in (HERE, HERE.parent / "phase7", HERE / "phase7", HERE.parent / "phase0", HERE.parent):
    if _c.exists() and str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

import vectorize as V  # noqa: E402

_A = None


def _analysis():
    """imported late so the viewer starts without phase7 on the path"""
    global _A
    if _A is None:
        import analysis
        _A = analysis
    return _A


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------

def axis_values(a: dict) -> list:
    """one axis: an explicit list, or from..to in N steps, rounded to whole numbers if asked"""
    if a.get("values"):
        vals = list(a["values"])
    else:
        lo, hi = float(a["from"]), float(a["to"])
        n = max(1, int(a.get("steps", 2)))
        vals = [lo] if n == 1 else [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    if a.get("int"):
        vals = [int(round(v)) for v in vals]
        seen, out = set(), []
        for v in vals:                      # an integer slider with more steps than values
            if v not in seen:
                seen.add(v); out.append(v)
        return out
    return [round(float(v), 6) for v in vals]


def expand(axes: list) -> tuple[list, dict]:
    """-> ([{key: value} per option], {key: values})"""
    grid = {a["key"]: axis_values(a) for a in axes}
    keys = list(grid)
    rows = [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]
    return (rows or [{}]), grid


def count(axes: list) -> int:
    n = 1
    for a in axes:
        n *= max(1, len(axis_values(a)))
    return n


def payload(base: dict, p: dict) -> dict:
    """a flat {axis: value} onto the shape the generator wants, the same names explore.py uses"""
    body = json.loads(json.dumps(base))          # a deep copy: targets and pca are nested
    targets = dict(body.get("targets") or {})
    dirs = dict(body.get("dirs") or {})
    pca = list(body.get("pca") or [])
    for k, v in p.items():
        if k.startswith("target:"):
            targets[k.split(":", 1)[1]] = float(v)
        elif k.startswith("dir:"):
            dirs[k.split(":", 1)[1]] = float(v)
        elif re.fullmatch(r"pca\d+", k):
            i = int(k[3:]) - 1
            while len(pca) <= i:
                pca.append(0.0)
            pca[i] = float(v)
        else:
            body[k] = v
    if targets:
        body["targets"] = targets
    if dirs:
        body["dirs"] = dirs
    if pca:
        body["pca"] = pca
    return body


def column(key: str) -> str:
    """the axis name as it appears in parameters.csv"""
    return key.replace("target:", "target_").replace("dir:", "dir_").replace(":", "_")


# ---------------------------------------------------------------------------
# the painted area
# ---------------------------------------------------------------------------

def mask_polygon(mask: np.ndarray, tile_m: float):
    """
    The painted mask as one polygon in metres, x east and y north from the tile centre.

    Rows are run-length encoded into strips and unioned, which is a few hundred boxes
    rather than the sixteen thousand cells of the raster.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union
    H, W = mask.shape
    px = tile_m / W
    x0 = -W * px / 2.0
    y1 = H * px / 2.0
    strips = []
    for r in range(H):
        row = mask[r]
        if not row.any():
            continue
        d = np.diff(np.concatenate(([0], (row > 0).view(np.int8), [0])))
        for a, b in zip(np.flatnonzero(d > 0), np.flatnonzero(d < 0)):
            strips.append(box(x0 + a * px, y1 - (r + 1) * px, x0 + b * px, y1 - r * px))
    if not strips:
        return None
    g = unary_union(strips)
    return g if not g.is_empty else None


def decode_mask(b64: str, H: int, W: int) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), np.uint8).reshape(H, W)


def decode_channels(b64: str, H: int, W: int, C: int = 4) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), np.uint8).reshape(C, H, W).copy()


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------

JOB = {
    "running": False, "done": 0, "total": 0, "name": "", "out": "", "zip": "",
    "error": "", "cancelled": False, "seconds": 0.0, "per_s": 0.0, "eta_s": 0.0,
    "note": "", "failed": 0, "started": 0.0,
}
_LOCK = threading.Lock()
_STOP = threading.Event()
_THREAD: threading.Thread | None = None


def progress() -> dict:
    with _LOCK:
        j = dict(JOB)
    if j["running"] and j["started"]:
        j["seconds"] = round(time.time() - j["started"], 1)
        if j["done"]:
            per = j["seconds"] / j["done"]
            j["per_s"] = round(per, 2)
            j["eta_s"] = round(per * (j["total"] - j["done"]), 1)
    return j


def cancel():
    _STOP.set()
    with _LOCK:
        JOB["note"] = "stopping..."
    return progress()


def busy() -> bool:
    return bool(_THREAD and _THREAD.is_alive())


def start(spec: dict, generate, lut, root: Path):
    """
    `generate(engine, payload, canvas) -> (channels, info)` is whatever the server has
    loaded; everything after it happens here.
    """
    global _THREAD
    if busy():
        raise RuntimeError("an export is already running")
    axes = spec.get("axes") or []
    n = count(axes)
    if n < 1:
        raise ValueError("nothing to sweep")
    if n > 4000:
        raise ValueError(f"{n} options is more than this will write in one go (cap 4000)")
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", (spec.get("name") or time.strftime("space-%Y%m%d-%H%M"))).strip("-")
    out = (root / "spaces" / name).resolve()
    if root.resolve() not in out.parents:
        raise ValueError("bad output name")
    out.mkdir(parents=True, exist_ok=True)
    _STOP.clear()
    with _LOCK:
        JOB.update(running=True, done=0, total=n, name=name, out=str(out), zip="", error="",
                   cancelled=False, seconds=0.0, per_s=0.0, eta_s=0.0, note="starting",
                   failed=0, started=time.time())
    _THREAD = threading.Thread(target=_run, args=(spec, generate, lut, out, name), daemon=True)
    _THREAD.start()
    return progress()


def _note(s: str):
    with _LOCK:
        JOB["note"] = s


def _run(spec, generate, lut, out: Path, name: str):
    try:
        _work(spec, generate, lut, out, name)
    except Exception as e:                      # noqa
        with _LOCK:
            JOB["error"] = f"{type(e).__name__}: {e}"
    finally:
        with _LOCK:
            JOB["running"] = False
            JOB["seconds"] = round(time.time() - JOB["started"], 1)
            if not JOB["error"]:
                JOB["note"] = "cancelled" if JOB["cancelled"] else "done"


def _work(spec, generate, lut, out: Path, name: str):
    engine = spec.get("engine", "gan")
    tile_m = float(spec.get("tile_m") or 400.0)
    anchor = spec.get("anchor") or None
    lat = float((anchor or {}).get("lat", 52.0))
    lon = (anchor or {}).get("lon")
    vopts = spec.get("vectorize") or {}
    aopts = dict(spec.get("analysis") or {})
    if "days" in aopts:
        aopts["days"] = tuple(aopts["days"])
    thumbs = bool(spec.get("thumbs", True))
    real = spec.get("vec") or None
    canvas = spec.get("canvas") or {}

    # the painted area: what the model rebuilds, what the option model holds and what the
    # numbers describe. Everything outside it still casts shadows and blocks views.
    area = None
    if canvas.get("mask_b64"):
        H, W = int(canvas["H"]), int(canvas["W"])
        area = mask_polygon(decode_mask(canvas["mask_b64"], H, W), tile_m)
    if engine == "gan":
        area = None                              # the GAN makes a whole tile, so that is the option

    # How much of the city around the plot takes part. The app can hold 1.5 km of
    # OpenStreetMap and measuring all of it is wasted work: what shades a plot and blocks
    # its views is its own neighbourhood, and the raster cost grows with the square of the
    # side. Everything within this distance of the painted area casts shadows and blocks
    # views; everything beyond is dropped, from the measurements and from context.3dm.
    ctx_m = spec.get("context_m", 300.0)
    ring = None
    if area is not None and ctx_m is not None:
        from shapely.geometry import box as _b
        ctx_m = max(0.0, float(ctx_m))
        h = tile_m / 2.0
        ring = area.buffer(ctx_m).intersection(_b(-h, -h, h, h))   # 0 measures the plot alone

    rows, grid = expand(spec.get("axes") or [])
    width = max(3, len(str(len(rows))))
    cols = [column(a["key"]) for a in (spec.get("axes") or [])]

    # the surroundings, once
    _note("writing the context")
    if area is not None:
        H, W = int(canvas["H"]), int(canvas["W"])
        if real is not None:
            # the surroundings come from OpenStreetMap itself: true outlines, true heights,
            # blocks that are the exact complement of the street bands. Nothing here is read
            # back off the 3.125 m raster, so the context never inherits the model's grid.
            base_res = {"W": W, "H": H, "px_m": tile_m / W, "buildings": [], "streets": [],
                        "plates": [], "blocks": [], "greens": []}
        else:
            base_ch = decode_channels(canvas["channels_b64"], H, W)
            base_res = V.vectorize(base_ch, lut, tile_m, **vopts)
        ctx = V.parts_m(base_res, real, area)
        if ring is not None:
            ctx = V.clip_parts(ctx, ring, inside=True)
        ctx = V.clip_parts(ctx, area, inside=False)
    else:
        from shapely.geometry import box as _box
        h = tile_m / 2.0
        ctx = {"tile": _box(-h, -h, h, h), "buildings": [], "streets": [],
               "plates": [], "blocks": [], "greens": []}
    (out / "context.3dm").write_bytes(V.to_3dm_parts(ctx, anchor, "context"))

    names, par, obj = [], [], []
    jl = (out / "rows.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    try:
        for i, p in enumerate(rows, start=1):
            if _STOP.is_set():
                with _LOCK:
                    JOB["cancelled"] = True
                break
            opt = str(i).zfill(width)
            names.append(opt)
            par.append({column(k): v for k, v in p.items()})
            _note(f"option {opt}")
            ta = time.time()
            try:
                o, secs = _one(engine, generate, payload(spec.get("base") or {}, p), canvas,
                               lut, tile_m, lat, lon, anchor, real, area, ring, vopts, aopts,
                               out, opt, thumbs)
            except Exception as e:               # one bad option must not lose the run
                o = {"error": f"{type(e).__name__}: {e}"}
                secs = {}
                with _LOCK:
                    JOB["failed"] += 1
            obj.append(o)
            jl.write(json.dumps({"name": opt, "parameters": par[-1], "objectives": o,
                                 "seconds": secs}) + "\n")
            jl.flush()
            with _LOCK:
                JOB["done"] = i
            del ta
    finally:
        jl.close()

    _note("writing the tables")
    _csv(out / "parameters.csv", names, par, cols)
    _csv(out / "objectives.csv", names, obj, None)
    (out / "run.json").write_text(json.dumps({
        "name": name, "engine": engine, "tile_m": tile_m, "anchor": anchor,
        "axes": {k: v for k, v in grid.items()}, "options": len(names),
        "area_m2": round(area.area, 1) if area is not None else None,
        "context_m": ctx_m if ring is not None else None,
        "context_extent_m": round(ring.bounds[2] - ring.bounds[0], 1) if ring is not None else None,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
        "spec": {k: v for k, v in spec.items() if k not in ("canvas", "vec")},
    }, indent=1), encoding="utf-8")

    _note("zipping")
    zp = out.parent / f"{name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out.parent))
    with _LOCK:
        JOB["zip"] = str(zp)


def _one(engine, generate, body, canvas, lut, tile_m, lat, lon, anchor, real, area, ring,
         vopts, aopts, out: Path, opt: str, thumbs: bool):
    t = {}
    t0 = time.time()
    ch = generate(engine, body, canvas)
    t["generate"] = round(time.time() - t0, 2)

    t0 = time.time()
    res = V.vectorize(ch, lut, tile_m, **vopts)
    parts = V.parts_m(res, real, area)
    scene = V.clip_parts(parts, ring, inside=True) if ring is not None else parts
    t["fit"] = round(time.time() - t0, 2)

    t0 = time.time()
    o = _analysis().analyse(scene, lat=lat, lon=lon, area=area, **aopts)
    t["analyse"] = round(time.time() - t0, 2)

    t0 = time.time()
    data = V.to_3dm_parts(V.clip_parts(parts, area, inside=True), anchor, opt)
    (out / f"{opt}.3dm").write_bytes(data)
    if thumbs:
        try:
            (out / "thumbs").mkdir(exist_ok=True)
            _thumb(res, out / "thumbs" / f"{opt}.png")
        except Exception:                        # PIL is optional
            pass
    t["export"] = round(time.time() - t0, 2)

    o = {k: v for k, v in o.items() if not k.startswith("_") and k not in ("latitude", "longitude")}
    o["kb"] = round(len(data) / 1024, 1)
    return o, t


def _csv(path: Path, names: list, rows: list, first_cols: list | None):
    cols = list(first_cols or [])
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name"] + cols)
        for n, r in zip(names, rows):
            w.writerow([n] + ["" if r.get(c) is None else r.get(c, "") for c in cols])


def _thumb(res: dict, path: Path, size=320):
    from PIL import Image, ImageDraw
    k = size / res["W"]
    im = Image.new("RGB", (size, size), (185, 182, 176))
    dr = ImageDraw.Draw(im)
    for p in res["plates"]:
        dr.polygon([(x * k, y * k) for x, y in p["outer"]], fill=(58, 58, 60))
    for p in res["greens"]:
        dr.polygon([(x * k, y * k) for x, y in p["outer"]], fill=(111, 174, 69))
    for p in res["buildings"]:
        g = int(150 + 100 * min(1.0, max(0.0, (p["h"] - 4) / 56)))
        dr.polygon([(x * k, y * k) for x, y in p["outer"]], fill=(g, g, g - 6), outline=(60, 60, 60))
        for h in p["holes"]:
            dr.polygon([(x * k, y * k) for x, y in h], fill=(185, 182, 176), outline=(60, 60, 60))
    im.save(path)
