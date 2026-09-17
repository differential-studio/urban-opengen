"""
explore.py

A design space from the two models, in the shape the multiobjective viewer reads:
one Rhino model per option numbered 001, 002, 003, a single context model that
holds the surroundings so they are not reloaded with every option, and two CSVs
whose first column is the option name and whose first row is the column names.

    python explore.py space.json --dry-run     # what the grid comes to, and roughly how long
    python explore.py space.json               # run it

The grid is full factorial: every parameter gets a number of steps and the count
is their product, times the number of variants per combination. Inference runs
against the phase 4 server, so whatever snapshot or checkpoint is loaded in the
viewer is what the space is built from and there is no second copy of the model
loading code here. Everything after inference happens locally: the fit
(`phase4/vectorize.py`), the analysis (`phase7/analysis.py`) and the export.

Two engines, one pipeline:

  gan       the whole tile is generated. Parameters are the latent and the city,
            and the metric targets, which work on this model because the viewer
            solves for them and re-measures rather than conditioning on them.
            The context model is just the tile boundary.
  inpaint   a real site is loaded from OpenStreetMap once, a rectangle of it is
            rebuilt per option, and the context model is the real geometry
            outside that rectangle. Parameters are the seed and the guidance
            strength, which are the things this model genuinely responds to, and
            the hole targets if you want them recorded.

A run is resumable: options whose model already exists are skipped and their
rows are read back from rows.jsonl, so an overnight inpainter run survives being
interrupted.

Spec, with only `engine` and `parameters` required:

    {
      "engine": "gan",
      "server": "http://127.0.0.1:5000",
      "out": "C:/opengan/spaces/run01",
      "lat": 41.39, "lon": 2.17,
      "variants": 3,
      "context_m": 300,
      "fixed": {"city_a": 6, "city_b": 40, "psi": 1.0},
      "parameters": {
        "target:far":       {"min": 1.5, "max": 4.0, "steps": 4},
        "target:green_pct": {"min": 5,   "max": 25,  "steps": 3},
        "city_mix":         {"values": [0.0, 0.5, 1.0]}
      },
      "vectorize": {"hstep_m": 3.0},
      "analysis":  {"days": ["equinox", "winter"]}
    }

`context_m` is how much of the city around the rebuilt area takes part: everything
within that distance of it casts shadows and blocks views, everything beyond is
dropped. A site can be 1.5 km across and measuring all of it is wasted work, since
the analysis raster costs the square of the side. 300 m is the default and already
gives the same numbers as the whole site on a dense Barcelona block; 0 measures the
plot as if it stood alone.

Parameter names map onto the server's payloads directly. `target:<metric>` goes
into the targets, `pca1`..`pcaN` into the PCA vector, `dir:<name>` into the
fitted metric directions, anything else is passed through: seed_a, seed_b, mix,
psi, city_a, city_b, city_mix for the GAN, and seed, cfg, steps, resample for the
inpainter. A requested target lands in parameters.csv as `target_far`; what the
model actually produced lands in objectives.csv as `far`, measured off the
geometry, so the two are never confused.
"""
from __future__ import annotations

import argparse
import base64
import csv
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parent / "phase0", HERE.parent / "phase4", HERE.parent):
    if cand.exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))
from tile_codec import HeightLUT  # noqa: E402
import vectorize as V  # noqa: E402
import analysis as A  # noqa: E402

ROUGH_S = {"gan": 0.3, "inpaint": 10.0, "stub": 0.02}     # inference only, before the fit and analysis
ROUGH_REST_S = 1.5                                        # fit, analysis, export, per option


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------

def axis_values(spec) -> list:
    """one parameter's values: an explicit list, a fixed value, or min..max in N steps"""
    if isinstance(spec, (int, float, str)):
        return [spec]
    if "values" in spec:
        return list(spec["values"])
    if "value" in spec:
        return [spec["value"]]
    lo, hi = float(spec["min"]), float(spec["max"])
    n = int(spec.get("steps", 2))
    if n <= 1:
        return [lo]
    vals = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    return [round(v, 6) for v in vals]


def expand(spec: dict):
    """-> (list of parameter dicts, one per option, {axis: values} for the record)"""
    axes = {k: axis_values(v) for k, v in (spec.get("parameters") or {}).items()}
    seed_name = spec.get("seed_name") or ("seed" if spec.get("engine") == "inpaint" else "seed_a")
    variants = int(spec.get("variants", 1))
    if variants > 1 and seed_name not in axes:
        axes[seed_name] = list(spec.get("seeds") or range(variants))
    names = list(axes)
    rows = []
    for combo in itertools.product(*(axes[n] for n in names)):
        p = dict(spec.get("fixed") or {})
        p.update(dict(zip(names, combo)))
        rows.append(p)
    return rows, axes


def payload(p: dict, engine: str, base: dict) -> dict:
    """a flat parameter dict into the shape the server's endpoint wants"""
    body = dict(base)
    targets, dirs, pca = {}, {}, {}
    for k, v in p.items():
        if k.startswith("target:"):
            targets[k.split(":", 1)[1]] = float(v)
        elif k.startswith("dir:"):
            dirs[k.split(":", 1)[1]] = float(v)
        elif k.startswith("pca") and k[3:].isdigit():
            pca[int(k[3:])] = float(v)
        else:
            body[k] = v
    if targets:
        body["targets"] = targets
    if dirs:
        body["dirs"] = dirs
    if pca:
        body["pca"] = [pca.get(i + 1, 0.0) for i in range(max(pca))]
    return body


# ---------------------------------------------------------------------------
# inference, against the phase 4 server
# ---------------------------------------------------------------------------

def post(server: str, path: str, body: dict, timeout=1200):
    import requests
    r = requests.post(server.rstrip("/") + path, json=body, timeout=timeout)
    j = r.json()
    if not r.ok:
        raise RuntimeError(j.get("error", r.reason))
    return j


def decode(res: dict) -> np.ndarray:
    C, H, W = int(res.get("C", 4)), int(res["H"]), int(res["W"])
    return np.frombuffer(base64.b64decode(res["channels_b64"]), np.uint8).reshape(C, H, W).copy()


def mask_b64(rect, H, W, tile_m):
    """a rectangle in metres, x east and y north from the tile centre, as the fill mask"""
    x0, y0, x1, y1 = [float(v) for v in rect]
    px = tile_m / W
    c0 = int(round((x0 + tile_m / 2) / px)); c1 = int(round((x1 + tile_m / 2) / px))
    r0 = int(round((tile_m / 2 - y1) / px)); r1 = int(round((tile_m / 2 - y0) / px))
    m = np.zeros((H, W), np.uint8)
    m[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)] = 1
    return base64.b64encode(m.tobytes()).decode("ascii"), int(m.sum())


class Engine:
    """holds whatever is shared by every option: the site, the context, the base payload"""

    def __init__(self, spec):
        self.spec = spec
        self.kind = spec["engine"]
        self.server = spec.get("server", "http://127.0.0.1:5000")
        self.real = None
        self.rect = None
        self.area = None
        self.ring = None
        self.base = {}
        self.tile_m = float(spec.get("tile_m", 400.0))
        self.lat = float(spec.get("lat", 52.0))
        self.lon = spec.get("lon")
        if self.kind == "inpaint":
            site = spec["site"]
            res = post(self.server, "/api/osm/fetch",
                       {"lat": site["lat"], "lon": site["lon"], "extent_m": site.get("extent_m", 800)})
            self.canvas = res
            self.real = res.get("vector")
            self.tile_m = float(res.get("tile_m") or self.tile_m)
            self.lat, self.lon = float(site["lat"]), float(site["lon"])
            self.rect = spec["mask"]
            from shapely.geometry import box as _box
            self.area = _box(*[float(v) for v in self.rect])
            # only the neighbourhood takes part: what shades a plot and blocks its views is
            # the city around it, and the analysis raster costs the square of the side
            ctx = spec.get("context_m", 300.0)
            if ctx is not None:
                h = self.tile_m / 2.0
                self.ring = self.area.buffer(max(0.0, float(ctx))).intersection(_box(-h, -h, h, h))
            mb, n = mask_b64(self.rect, res["H"], res["W"], self.tile_m)
            if n == 0:
                raise SystemExit("the mask rectangle does not overlap the site")
            self.base = {"channels_b64": res["channels_b64"], "H": res["H"], "W": res["W"],
                         "mask_b64": mb, "tile_m": self.tile_m}
            print(f"site {site['lat']:.4f}, {site['lon']:.4f}  {self.tile_m:.0f} m canvas at {res['W']} px, "
                  f"mask {n} px ({100.0 * n / (res['H'] * res['W']):.1f} % of it)")
        elif self.kind == "stub":
            self.pool = np.load(spec["stub_npz"])["x"]

    def generate(self, p: dict) -> np.ndarray:
        if self.kind == "gan":
            return decode(post(self.server, "/api/generate", payload(p, self.kind, self.base)))
        if self.kind == "inpaint":
            return decode(post(self.server, "/api/dif/fill", payload(p, self.kind, self.base)))
        import hashlib                                  # the stub stands in for a model: same
        h = hashlib.md5(json.dumps(p, sort_keys=True).encode()).hexdigest()   # parameters, same tile
        return self.pool[int(h[:8], 16) % len(self.pool)]


# ---------------------------------------------------------------------------
# one option, end to end
# ---------------------------------------------------------------------------

def thumbnail(res: dict, path: Path, size=320):
    from PIL import Image, ImageDraw
    W, H = res["W"], res["H"]
    k = size / W
    im = Image.new("RGB", (size, size), (185, 182, 176))
    dr = ImageDraw.Draw(im)
    for p in res["plates"]:
        dr.polygon([(x * k, y * k) for x, y in p["outer"]], fill=(58, 58, 60))
    for p in res["greens"]:
        dr.polygon([(x * k, y * k) for x, y in p["outer"]], fill=(111, 174, 69))
    for p in res["buildings"]:
        t = min(1.0, max(0.0, (p["h"] - 4) / 56))
        g = int(150 + 100 * t)
        dr.polygon([(x * k, y * k) for x, y in p["outer"]], fill=(g, g, g - 6), outline=(60, 60, 60))
        for hole in p["holes"]:
            dr.polygon([(x * k, y * k) for x, y in hole], fill=(185, 182, 176), outline=(60, 60, 60))
    im.save(path)


def one_option(eng: Engine, p: dict, name: str, out: Path, lut, vopts: dict, aopts: dict, thumbs: bool):
    t0 = time.time()
    ch = eng.generate(p)
    t1 = time.time()
    res = V.vectorize(ch, lut, eng.tile_m, **vopts)
    # everything casts shadows and blocks views, including the city around an infill, but the
    # numbers are reported for the plot being designed. On a real Barcelona block the same plot
    # reads 0.19 h of winter sun in context against 1.34 h with its surroundings deleted.
    parts = V.parts_m(res, eng.real, eng.rect)
    scene = V.clip_parts(parts, eng.ring, inside=True) if eng.ring is not None else parts
    t2 = time.time()
    obj = A.analyse(scene, lat=eng.lat, lon=eng.lon, area=eng.area, **aopts)
    t3 = time.time()
    data = V.to_3dm(res, eng.real, eng.rect, {"lat": eng.lat, "lon": eng.lon} if eng.lon is not None else None,
                    name, only="fit" if eng.real else None)
    (out / f"{name}.3dm").write_bytes(data)
    if thumbs:
        (out / "thumbs").mkdir(exist_ok=True)
        thumbnail(res, out / "thumbs" / f"{name}.png")
    # latitude and longitude describe the run, not the option, and live in run.json
    obj = {k: v for k, v in obj.items() if not k.startswith("_") and k not in ("latitude", "longitude")}
    obj["kb"] = round(len(data) / 1024, 1)
    return obj, (round(t1 - t0, 2), round(t2 - t1, 2), round(t3 - t2, 2), round(time.time() - t3, 2))


def write_csv(path: Path, rows: list, names: list, first="name"):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([first] + cols)
        for n, r in zip(names, rows):
            w.writerow([n] + [r.get(c, "") for c in cols])


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("spec")
    ap.add_argument("--dry-run", action="store_true", help="print the grid and an estimate, generate nothing")
    ap.add_argument("--out", default=None, help="override the spec's output folder")
    ap.add_argument("--no-thumbs", action="store_true")
    ap.add_argument("--lut", default=None, help="height_lut.json (default: the one named in the spec)")
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    rows, axes = expand(spec)
    n = len(rows)
    print(f"engine {spec['engine']}, {len(axes)} axes:")
    for k, v in axes.items():
        show = ", ".join(f"{x:g}" if isinstance(x, (int, float)) else str(x) for x in v[:6])
        print(f"  {k:22s} {len(v):3d} x  [{show}{', ...' if len(v) > 6 else ''}]")
    per = ROUGH_S.get(spec["engine"], 1.0) + ROUGH_REST_S
    print(f"  {'':22s} ------")
    print(f"  {'options':22s} {n:3d}     roughly {n * per / 60:.0f} min at {per:.1f} s each, before measuring")
    if args.dry_run:
        return

    out = Path(args.out or spec.get("out") or "space")
    out.mkdir(parents=True, exist_ok=True)
    lut = HeightLUT.load(args.lut or spec.get("lut") or "height_lut.json")
    vopts = spec.get("vectorize") or {}
    aopts = spec.get("analysis") or {}
    if "days" in aopts:
        aopts["days"] = tuple(aopts["days"])
    width = max(3, len(str(n)))
    eng = Engine(spec)

    done = {}
    jl = out / "rows.jsonl"
    if jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["name"]] = d
        print(f"resuming: {len(done)} options already recorded")

    # the surroundings, written once
    ctx = out / "context.3dm"
    if not ctx.exists():
        if eng.real:
            base = V.vectorize(decode(eng.canvas), lut, eng.tile_m, **vopts)
            ctx.write_bytes(V.to_3dm(base, eng.real, eng.rect,
                                     {"lat": eng.lat, "lon": eng.lon}, "context", only="real"))
        else:
            empty = {"W": 1, "H": 1, "tile_m": eng.tile_m, "px_m": eng.tile_m,
                     "buildings": [], "blocks": [], "greens": [], "plates": [], "streets": [],
                     "axis": {}, "street_axis": {}, "stats": {}}
            ctx.write_bytes(V.to_3dm(empty, None, None, {"lat": eng.lat, "lon": eng.lon} if eng.lon else None, "context"))
        print(f"context.3dm  {ctx.stat().st_size / 1024:.0f} KB")

    t_start = time.time()
    names, par, obj = [], [], []
    with open(jl, "a", encoding="utf-8") as log:
        for i, p in enumerate(rows, start=1):
            name = str(i).zfill(width)
            names.append(name)
            par.append({("target_" + k.split(":", 1)[1]) if k.startswith("target:") else k.replace(":", "_"): v
                        for k, v in p.items()})
            if name in done:
                obj.append(done[name]["objectives"])
                continue
            try:
                o, t = one_option(eng, p, name, out, lut, vopts, aopts, not args.no_thumbs)
            except Exception as e:                       # one bad option must not lose the run
                print(f"  {name}  FAILED  {type(e).__name__}: {e}")
                o, t = {"error": f"{type(e).__name__}: {e}"}, (0, 0, 0, 0)
            obj.append(o)
            log.write(json.dumps({"name": name, "parameters": par[-1], "objectives": o,
                                  "seconds": {"generate": t[0], "fit": t[1], "analyse": t[2], "export": t[3]}}) + "\n")
            log.flush()
            elapsed = time.time() - t_start
            per = elapsed / max(i - len(done), 1)
            left = per * (n - i)
            num = lambda v, d: f"{v:.{d}f}" if isinstance(v, (int, float)) else "  -"   # noqa: E731
            sun = next((k for k in o if k.endswith("_street_h")), None)
            print(f"  {name}/{n}  gen {t[0]:5.2f}  fit {t[1]:5.2f}  analyse {t[2]:5.2f}  export {t[3]:5.2f} s"
                  f"   FAR {num(o.get('far'), 2)}  street sun {num(o.get(sun) if sun else None, 1)} h"
                  f"   {left / 60:.0f} min left")

    write_csv(out / "parameters.csv", par, names)
    write_csv(out / "objectives.csv", obj, names)
    (out / "run.json").write_text(json.dumps({
        "spec": spec, "axes": {k: v for k, v in axes.items()}, "options": n,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t_start, 1),
        "code": {"vectorize": getattr(V, "__file__", ""), "analysis": getattr(A, "__file__", "")},
    }, indent=1))
    ok = sum(1 for o in obj if "error" not in o)
    print(f"\n{ok} of {n} options in {(time.time() - t_start) / 60:.1f} min -> {out}")
    print("  001.3dm .. plus context.3dm, parameters.csv, objectives.csv, run.json")


if __name__ == "__main__":
    main()
