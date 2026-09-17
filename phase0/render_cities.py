"""
render_cities.py

Render a whole list of cities with osm_tiles.py, unattended and resumable. Fetching from
Overpass is network-bound, so this is the job to run on the CPU while the GPU is busy
training.

    python render_cities.py --list cities_60.json --out C:\\opengan\\images_v2 --cache C:\\opengan\\osm_cache --contact you@example.com --px 128 --raster

Resumable by default: a city that already has <City>_grid.json in the output folder is
skipped, so the script can be stopped and restarted freely, and a city that failed
half-way keeps everything it already fetched in the cache and only re-fetches the rest.

A report is written to <out>/render_report.json and printed as a table at the end, with
per-city tile counts, feature counts and timings, so a city that came back suspiciously
empty is visible at a glance rather than after training.

    --only Tokyo,Osaka       just these
    --skip Sao_Paulo         all but these
    --redo                   re-render even if the city is already done (cache still used)
    --grid 25                tiles per side, default 25 (a 10 km window at 400 m)
    --px 128                 pixels per tile; 128 is the training size, 100 matches the old set
    --raster                 also save the whole window as one image, for random-crop training
    --pause 20               seconds between cities, be polite to the public servers

Expect roughly ten to twenty minutes per city on the first pass, so a 60-city run is a
long overnight job. Everything is cached, so a second pass is minutes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import osm_tiles  # noqa: E402


def load_list(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict):                      # {"names": [...]} or {"City": [lat, lon]}
        if "names" in data:
            return [{"city": n} for n in data["names"]]
        return [{"city": k, "lat": v[0], "lon": v[1]} for k, v in data.items()]
    return [c if isinstance(c, dict) else {"city": c} for c in data]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default=str(HERE / "cities_60.json"),
                    help="json: [{city, lat, lon}] or [{city, address}] or {\"names\": [...]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="osm_cache")
    ap.add_argument("--contact", required=True, help="your email, for the OSM User-Agent")
    ap.add_argument("--grid", type=int, default=25)
    ap.add_argument("--px", type=int, default=None)
    ap.add_argument("--raster", action="store_true")
    ap.add_argument("--class-widths", action="store_true")
    ap.add_argument("--street-width", type=float)
    ap.add_argument("--default-height", type=float)
    ap.add_argument("--conventions")
    ap.add_argument("--only", help="comma-separated city names")
    ap.add_argument("--skip", help="comma-separated city names")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--pause", type=float, default=20.0)
    args = ap.parse_args()

    cities = load_list(Path(args.list))
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        cities = [c for c in cities if c["city"] in want]
    if args.skip:
        drop = {s.strip() for s in args.skip.split(",")}
        cities = [c for c in cities if c["city"] not in drop]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "render_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    todo = [c for c in cities if args.redo or not (out / c["city"] / f"{c['city']}_grid.json").exists()]
    print(f"{len(cities)} cities in the list, {len(cities) - len(todo)} already done, {len(todo)} to render")
    t_all = time.time()

    for i, c in enumerate(todo, 1):
        city = c["city"]
        print(f"\n=== [{i}/{len(todo)}] {city} " + "=" * 40)
        t0 = time.time()
        a = Namespace(
            mode="grid", city=city, grid=args.grid, out=args.out, cache=args.cache,
            contact=args.contact, conventions=args.conventions, px=args.px, raster=args.raster,
            class_widths=args.class_widths, street_width=args.street_width,
            default_height=args.default_height,
            center=(f"{c['lat']},{c['lon']}" if "lat" in c else None),
            address=c.get("address") or (None if "lat" in c else city.replace("_", " ")),
        )
        try:
            summary = osm_tiles.mode_grid(a)
            summary["seconds"] = round(time.time() - t0)
            summary["ok"] = True
            report[city] = summary
        except SystemExit as e:                     # osm_tiles exits on an unrecoverable fetch
            report[city] = {"city": city, "ok": False, "error": str(e), "seconds": round(time.time() - t0)}
            print(f"  FAILED: {e}")
        except Exception as e:  # noqa
            report[city] = {"city": city, "ok": False, "error": f"{type(e).__name__}: {e}",
                            "seconds": round(time.time() - t0)}
            traceback.print_exc()
        report_path.write_text(json.dumps(report, indent=1))
        if i < len(todo):
            time.sleep(args.pause)

    print(f"\nall done in {(time.time() - t_all) / 3600:.1f} h\n")
    print(f"{'city':22s} {'tiles':>6s} {'empty':>6s} {'buildings':>10s} {'streets':>8s} {'green':>7s} {'min':>5s}")
    bad = []
    for city in sorted(report):
        r = report[city]
        if not r.get("ok"):
            bad.append((city, r.get("error", "?")))
            continue
        print(f"{city:22s} {r['tiles']:6d} {r['empty']:6d} {r['buildings']:10d} {r['streets']:8d} "
              f"{r['greens']:7d} {r['seconds'] / 60:5.1f}")
        if r["empty"] > r["tiles"] * 0.6 or r["buildings"] < 2000:
            bad.append((city, f"thin: {r['empty']} empty tiles, {r['buildings']} buildings"))
    if bad:
        print("\nworth a look:")
        for city, why in bad:
            print(f"  {city:22s} {why}")
    print(f"\nreport: {report_path}")


if __name__ == "__main__":
    main()
