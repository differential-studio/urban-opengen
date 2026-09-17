"""
scan_dataset.py

Phase 0: does every city actually have its 625 tiles?

One question, counted from filenames. A city is complete when <City>_0.png through
<City>_624.png are all present, and incomplete otherwise. Nothing here looks inside a tile,
and nothing here is a judgement about whether a tile is any good: the only thing that puts a
city on the re-render list is a file that is not there.

    python scan_dataset.py --images C:\\opengan\\images_all --cities cities_60.json

Reads. Never writes, never deletes, never renders. It prints a table and, at the end, the
commands you would run to re-render the incomplete cities, for you to run or not.

--check-blank additionally opens every tile and counts the ones with nothing drawn in them.
That is slow and it is only ever reported, never acted on: a blank tile can perfectly well be
a correct tile of open water, and telling those apart is not this script's job.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TILE_RE = re.compile(r"^(?P<city>.+)_(?P<idx>\d+)\.png$", re.IGNORECASE)


def runs(nums: list[int]) -> str:
    """[1,2,3,7,9,10] -> '1-3, 7, 9-10', so a long gap reads as a gap"""
    if not nums:
        return ""
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    return ", ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="the tile root, one folder per city")
    ap.add_argument("--cities", default="cities_60.json", help="the city list the dataset claims to cover")
    ap.add_argument("--grid", type=int, default=25)
    ap.add_argument("--csv", default=None, help="also write a row per city here")
    ap.add_argument("--contact", default="you@example.com", help="the email to put in the re-render commands")
    ap.add_argument("--check-blank", action="store_true",
                    help="also open every tile and count the empty ones. Reported only, never a "
                         "reason to re-render: an empty tile can be a correct tile of open water")
    args = ap.parse_args()

    root = Path(args.images)
    if not root.is_dir():
        sys.exit(f"no such folder: {root}")
    want = args.grid * args.grid
    listed = json.loads(Path(args.cities).read_text()) if Path(args.cities).exists() else []
    centres = {c["city"]: (c["lat"], c["lon"]) for c in listed}
    names = [c["city"] for c in listed] or sorted(p.name for p in root.iterdir() if p.is_dir())
    if not listed:
        print(f"note: {args.cities} not found, scanning whatever folders are under --images\n")

    blank_col = f"{'blank':>6}" if args.check_blank else ""
    rows, todo = [], []
    print(f"{'city':22} {'files':>6} {'missing':>8}{blank_col}  {'stray':>6}  state")
    print("-" * (72 + len(blank_col)))
    for city in names:
        d = root / city
        if not d.is_dir():
            todo.append(city)
            print(f"{city:22} {'-':>6} {'-':>8}{'':>{len(blank_col)}}  {'-':>6}  no folder")
            rows.append({"city": city, "files": 0, "missing": want, "blank": "", "stray": 0,
                         "state": "no folder", "missing_idx": f"0-{want - 1}"})
            continue
        have, stray = {}, []
        for p in d.glob("*.png"):
            m = TILE_RE.match(p.name)
            if not m or m.group("city").lower() != city.lower():
                stray.append(p.name)
                continue
            i = int(m.group("idx"))
            (have.__setitem__(i, p) if i < want else stray.append(p.name))
        missing = [i for i in range(want) if i not in have]

        blank = ""
        if args.check_blank:
            import numpy as np
            from PIL import Image
            n = 0
            for p in have.values():
                try:
                    with Image.open(p) as im:
                        if not np.asarray(im.convert("RGB")).any():
                            n += 1
                except Exception:
                    n += 1
            blank = n

        state = "complete" if not missing else f"{len(missing)} missing"
        if missing:
            todo.append(city)
        bc = f"{blank:>6}" if args.check_blank else ""
        print(f"{city:22} {len(have):>6} {len(missing):>8}{bc}  {len(stray):>6}  {state}")
        rows.append({"city": city, "files": len(have), "missing": len(missing), "blank": blank,
                     "stray": len(stray), "state": state, "missing_idx": runs(missing)[:400]})

    done = len(names) - len(todo)
    print(f"\n{done} of {len(names)} cities have all {want} tiles, {len(todo)} do not")
    if args.check_blank:
        print("the blank column is information only: no city is on the list below because of it")

    for r in rows:
        if r["missing"] and r["missing"] < want and r["missing_idx"]:
            print(f"  {r['city']}: missing {r['missing_idx']}")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv}")

    if not todo:
        print("\nNothing to do: every city is a full 25 x 25.")
        return
    print(f"\nThese {len(todo)} are short of {want} tiles. A city renders whole, so each command "
          f"below\nreplaces all {want} tiles of that one city and touches no other city:\n")
    for city in todo:
        if city not in centres:
            print(f"  # {city}: no centre in {args.cities}, pass --center lat,lon yourself")
            continue
        lat, lon = centres[city]
        print(f"  python osm_tiles.py grid --city {city} --center {lat},{lon} "
              f"--out {args.images} --grid {args.grid} --contact {args.contact}")


if __name__ == "__main__":
    main()
