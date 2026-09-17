"""
check_cities.py

Reverse-geocode every coordinate in a city list and print what OSM thinks is actually
there. Three minutes now against a typo that costs half a day of fetching, so run this
before render_cities.py on any list you did not hand-verify.

    python check_cities.py --list cities_120.json

Anything printed under "check by hand" is a coordinate whose reverse geocode did not
look like the city name. That is not always wrong: a centre point can land in a named
district rather than the city, and a few of these names are deliberately not the OSM
name (Cordoba_AR, Hamilton_NZ). Read the returned place and decide.

Nominatim asks for one request per second and a real contact in the User-Agent, so this
is deliberately slow and deliberately identifies itself.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

URL = "https://nominatim.openstreetmap.org/reverse?"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True)
    ap.add_argument("--contact", required=True, help="your email, Nominatim requires a contact")
    ap.add_argument("--pause", type=float, default=1.1)
    a = ap.parse_args()

    rows = json.loads(Path(a.list).read_text())
    ua = f"urban-opengen-coord-check/1.0 ({a.contact})"
    bad = []
    for i, r in enumerate(rows, 1):
        if "lat" not in r:
            continue
        q = urllib.parse.urlencode({"lat": r["lat"], "lon": r["lon"], "format": "json", "zoom": 10})
        try:
            req = urllib.request.Request(URL + q, headers={"User-Agent": ua})
            d = json.load(urllib.request.urlopen(req, timeout=30))
            ad = d.get("address", {})
            got = ad.get("city") or ad.get("town") or ad.get("municipality") or ad.get("county") or ad.get("state") or ""
            cc = (ad.get("country_code") or "").upper()
            want = r["city"].replace("_NZ", "").replace("_AR", "").replace("_", " ")
            ok = want.lower()[:5] in got.lower() or (got.lower()[:5] in want.lower() if got else False)
            print(f"[{i:3d}/{len(rows)}] {'ok ' if ok else 'CHK'} {r['city']:20s} -> {got}, {cc}")
            if not ok:
                bad.append((r["city"], got, cc, d.get("display_name", "")[:100]))
        except Exception as e:  # noqa
            print(f"[{i:3d}/{len(rows)}] ERR {r['city']:20s} {e}")
            bad.append((r["city"], f"error: {e}", "", ""))
        time.sleep(a.pause)

    if bad:
        print("\ncheck by hand:")
        for city, got, cc, full in bad:
            print(f"  {city:20s} got '{got}' {cc}\n      {full}")
    else:
        print("\nall coordinates land in the expected city")


if __name__ == "__main__":
    main()
