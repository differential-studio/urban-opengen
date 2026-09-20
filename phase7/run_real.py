import sys, time, json
import numpy as np
sys.path.insert(0, '..'); sys.path.insert(0, '.')
from pathlib import Path
from tile_codec import HeightLUT
import vectorize as V
import analysis as A

LAT = {'Barcelona': 41.39, 'Paris': 48.86, 'Berlin': 52.52, 'Istanbul': 41.01, 'Osaka': 34.69,
       'Jakarta': -6.21, 'Cairo': 30.04, 'Los_Angeles': 34.05, 'Buenos_Aires': -34.61}
# the LUT and the tile arrays these scripts read, found relative to the repository
# rather than to wherever you happen to be standing when you run them
ROOT = Path(__file__).resolve().parent.parent
lut = HeightLUT.load(ROOT / 'assets' / 'height_lut.json')
real = np.load(ROOT / 'data' / 'real_v2.npz')
gen = np.load(ROOT / 'data' / 'fakes003000_ch.npz')['x']

rows = []
def run(name, ch, lat, kind):
    t0 = time.time(); res = V.vectorize(ch, lut); t1 = time.time()
    parts = V.parts_m(res)
    t2 = time.time(); a = A.analyse(parts, lat=lat); t3 = time.time()
    a.update(name=name, kind=kind, ms_vector=round((t1-t0)*1000), ms_analysis=round((t3-t2)*1000))
    rows.append(a); return a

for n in ['Barcelona_184', 'Barcelona_411', 'Barcelona_16', 'Paris_287', 'Paris_312', 'Berlin_287',
          'Berlin_312', 'Istanbul_312', 'Osaka_312', 'Jakarta_312', 'Cairo_312', 'Los_Angeles_312',
          'Buenos_Aires_312']:
    city = n.rsplit('_', 1)[0]
    run(n, real[n], LAT[city], 'real')
for r, c in [(10, 2), (17, 9), (25, 7), (31, 12)]:
    run(f'gen_{r}_{c}', gen[r*32+c], 41.39, 'generated')

cols = ['far', 'coverage_pct', 'green_pct', 'mean_height_m', 'facade_to_floor',
        'sun_equinox_ground_h', 'sun_winter_ground_h', 'sun_winter_ground_above_2h_pct',
        'sun_winter_street_h', 'sun_winter_facade_h', 'sun_winter_facade_above_2h_pct',
        'view_sky_pct', 'view_building_pct', 'view_green_pct', 'view_overlooked_pct',
        'view_obstruction_m']
hdr = ['tile', 'lat', 'FAR', 'cov%', 'grn%', 'h(m)', 'f/flr', 'sunEQ', 'gndW', '>2h%', 'stW', 'facW', 'f>2h%',
       'sky%', 'bld%', 'grn%', 'ovl%', 'obst', 'ms']
w = [16, 6, 5, 6, 5, 6, 6, 6, 6, 6, 5, 5, 6, 5, 5, 5, 5, 5, 9]
print(''.join(h.rjust(x) for h, x in zip(hdr, w)))
for a in rows:
    vals = [a['name'], f"{a['latitude']:.1f}"] + [f"{a[c]:.2f}" if isinstance(a[c], float) else str(a[c]) for c in cols] + [f"{a['ms_vector']}+{a['ms_analysis']}"]
    print(''.join(str(v).rjust(x) for v, x in zip(vals, w)))
json.dump([{k: v for k, v in a.items() if not k.startswith('_')} for a in rows], open('real_tiles.json', 'w'), indent=1)
