"""
How much does the vectoriser change the answer?

The analysis runs on the fitted polygons, because that is the model you can
download. This asks the same questions of the raster the model actually drew,
resampled to the analysis grid with no fitting at all, and reports the gap. If
the two agree, the sun numbers are a property of the scheme rather than of the
vectoriser's settings.
"""
import sys, math
import numpy as np
from scipy import ndimage
sys.path.insert(0, '..'); sys.path.insert(0, '.')
from tile_codec import HeightLUT, grey_to_metres
import vectorize as V, analysis as A

lut = HeightLUT.load('/mnt/user-data/uploads/opengan/dataset128_sdf/height_lut.json')
real = np.load('../data/real_v2.npz')
LAT = {'Barcelona': 41.39, 'Paris': 48.86, 'Berlin': 52.52, 'Istanbul': 41.01, 'Osaka': 34.69,
       'Jakarta': -6.21, 'Cairo': 30.04, 'Los_Angeles': 34.05, 'Buenos_Aires': -34.61}
PX = 2.0


def raster_fields(ch, tile_m=400.0, px_m=PX):
    """height and class straight off the tile channels, nearest-resampled, no fitting"""
    fp = ch[0] >= 128; st = ch[2] >= 128; gr = ch[3] >= 128
    h = grey_to_metres(np.asarray(lut.inverse)[ch[1]]).astype(np.float32) * fp
    n = int(round(tile_m / px_m))
    z = ch.shape[1] / n
    idx = (np.arange(n) * z).astype(int)
    take = lambda a: a[np.ix_(idx, idx)]  # noqa: E731
    cls = np.full((n, n), A.CLS_OPEN, np.uint8)
    cls[take(st)] = A.CLS_STREET
    cls[take(gr)] = A.CLS_GREEN
    cls[take(fp)] = A.CLS_BUILDING
    return take(h), cls


def ground_sun(H, cls, lat, decl, thr=2.0):
    alt, az, w = A.solar_positions(lat, decl, 30)
    acc = np.zeros_like(H)
    for a, z, hrs in zip(alt, az, w):
        M = A.horizon_field(H, a, z, PX, 128)
        acc += np.where(M <= 0.05, hrs, 0.0).astype(np.float32)
    open_ = cls != A.CLS_BUILDING
    street = cls == A.CLS_STREET
    return (float(acc[open_].mean()), 100.0 * float((acc[open_] >= thr).mean()),
            float(acc[street].mean()) if street.any() else float('nan'))


names = ['Barcelona_184', 'Barcelona_411', 'Barcelona_16', 'Paris_287', 'Paris_312', 'Berlin_287',
         'Berlin_312', 'Istanbul_312', 'Osaka_312', 'Jakarta_312', 'Cairo_312', 'Los_Angeles_312',
         'Buenos_Aires_312']
print(f"{'tile':17s}{'built %':>18s}{'mean height m':>16s}{'ground sun h':>16s}{'above 2 h %':>16s}{'street sun h':>16s}")
print(f"{'':17s}{'raster  fit':>18s}{'raster  fit':>16s}{'raster  fit':>16s}{'raster  fit':>16s}{'raster  fit':>16s}")
d = {k: [] for k in ('built', 'h', 'sun', 'above', 'street')}
for n in names:
    lat = LAT[n.rsplit('_', 1)[0]]
    decl = -23.44 if lat >= 0 else 23.44
    ch = real[n]
    Hr, clsr = raster_fields(ch)
    sr = ground_sun(Hr, clsr, lat, decl)
    parts = V.parts_m(V.vectorize(ch, lut))
    Hf, clsf, _ = A.rasterize_parts(parts, PX)
    sf = ground_sun(Hf, clsf, lat, decl)
    br, bf = 100.0 * (clsr == A.CLS_BUILDING).mean(), 100.0 * (clsf == A.CLS_BUILDING).mean()
    hr = Hr[Hr > 0].mean() if (Hr > 0).any() else 0.0
    hf = Hf[Hf > 0].mean() if (Hf > 0).any() else 0.0
    print(f"{n:17s}{br:9.1f}{bf:9.1f}{hr:8.1f}{hf:8.1f}{sr[0]:8.2f}{sf[0]:8.2f}{sr[1]:8.1f}{sf[1]:8.1f}{sr[2]:8.2f}{sf[2]:8.2f}")
    d['built'].append(bf - br); d['h'].append(hf - hr); d['sun'].append(sf[0] - sr[0])
    d['above'].append(sf[1] - sr[1]); d['street'].append(sf[2] - sr[2])
print()
for k, unit in [('built', 'percentage points of the tile'), ('h', 'm'), ('sun', 'h'),
                ('above', 'percentage points'), ('street', 'h')]:
    a = np.asarray(d[k])
    print(f"  fit minus raster, {k:6s}: mean {a.mean():+.2f}, mean absolute {np.abs(a).mean():.2f}, worst {a[np.argmax(np.abs(a))]:+.2f}  ({unit})")
