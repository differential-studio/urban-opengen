"""plan, winter ground sun, facade winter sun and facade sky view, for a few tiles"""
import sys
import numpy as np
sys.path.insert(0, '..'); sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from tile_codec import HeightLUT, channels_to_rgb
import vectorize as V, analysis as A

lut = HeightLUT.load('/mnt/user-data/uploads/opengan/dataset128_sdf/height_lut.json')
real = np.load('../data/real_v2.npz')
TILES = [('Barcelona_411', 41.39, 'Barcelona Eixample'), ('Los_Angeles_312', 34.05, 'Los Angeles downtown'),
         ('Berlin_312', 52.52, 'Berlin, park edge'), ('Osaka_312', 34.69, 'Osaka towers')]

fig, axes = plt.subplots(len(TILES), 4, figsize=(15.5, 3.7 * len(TILES)))
for row, (name, lat, label) in enumerate(TILES):
    ch = real[name]
    res = V.vectorize(ch, lut)
    parts = V.parts_m(res)
    a = A.analyse(parts, lat=lat, maps=True)
    M, P = a['_maps'], a['_points']
    x0, y1 = M['anchor']; px = M['px_m']
    R, C = M['height'].shape
    ext = [x0, x0 + C * px, y1 - R * px, y1]

    ax = axes[row, 0]
    ax.imshow(channels_to_rgb(ch, lut), extent=ext)
    ax.set_ylabel(f"{label}\nFAR {a['far']:.2f}  cover {a['coverage_pct']:.0f}%", fontsize=9)
    if row == 0: ax.set_title('the tile', fontsize=10)

    ax = axes[row, 1]
    gs = np.where(M['class'] == A.CLS_BUILDING, np.nan, M['ground_sun']['winter'])
    im = ax.imshow(gs, extent=ext, cmap='inferno', vmin=0, vmax=a['sun_winter_daylength_h'])
    plt.colorbar(im, ax=ax, fraction=0.046, label='h')
    ax.set_xlabel(f"street {a['sun_winter_street_h']:.1f} h, {a['sun_winter_ground_above_2h_pct']:.0f}% of open above 2 h", fontsize=8)
    if row == 0: ax.set_title(f"winter solstice, sun on the ground\n(day is {a['sun_winter_daylength_h']:.1f} h long here)", fontsize=10)

    for col, (key, vals, cmap, lo, hi, txt) in enumerate([
            ('sun', P['sun']['winter'], 'inferno', 0, a['sun_winter_daylength_h'],
             f"mean {a['sun_winter_facade_h']:.1f} h, {a['sun_winter_facade_above_2h_pct']:.0f}% of wall above 2 h"),
            ('sky', P['view']['sky'] * 100, 'viridis', 0, 60,
             f"sky {a['view_sky_pct']:.0f}%, overlooked {a['view_overlooked_pct']:.0f}%, view reaches {a['view_obstruction_m']:.0f} m")]):
        ax = axes[row, 2 + col]
        ax.imshow(np.where(M['class'] == A.CLS_BUILDING, 0.85, 1.0), extent=ext, cmap='gray', vmin=0, vmax=1)
        o = np.argsort(vals)
        sc = ax.scatter(P['xy'][o, 0], P['xy'][o, 1], c=vals[o], s=2.5, cmap=cmap, vmin=lo, vmax=hi, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.046, label='h' if key == 'sun' else '%')
        ax.set_xlabel(txt, fontsize=8)
        if row == 0: ax.set_title('winter sun on the walls' if key == 'sun' else 'how much sky each wall sees', fontsize=10)
    for ax in axes[row]:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
plt.tight_layout()
plt.savefig('../out/analysis_real_tiles.png', dpi=105)
print('saved')
