# Phase 0: data audit and re-encoding

Three files, no torch dependency. Needs `numpy`, `pillow`, `scipy`, `matplotlib`
(all already in the project's requirements).

Copy this folder next to `images/` (or point `--images` at it) and run from inside it.

## 1. Audit the raw tiles

    python audit_tiles.py --images ../images --out audit

Prints one line per city and a summary, and writes to `audit/`:

| file | what |
|---|---|
| `summary.txt` | headline numbers: blank / near-blank / usable, grey range and percentiles |
| `cities.csv` | per city: tile counts, usable percent, mean coverage / street / green |
| `tiles.csv` | per tile: status, content percent, per-class percent, grey stats |
| `blank_grids.png` | a 25 x 25 map per city showing where the blanks fall |
| `city_blank_bars.png` | usable percent per city, sorted |
| `grey_histogram.png` | building grey-value distribution, linear and log |

`blank_grids.png` is the one to look at first. If the blanks cluster along one edge of a
city, that window was not centred on the built-up area and is worth re-centring when the
Grasshopper pipeline is re-run.

## 2. Build the 4-channel dataset

    python reencode_tiles.py --images ../images --out dataset128_sdf --size 128 --encoding sdf
    python reencode_tiles.py --images ../images --out dataset128 --size 128            # hard masks

Drops tiles under `--min-content-pct` (default 5 percent non-background), builds a
corpus-wide height look-up table, encodes every kept tile as four channels
(footprint, height, street, green) in one of two encodings:

    mask   footprint / street / green are 0 or 255; height is 0 outside the footprint,
           51..255 inside; resized with nearest-neighbour
    sdf    footprint / street / green are truncated signed distance fields, byte 128 on
           the class edge, >128 inside, <128 outside, saturating 8 native px away;
           height is exact inside footprints and, outside, the nearest building's
           height blurred by 1.5 px (a tile with no buildings gets a flat field at the
           corpus mean); fields are computed at native size and resampled bilinearly

Use `sdf` for training. Hard masks are a shortcut for the discriminator (a generator
never outputs exactly 0 or 255, and nearest-neighbour resizing stamps a fixed aliasing
pattern on every real tile); three runs on mask data had the ADA augment value climb
past 1.0 without ever flattening. Both encodings decode with the same rule, class present
where byte >= 128 and height read only where the footprint is, so nothing downstream
cares which one a dataset used. `encoding.json` records it.

Writes to the output folder:

| file | what |
|---|---|
| `tiles.npz` | `x` uint8 [N, 4, S, S], `city` int16 [N], `idx` int16 [N] |
| `meta.csv` | per kept tile: city, grid row/col, coverage, street, green, height, FAR proxy, building count, street components |
| `height_lut.json` | forward and inverse height mapping. Keep this with any trained model, it is needed to decode generated tiles back to grey values |
| `cities.json` | city name to integer label |
| `encoding.json` | encoding and its parameters |
| `preview.png` | original, four channels, and round-trip decode side by side |
| `stats.txt` | kept / dropped counts and a lossless round-trip check |

### Height mapping

The renderer's grey values (observed 105..241) are histogram-equalised across the corpus
with a guaranteed minimum spacing of 4 bytes between neighbouring levels, so the common
105..121 band spreads over roughly 160 bytes instead of 16, while rare tall buildings keep
their own distinct values. The mapping is exactly invertible; `stats.txt` confirms it.

`tile_metrics()` converts grey back to metres with the renderer's mapping, grey 100 to
255 is 4 m to 100 m linear (`grey_to_metres` / `metres_to_grey` in `tile_codec.py`).
`far_proxy` counts floors at 3 m. The encoding itself does not depend on this mapping.

### Loading in PyTorch

    import numpy as np, torch
    d = np.load("dataset/tiles.npz")
    x = torch.from_numpy(d["x"]).float() / 127.5 - 1.0   # [N, 4, S, S] in [-1, 1]
    city = torch.from_numpy(d["city"]).long()

### Decoding a generated tile

    from tile_codec import HeightLUT, channels_to_rgb, tile_metrics
    lut = HeightLUT.load("dataset/height_lut.json")
    ch = ((g_out.clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()   # [4, S, S]
    rgb = channels_to_rgb(ch, lut)      # back to the red / green / grey palette
    m = tile_metrics(ch, lut)           # the same metrics as meta.csv

## Notes from the first audit (Barcelona sample)

* Tiles are 100 x 100 px, RGBA with a constant alpha of 255. No off-palette pixels, so
  the class masks are exact.
* Streets are drawn on top of buildings. A building under a street is lost; that is a
  renderer decision, not something to fix here.
* Files of exactly 373 bytes are fully black tiles.

## Tiles straight from OpenStreetMap: `osm_tiles.py`

The Grasshopper renderer produced the original tiles. `osm_tiles.py` does the same job in
Python for any place on earth, which makes the pipeline reproducible and is what the
future app will run when a user types an address. It fetches buildings, streets and
green areas from the Overpass API in cached 2.5 km chunks, projects them to a local
metric frame, rasterises them at 4 m per pixel (supersampled 2x, majority vote) and
writes the same palette encoding as before, so `reencode_tiles.py` accepts the output
unchanged.

    pip install requests shapely pyproj

Regenerate one city in the training layout (25 x 25 tiles of 400 m, ten minutes of
polite fetching the first time, seconds afterwards from the cache):

    python osm_tiles.py grid --city Sao_Paulo --center -23.5505,-46.6333 --out C:\opengan\images_all --contact you@example.com

One image of any size around an address, for the app:

    python osm_tiles.py canvas --address "Carrer de Mallorca 401, Barcelona" --size-m 1200 --out site.png --contact you@example.com

Old tiles next to new ones for the same window, with per-class IoU, searching around
the given centre for the alignment the Grasshopper run used:

    python osm_tiles.py compare --city Barcelona --center 41.3874,2.1686 --images C:\opengan\images_all --idx 312 183 88 --search-m 800 --contact you@example.com

Every grid run writes `<City>_grid.json` (centre, projection, tile origin rule) so each
tile is georeferenced, and `conventions.json` with the settings used.

The conventions, which the Grasshopper tiles never wrote down, live in `CONVENTIONS` at
the top of the file and can be overridden with `--conventions my.json` or the flags:
street width 8 m for every class (that is what the old tiles have: 2 px at 4 m/px on
every road; `--class-widths` switches to a per-class table); street classes motorway to
living_street plus pedestrian and the `_link` variants, with service roads, footways,
paths, cycleways and tracks excluded; building height from the `height` tag, else
`building:levels` x 3 m, else 7 m (`--default-height`); green from parks, gardens,
pitches, grass, forest, meadow, cemeteries and similar landuse, leisure and natural
tags; draw order green, then buildings, then streets. Multipolygon relations (parks
with ponds, buildings with courtyards) are assembled from their member ways.

### All the cities at once: `render_cities.py`

Fetching is network-bound, so the whole corpus should be rendered on the CPU while the
GPU trains:

    python render_cities.py --list cities_60.json --out C:\opengan\images_v2 --cache C:\opengan\osm_cache --contact you@example.com --px 128 --raster

`cities_60.json` holds the 60 cities of the current dataset with their coordinates. The
run is resumable: a city that already has its `_grid.json` is skipped, and anything
already fetched stays in the cache, so the job can be stopped and restarted at will. A
report lands in `<out>/render_report.json` and is printed as a table at the end, with a
"worth a look" list flagging cities that came back suspiciously thin, which is how the
old São Paulo problem would have been caught. Expect ten to twenty minutes per city on
the first pass.

`--px 128` renders natively at the training resolution (3.125 m per pixel) instead of
the old 4 m, so `reencode_tiles.py --size 128` no longer has to upsample. `--raster`
additionally writes the whole 10 km window as one `<City>_raster.png`, 3200 x 3200 px,
about 41 MB of pixels per city: that is the form the inpainting work wants, because
training can then crop 128 px windows at random positions instead of being stuck with a
fixed grid, and cropping 512 px and downsampling gives the larger scale for free.

### When Overpass says no

The public servers hand out a couple of query slots per client and throttle anyone who
retries blind. The fetcher therefore asks `/api/status` for a free slot before every
call, rotates over four mirrors, and backs off up to two minutes between attempts. It
only splits a box into quarters when a server says the query itself timed out, which is
the one case where a smaller box helps: splitting on a rate limit just multiplies the
requests and makes the throttling worse.

If a whole city still fails, the servers are busy rather than broken. Wait an hour and
run the batch again; everything already fetched is in the cache, so it resumes where it
stopped. For a corpus of hundreds of cities, stop using the public API and parse a
Geofabrik `.pbf` extract locally instead.

Two honest caveats. Heights are the weak point of OSM: many cities tag a minority of
buildings, so the 7 m fallback shows up a lot; the old tiles have the same problem in a
different disguise. And the public Overpass servers rate-limit: keep `--contact` real,
let the 1 s pause between chunks be, and expect a city to take five to ten minutes the
first time. For hundreds of cities, switch the fetch to a local Geofabrik extract.
