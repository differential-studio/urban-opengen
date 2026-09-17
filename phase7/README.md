# Phase 7: analysis

Direct sun and view analysis for a generated or real tile, plus the density numbers, computed on
the same geometry the viewer draws and the 3dm export writes.

This exists so that options can be compared on something the model does not control. FAR, coverage
and green share are inputs: the GAN's target solver hits them and the inpainter is conditioned on
them, so plotting them against each other says nothing. Sun and outlook are emergent. Two schemes
at FAR 4 can differ by a factor of two in sunlit wall, and that difference is the reason to build a
design space at all.

It is not a daylight report. There is no diffuse light, no reflection, no glazing, no material and
no ground albedo. What it computes is exact for the massing it is given, which is what makes two
options comparable.

    python analysis.py --selftest

## Running it, start to finish

Once, to install the one extra library the Rhino export needs:

    C:\opengan\venv\Scripts\activate
    pip install rhino3dm

Check the analysis behaves on your machine. This generates nothing and takes a few seconds:

    python "G:\...\gan_project\phase7\analysis.py" --selftest

Start the viewer the way you normally do, and load a snapshot in it. The design space runner asks
this server for tiles, so whatever snapshot is loaded is what the space is built from:

    cd C:\opengan\stylegan3
    python "G:\...\gan_project\phase4\serve.py" --repo . ^
        --lut ..\dataset128_sdf\height_lut.json --cities ..\dataset128_sdf\cities.json

Open `phase7/space_barcelona.json` and change the ranges if you want. Then see how big the run is
before committing to it. This only prints, it generates nothing:

    python "G:\...\gan_project\phase7\explore.py" "G:\...\gan_project\phase7\space_barcelona.json" --dry-run

If the count looks right, drop `--dry-run` and let it run. It prints a line per option and an
estimate of the time left:

    python "G:\...\gan_project\phase7\explore.py" "G:\...\gan_project\phase7\space_barcelona.json"

You get `C:\opengan\spaces\run01` containing `001.3dm`, `002.3dm` and so on, `context.3dm`,
`parameters.csv`, `objectives.csv`, `run.json`, and a `thumbs` folder of small plan images. Point
the multiobjective viewer at it.

Optionally, add the real-world comparison columns to that run:

    python "G:\...\gan_project\phase7\corpus.py" C:\opengan\dataset128_sdf\meta.csv ^
        --space C:\opengan\spaces\run01 --background C:\opengan\spaces\corpus.csv

If a run is interrupted, running the same command again picks up where it stopped.

## What comes out

One flat dictionary per tile, ready to be a row in a CSV.

| group | columns |
|---|---|
| form | `far`, `coverage_pct`, `open_space_ratio`, `green_pct`, `street_pct`, `mean_height_m`, `max_height_m`, `building_count`, `mean_footprint_m2`, `gfa_m2`, `facade_area_m2`, `facade_to_floor` |
| sun, per day | `sun_<day>_daylength_h`, `sun_<day>_ground_h`, `sun_<day>_ground_above_2h_pct`, `sun_<day>_street_h`, `sun_<day>_facade_h`, `sun_<day>_facade_above_2h_pct` |
| view | `view_sky_pct`, `view_building_pct`, `view_green_pct`, `view_street_pct`, `view_open_pct`, `view_overlooked_pct`, `view_obstruction_m` |

Days default to `equinox` and `winter`, and winter means the solstice of the hemisphere the tile is
in, so a Buenos Aires tile is analysed in June. The ground numbers cover everything that is not
built, `street` restricts them to the street surface, which is the public realm and the thing a
scheme is usually judged on: in a perimeter block the courtyards drag the open-ground average down.
Facade numbers are weighted by facade area, courtyard walls included.

`view_*` are shares of the outward hemisphere of a wall, by solid angle, so an unobstructed wall
reads 50 percent sky and 50 percent ground. `view_overlooked_pct` is the part of the view taken by
building surfaces within 25 m, which is the privacy reading, and `view_obstruction_m` is how far a
view reaches before it meets something, over the rays that meet anything at all.

## Three simplifications, stated once

* Nothing outside the tile casts a shadow or blocks a view, so points near the edge are optimistic.
  `margin_m` excludes a border from the ground statistics.
* The massing is the extruded footprints. A pitched roof, a setback and a balcony do not exist.
* Facades are sampled every 10 m at 2 levels and stand 1.5 cells clear of the wall, so these are
  block-scale readings, not room-scale ones.

Day length is geometric, the sun's centre crossing a flat horizon. A published sunrise table adds
the solar disc and refraction, eight or nine minutes at European latitudes, which is not direct sun
anyone can use.

## How it is fast

For one sun position, `horizon_field` returns, for every cell, the lowest height from which the sun
is visible there. A point at (x, y, z) sees the sun when z is above that, so one field answers the
question for the ground and for every facade sample above it. The field is the running maximum
along the ray toward the sun of the obstruction height minus the height the ray has climbed, and
that maximum is computed by doubling: the maximum over 2k steps is the maximum over k steps
combined with the same field shifted k steps and lowered by what the ray climbs in k steps. So 128
steps of marching cost 8 array shifts, and a whole day of sun positions takes about a fifth of a
second. Only the view rays are marched one by one, batched over every ray at once.

A 400 m tile at 2 m resolution, two days of sun and about 4000 facade samples takes 0.4 to 1.0
seconds, roughly the same again as the vectoriser that feeds it. Raise `px_m` to 3 and drop
`facade_levels` to 1 for a batch run.

## The self-test

`--selftest` checks the things that have an answer known in advance, which is the only way to trust
this kind of code:

1. day length at six latitudes including the southern hemisphere, against `acos(-tan(lat) tan(decl))`
2. the horizon field against a brute-force ray march, on four towers that genuinely shade each other
3. an isolated block at Barcelona's latitude on the winter solstice: the sun is south of east all
   day, so the south wall sees the whole day, the north wall none, and east and west half each
4. the noon shadow of a 30 m block reaches `h / tan(noon altitude)` to the north, to within a metre
5. an unobstructed wall sees exactly 50 percent sky
6. an empty tile at the equinox gets twelve hours everywhere
7. four towers moved from 140 m apart to 45 m: less sky, more overlooking, shorter views, less sun
   on the walls, less wall above two hours

Three real bugs were found this way and would not have been found by looking at pictures: sun
samples near sunrise were being dropped so the day was short, the march step could be finer than a
cell so a ray only ever sampled the cell it started in, and a facade sample one metre off its wall
sat on the raster boundary of its own building and was shadowed by it.

## Does the vectoriser change the answer?

The analysis runs on the fitted polygons, because that is the model you can download. `validate_fit.py`
asks the same questions of the raster the model actually drew, with no fitting at all, and reports
the gap over thirteen real tiles from nine cities:

| quantity | mean difference | mean absolute | worst |
|---|---|---|---|
| built area | +0.15 pp | 0.48 pp | +1.05 pp |
| mean height | +0.31 m | 0.35 m | +0.96 m |
| ground sun | +0.14 h | 0.15 h | +0.40 h |
| street sun | +0.14 h | 0.16 h | +0.54 h |
| open ground above 2 h | +2.03 pp | 2.05 pp | +6.71 pp |

So the sun numbers are a property of the scheme and not of the vectoriser's settings. The fit is
consistently a touch sunnier because it removes single-pixel speckle and straightens narrow gaps.
The threshold metric moves most, as a threshold metric will.

## What it says about real places

Thirteen tiles, winter solstice, at each city's own latitude:

| tile | FAR | cover | street sun | wall above 2 h | sky | overlooked | view reaches |
|---|---|---|---|---|---|---|---|
| Barcelona Eixample | 4.29 | 70 % | 1.1 h | 26 % | 20 % | 65 % | 8 m |
| Paris, dense | 3.62 | 59 % | 1.1 h | 30 % | 26 % | 50 % | 13 m |
| Los Angeles downtown | 4.94 | 33 % | 3.7 h | 52 % | 34 % | 18 % | 36 m |
| Osaka towers | 2.62 | 22 % | 5.0 h | 58 % | 40 % | 14 % | 36 m |
| Berlin, park edge | 1.18 | 18 % | 5.7 h | 33 % | 36 % | 23 % | 24 m |
| Istanbul | 0.67 | 31 % | 7.2 h | 50 % | 40 % | 24 % | 10 m |

Barcelona and Los Angeles carry almost the same floor area. Los Angeles puts it in towers and gets
three times the sun in the street and twice the sunlit wall; Barcelona puts it in perimeter blocks
and gets a street wall, continuous frontage and eight-metre views. Neither is the right answer.
That is the trade-off the design space is meant to let someone argue about with numbers.

## The design space (from the viewer, or `explore.py`)

The usual way in is the viewer itself: **Export design space** at the bottom of the panel opens a
popup listing every slider the active tab can sweep, each with a range and a number of steps, and
a running count of the options and of how long they will take. The server then works through the
grid on a background thread and writes the same folder described below. `phase4/export_space.py`
does that work; the rest of this section describes the command line runner, which takes a JSON
spec and produces the identical output.

`explore.py` turns the two models into a design space in the shape the multiobjective viewer
reads: `001.3dm`, `002.3dm` and so on, one `context.3dm` holding the surroundings so they are not
reloaded with every option, and `parameters.csv` and `objectives.csv` whose first column is the
option name and whose first row is the column names.

    python explore.py space.json --dry-run     # the grid, and roughly how long
    python explore.py space.json

The grid is full factorial: each parameter gets a number of steps, the option count is their
product times the variants per combination, and `--dry-run` prints both the count and an estimate
before anything is generated. Inference goes to the running phase 4 server, so the space is built
from whatever snapshot or checkpoint is loaded in the viewer and there is no second copy of the
model loading code. Everything after inference is local: the fit, the analysis and the export.

| engine | the whole tile | parameters | context model |
|---|---|---|---|
| `gan` | generated | the latent, the city, and the metric targets, which work here because the viewer solves for them and re-measures rather than conditioning | the tile boundary |
| `inpaint` | a real site loaded once from OpenStreetMap, a rectangle of it rebuilt per option | the seed and the guidance strength, which are what this model responds to, plus the hole targets if you want them recorded | the real geometry outside the rectangle |

Parameter names map onto the server's payloads: `target:<metric>` into the targets, `pca1..N` into
the PCA vector, `dir:<name>` into the fitted directions, anything else passed through (`seed_a`,
`psi`, `city_a`, `city_mix` for the GAN, `seed`, `cfg`, `steps` for the inpainter). A requested
target is written to parameters.csv as `target_far`; what the model actually produced is written to
objectives.csv as `far`, measured off the geometry, so the two can never be confused. That
distinction is the point of the whole exercise.

```json
{
  "engine": "gan",
  "server": "http://127.0.0.1:5000",
  "out": "C:/opengan/spaces/run01",
  "lut": "C:/opengan/dataset128_sdf/height_lut.json",
  "lat": 41.39, "lon": 2.17,
  "variants": 3,
  "fixed": {"city_a": 6, "city_b": 40, "psi": 1.0},
  "parameters": {
    "target:far":       {"min": 1.5, "max": 4.0, "steps": 4},
    "target:green_pct": {"min": 5,   "max": 25,  "steps": 3},
    "city_mix":         {"values": [0.0, 0.5, 1.0]}
  },
  "analysis": {"days": ["equinox", "winter"]}
}
```

That is 4 x 3 x 3 x 3 variants = 108 options.

**An infill is judged in its context.** Everything casts shadows and blocks views, including the
city around the plot, but the numbers are reported for the plot itself, through `analyse(...,
area=...)`. It matters more than it sounds: the same Barcelona plot reads 0.19 hours of winter sun
on the ground in its context and 1.34 hours with the surroundings deleted.

**But only the neighbourhood takes part.** `context_m` (300 m by default, the *context* slider in
the viewer's popup) keeps the geometry within that distance of the plot and drops the rest, from
the analysis and from `context.3dm`. A site can be 1.5 km across and measuring all of it is wasted
work: the analysis raster costs the square of the side, and on a dense Barcelona block 100 m of
surroundings already gives the same ground-sun hours to two decimals as the full 400 m tile.
`context_m: 0` measures the plot as if it stood alone, which is the 1.34 hours above.

A run is resumable. Options whose model already exists are skipped and their rows are read back
from `rows.jsonl`, so an overnight inpainter run survives being interrupted, and one option that
fails is recorded with its error rather than losing the run. `run.json` records the spec, the axes,
the timings and the code paths, so a space can be rebuilt.

Every option also gets `thumbs/001.png`, a small plan of the fitted geometry, which is outside the
viewer's contract and harmless to ignore. `--no-thumbs` turns it off.

## Against the real world (`corpus.py`)

`meta.csv` from phase 0 already holds the morphology of every real tile in the training set,
measured by the same code that measures a generated one. So an option does not have to be read in a
vacuum:

    python corpus.py meta.csv --background corpus.csv    # the cloud to plot behind a run
    python corpus.py meta.csv --space <run folder>       # annotate that run's objectives.csv

`--space` adds a percentile for every shared column and the nearest real tiles by morphology,
standardised so no column dominates. An option that reads "FAR 1.36, 81st percentile of real
fabric, nearest real tile London_593 at 0.35 standard deviations" is interpretable in a way a
latent coordinate never will be, and the percentile says how unusual a *combination* is, which the
raw numbers do not.

The match uses only what both sides measure: density, grain and height. The sun and view columns
have no counterpart in `meta.csv` yet, because computing them over 30,519 tiles is an overnight job
on the machine that holds them rather than something to slip into a lookup.

**This is for whole tiles, not for infills.** The corpus is 400 m tiles; an infill option is a plot
of one or two hectares, and a plot's FAR, coverage and mean height are not on the same scale as a
tile's — a 150 m courtyard block reads FAR 4 where the Eixample tile around it reads 2. So an
infill run is refused unless you pass `--force`, and what you get then is a rough bearing rather
than a like-for-like comparison. Comparing infills against each other, which is what the design
space is for, is unaffected.

**Point it at the dataset the model was trained on.** `images_all` and `images_v2` use the same tile
names for different ground, so a v1 corpus and a v2 tile that share a name are not the same place
at all: Barcelona_411 is 8.7 percent built in one and 70 percent in the other. The GAN trained on
v1, so `dataset128_sdf/meta.csv` is its corpus; the inpainter trains on the v2 rasters and wants a
v2 one. Getting this wrong is silent, since everything still runs.

Checked by taking real tiles through the fit and the analysis and asking the corpus to find them
again. Measurements agree with `meta.csv` to a few percent, five of eight test tiles have a tile
from their own city in the top five, and one finds itself. The rest are honest misses rather than
errors: a sparse tile looks like sparse tiles anywhere, and a very fine-grained dense Tokyo block
really does look like Manila. Morphology is not city-specific, which is the finding, not a fault.

That exercise also caught a real bug. The fit splits a building at its height steps and at narrow
necks, so the polygons are parts, not buildings, and `building_count` was counting the pieces:
Tokyo_254 read 207 buildings where the raster reads 108. It now counts the connected pieces of
their union, which is what the raster counts and what anyone reading the number expects, and
`mean_footprint_m2` follows. `part_count` keeps the old number.

## Using it

```python
import vectorize, analysis
res = vectorize.vectorize(channels, lut, tile_m=400)     # the fit
parts = vectorize.parts_m(res)                           # shapely geometry in metres
row = analysis.analyse(parts, lat=41.39)                 # the numbers
```

`parts_m(res, real, rect)` composes a real OpenStreetMap area with the piece the model rebuilt, so
the same call analyses a fill in its real context. `analyse(..., maps=True)` also returns the ground
sun rasters and the per-facade sun and view values, which is what `draw.py` plots and what would
paint the model by sun hours in the viewer.

Dependencies: numpy, scipy, shapely, pillow. No torch, no solar library.
