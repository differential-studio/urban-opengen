# Urban OpenGen

Open generative models of urban morphology, trained on sixty cities across every
continent. Open code, open data, open weights, and everything here runs on one consumer
GPU. 

The YouTube video below walks through the full pipeline in a clear, practical way: how we gather and prepare urban data, turn it into a consistent training dataset, train the models, and use the resulting generators to explore new urban scenarios:
https://www.youtube.com/watch?v=gjkFAR7kbEY

There are two models.

**The inpainter** takes a real place with an area cut out of it and fills that area with
streets, blocks and buildings that continue the fabric around them. Type an address, draw
a shape, optionally ask for a floor area ratio or a share of green, and it rebuilds the
site in 3D. It is a conditional diffusion model, 23.2 million parameters, trained from
scratch, and it carries no city label: the surroundings are the style, which is why it
works on cities it was never trained on.

**The generator** draws a whole 400 m × 400 m tile out of nothing, conditioned on which of
the sixty cities it should resemble, with sliders that move real measured quantities:
floor area ratio, coverage, green share, street share. You can blend Barcelona into Tokyo
and watch the fabric change continuously. It is StyleGAN2-ADA.

Neither one outputs a picture. Every tile is four channels, building footprint, building
height in metres, street and green, so anything the models make can be measured, traced
into polygons, exported as GeoJSON or a Rhino `.3dm`, and analysed for sun hours and
views. That is the point of the project: a design instrument rather than an image
generator.

---

## Try it

You need Python 3.10 or newer and, realistically, an NVIDIA GPU. Both models run on CPU
but slowly.

```
git clone https://github.com/differential-studio/urban-opengen
cd urban-opengen
pip install -r requirements.txt
```

If pip gives you a CPU-only PyTorch, install the CUDA build for your machine from
pytorch.org first, then re-run the line above.

### The inpainter, which is the quickest thing to see working

Download the inpainter checkpoint from `DOWNLOADS.md`, then:

```
python phase4\serve.py --lut assets\height_lut.json --cities assets\cities.json ^
    --inpaint <path to>\latest.pt --osm-contact you@example.com
```

Open http://127.0.0.1:5000, type an address, draw a shape over part of it, and generate.
The site loads live from OpenStreetMap, so put a real email in `--osm-contact`: their
usage policy asks for a way to reach whoever is making the requests.

Nothing else is needed for this. No NVIDIA repository, no dataset, no training.

### The generator

This one runs inside NVIDIA's stylegan3 code, which we cannot redistribute, so you clone
it yourself and patch it:

```
git clone https://github.com/NVlabs/stylegan3
python phase1\apply_patches.py stylegan3
```

Download the generator snapshot from `DOWNLOADS.md`, then:

```
python phase4\serve.py --repo stylegan3 --network <path to>\network-snapshot-003000.pkl ^
    --lut assets\height_lut.json --cities assets\cities.json --directions assets\directions.npz
```

Add `--inpaint <path to>\latest.pt` to the same command to have both models in one page.

The three files in `assets/` are not weights. They are the height lookup table, the city
label order and the fitted slider directions, 72 KB in total, the same for every
snapshot, which is why they are in the repository rather than in the downloads.

---

## What to expect

This is a research build, trained by two people on a single laptop GPU, and it is honest
about where it falls short.

| | |
|---|---|
| **Resolution** | A tile is 400 m across 128 pixels, about 3 m per pixel, so a typical building is three to five pixels wide. The fabric is coarse and buildings merge into their neighbours. This is the main limitation and more resolution is the main fix. |
| **Grain** | Measured against real tiles the models draw two thirds the buildings at one and a half times the footprint, and half the blocks. Area shares (density, coverage, street, height) come out within 0.15 standard deviations of the real corpus; the fine structure does not. Full numbers in section 7. |
| **Slider strength** | Coverage and street share are reliable. Mean height and green share are weaker. `assets/directions.csv` gives the R² of every one, and the viewer shows it as a dot beside each slider. |
| **Height data** | In cities where OpenStreetMap carries no height tags, most buildings sit at a 7 m default, so the model learns those cities as flat. Named and quantified in section 1. |
| **Not a daylight tool** | The analysis in `phase7/` computes direct sun and view exactly for the massing it is given, with no diffuse light, reflection, glazing or material. It is for comparing schemes, not for a report. |

Read the Lessons in section 3 before training anything. Three runs failed in instructive
ways and the current encoding is the result.

---

## Why this exists

Cities are the largest things people build, and we still design them one plot at a time.
The way a city is put together at the scale of a few blocks, its morphology, decides most
of what matters afterwards: how far people walk, how much land goes to roads, how much
daylight reaches the street, how much energy the buildings burn, whether there is a park
within reach. Those relationships are real and measurable, but they are spread over
thousands of cities and millions of blocks, and no planner, architect or student can hold
that knowledge in their head.

Generative models can. A model trained on how sixty very different cities actually arrange
their blocks learns a compressed picture of what has been tried, and a latent space you can
move through continuously: from Barcelona's chamfered grid to Tokyo's fine grain to the
loose fabric of a fast-growing African city, with everything in between that no city has
built yet. Coupled with hard morphological metrics, that becomes a design instrument rather
than a picture generator. "Show me a block with FAR 3, 20 percent green and a street network
that still connects" is a question a model can answer, and a question worth asking at the
start of a masterplan rather than the end.

We think this should be a shared instrument. The training data is derived from public
geography, the maths is published, and the value is in the breadth of cities, metrics and
eyes looking at the output. If you work on urban design, planning, computational design,
GIS, or machine learning and any of this is useful to you, the intent is that you take it,
run it, add your cities and your metrics, and send the improvements back. The scripts,
the encoding, the patches, the viewer and the findings from every failed run are here so
that nobody has to repeat them.

---

## What is in here

| you want to | go to |
|---|---|
| run the app on a trained model | `phase4/`, and `DOWNLOADS.md` for the weights |
| build the tile dataset, for our sixty cities or your own | `phase0/` |
| train the GAN, then find out whether it is any good | `phase1/`, then `phase3/` |
| train or sample the diffusion inpainter | `phase6/` |
| measure an option: sun, views, density, the design space | `phase7/` |

The folders are numbered by the order we built them, which is not the order you need them.
There is no `phase2/` and no `phase5/` here: phase 2 is the slider fitting, which lives in
`phase1/` because it runs against the training repo, and phase 5 is the film, which is a
communication piece rather than a tool and is not published. Every folder has its own
README with the commands for that step. The full layout:

```
urban-opengen/
├── images/                 not in the repository: where the raw tiles go, one folder per city
├── phase0/                 data: audit, encoding, metrics and the OpenStreetMap renderer
│   ├── tile_codec.py       the tile format: palette, height mapping, encodings, metrics
│   ├── reencode_tiles.py   builds tiles.npz from the raw tiles
│   ├── osm_tiles.py        renders tiles straight from OpenStreetMap, for any place
│   ├── render_cities.py    batch driver: a whole city list, resumable, with a report
│   ├── cities_60.json      the 60 cities of dataset v1 with coordinates
│   ├── cities_120.json     120 more, 20 per continent, each with the fabric type it was picked for
│   ├── check_cities.py     reverse-geocodes a city list before spending a night fetching it
│   ├── audit_tiles.py      raw-tile audit of the Grasshopper set (blanks, grey histogram, palette)
│   ├── audit_osm.py        per-city audit of rendered OSM tiles: coverage, missing buildings, missing heights
│   └── README.md
├── phase1/                 training: StyleGAN2-ADA, and the slider fitting (that is phase 2)
│   ├── apply_patches.py    patches a fresh NVlabs stylegan3 clone
│   ├── npz_dataset.py      dataset class installed by the patch
│   ├── fit_directions.py   Phase 2: sample, measure, regress w -> metric
│   ├── fix_build.py        finds compiler flags for the CUDA kernels (Windows)
│   ├── diagnose_build.py   verbose build log
│   └── README.md
├── phase3/                 evaluation: generated tiles against the real corpus
│   ├── evaluate.py         generated metric distributions vs the real corpus
│   └── README.md
├── phase6/                 the inpainter: the conditional diffusion model
│   ├── prepare.py          city rasters to a per-city field cache
│   ├── data.py             scale-aware crops, hole masks, hole metrics, on the GPU
│   ├── unet.py             conditional UNet, no diffusion library
│   ├── diffusion.py        schedule, loss, DDIM, RePaint, MultiDiffusion canvas
│   ├── train.py            training loop with resume, warm start, previews
│   ├── sample.py           inpaint a region of a real city
│   └── README.md
├── phase7/                 analysis: sun, views, density, and the design space
│   ├── analysis.py         direct sun, views from the walls, density and envelope, with a self-test
│   ├── explore.py          the grid, one model and one row per option, in the viewer's shape
│   ├── validate_fit.py     the same questions asked of the raw raster, to size what the fit changes
│   ├── draw.py             plan, ground sun, wall sun and sky view, side by side
│   └── README.md
├── phase4/                 the app: viewer and inference server
│   ├── serve.py            Flask inference server
│   ├── index.html          the page, vanilla JS + Three.js
│   ├── regular.js          the Clean and Regular geometry modes, in the browser
│   ├── vectorize.py        the Vector geometry mode and the GeoJSON / Rhino 3dm exports, on the server
│   ├── vendor/             three.module.js, OrbitControls.js, BufferGeometryUtils.js (r160, MIT)
│   └── README.md
├── assets/                 what the viewer needs besides the weights, 72 KB
│   ├── height_lut.json     grey level to height in metres, written with the dataset
│   ├── cities.json         the 60 city names and their class labels
│   ├── directions.npz      the fitted metric directions behind the sliders
│   ├── directions.csv      the same fit as a readable table, with R² per metric
│   └── README.md
├── requirements.txt        the Python dependencies, thin on purpose
├── DOWNLOADS.md            where the dataset and the two sets of weights live
└── README.md               this file
```

The numbers are the technical plan's and the names stayed because the scripts find each
other by folder name.

---

## 1. The data

### What a tile is

Each city is a 10 km × 10 km window cut into a 25 × 25 grid, so one tile covers
400 m × 400 m and there are 625 tiles per city. In the released dataset (v2) a tile is
128 × 128 px, 3.125 m per pixel; the original Grasshopper set (v1, below) was 100 × 100 px
at 4 m. Either way a tile is a flat-colour PNG in a fixed palette:

| colour | class |
|---|---|
| `(0, 0, 0)` | open ground, water, anything unbuilt |
| `(v, v, v)` grey | building; `v` encodes height: grey 100 is 4 m, grey 255 is 100 m, linear between |
| `(255, 0, 0)` | street, drawn as a filled centreline stroke on top of everything |
| `(128, 255, 0)` | greenery |

No anti-aliasing, so every pixel is exactly one of those classes and the masks are exact.

### Dataset v1: the Grasshopper tiles

The first tiles came from a Grasshopper definition (`Archive/240823 - Interpreter.gh`)
that takes building footprints with heights, a street network and green areas from
OpenStreetMap, centres a 10 km window on the city, slices it into the grid and renders each
cell into the palette above. Sixty cities, chosen to span continents, densities and street
grammars, at 100 x 100 px and 4 m per pixel. The GAN in this repository was trained on that
set, which is why it is released: without it those weights cannot be reproduced.

One exception, and it matters if you are checking our work. São Paulo came out of the
Grasshopper run almost empty, four usable tiles, and it was repaired **after** the GAN had
been trained, by dropping in the v2 render of that city. So the released v1 differs from
what the GAN actually saw, in exactly one city: the snapshots were trained with São Paulo
effectively absent. Everything else is the training set.

About a quarter of the raw tiles in either set are empty, because a 10 km window centred
on a coastal city contains a lot of sea. Those are dropped, see below.

### Mixing tile sizes

`reencode_tiles.py --size 128` resamples every tile to one size before anything is trained,
so a folder holding both 100 px and 128 px cities works without changes: nearest neighbour
for the `mask` encoding, bilinear for `sdf`.

It is not quite free, though, and the reason is in the Lessons section below: nearest
resizing from 100 to 128 stamps a fixed column-duplication pattern on every tile, which a
discriminator will happily use as a free feature. The `sdf` encoding, which is what the
released run uses, resamples bilinearly into distance fields and removes it.

What is left is subtler. A city already at 128 px is passed through untouched while the
other 59 are upsampled, so it arrives with genuinely sharper edges than its neighbours, and
in a city-conditional model that is a tell. If you are retraining and you care, downsample
the odd city to 100 px first so every city takes the identical path:

```python
from PIL import Image
from pathlib import Path
for p in Path("images/Sao_Paulo").glob("*.png"):
    Image.open(p).resize((100, 100), Image.NEAREST).save(p)
```

The released v1 has already had this done, so it is uniform 100 px throughout.

### Dataset v2: rendered from OpenStreetMap

`phase0/osm_tiles.py` does the same job in Python, from OpenStreetMap alone, for any
centre or address, and is the replacement for the Grasshopper step: reproducible, one
licence (ODbL), and the same code the "type an address, rebuild this site" tool runs at
inference time, so training data and inference input come from one path. Its
conventions are in the file and saved next to every output: every street class 8 m
wide, height from the `height` tag, else `building:levels` times 3 m, else 7 m; green
from the usual landuse, leisure and natural tags; draw order green, building, street.
Tiles are 128 px at 3.125 m per pixel, and `--raster` also writes the whole 10 km window
as one 3200 x 3200 image, which is what the inpainter trains on (random crops at any
scale instead of a fixed grid). `render_cities.py` runs a whole list unattended and
resumably, and `cities_60.json` and `cities_120.json` are the lists.

Findings from rendering the 60 (see also `audit_osm.py`):

* **Fetching is the bottleneck, and it is the server, not the query.** The public
  Overpass mirrors throttle by slot; the fetcher asks `/api/status` for a free one,
  rotates four mirrors and backs off, and splits a box only when the server says the
  query itself timed out. Splitting on every busy answer, the first version, multiplied
  the load and made the throttling worse. A city takes 4 minutes on a quiet day and over
  an hour on a loaded evening, so the 60 took two nights. Dense cities (Mumbai, Manila,
  Athens) get 504s from the two main mirrors and are quietly served by the third. Beyond
  about 200 cities the honest answer is Geofabrik extracts parsed locally.
* **Polygon counts do not measure coverage.** Beijing returned 21k buildings and Jakarta
  292k, and both are fine: Beijing's are 2000 m² slabs, Jakarta's are 600 m² houses.
  Judge a city by rendered coverage and buildings-per-tile, not by the feature count.
* **Heights are the weak channel.** In the HOT-mapped cities most footprints carry no
  height and no levels tag, so they sit at the 7 m default: Jakarta 84 percent, Dar es
  Salaam 43 percent, Beijing 52 percent, against Berlin 19 percent. A model trained
  naively learns that Jakarta is flat. Options are to drop those cities, to mask the
  height loss for untagged pixels, or to accept it and say so; not decided yet.
* **Some cities are simply not mapped.** Luanda returned 1,787 buildings and 8,758
  street segments: the network is there, the buildings are not. Re-fetching cannot
  invent them. Luanda is in the release as rendered, and is the first city to replace.
* **The Grasshopper set was coarser than it looked.** The same tiles from OSM carry 77
  to 118 distinct height levels against 12 to 24 in the old renders.

The 120 additional cities were chosen for fabric type rather than size, 20 per
continent: medinas, Japanese and Korean fine grain, Soviet microrayon, colonial grids,
Gulf superblocks, North American sprawl, Latin American hill cities, and Brasília,
Savannah, Ahmedabad and Hanoi for the block types nothing else has. Oceania is the weak
continent, its tail is small regional towns and Pacific capitals, and the audit decides
which of those stay. These are for the inpainter; the GAN stays on the 60 it was trained
on, because 180 one-hot classes would make its conditioning harder, not better.

---

## 2. Dataset preparation (Phase 0)

`phase0/reencode_tiles.py` turns the PNG folders into one training file. It makes two
passes over the images.

**Pass 1** reads every tile, drops the ones with less than 5 percent non-background
content (`--min-content-pct`), and accumulates a histogram of building grey values over
the whole corpus. From that histogram it builds the height look-up table.

**Pass 2** encodes every kept tile to four channels, resizes to the training resolution,
computes the morphology metrics for the record, and writes everything out.

```
python reencode_tiles.py --images C:\opengan\images_all --out C:\opengan\dataset128_sdf --size 128 --encoding sdf
```

Outputs:

| file | what |
|---|---|
| `tiles.npz` | `x` uint8 `[N, 4, S, S]`, `city` int16 `[N]`, `idx` int16 `[N]` |
| `height_lut.json` | forward and inverse height mapping; keep it with every trained model |
| `cities.json` | city name to class label |
| `encoding.json` | which encoding and parameters were used |
| `meta.csv` | one row per kept tile with grid position and all metrics |
| `preview.png` | original, the four channels, and the round-trip decode, side by side |
| `stats.txt` | kept / dropped / unreadable counts, grey range, and a lossless round-trip check |

On the 60-city set: 36,840 tiles found, 30,519 kept, 6,321 dropped, 151 distinct grey
levels in the range 105..255.

`height_lut.json` and `cities.json` have to travel with the model, not with the dataset:
without them a snapshot decodes to nothing and its city labels point at the wrong cities.
The pair for the released snapshots is in `assets/`.

### Why four channels and not RGB

The raw palette is fine for a human and hostile to a network. Building height lives in
a narrow band of greys (99 percent of building pixels sit between grey 105 and 125), so
after normalising to `[-1, 1]` the difference between a five-storey and an eight-storey
building is about 0.03, below anything a GAN will learn. Feeding RGB also asks the model
to infer two things, "is there a building" and "how tall", from one badly scaled number.

So each tile becomes:

| channel | content |
|---|---|
| 0 | building footprint |
| 1 | building height, rank-normalised across the corpus |
| 2 | street |
| 3 | green |

The height channel is histogram-equalised over the whole corpus with a guaranteed
minimum spacing of 4 bytes between neighbouring grey levels, so the common 105..125
band spreads over most of the byte range while rare tall buildings keep distinct
values. The mapping is exactly invertible; `stats.txt` confirms the round trip.
Metrics convert height back to metres with the renderer's linear mapping and count
floors at 3 m for the FAR proxy.

### Why signed distance fields and not masks

The first version stored channels 0, 2 and 3 as hard 0/255 masks. Three training runs
with different model sizes and dataset sizes all failed the same way: the ADA
augmentation strength climbed at its maximum rate from the first tick and went past 1.0,
which means the discriminator could tell real from fake by something no augmentation can
hide. It could: a generator never outputs exactly 0 or 255, and nearest-neighbour
resizing from 100 to 128 px stamps a fixed column-duplication pattern on every real tile.
Both are free features for the discriminator, and in the mask dataset 90 percent of all
bytes were exactly 0 or 255.

The `sdf` encoding removes that. Footprint, street and green are stored as truncated
signed distance fields: byte 128 on the class boundary, brighter inside, darker outside,
saturating 8 native pixels (32 m) away. The height channel keeps exact heights inside
footprints and, outside them, the height of the nearest building blurred by 1.5 px, so
the field has no cliffs; tiles with no buildings at all get a flat field at the corpus
mean height. Fields are computed at native resolution and resampled bilinearly, which is
what a distance field wants, so the aliasing fingerprint is gone too.

Decoding is the same rule for both encodings: a class is present where the byte is
`>= 128`, and height is read only where the footprint is. Every consumer (training
previews, metrics, the direction fitter, the viewer) is therefore encoding-agnostic.
Thresholded SDF output also has sub-pixel edges, which the viewer's clean-geometry mode
benefits from. Only 2 percent of bytes in the SDF dataset are 0 or 255.

`tile_codec.py` holds all of this: the palette, the height mapping, `HeightLUT`,
`encode_tile`, `channels_to_rgb`, and `tile_metrics`. It has no torch dependency so the
same file serves the audit, the encoder, training previews, the direction fitter and the
viewer.

### Metrics

`tile_metrics()` computes, on any 4-channel tile, real or generated:

| group | metrics |
|---|---|
| area shares | coverage, street, green and open percent, and the same as m² |
| buildings | count, mean footprint, mean and max height in metres, gross floor area, FAR proxy |
| streets | connected components, centreline length in metres (Zhang-Suen skeleton), intersections, dead ends, density in km/km² |
| blocks | count and mean block area, from the regions the street network encloses |

These are the columns in `meta.csv`, the sliders `fit_directions.py` can fit, and the
readout in the viewer.

---

## 3. Training (Phase 1)

### Model

StyleGAN2-ADA, from the NVlabs `stylegan3` repository (`--cfg=stylegan2`), trained
city-conditional over the 60 class labels. The reasons, in order of importance:

1. ADA, adaptive discriminator augmentation, exists for datasets of this size and is the
   single largest difference from the DCGAN attempts that preceded this project.
2. The `W` latent space is what makes a slider UI possible: it is smooth, disentangled
   enough for linear directions to mean something, and a single forward pass, so sliders
   update in real time. A diffusion model would need tens of steps per frame.
3. City conditioning is nearly free and immediately gives a meaningful control: the
   viewer interpolates the class embedding between two cities.

Fine-tuning Stable Diffusion or training a LoRA was considered and rejected: its prior
is photographs, its VAE smears the height values so the output stops being measurable
data, and a text prompt is not a slider.

### The patch

`phase1/apply_patches.py` modifies a fresh `stylegan3` clone, idempotently:

* a dataset class that reads `tiles.npz` and applies 8-way dihedral augmentation
  (`--dihedral=1`), legitimate for city tiles in a way it is not for faces;
* `--augpipe` choices `blit`, `geom`, `bg`, `bgc`; only geometry pipelines make sense
  here, since colour augmentation would corrupt the height channel;
* 4-channel preview grids (`fakes*.png`, palette-rendered) plus raw channel grids
  (`fakes*_ch.png`, unthresholded, the honest view);
* compatibility fixes for current torch and Python, and a graceful fallback when the
  custom CUDA kernels fail to build.

Setting it up:

```
git clone https://github.com/NVlabs/stylegan3
python phase1/apply_patches.py stylegan3
pip install click requests tqdm psutil imageio packaging ninja
```

### The run

From inside `stylegan3`:

```
python train.py --outdir=runs_sdf --cfg=stylegan2 --data=..\dataset128_sdf\tiles.npz ^
    --gpus=1 --batch=32 --batch-gpu=16 --gamma=4 --cbase=16384 --cmax=256 ^
    --cond=1 --dihedral=1 --aug=ada --augpipe=bg --metrics=none --kimg=3000 --snap=10 --workers=4
```

| flag | why |
|---|---|
| `--cbase=16384 --cmax=256` | a small generator and discriminator; the full-size model collapsed on this data |
| `--batch=32 --batch-gpu=16` | fits an 8 GB card at 128 px |
| `--gamma=4` | R1 regularisation; the most important knob on a dataset this size |
| `--aug=ada --augpipe=bg` | adaptive augmentation, geometry only |
| `--metrics=none` | FID needs an RGB Inception network and is meaningless on semantic maps |
| `--snap=10` | a snapshot and preview every 40 kimg |

About 29 s per kimg on an RTX 4070 Laptop, so roughly 24 hours to 3000 kimg. Resume
with `--resume=runs_sdf/<id>/network-snapshot-XXXXXX.pkl`.

The Windows toolchain notes (Visual Studio, CUDA 13, the `/permissive-` flag, and the
`bias_act_plugin` import bug) are in `phase1/README.md`. The kernels are a speedup, not
a requirement.

### Reading the log

The number to watch is `augment` in the tick lines. ADA raises it whenever the
discriminator is over-confident on real images and lowers it otherwise, by at most about
0.008 per tick. A healthy run on this data rises to somewhere between 0.2 and 0.6 within a
few hundred kimg and then flattens. A run whose `augment` climbs at the maximum rate from
tick 1 and crosses 1.0 is not going to recover: the discriminator has found a shortcut,
and the fix is in the data, not the hyperparameters.

Look at `fakes*_ch.png`, not only `fakes*.png`. Early in training the generator sits near
the dataset mean, which can be under the 128 threshold, so the palette preview can be
black while the raw channels show a model that is learning normally. Each row of a
preview grid is one city; within a row the tiles should differ, since each column is a
different latent. Identical tiles along a row mean the generator is ignoring the latent.

### Lessons

The failures, so nobody repeats them:

* **DCGAN on raw RGB at 100 px** (the original attempt): no augmentation, entangled `z`
  with 100 uninterpretable sliders, height unlearnable for the dynamic-range reason above,
  blank tiles teaching the model that black is a valid answer, and a non-power-of-two
  resolution forcing awkward transposed convolutions. Five independent problems.
* **Conditioning on continuous metrics during training** made things worse, which is
  expected: a projection discriminator on continuous attributes degrades the base
  generator before it gives any control. Directions are found afterwards instead (Phase 2).
* **Full-size StyleGAN2 on 34 cities of hard masks**: discriminator overfit, `augment`
  above 1.0, identical tiles per row.
* **Small StyleGAN2 on 60 cities of hard masks at 128 px**: slower, same curve, same
  ending at 1256 kimg. Diagnosis and fix in the SDF section above.
* **The SDF run (the one that worked) stopped improving at 1600 kimg.** Mean distance
  to the real corpus went 0.232 at 840, 0.207 at 1600, 0.210 at 2640, 0.207 at 3000.
  The augment curve stayed healthy (0.3) throughout, so this is not overfitting; it is
  the model's ceiling. The full numbers are in the evaluation section. The practical
  lesson: evaluate every 800 kimg with `evaluate.py` and stop when the mean distance
  is flat for two snapshots, rather than training to a round number.
* **The residual gap is grain, at every snapshot.** Real tiles have 29 buildings at
  547 m² and 24 blocks; the generator draws 20 buildings at 800 to 900 m² and 12 or 13
  blocks, at 840 kimg and at 3000 alike. It merges neighbours and does not close
  street loops. More kimg does not fix this; more resolution or a different objective
  might, which is one reason the inpainter is a diffusion model.

---

## 4. Sliders (Phase 2)

Training is unconditional on metrics on purpose. The directions are found afterwards:

1. Sample many latents `w` from the trained generator (20,000 by default).
2. Generate each tile and measure it with `tile_metrics()`.
3. Ridge-regress `w -> metric` for every metric. The coefficient vector is that metric's
   direction in `W`; its R² says how linearly controllable the metric is.
4. Gram-Schmidt the directions in priority order so "more green" does not also drag
   density along.
5. Verify by walking a fixed tile along each direction and re-measuring: the metric has to
   move monotonically, or the direction is not trusted.

```
python phase1/fit_directions.py --repo stylegan3 --network runs_sdf/<id>/network-snapshot-003000.pkl ^
    --lut ..\dataset128_sdf\height_lut.json --out directions --n 20000
```

It writes `directions.npz` with the orthogonalised directions, the raw Jacobian of the
fit, and the metric statistics, plus `directions.csv` and one `walk_<metric>.png` per
metric. The fit for the released snapshot is already in `assets/`, so you only need to
run this if you train your own. The Jacobian is what lets the viewer solve for absolute targets ("FAR 2.5")
rather than only nudge relatively. A new metric costs a function in `tile_codec.py` and
a re-run of this script, ten minutes, no retraining.

---

## 5. Viewer (Phase 4)

`phase4/serve.py` runs the generator behind a small Flask server; `index.html` is the
page. Any `network-snapshot-*.pkl` can be loaded from the page, so it doubles as a way to
compare snapshots while training runs.

```
python phase4\serve.py --repo <your stylegan3 clone> --network <network-snapshot-003000.pkl> ^
    --lut assets\height_lut.json --cities assets\cities.json --directions assets\directions.npz
```

then open http://127.0.0.1:5000. Leave `--network` off and pick a snapshot in the page
instead.

The three files under `assets/` are in this repository on purpose. They are not weights:
`height_lut.json` turns the grey of a footprint pixel back into metres, `cities.json`
fixes the order of the sixty labels the city conditioning was trained against, and
`directions.npz` holds the latent directions fitted after training, which is what makes
the metric sliders move a real quantity. They are the same for every released snapshot
and they total 72 KB, so the download is only ever the weights. `assets/README.md` says
which run each one came from and what to regenerate if you train your own.

What it shows: the generated tile as a ground plane in the original palette, every
footprint pixel extruded to its decoded height in metres, the four raw channels in the
corner, and the live metrics of the tile on screen.

Controls, all combined into one `w` on the server:

| control | effect |
|---|---|
| seed A, seed B, mix | interpolation between two latents in `W` |
| truncation | `truncation_psi` |
| city A, city B, blend | interpolation of the class vector, so one city's fabric morphs into another's |
| PC 1..N | GANSpace-style principal components of `W`, computed at load time; useful before Phase 2 has run |
| metric sliders | the fitted directions, each with a dot showing the R² of its fit |
| targets | absolute values (FAR 2.5, green 15 %) solved with damped Newton steps on the Jacobian, re-measuring the real tile after each step; the readout turns amber when the model cannot get there, which is reported rather than faked |

**Geometry** (bottom right of the 3D view) switches between *Voxels*, one box per
footprint pixel, and *Clean*, which traces each building as a polygon with courtyard
holes, simplifies the outline with Douglas-Peucker, extrudes it to the median height of
its pixels, merges streets and greens into plates, and scatters sphere trees over green
cells. *Regular* adds the mesh regulariser (`phase4/regular.js`): it measures the
direction the city is built along, resamples the distance fields in a frame rotated to
it, and shapes every building, block and park in that frame, as a cover of aligned
rectangles or as an orthogonalised outline, with the height map split into steps so a
block becomes a stepped massing model rather than one flat extrusion. Streets are the
gaps between regularised blocks. *Vector* is the mode meant for export and runs on the
server (`phase4/vectorize.py`): buildings are fitted rather than traced. Each one is
measured in a frame rotated to its own axis, every boundary sample of the distance field
votes for the wall line it lies on, the few lines with enough votes cut the frame into
cells, and the union of the cells the field says are inside is the building, so a
slightly crooked 30 m wall becomes one straight 30 m wall instead of a staircase. Corners
the field says are cut are chamfered, height plateaus become parts that share the same
walls, and blobs joined by a neck are split. Streets are the skeleton of the street field
as a cleaned graph, straightened, snapped to the grid where they are close to it and
re-intersected at the junctions, each with its width read off the mask; plates, blocks
and greens follow from that. The result is rasterised back and compared with the tile it
came from (footprint IoU 0.85 to 0.95 on real tiles, about 0.8 on generated ones), and
*Download GeoJSON* writes buildings with heights, streets with widths, greens and blocks in
local metres, and *Download 3DM* writes the same as a Rhino file (via `rhino3dm`): building
solids with courtyards, outlines, centrelines with widths, plates, blocks and greens on
layers, in metres, geo-anchored when the tile came from an address. With `--images` the
same view shows real OSM tiles for comparison. Full details in `phase4/README.md`.

**Latent fill** (panel, right) paints a mask over part of the tile and rebuilds it using
the GAN alone, with no second model: `w` is optimised in `W+` to match everything outside the mask, using a
multi-scale loss weighted by a halo around the hole, and the generated patch is
composited back over a few pixels of feather. Because the channels are distance fields
the blend is smooth and the outlines stay continuous. The city selector sets the style
the fill reaches for, so the same hole can be rebuilt as Barcelona or as Tokyo, and fills
chain. Fill quality tracks how well the model can reproduce the context: a hole in a
generated tile comes back almost perfectly, a moderate real tile gets streets that
continue through the hole, and the densest real tiles are still beyond the current
snapshot, which makes this a useful read on training progress. Details in
`phase4/README.md`.

**Diffusion fill** (panel, right) is the other one, and the one you probably want. Load an
inpainter checkpoint, type an address to pull the real fabric from OpenStreetMap, then
choose the area to rebuild on the ground in the 3D view: drag a rectangle, or switch to
Shape and click the corners of as many outlines as you like. Set any subset of the six
targets, press Fill, and the model rebuilds the selection in the context it sits in. By
default the board cuts real geometry against the bounding box of what you selected; turn
on "keep the drawn shape" and the seam follows the outlines instead, so the city between
two drawn plots stays as OpenStreetMap drew it rather than being round-tripped through the
regulariser. *Export design space* then sweeps a grid over the targets and writes one
Rhino model and one row of parameters and objectives per option.

This is a testing tool. The deployable version is a static page with the generator
exported to ONNX and run in the browser with ONNX Runtime Web; the LUT inverse is all the
page needs to decode.

---

## 6. The inpainter (Phase 6)

The GAN draws a tile out of nothing. The inpainter does the thing an architect actually
asks for: here is a real piece of city with an area taken out of it, rebuild what is
inside so that it continues what is around it.

It is a pixel-space conditional diffusion model, 23.2 million parameters, written against
PyTorch with no diffusion library in between. Input is nine channels: the noisy fields,
the same four context fields with the hole zeroed, and the mask. Cosine schedule,
v-prediction, min-SNR-5 weighting, and a loss weighted 1 inside the hole and 0.1 outside,
so most of the gradient goes where the work is. Sampling is DDIM with RePaint, which
re-injects the known surroundings at every step and is why the seams close.

### Scale

Training crops whole-city rasters at a random physical scale, 0.75 to 2.0 times the 400 m
tile, so 300 m to 800 m per window, and feeds the scale to the network as part of the
condition. One set of weights therefore produces individual plots at the small end and
district structure at the large end. A canvas bigger than one window is filled by
MultiDiffusion: overlapping windows denoised together and averaged where they overlap, so
there is no tiling seam. `--extent 1600 --tile-m 800` is a 1.6 km canvas built from 800 m
windows.

### The condition

Six numbers measured from the hole before it was emptied: coverage, green share, street
share, mean height, floor area ratio and block size. Each arrives with a given flag, so
any subset can be withheld, and the all-withheld case is the unconditional model that
classifier-free guidance needs.

One number is worth more than the loss when reading a training log. Reconstruction error
is dominated by the fabric around the hole, so a model ignoring the condition entirely
still scores well. `condition delivered` is the honest one: it moves a target and measures
what comes out. A value near 1 means the request arrived. Watch it rather than the loss.

Guidance is a multiplier on the distance from the unconditional prediction, so it does not
just sharpen the response, it extrapolates it. At `--cfg 2.0` a delivered value of about 2
is arithmetic, not a bug, and pushing it further leaves the region the model was trained
on and the fabric smears. There is also a metric guidance path that differentiates the
measurement itself at each step, which works whether or not the network learned to use the
condition.

### Running it

`prepare.py` turns the city rasters into a per-city field cache, `train.py` trains with
resume, warm start and previews, `sample.py` inpaints a region of a real city and writes
the four channels in the same layout as the training tiles, so the phase 4 configurator
displays them directly. Commands are in `phase6/README.md`.

---

## 7. Evaluation (Phase 3)

FID is ignored; it is near meaningless on semantic maps. `phase3/evaluate.py` samples
tiles from one or more snapshots (16 per city label, truncation 1.0 so the distribution
is not flattered), measures them with the same `tile_metrics()` that produced
`meta.csv`, and reports for every metric the 1-Wasserstein distance between generated
and real, divided by the real standard deviation. That reads as "how many SDs apart the
two distributions are": below 0.1 is a match, above 0.5 is a different kind of city.
It also reports per-city conditioning error, the mean absolute difference between each
label's median coverage and its real city's, and names the best snapshot.

```
python phase3\evaluate.py --repo stylegan3 --lut ..\dataset128_sdf\height_lut.json --meta ..\dataset128_sdf\meta.csv ^
    --snapshot <run>\network-snapshot-001600.pkl <run>\network-snapshot-003000.pkl --per-city 16 --device cpu --out eval
```

### The SDF run, 840 to 3000 kimg

| metric | real | 840 | 1600 | 2640 | 3000 | distance at 3000 |
|---|---|---|---|---|---|---|
| FAR proxy | 0.62 | 0.60 | 0.62 | 0.59 | 0.62 | 0.03 |
| coverage % | 14.9 | 14.7 | 16.0 | 14.9 | 15.5 | 0.05 |
| green % (median) | 0.50 | 0.00 | 0.00 | 0.00 | 0.42 | 0.14 |
| street % | 15.2 | 16.3 | 16.1 | 15.5 | 16.1 | 0.11 |
| mean height m | 12.2 | 11.8 | 12.0 | 11.9 | 12.1 | 0.13 |
| building count | 29 | 22 | 23 | 20 | 20 | 0.38 |
| mean footprint m² | 547 | 806 | 796 | 781 | 897 | 0.16 |
| street length m | 4373 | 3665 | 3874 | 3521 | 3804 | 0.34 |
| intersections | 45 | 46 | 48 | 39 | 48 | 0.08 |
| block count | 24 | 11 | 12 | 12 | 13 | 0.65 |
| **mean distance** | | 0.232 | 0.207 | 0.210 | 0.207 | |
| per-city coverage error, pp | | 3.89 | 3.35 | 3.65 | 3.75 | |

What this says. The area shares are learned: FAR, coverage, street share and height are
all within 0.15 SD, and city conditioning works to within 4 percentage points of
coverage. Snapshot 3000 is the one to use for the viewer, the film and the fitted
directions, for one reason: it is the first snapshot where the median tile has any
green at all. At every earlier snapshot more than half of generated tiles are barren,
which is visible on screen in a way that 0.4 pp of conditioning error is not.

What is not learned is grain, and no snapshot moves it: two thirds the buildings at
one and a half times the footprint, half the blocks, 13 percent less street. The
generator merges adjacent buildings and draws streets that do not enclose. This is the
ceiling of a small StyleGAN2 at 3.1 m per pixel, and it is the yardstick for the
inpainter: `building_count`, `mean_footprint_m2` and `block_count` have to beat 0.38,
0.16 and 0.65 for the diffusion model to be a step forward rather than sideways.

Still to add: the slider test (moving a fitted direction must move the measured metric
monotonically, `fit_directions.py` writes the walk images but the number is not yet
reported) and a seam and target-error evaluation for the inpainter on held-out tiles.

---

## 8. Analysis (Phase 7)

Density is an input here: the GAN's target solver hits a requested FAR and the inpainter is
conditioned on one. So the numbers worth plotting an option against are the ones the models do not
control. `phase7/analysis.py` computes them from the fitted geometry, which is the same geometry the
viewer draws and the 3dm export writes:

* direct sun hours on open ground, on the street surface alone, and on every wall, on the equinox
  and on the winter solstice of the hemisphere the tile is in, from true solar geometry at the
  tile's own latitude
* from every wall sample, the share of its outward hemisphere taken by sky, by other buildings, by
  greenery and by street, the share taken by buildings within 25 m, which reads as overlooking, and
  how far a view reaches before it meets something
* floor area ratio, coverage, open space ratio, heights, facade area and facade-to-floor ratio,
  straight off the polygons

It is meant for comparing rough schemes, not for a daylight report: no diffuse light, no reflection,
no glazing, no material. What it computes is exact for the massing it is given, which is what makes
two options comparable. Nothing outside the tile casts a shadow, so points near the edge are
optimistic.

It is fast because one field per sun position answers the question for every height at once. For a
given sun, `horizon_field` returns the lowest height from which that sun is visible in each cell, so
the ground reads it at zero and every wall sample reads it at its own height. The field is a running
maximum along the ray toward the sun, built by doubling, so 128 marching steps cost 8 array shifts.
A 400 m tile with two days of sun and 4000 wall samples takes 0.4 to 1.0 seconds.

    python phase7/analysis.py --selftest

checks only things with an answer known in advance: day length at six latitudes against
`acos(-tan(lat) tan(decl))`, the horizon field against a brute-force ray march on towers that shade
each other, an isolated block whose south wall must see the whole winter day and whose north wall
none, a noon shadow that must reach `h / tan(altitude)`, an unobstructed wall that must read exactly
50 percent sky, and four towers that must get worse on every reading when moved closer. It found
three real bugs that looking at pictures would not have.

`validate_fit.py` sizes the other risk, that the answers are an artefact of the vectoriser. It asks
the same questions of the raw raster the model drew. Over thirteen real tiles from nine cities the
ground sun moves by 0.15 h on average and 0.40 h at worst, and built area by half a percentage
point, so these are properties of the scheme.

What it says about real fabric, on the winter solstice at each city's own latitude:

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
and gets continuous frontage, a street wall and eight-metre views. Neither is the right answer, and
having both measured is the point.

### The design space

`phase7/explore.py` turns either model into a set of options in the shape a multiobjective viewer
reads: `001.3dm`, `002.3dm` and so on, one `context.3dm` for the surroundings, and two CSVs whose
first column is the option name. The grid is full factorial, `--dry-run` prints the count and an
estimate before anything runs, and inference goes to the phase 4 server so the space is built from
whatever snapshot or checkpoint the viewer has loaded.

The parameters are the things each model actually responds to. For the GAN that is the latent, the
city and the metric targets, which work there because the viewer solves for them and re-measures
rather than conditioning on them. For the inpainter it is the seed and the guidance strength: its
metric conditioning is weak next to the pull of the surrounding fabric, so a grid over its targets
would be a grid over a wish. A requested target is recorded as `target_far` in parameters.csv and
what the model produced as `far` in objectives.csv, measured off the geometry, so the two are never
confused.

An infill is judged in its context: everything casts shadows and blocks views, including the city
around the plot, while the numbers are reported for the plot itself. The same Barcelona plot reads
0.19 hours of winter sun in its context and 1.34 with its surroundings deleted, so this is not a
detail.

## Requirements

`requirements.txt` covers everything in one line of pip. What it is for, in case you want
to install by hand:

* Python 3.10 or newer, tested to 3.14
* `numpy`, `pillow`, `scipy` for the tile codec; `requests` and `shapely` for the
  OpenStreetMap rasteriser; `pandas` for the evaluation
* PyTorch with CUDA for either model, plus `click tqdm psutil imageio packaging ninja`
  for training the GAN inside stylegan3
* `flask` for the viewer, `rhino3dm` only if you want the `.3dm` export
* A GPU with 8 GB or more to train at 128 px. Running the trained models needs much less,
  and the whole pipeline also runs on CPU at 32 px for testing.

---

## Roadmap and how to help

In rough order of leverage:

1. **Resolution.** At 3.1 m per pixel a typical building is 3 to 5 pixels wide, and the
   grain gap in the evaluation is the price. `osm_tiles.py --px 256` renders 1.56 m
   tiles today; what it costs is a 4x larger model and training time.
2. **More cities.** `cities_120.json` is queued; the rasteriser takes any address. Cities
   from under-represented fabric types are worth more than big ones.
3. **More metrics.** Anything computable from four masks: block compactness, street
   orientation entropy, courtyard share, frontage continuity. One function in
   `tile_codec.py`, then re-run `fit_directions.py`.
4. **The height problem.** Decide what to do about cities where most footprints have
   no height tag (mask the loss, drop them, or say so in the output).
5. **The ONNX export and static demo** (Phase 4, serverless).
6. **A finer inpainter.** The model is trained and released, see section 6, but it works
   at six metres a pixel and blurs anything smaller than about a storey. A rowhouse grid
   is where this shows. Higher resolution, more cities and longer runs are the obvious
   next step and the code to do all three is here.

Open a discussion or a pull request. Failed experiments with their logs are as welcome as
successful ones; the augment curve of a bad run is worth more than a screenshot of a good
one.

---

## Licence and attribution

Open source, with credit required everywhere. The parts are different kinds of work and
carry the licence that fits each; `LICENSE.md` is the map, the licence files govern, and
`RELEASE.md` says what is in the public repository (the pipeline and the configurator),
what comes as a download (the weights and both datasets) and what is not published (the
film and the archive).

| part | licence |
|---|---|
| code, all phases | Apache License 2.0 (`LICENSE`, keep `NOTICE` with any copy) |
| diffusion inpainter weights | CC BY 4.0 (`LICENSE-MODELS.md`) |
| GAN generator weights | non-commercial research use only, because they were trained with NVIDIA's StyleGAN2-ADA code under the NVIDIA Source Code License (`LICENSE-MODELS.md`) |
| dataset v2, rendered from OpenStreetMap | ODbL 1.0, © OpenStreetMap contributors (`LICENSE-DATA.md`) |
| dataset v1, the Grasshopper tiles | ODbL 1.0, © OpenStreetMap contributors (`LICENSE-DATA.md`) |

Credit line, for anything built on or shown from this work:

    Urban OpenGen by Differential (differential.studio), github.com/differential-studio/urban-opengen

`CITATION.cff` gives the same in machine-readable form. NVIDIA's `stylegan3` is not
redistributed: `apply_patches.py` modifies a clone you make yourself under NVIDIA's own
licence, which permits non-commercial research use only. Three.js (vendored) is MIT.

Urban OpenGen is developed by Differential.
