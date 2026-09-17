# Phase 6: the conditional diffusion inpainter

Fills a region of a city with fabric that continues its surroundings, at a chosen scale,
steered by any subset of six metric targets. This is the model behind "load an address,
draw a site, set FAR and green share, weave it in". It is trained from scratch on the
city rasters from `phase0/osm_tiles.py`; the StyleGAN stays as it is for the sliders.

    prepare.py     rasters -> per-city field cache (signed distances, height), once
    data.py        scale-aware crops, hole masks, hole metrics, all on the GPU
    unet.py        23M-parameter conditional UNet, no external diffusion library
    diffusion.py   cosine schedule, v-prediction, DDIM, RePaint, MultiDiffusion canvas
    train.py       the loop: AMP, EMA, resume, preview grids, held-out validation loss
                   and the conditioning probe
    sample.py      inpaint a region of a real city from a checkpoint

## Run order

Everything below is run from the repository root, with a Python that has torch and scipy
installed. On Windows that usually means calling the virtual environment's interpreter by
its full path rather than `python`, because the system Python has neither.

1. Build the cache from the rasters. Rendered cities only; needs `<City>_raster.png`,
   which `render_cities.py --raster` writes. About ten seconds per city, once.

       python phase6\prepare.py --images C:\opengan\images_v2 --out C:\opengan\cache_v2

   Cities the audit says to leave out: `--drop Luanda,Papeete`. The height LUT is rebuilt
   from the whole set and saved next to the cache; a checkpoint carries its own copy.

2. Smoke test, ten minutes on the GPU. Proves the loop on your data and gives you a first
   `preview_*.png` to look at (context with the hole in grey, the fill, the truth).

       python phase6\train.py --cache C:\opengan\cache_v2 --out C:\opengan\inpaint\smoke --tiny --steps 3000 --preview-every 500

3. The real run. Batch 32, 100k steps, previews every 2000 steps, checkpoints every 10k.
   Expect 4 to 6 steps per second on a 4070-class GPU, so 5 to 7 hours. Stop it with
   Ctrl-C any time and continue with `--resume`.

       python phase6\train.py --cache C:\opengan\cache_v2 --out C:\opengan\inpaint\run01
       python phase6\train.py --cache C:\opengan\cache_v2 --out C:\opengan\inpaint\run01 --resume

   Three numbers land in `log.txt` and in `metrics.jsonl` next to it, one record per
   preview, so a finished run can be plotted rather than read.

   **target error**, per metric and per scale: the fill is asked for the true metrics of
   the hole that was just cut, and this is how far it landed from them. Read it as "is the
   fill realistic and does it sit where the neighbourhood implies". It is deliberately not
   a test of whether a request is obeyed: the request here always agrees with the fabric
   around the hole, so a model ignoring the condition entirely would still score well. The
   crops and the holes are frozen, so consecutive previews are comparable.

   **condition delivered**: the one that says whether the targets are controls or
   decoration. The same holes are filled twice per metric, once asking 1.5 standard
   deviations below the corpus mean and once above, everything else withheld, from the
   same noise. The number is how far the two fills actually moved apart as a fraction of
   how far they were asked to. 1.0 is a request delivered in full, 0 is a network that
   ignored it and produced what the context implied both times. This is what predicts
   whether the sliders in the phase 4 viewer will do anything, and it is the number to
   watch if they do not.

   **val loss**: the same frozen crops, holes, timesteps and noise every time, drawn from
   `--holdout` cities the model never trains on (four by default, spread through the city
   list). Timesteps are frozen along with everything else because diffusion loss swings
   far more with which timestep was drawn than with the state of the model, so a
   validation number that redraws them is mostly noise. Training loss going flat says very
   little on its own; this one going flat while the previews still improve is normal, and
   this one turning up while training loss keeps falling is the only real evidence of
   overfitting you will get.

   The held-out city names travel inside the checkpoint, so `--resume` validates on the
   same set and warns if the split has changed under it.

4. Inpaint something.

       python phase6\sample.py --ckpt C:\opengan\inpaint\run01\latest.pt --cache C:\opengan\cache_v2 --city Berlin --extent 800 --far 2.5 --green 0.15 --out C:\opengan\inpaint\out\berlin

   Outputs `candidate_k.png` in the tile palette, `candidate_k.npy` (the four field
   channels, same layout as the training tiles, so the phase 4 viewer can show them),
   `context.png`, and `metrics.json` with the targets, the truth that was there, and what
   each candidate achieved.

## Scale

One model covers a range of physical scales. A training crop is `side` raster pixels
resampled to 128, with `side` drawn log-uniformly so that one window spans anything from
300 m to 800 m (`--scale-lo 0.75 --scale-hi 2.0`, times 400 m). The signed distance
band is always 8 working pixels whatever the scale, and log2(scale) is given to the
network as a condition, so the same weights produce fine-grained plots at 400 m per
window and district structure at 800 m per window.

At inference, `--tile-m` picks the metres one 128 px window covers and `--extent` the
canvas size. A canvas larger than one window is filled by MultiDiffusion: the model runs
on overlapping windows (stride 64) at every denoising step and the predictions are
blended where they overlap, so a 1.2 km site is one coherent fill, not nine tiles with
seams. The two mechanisms combine: `--extent 1600 --tile-m 800` is a 256 px canvas of
coarse fabric, `--extent 1600 --tile-m 400` the same ground at full resolution through
49 windows.

## Targets

Six, all physical and scale-free, all optional:

    --coverage   footprint share of the hole, 0..1
    --green      green share, 0..1
    --street     street share, 0..1 (streets are 8 m, so this is also length density)
    --height     mean building height in the hole, metres
    --far        floor area ratio of the hole (3 m floors)
    --bsize      equivalent building side, metres: 4 x footprint area / perimeter, the grain

Withheld targets are inferred from context. Training withholds each metric 30% of the
time and all of them 15% of the time, so any subset is in-distribution and the
all-withheld case is the unconditional model that classifier-free guidance (`--cfg`,
default 2) needs. `metrics.json` always reports what was achieved; a target that was
not met is visible there rather than silently approximated.

`--keep-streets` cuts the existing street band out of the hole so only the plots between
streets are rebuilt. `--all` masks the whole canvas: generation from nothing at any size.

## Masks in training

Rectangle 17%, union of up to three rectangles 8%, brush strokes 16%, rectangle minus the
street band 9%, part of one quarter 12%, whole quarters 22%, whole quarters grown into the
street between them 16%. `--mask-probs` overrides the mix, `--mask-margin` the context ring, `--mask-quarters` how
far a quarter selection may walk.

Two rules behind that list, both of them about matching what actually gets asked at
inference.

**Every mask keeps `--mask-margin` working pixels (6 by default) clear of the border, and
there is no whole-tile or half-tile case any more.** The viewer renders the window around
whatever is selected, so the model is never handed a hole that runs off the edge of its own
view: there is always fabric on all four sides. The hole may be most of the tile and it may
come close to the edge, but the ring is always there. This used not to hold, because a
canvas larger than one window was covered with overlapping windows and some of those saw
the hole leave their edge. Phase 4 now renders at `extent / 128` metres per pixel instead
of a fixed 3.125, so any extent inside the trained scale range arrives as a single window
and the stitched path is not used. `sample_canvas` stays in `diffusion.py` for `sample.py`,
which can still cover a canvas of any size.

**Three of the seven kinds are quarter-shaped, and they are the plurality.** A quarter is a
connected piece of ground the street network encloses, found with a flood fill on the GPU:
a seed in the middle of the window, then up to `--mask-quarters` more (six by default),
each reached by stepping over the street from what is already selected. That is a ceiling
rather than a quota: a walk that runs out of room, at the window edge or against a road too
wide to step over, simply stops adding, so raising it widens the spread of selection sizes
instead of making every mask large. The walk is done once per batch and all three kinds are
derived from it, since a sample only ever uses one of them.

The masks cost about 170 max-pool calls on a `[batch, 1, 128, 128]` tensor per step, which
is single-digit milliseconds on a training GPU. The first 200 steps of the log print a
`[data x% prep y% gpu z%]` breakdown and mask generation sits in `prep`: if that climbs out
of the low single digits, lower `--mask-quarters`. On CPU it is slow enough to notice, which
only matters for `--tiny` smoke tests. `quarters` leaves the streets alone, which is the viewer's "keep the existing
streets"; `qgrown` pushes the selection out into the street around it so neighbouring
quarters join and the street between them has to be rebuilt too, which is what the film
does; `qpart` is a piece of a single block, one plot rather than the whole thing.

They are the plurality for a reason beyond matching the app. Cut a rectangle across a
street grid and most of the model's work is putting the street network back from the stubs
at the boundary, with the six numbers a small residual on a largely determined answer. Cut
a quarter and the street network is given, because the boundary of the mask is the street.
What is left undetermined is how much of the block is built, how tall and how green, which
is exactly what the condition controls. If the conditioning is going to be learned
anywhere, it is here, which is why the `condition delivered` line is the one to watch when
this mix changes.

A crop with no real street network floods to the whole tile, and an unlucky pick can come
back empty; either falls back to the plain rectangle for that sample.

## Model and loss

UNet, base 64, channel multipliers 1 2 3 4, two residual blocks per level, attention at
32 and 16 px, adaptive group norm carrying one embedding that sums the timestep, the
scale token and an MLP over the 12-dim condition (six values, six given flags). Input is
nine channels: the noisy fields, the context fields with the hole zeroed, the mask.
Cosine schedule, v-prediction, min-SNR-5 weighting, loss weight 1 in the hole and 0.1
outside it. `--base 96` roughly doubles the parameters if 23M underfits.

## Not built yet

Coarse-to-fine cascading (generate the network at 800 m per
window, lock it, refine plots at 400 m); the style-reference embedding for "rebuild this
as Kyoto" on a city the model has not seen; the viewer integration. All of them sit on
top of this code rather than changing it.
