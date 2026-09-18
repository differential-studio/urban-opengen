# Downloads

The repository is the pipeline. The things it produces are too large for git and are
published as downloads. Each is licensed separately; see `LICENSE.md`.

What is **not** here is already in the repository. `assets/` carries the three small files
the viewer needs besides the weights: `height_lut.json` (grey level back to metres),
`cities.json` (the sixty labels in the order the model was trained on) and
`directions.npz` (the latent directions behind the metric sliders). They are the same for
every snapshot below, so the downloads are only ever the weights.

## Dataset v1: the tiles the GAN was trained on

Sixty cities, the same 10 km window and 25 by 25 grid, rendered from OpenStreetMap through
a Grasshopper definition at 100 by 100 pixels and 4 m per pixel.

This is the set the released GAN snapshots were trained on, with one repair: São Paulo came
out of the original run almost empty and was replaced afterwards with the v2 render of that
city, downsampled to match. So the GAN was trained with São Paulo effectively missing. If
you are reproducing that training, this is the set you need, and that is the one difference.

    https://drive.google.com/drive/folders/1zt-Uuqrdzz14QmBHFDgNJ-sq8T2ZjdVZ?usp=sharing

Licence: Open Database License 1.0, © OpenStreetMap contributors. See `LICENSE-DATA.md`.

## Dataset v2: the rasters the inpainter was trained on

Sixty cities, one 10 km window each, centred on the city core and cut into a 25 by 25 grid
of 400 m tiles. That is 625 tiles per city and 37,500 in all, of which 30,957 carry built
content; the rest are sea, or ground OpenStreetMap has nothing for. Every tile is 128 by
128 pixels at 3.125 metres per pixel, four classes: building footprints in grey with the
grey level carrying height, streets in red, planted ground in green, everything else black.

Every city was rendered by the same code with the same conventions, which is the point of
it. Opportunistic crops from different sources cannot be compared to each other; these can.

    https://drive.google.com/drive/folders/1d6CEbB7EkV2rp0jqudk6_2RlZFDc48DY?usp=sharing

Also in the download: the whole 10 km window per city as one 3200 by 3200 raster, the
per-city georeference (`<City>_grid.json`, centre, CRS and per-tile coordinates), and
`conventions.json`, which records the street widths, the green tags and the height
assumptions the renderer used.

Licence: Open Database License 1.0, © OpenStreetMap contributors. See `LICENSE-DATA.md`.
If you build on it, credit OpenStreetMap and us.

## Diffusion inpainter weights

A 23.2 million parameter conditional UNet, trained from scratch on dataset v2. Give it a
real piece of city with an area taken out, and six numbers measured from what was there,
and it rebuilds the area so that the streets and blocks continue into the surrounding
fabric. The same weights work from 300 m to 800 m windows, because the scale goes into the
network as part of the condition.

    https://drive.google.com/drive/folders/19MTAnj_ZyVp2yGEHh6WNb9v8aumvkApo?usp=sharing

Licence: Creative Commons Attribution 4.0. Commercial use is fine, credit is required.
See `LICENSE-MODELS.md`.

## GAN generator snapshots

StyleGAN2-ADA, patched for four-channel tiles and geometry-only augmentation, conditioned
on the sixty city labels. Draws a whole 400 m tile from a latent vector. Latent directions
fitted afterwards to the measured metrics turn floor area ratio, coverage, green share and
street share into sliders that move the real quantity.

    https://drive.google.com/drive/folders/149r6eHQ5I96zi5G-f5zJgcH3W0o_xU4y?usp=sharing

Licence: non-commercial research and educational use only. These weights were produced
with NVIDIA's stylegan3 repository, whose licence covers works derived from it. Credit
both Differential and NVIDIA. See `LICENSE-MODELS.md` before using them for anything.

## Which dataset do you want

v2, unless you are reproducing the GAN. It is finer at 3.125 m per pixel, its renderer is
in this repository so you can extend it to any city yourself, and it carries the whole-city
rasters the inpainter needs. Its one weakness is coverage: some cities have gaps where
OpenStreetMap does, and a few have blank bands where a fetch came back short. v1 is coarser
and its renderer is a Grasshopper file, but it is the set the GAN actually saw.

## Which model do you want

If you are rebuilding part of an existing place, you want the inpainter: it sees the
context and is licensed for commercial work. If you want whole tiles out of nothing, or
you want to move through a latent space, you want the generator, and you are limited to
non-commercial use by NVIDIA's licence rather than by ours.
