# Data

## Dataset v2: rendered from OpenStreetMap

Everything `phase0/osm_tiles.py` and `phase0/render_cities.py` produce (the city rasters,
the 400 m tiles, the grid files) and everything `phase6/prepare.py` derives from them (the
field caches) contains OpenStreetMap data. OpenStreetMap is © OpenStreetMap contributors
and is licensed under the **Open Database License 1.0** (ODbL).

We release dataset v2 under the same licence, the ODbL, because it is a derivative
database in the licence's sense and that is what the ODbL requires. In practice:

* you may copy, share and build on it, including commercially;
* you must credit OpenStreetMap: "© OpenStreetMap contributors" and a link to
  https://www.openstreetmap.org/copyright, and we ask that you credit the rendering:
  "rendered by Differential for Urban OpenGen";
* if you publish a database derived from it, that database goes out under the ODbL too;
* if you publish only a *produced work* (a picture, a model, a paper), the attribution is
  all that is required.

Full text: https://opendatacommons.org/licenses/odbl/1-0/
The renderer's conventions (street widths, height fallbacks, green tags) are in
`phase0/osm_tiles.py` and saved next to every render, so the dataset is reproducible from
OpenStreetMap by anyone.

## Dataset v1: the Grasshopper tiles

The original 60-city tile set was rendered with a Grasshopper definition, from building,
street and green layers taken from OpenStreetMap. Same source, same licence: it is released
under the **ODbL** on exactly the terms above, and the same attribution applies.

It is the set the GAN snapshots were trained on, which is why it is in the release: without
it those weights cannot be reproduced. Tiles are 100 x 100 px at 4 m per pixel, except São
Paulo, which is the v2 render at 128 px because the original city was incomplete.

Dataset v2 is the one to build on. It is finer, its renderer is in this repository, and it
carries the whole-city rasters the inpainter trains on. v1 is here for reproducibility.
