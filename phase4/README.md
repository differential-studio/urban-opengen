# Phase 4: viewer

A local page that renders a generated tile as an extruded 3D block in Three.js and drives
the latent space with sliders. The generator runs in a small Flask server on your GPU;
the page only talks to it over HTTP. Any `network-snapshot-*.pkl` can be loaded from the
page, so it doubles as a way to compare snapshots while training runs.

## Run

From inside the stylegan3 folder, with the venv active:

    python "G:\...\gan_project\phase4\serve.py" --repo . --lut ..\dataset64\height_lut.json --cities ..\dataset64\cities.json

then open http://127.0.0.1:5000. The panel has two tabs: **GAN** (snapshot, latents,
cities, PCA and metric directions, and the latent-search inpainting) and **Diffusion** (the
phase 6 inpainter, an address loaded from OpenStreetMap, and the metric targets). The
*Selection* block under them is shared, and so is the 3D view. Pick a snapshot from the
dropdown in the GAN tab. It scans
`<repo>\runs` recursively. Loading takes a few seconds: it also samples 4000 latents to
compute the PCA sliders.

Options:

| flag | what |
|---|---|
| `--network runs\...\network-snapshot-000600.pkl` | load a snapshot at start instead of from the page |
| `--directions directions\directions.npz` | named metric sliders from `fit_directions.py` |
| `--runs <folder>` | where to look for snapshots, default `<repo>\runs` |
| `--pca 8` | number of PCA sliders |
| `--port 5000` | |
| `--images C:\opengan\images_all` | offer real OSM tiles as references in the viewer (see Reference tiles below) |
| `--inpaint C:\opengan\inpaint` | a phase 6 checkpoint, or a folder of them: turns on the diffusion inpainter panel |
| `--osm-contact you@example.com` | the email OpenStreetMap sees in the User-Agent when an address is loaded; their policy asks for a real one |
| `--osm-cache <folder>` | where Overpass replies are cached, default `<phase4>/osm_cache` |
| `--cpu` | force CPU inference |

Use the LUT and cities file from the dataset the snapshot was trained on. Loading a 128 px
snapshot with the 64 px LUT is harmless (the LUT only depends on the tiles' grey values,
which are the same), but keep them paired anyway.

Needs `pip install flask`. Three.js is vendored in `vendor/`, so the page works offline.

## What the sliders do

Everything is combined into one `w` on the server:

| control | effect |
|---|---|
| seed A, seed B, A to B | `w = lerp(w_A, w_B, mix)` in W space |
| truncation | `truncation_psi` in the mapping network |
| city A, city B, blend | the class vector is `lerp(onehot_A, onehot_B, blend)`, so you can morph one city's style into another's |
| PC 1..N | GANSpace-style: `w += alpha * std_i * pc_i`, principal components of W sampled at load time, in units of each axis' standard deviation |
| metric directions | `w += alpha * w_scale * dir_k` from `directions.npz`, present only when the server was started with `--directions`. The dot before each name shows the R² of the fit: green is reliably controllable, amber weak, red mostly noise |
| targets | type an absolute value, e.g. FAR 2.5 or green 15 %. The server takes damped minimum-norm Newton steps using the fitted Jacobian, re-measuring the real tile after each, up to `--target-iters` (4). The readout turns green when within `--target-tol` (3 %) and amber when the model got as close as it could. Targets solve on top of whatever the sliders produce, so the sliders choose the starting tile |
| height scale, cell gap | display only |

The PCA sliders are what makes this useful before Phase 2 has been run: they expose the
model's main axes of variation immediately. Expect PC 1 to be something like density or
city identity and later components to be subtler. Once `fit_directions.py` has run, the
named sliders and the targets are the ones with a defined meaning.

A target the model cannot reach is reported honestly rather than faked: FAR 6 with 50 %
green exists in none of the training cities, so the solver will stop short and show
amber. The `not converged` note in the top-left latency line says the same.

## What you see

The ground plane is the palette tile (streets red, greenery green, open ground black).
Every footprint pixel becomes a box whose height is the decoded building height in
metres, so a 400 m tile at 128 px is 128 x 128 cells of 3.1 m. Colour is a ramp on
height. The corner shows the palette tile and the four raw channels, unthresholded, which
is the honest view of what the network produced. The panel top right shows the same
metrics `meta.csv` and `fit_directions.py` compute, on the tile currently displayed.

Latency top left is server inference time only.

**Geometry** (button bottom right of the 3D view) switches between two renderings of the
same channels. *Voxels* is one box per footprint pixel, the honest view of what the network
produced. *Clean* traces each building as a polygon with courtyard holes, simplifies the
outline with Douglas-Peucker to remove the pixel staircase, and extrudes it once to the
median height of its pixels; streets and greens become merged plates the same way. Three
controls: *simplify* is the tolerance in cells (0 keeps exact pixel outlines, around 0.6 gives
clean orthogonal corners, above 1 starts eating small features), *min size* drops buildings
under that many pixels, *plates* sets the street plate thickness in metres, *trees* scatters
sphere-on-trunk trees over green cells that are not under a building or street. Tree
positions and sizes are hashed from the pixel index, so they stay put while you adjust the
other sliders. Clean mode uses a model-board palette: white buildings, dark grey streets,
green parks, light grey ground. It rebuilds on every regeneration, in roughly 10 to 40 ms.

*Regular* is Clean plus the mesh regulariser in `regular.js`. The idea: find the direction
the city is built along, resample the distance fields in a frame rotated to that
direction, and do every shape operation in that frame, where "rectangular" means
axis-aligned and every fit is one-dimensional. The settings, top to bottom:

| setting | what it does |
|---|---|
| axis | *tile*: one dominant direction for the whole tile, measured from the boundary gradients of the fields with the angle-doubling trick (edges 90 degrees apart add up instead of cancelling). *per building*: each building gets its own axis when it is confident and more than 6 degrees off the tile's, otherwise the tile's. *none*: 0 degrees. The readout shows the angle; hover it for the confidence |
| axis from | which outlines vote for the direction: streets, buildings or both. Streets are the stronger signal in grid cities |
| upsample | resampling of the distance fields in the rotated frame, samples per native pixel. Bilinear sampling of an SDF and thresholding at 128 gives clean rotated outlines with no staircase, which is why the dataset is encoded that way |
| despeckle | morphological opening radius in native pixels. Removes pimples and thin spurs and splits masses joined by a thin neck |
| fill gaps | morphological closing radius. Fills notches and gaps narrower than twice this. Keep it at 0 on dense tiles, it swallows narrow streets |
| buildings | *rectangles*: greedy cover with the largest inscribed rectangles, each extruded to the median height inside it. *ortho*: snap every edge of the traced outline to the frame axes at its length-weighted position, merge same-direction neighbours, then remove edges shorter than *min edge* by merging their two parallel neighbours, shortest first. *boxes*: bounding rectangle. *trace*: Douglas-Peucker only, in the rotated frame |
| min edge | ortho: edges shorter than this are removed. Shown in metres of the current tile |
| box if fill | a hole-free piece that fills its bounding rectangle above this ratio becomes that rectangle, whatever the method |
| min rect | rectangles: the smallest rectangle kept |
| height step | 0 keeps one height per building. Above 0, the height map inside a building is quantised in steps of this many metres, majority-filtered so single-pixel steps vanish, and the building is split into parts that are shaped and extruded separately, each to its own median. 6 m (two floors) turns a flat block into a stepped massing model |
| streets | *blocks*: the not-street area is regularised exactly like buildings (with the *blocks* method) and drawn as plates on one street-coloured ground plate, so streets are the gaps between blocks and their edges are as straight as the blocks'. *trace*: outline the street pixels directly |
| green | shape method for green areas |

**Detail** (the tick box under *board*) adds the model-shop treatment on top of Clean,
Regular: a kerb line along the top edge of every block, a parapet around every roof (the
outline offset inward and extruded, skipped where the offset would fold or the building has
a courtyard), storey lines every 3.2 m up the walls, a soft edge on the corners and roofs,
and dashed centre lines down the streets, found by scanning the street mask along the tile's
own direction for short runs. One key light casts a soft shadow over a hemisphere wash, and
the shadow map is only recomputed when the geometry changes, so orbiting stays smooth. It
costs 20 to 60 ms on a rebuild; turn it off on large OSM areas if that matters. The same
treatment is what the phase 5 film uses.

Ortho refuses to snap an outline that has long edges closer to the diagonal than to an
axis in its frame (it would fold the polygon), and keeps the traced outline for that piece
instead, so on a tile with curvy streets most blocks stay traced while the buildings snap.
Per-building axis is the fix when buildings sit at different angles from the street grid.
The stat line reports how many pieces were traced, covered with rectangles or boxed.

The regulariser has no three.js dependency (`regular.js` exports `regularise(tile, options)`
returning polygons in tile pixel coordinates) so the same code can drive the phase 5
animation or an export. On a dense real tile it takes 200 to 400 ms, on a generated one
under 200 ms.

*Vector* is the fourth mode and the one meant for export. It runs on the server
(`vectorize.py`, needs `shapely`, which the OSM rasteriser already needs) and differs from
Regular in one idea: buildings are fitted, not traced. Tracing an outline and simplifying
it keeps every wobble of the raster; a fit decides in advance what a building may be and
finds the best instance of it.

1. Every building gets its own axis, measured from the gradient of the distance field along
   its edge with the angle-doubling vote (so the four sides of a rectangle agree). If that is
   not confident, the direction of the nearest street; if there is none, the tile's.
2. In a frame rotated to that axis the distance field is resampled at four samples per pixel
   and every boundary sample votes for the line it lies on: an edge running across the frame
   votes for a vertical line at its sub-pixel zero crossing, an edge running along it for a
   horizontal one. The few lines with enough votes (at least 4 m of edge behind them, never
   closer than *line spacing*) are the only places a wall can be.
3. Those lines cut the frame into cells; a cell is inside by majority of the field, and the
   union of the inside cells is the building. A wall that is 30 m long and slightly crooked
   in the raster becomes one straight 30 m wall. Gaps longer than 18 m are subdivided so a
   shapeless blob is still approximated rather than dropped; on a clean shape the
   subdivision is invisible because the cells merge again.
4. Convex corners the field says are cut are chamfered (a Barcelona block keeps its
   chamfers); a building whose height map has two plateaus is split into parts that share
   the same lines, so a tower on a podium is two boxes that meet exactly.
5. A mass joined at a narrow neck is split there first (watershed on the distance
   transform), so a generated blob that is really three buildings comes out as three.

Streets are the skeleton of the street field turned into a graph and cleaned: dead-end
stubs shorter than *street spur* (and always shorter than 1.5 street widths) go, junction
clusters closer than 10 m become one junction, loops the skeleton draws around a blob
collapse, and the path between two junctions is simplified into straight runs. Runs within
*street snap* degrees of the tile's grid axis are snapped to it, and every junction is
re-placed at the least-squares intersection of the runs that meet there (never moving more
than 2.5 px). The width of every street is twice the mean distance transform of the mask
along its skeleton, so a shallow generated field still gives the right width. Plates are
the runs buffered by half their width, extended half a width into every junction so the
corners are mitred; blocks are the tile minus the plates; greens are traced from the field
and clipped to the blocks. Buildings that merely touch a street keep their shape (the plate
is under them); one the raster drew well into the street is cut back to the block.

The stat line reports how many buildings, streets, blocks and greens came out, the server
time, and the round trip: the result is rasterised back and compared with the tile it came
from (IoU per class). On real OSM tiles the footprint IoU is 0.85 to 0.95 and the street IoU
0.65 to 0.75 (a straightened 8 m street is only ever half over its wobbly 2.5-pixel
original); on generated tiles 0.80 to 0.86 and 0.55 to 0.70. *Download GeoJSON* writes
the buildings with `height` and `building:levels`, the streets as `highway` lines with
`width`, the greens and the blocks, in local metres (x east, y north, origin at the tile
centre), so the tile can go straight into QGIS or back into OSM tooling. *Download 3DM*
writes a Rhino file through the `rhino3dm` package (`pip install rhino3dm`): buildings as
capped extrusion solids with courtyards as inner profiles, named and carrying `height_m`
and `levels` as user text, on a *Buildings* layer; their footprints on *Building outlines*;
street centrelines on *Streets* with `width_m`; and closed polylines on *Street plates*,
*Blocks* and *Greens*, plus the tile boundary. Units are metres, origin at the tile centre,
x east, y north, z up. Both exports write what the board shows: with an address loaded, the
real OpenStreetMap outlines go out as they are and the fit only covers the rectangle the
model rebuilt, and the place's latitude and longitude are written into the file's
EarthAnchorPoint so Rhino can place it on the planet. A dense real tile takes about a
second on the server, a generated one 100 to 400 ms; sliders in the Vectorise section
apply on release, and only the newest tile is fetched while one is in flight, so dragging a
latent slider stays responsive.

`vectorize.py` is also a library and a command line (`python vectorize.py tiles.npz --lut
height_lut.json --n 8`): `vectorize(ch, lut, tile_m, **options)` returns the same
dictionary the page gets, `parts_m()` turns it into shapely geometry in metres,
`to_geojson()` and `to_3dm()` write the exports, `rasterize()` and `roundtrip()` do the
check.

**Light or dark.** The button next to *Reset camera* swaps the whole app: the panel, the
scene background, the model board, and the ground the tile is drawn on, which is black on
dark and a mid grey on light so that white towers still read against it. Red, green and the
building greys are data, not decoration, so they never change. The choice is remembered.

**Reference tiles.** Start the server with `--images C:\opengan\images_all` and a
*Reference tile* section appears in the panel: pick a city and a tile and the real OSM
tile is encoded exactly like the training data (same LUT, same SDF settings, read from
`encoding.json` next to the LUT) and shown through the same viewer and geometry
settings. That is the honest way to tune the regulariser: a real Barcelona tile should
come out looking like Barcelona before the settings are trusted on generated tiles.
Moving any slider returns to the generated tile.

Geometry settings and the chosen mode are remembered in the browser between visits.

## The inpainter, and any address

Started with `--inpaint`, the panel grows an *Inpainter (diffusion)* section: the second
model, the one from phase 6. It is a conditional diffusion model trained to rebuild a hole
in a real city from the fabric around it and six measured numbers, so unlike the latent
search below it does not use the GAN at all. The server does not even need a snapshot
loaded for it to work.

    python <phase4>\serve.py --repo . --lut ..\dataset128_sdf\height_lut.json ^
        --cities ..\dataset128_sdf\cities.json --inpaint C:\opengan\inpaint

Pick a checkpoint (`ckpt_*.pt` or `latest.pt`, newest first) and press Load. The status
line reports its size, how far it trained and the window sizes it was trained at.

**The place.** Type an address, or a `lat,lon` pair, and press Find: the hits come from
Nominatim. Pick one, choose how much ground to load (400 m is one training tile, up to
1200 m) and press *Load from OpenStreetMap*. The server pulls the buildings, streets and
green areas from Overpass with the phase 0 rasteriser and sends back two things: the raster,
encoded with the same signed distance fields the model was trained on, and the geometry
itself, as outlines in metres.

The geometry is what you see. Buildings keep their real outline and their real height rather
than being read back off a 3.125 m grid, the blocks are the exact complement of the street
bands, so the kerbs land where they land, and the centre lines come from the ways rather
than from scanning pixels. The raster is only what the model reads. Overpass is rate-limited
and can take from a few seconds to a minute or two; the progress line and the bar under the
button follow the chunks, and replies are cached, so the same area comes back instantly next
time. The board resizes to the area, and the metrics, the geometry modes and the export all
work on it unchanged.

**The area to rebuild.** With the Diffusion tab open, drag a rectangle on the ground in the
3D view: that is the selection, and it is drawn as a blue rectangle on the board. While the
tool is on the camera does not orbit; the *Select area in 3D* button toggles it. The canvas
in the *Selection* block is the same mask seen from above, with a brush, a rectangle and an
eraser for anything that is not a simple box, *All* to take the whole board (which turns the
fill into pure generation) and *Undo* to go back to before the last fill.

After a fill, the board shows the real OpenStreetMap geometry everywhere except inside the
selection, where it shows what the model generated, traced and extruded the same way. Real
outlines that straddle the edge are clipped rather than dropped, so the seam is where the
selection is. Undo puts the real geometry back.

**The targets.** Six sliders, one per metric the model conditions on: coverage, green,
street, mean height, FAR and mean block size. Each has a tick box. Ticked, the number is
handed to the model as a request; unticked, the model infers it from what surrounds the
hole, which is usually what you want for most of them. After a fill, the number the fill
actually measures is printed under each slider, and where a target was asked for, the
readout says what was asked next to what was measured and turns green within 12 %. A model
that cannot reach a request says so this way rather than pretending.

| control | what it does |
|---|---|
| steps | denoising steps. 30 is enough for a look, 60 is cleaner, beyond that little changes |
| guidance | classifier-free guidance: how hard the model is pushed toward the requested numbers. 1 ignores them, 2 is the default, above 4 the fabric starts to distort |
| harmonise | RePaint resampling: after each step the fill is pushed back up the noise schedule and re-predicted, so it re-knits with the edge of the hole. One extra pass per level |
| seed | same seed and same request gives the same fill |
| keep the existing streets | the street band is cut out of the mask, so the network stays and only the plots between the streets are rebuilt |

A canvas larger than the model's window (128 px, 400 m) is covered with overlapping windows
and stitched with MultiDiffusion, exactly as `phase6/sample.py` does it, so an 800 m area is
filled by nine windows that agree with each other rather than nine separate fills. Cost
scales with the area: a 400 m fill at 40 steps is a second or two on a 4070, an 800 m one
under ten.

The viewer's tiles are encoded with the GAN dataset's height LUT and the checkpoint carries
the LUT of the corpus it was trained on. Both are rank-normalised, so the same byte can mean
different metres in each; the server converts the height channel both ways through the grey
value they agree on, and says so on load when the two differ.

## Inpainting by latent search

The *Inpaint (latent search)* section paints a mask over part of the tile and rebuilds that
part. No second model and no extra training: the generator can only make tiles that look
like the training set, so if you search its latent space for the tile that best matches
the part of the image you are keeping, whatever it puts in the hole is by construction a
plausible piece of city that continues the streets and blocks around it.

What happens on Fill:

1. the mean of `W` for the selected city is computed (cached) and used as the starting
   point of the search
2. `w` is optimised in `W+`, one vector per synthesis layer, with Adam and the schedule
   from NVIDIA's projector: a little noise on `w` early to get off the flat starting
   point, learning rate ramped up briefly then down to a crawl
3. the loss is squared error against the tile, but only where the mask says keep, summed
   over four resolutions so the big shapes are matched before the pixels, and weighted by
   a smooth halo around the hole so the street and block edges the fill has to meet count
   for more than the far corners of the tile
4. the result is composited: the original tile outside the hole, the generated tile
   inside, blended over a few pixels

Point 4 is where the SDF encoding pays off again. The channels are distance fields, so
the blend happens on smooth quantities and the decoded outline stays continuous. Blending
hard masks would leave a ragged seam.

| control | what it does |
|---|---|
| Brush / Rect / Erase | how the mask is painted. Right-click or alt while painting also erases |
| brush | brush diameter in tile pixels |
| steps | optimisation steps. On a 4070 at 128 px expect roughly 40 to 60 steps per second, so 400 steps is under ten seconds; on CPU it is about 100 times slower |
| prior | how hard the search is pulled back toward the average tile of the city. 0 is the usual setting; raise it if a fill comes out as nonsense |
| context | how much the fit has to match the far parts of the tile. 1 matches the whole tile, which gives the most coherent fill. Turn it down when the model cannot reproduce this tile: the fit then concentrates on the neighbourhood of the hole and the seam lines up much better |
| focus | radius of the halo around the hole that the fit weights most heavily |
| feather | pixels over which the generated patch is blended into the original |

The city selector in the City section sets the style the fill reaches for, so the same
hole can be filled as Barcelona or as Tokyo. Fills chain: fill one area, paint another,
fill again. Undo restores the tile from before the last fill. It works the same on a
generated tile and on a real reference tile.

How well it works depends on how well the model can reproduce the tile around the hole,
which is worth knowing before reading too much into a bad result:

* a hole in a **generated** tile is filled almost perfectly, because the model can
  reproduce the context exactly. This is the honest test of the mechanism
* a hole in a **real** tile of moderate density, a Tokyo grid say, is filled with streets
  that continue through the hole and join at the intersections
* a hole in a **very dense** real tile, Barcelona Eixample at 65 % coverage, comes back as
  a vague mass at the current snapshot. The generator cannot yet produce fabric that
  dense, so the search has nowhere good to land. Turning *context* down to about 0.15
  helps a lot with the seam but not with the fill. This should improve as training runs
  on, and is a useful measure of it

`--inpaint-steps` sets the default step count for the panel. This method runs the generator
itself, so it only works at the generator's own resolution: on an area loaded from
OpenStreetMap that is larger than one tile its Fill button is disabled, and the diffusion
inpainter is the one to use.

## Exporting a design space

**Export design space** at the bottom of the panel opens a Colibri-style popup. Every slider
the active tab can sweep is listed with a tick box, a range and a number of steps; the grid
is full factorial, so the count is the product of the step counts and it is shown, with a
time estimate taken from what the model actually cost on this machine, while you change it.
Press Export and the server works through the grid on a background thread, reporting
progress, and writes the folder the multiobjective viewer reads:

    001.3dm  002.3dm  ...     one Rhino model per option
    context.3dm               the surroundings, written once instead of with every option
    parameters.csv            first column the option name, one column per swept slider
    objectives.csv            first column the option name, one column per measurement
    run.json                  the spec, the axes, the timings
    thumbs/001.png            a small plan of each option

into `<runs>/spaces/<name>/`, with a zip of it next to it for the browser.

What an option covers depends on the tab, and this is the part worth getting right:

* **Diffusion.** An option is *only* what the model put inside the area you painted, and the
  objectives describe that piece: `site_area_m2` is the painted area, and FAR, coverage and
  the sun and view hours are measured on it. The city around it is still there while the sun
  and view rays are cast, so a courtyard in the Eixample reads its real 0.2 h of winter sun,
  not the 1.3 h it would get standing alone. That surrounding city goes out once as
  `context.3dm` with a hole exactly where the options go, in the same coordinates and with
  the same EarthAnchorPoint, so an option drops straight into it.

  **`context.3dm` is OpenStreetMap, not the raster.** With an address loaded, the context is
  built from the vector geometry `dif.py` traced from the OSM ways: true building outlines
  with their real heights, blocks that are the exact complement of the street bands, the
  street surface derived from those blocks, greens, and centrelines from the ways themselves.
  Nothing in it is read back off the 3.125 m grid, so the surroundings never inherit the
  model's pixels; only the fill inside the hole does, because only the fill came from a model.
  Working on a generated or reference tile there is no OpenStreetMap to use, so the context
  falls back to the fit of that tile, and the popup says which of the two you are getting.
* **GAN.** An option is a whole generated tile, so there is nothing outside it and
  `context.3dm` holds only the tile boundary.

The **context** slider in the popup says how much of the surroundings takes part: everything
within that distance of the painted area casts shadows and blocks views, everything beyond is
dropped from both the measurements and `context.3dm`. The app can hold 1.5 km of
OpenStreetMap and measuring all of it is wasted work, because the analysis raster costs the
square of the side. 300 m is the default and on a dense Barcelona block already gives the
same numbers to two decimals as the whole 400 m tile; 0 measures the plot as if it stood
alone, which is a useful thing to compare against but not what the site does.

Which sliders are worth sweeping differs by model. On the GAN the seed and the fitted metric
directions move the tile a long way. On the diffusion inpainter the seed is by far the
strongest handle: the fill follows the fabric around the hole much more than it follows the
numbers asked for, so a sweep of guidance against a target is usually a flatter space than a
sweep of seeds. The measurements are made on the geometry either way, so what a target asked
for lives in `parameters.csv` and what the model actually produced lives in `objectives.csv`,
and the two are never confused.

The same run can be driven from the command line with `phase7/explore.py`, which takes a JSON
spec and talks to this server over HTTP. The measurements come from `phase7/analysis.py`; its
README explains what each objective column means.

## Files

| file | what |
|---|---|
| `serve.py` | Flask server: loads snapshots, PCA, generate endpoint, serves the page |
| `dif.py` | the diffusion inpainter and the OpenStreetMap loader: checkpoint loading, the fill endpoint's work, geocoding and the raster |
| `index.html` | the viewer, vanilla JS + Three.js modules |
| `regular.js` | the Clean and Regular geometry modes: tracing, simplification, the mesh regulariser (browser side) |
| `export_space.py` | the design space export behind the Export button: the grid, the background run, the option and context models, the two CSVs |
| `vectorize.py` | the Vector geometry mode and the GeoJSON and Rhino 3dm exports: fitted buildings, street graph with widths, plates, blocks, greens, round-trip check (server side) |
| `vendor/` | three.module.js, OrbitControls.js, BufferGeometryUtils.js (r160, MIT) |

`serve.py` imports `tile_codec.py` from `../phase0`, and `dif.py` imports the model from
`../phase6` and the rasteriser from `../phase0`, so keep the folder structure. The OSM
loader and `vectorize.py` need `requests`, `shapely` and `pyproj`, the same ones
`phase0/osm_tiles.py` uses; the 3dm export needs `rhino3dm` as well, and the design space
export imports `analysis.py` from `../phase7`.

## Notes

* The `height_lut.json` inverse is rebuilt on load by the current `tile_codec.py`, so a
  LUT written before the sub-threshold-height fix decodes correctly without regenerating.
* Requests are debounced and coalesced, so dragging a slider sends at most one request
  per 50 ms and never queues up.
* This is a testing tool, not a deployable demo. A static, serverless version needs the
  generator exported to ONNX and run with ONNX Runtime Web; that's a separate piece of
  work worth doing once a snapshot is good enough to show people.
