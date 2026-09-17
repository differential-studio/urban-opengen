# assets

Three small files the viewer needs to run, plus one that is only there to be read.

They are not weights and they do not change when you swap a snapshot, which is why they
live in the repository rather than in the downloads. Together they are 72 KB.

| file | what it is | what breaks without it |
|---|---|---|
| `height_lut.json` | the forward and inverse mapping between the grey level of a footprint pixel and a height in metres, written by `phase0/reencode_tiles.py` with the dataset | nothing decodes to metres, so no extrusion and no FAR |
| `cities.json` | the 60 city names and the class label each one was given during training | the city conditioning points at the wrong city; the blend between two cities is meaningless |
| `directions.npz` | the latent directions fitted after training by `phase1/fit_directions.py`, orthogonalised, with the Jacobian and the metric statistics | the metric sliders do nothing and absolute targets cannot be solved |
| `directions.csv` | the same fit as a table: R², slope per unit of alpha, how monotonic the walk was | nothing, it is documentation |

## Which run these came from

`height_lut.json` and `cities.json` are from `dataset128_sdf`, the 128 px SDF encoding of
dataset v1. Every released snapshot was trained on it, so one copy covers all of them.

`directions.npz` was fitted on `network-snapshot-003000.pkl`, the released generator, with
20,000 samples. The R² column in `directions.csv` is the honest measure of how far each
slider can be trusted: coverage and street share are the reliable ones, mean height and
green share are weaker. See section 4 of the root `README.md`.

## If you train your own

Re-run `phase0/reencode_tiles.py` and keep the `height_lut.json` and `cities.json` it
writes next to your dataset, then re-run `phase1/fit_directions.py` against your snapshot
and point `--directions` at the `directions.npz` it produces. The files here are specific
to the released model and are not a general-purpose lookup.
