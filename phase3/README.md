# Phase 3: evaluation

Preview grids tell you the model has not collapsed. They do not tell you whether it
matches the corpus, or whether the last thousand kimg bought anything. `evaluate.py`
answers both, with the metrics the dataset already carries.

    python evaluate.py --repo . --lut ..\dataset128_sdf\height_lut.json --meta ..\dataset128_sdf\meta.csv \
        --snapshot runs_sdf\<run>\network-snapshot-001600.pkl runs_sdf\<run>\network-snapshot-002640.pkl \
        --per-city 16 --out eval

It samples `--per-city` tiles for every city label from each snapshot, measures each one
with `tile_metrics` (the same function that produced `meta.csv` for the real tiles), and
prints the median of each metric next to the real median plus a distance.

The distance is the 1-Wasserstein distance between the generated and the real sample for
that metric, divided by the real standard deviation. It reads roughly as "how many
standard deviations apart the two distributions are", and is comparable across metrics.
Under about 0.1 is a good match. Above 0.5 means the model is making a different kind of
city, whatever the previews look like.

Two details that matter. Sampling uses truncation 1.0, because truncation narrows the
distribution and would flatter the model. And measured metrics are cached per snapshot in
the output folder, so re-running to add a snapshot costs only the new one.

The last block reports the per-city conditioning error: for each city label, how far the
median generated coverage is from that city's real median. If the class conditioning is
working, this is small; if the model has collapsed the labels together, it is large and
roughly equal to the spread between cities.

`--device cpu` is the default because the GPU is usually busy training. 960 tiles take
about four minutes on a CPU.

## What it found on the first SDF run

Measured at three points of the 60-city 128 px run (960 tiles each, truncation 1.0):

| | kimg 840 | kimg 1600 | kimg 2640 |
|---|---|---|---|
| mean distance to the corpus | 0.232 | 0.207 | 0.210 |
| per-city coverage error | 3.89 pp | 3.35 pp | 3.65 pp |

The model stopped improving on these measures somewhere around kimg 1600. The gap that
remains is not a "train longer" gap, it is structural, and the individual metrics say
what it is: the model makes 20 buildings where the corpus has 29, each about 40 percent
larger than a real one, in 12 blocks where the corpus has 24, connected by 20 percent
less street. It under-resolves fine grain, consistently, at every checkpoint.

At 128 px one pixel is 3.1 m, so a small building is two or three pixels and neighbouring
buildings merge. That is a resolution and capacity limit rather than a training-time one,
which is worth knowing before spending another day of GPU on the same configuration.
