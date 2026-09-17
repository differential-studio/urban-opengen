# Phase 1 and 2: train StyleGAN2-ADA on the 4-channel tiles, then fit sliders

Everything here was smoke-tested end to end on CPU at 32 px (dataset class,
city conditioning, geometry-only ADA, dihedral augmentation, preview grids,
snapshots, and the direction-fitting script against a snapshot). What is missing
is the GPU hours.

## Files

| file | what |
|---|---|
| `apply_patches.py` | patches a fresh `stylegan3` clone, idempotent, prints what it did |
| `npz_dataset.py` | the dataset class the patch installs, reads `tiles.npz` from phase 0 |
| `fit_directions.py` | Phase 2: sample, measure, regress `w -> metric`, orthogonalise, verify by walking |

## 1. Set up the repo

    git clone https://github.com/NVlabs/stylegan3
    python apply_patches.py stylegan3
    pip install click requests tqdm psutil imageio packaging

The patch adds `--augpipe`, `--dihedral`, `.npz` support for `--data`, 4-channel preview
grids plus raw-channel previews, and six compatibility fixes (see the header of
`apply_patches.py`). Run it again after a fresh clone; it skips what is already applied.

### Windows toolchain

The repo compiles two small CUDA kernels on first run. They are a 2 to 3x speedup, not
a correctness requirement: patch 7 makes a failed build warn and fall back to the
reference implementation rather than crash. Getting them to compile on a current
toolchain took some work, recorded here because it will recur on any new machine.

What is needed: `pip install ninja`, Visual Studio 2022 (Community or Build Tools) with
the "Desktop development with C++" workload, and a CUDA Toolkit whose major version
matches the torch build. Check the first two with

    & "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property displayName
    nvcc --version

Two problems then appear, and neither error message points at its real cause:

1. **`fatal error C1004: unexpected end-of-file found`** in a torch header, alongside
   `error C3688: invalid literal suffix 'x1'` in CUDA's `driver_types.h`. torch 2.14
   requires `/std:c++20` for its own headers (designated initializers,
   `std::strong_ordering`), so dropping to C++17 is not an option, but CUDA 13.4's
   headers do not survive MSVC's traditional preprocessor at C++20. Adding
   `/permissive-` fixes it. `/Zc:preprocessor` looks like the narrower fix but fails
   further along, in `torch/nn/options/conv.h`.

2. **`ModuleNotFoundError: No module named 'bias_act_plugin'`** even when the compile
   succeeded. `custom_ops.get_plugin` discarded the module returned by
   `torch.utils.cpp_extension.load()` and re-imported it by name; torch >= 2.0 builds
   the module with `spec_from_file_location` and never registers it in `sys.modules`.
   Patch 10 uses the return value. This one masks the first: a working flag combination
   still reports failure, so it is easy to conclude wrongly that nothing compiles.

Run `fix_build.py` from inside the stylegan3 folder to search flag combinations
empirically. It writes the winner to `torch_utils/ops/build_flags.json`, which patch 9
makes the ops modules read. `diagnose_build.py` prints the full untruncated compiler
output if you need to see it.

Confirmed working: Windows 11, Python 3.14, torch 2.14.0+cu130, CUDA Toolkit 13.4,
MSVC 14.44, RTX 4070 Laptop, with `/permissive-`.

## 2. Smoke test on the GPU (minutes)

From inside the `stylegan3` folder, with the phase 0 dataset built at 64 px
(`python reencode_tiles.py --images ../images --out ../dataset64 --size 64`):

    python train.py --outdir=runs --cfg=stylegan2 --data=../dataset64/tiles.npz ^
        --gpus=1 --batch=32 --gamma=2 --cond=1 --dihedral=1 --aug=ada --augpipe=bg ^
        --metrics=none --kimg=200 --snap=5 --workers=4

Look at `runs/<id>/fakes000200_ch.png`, the raw four channels, **not** the palette
version. Early in training the generator sits near the dataset mean, which for these
sparse masks is below the 128 threshold the palette renderer uses, so `fakes*.png` can
be solid black while the model is in fact learning normally. The `_ch.png` files are
unthresholded and show the truth. Streets appear before block texture, which is
expected: thin red lines are lower-frequency than building footprints at 64 px.

Note that resuming resets the kimg counter to zero. `fakes000000.png` in a resumed run
is the state of the snapshot you resumed from, not a fresh network, and ADA's `augment`
value also restarts and ramps up again over `ada_kimg`.

## 3. The real run (a day or two)

Dataset at 128 px (`--size 128`), or 256 px after the Grasshopper re-render.

    python train.py --outdir=runs --cfg=stylegan2 --data=../dataset128/tiles.npz ^
        --gpus=1 --batch=32 --gamma=4 --cond=1 --dihedral=1 --aug=ada --augpipe=bg ^
        --metrics=none --kimg=5000 --snap=20 --workers=4

| flag | why |
|---|---|
| `--cfg=stylegan2` | StyleGAN2 architecture with ADA; StyleGAN3 configs are for alias-free video and cost more |
| `--cond=1` | city-conditional, 34 classes from `tiles.npz` |
| `--dihedral=1` | 8-way dataset augmentation; real data for city tiles, 8x effective set |
| `--aug=ada --augpipe=bg` | adaptive discriminator augmentation, geometry only. Never use `bgc` here: brightness / hue / saturation would corrupt the height channel |
| `--gamma=4` | R1 regularisation. The single most important knob on a small dataset. Try 2, 4, 8 and keep the one whose fakes look best at equal kimg |
| `--metrics=none` | FID needs an RGB Inception network, meaningless on semantic maps. Use `fit_directions.py` output and `dataset/meta.csv` distributions instead |
| `--batch=32` | fits a 12 GB card at 128 px. On 8 GB use `--batch=16 --batch-gpu=8`. At 256 px use `--batch=16` |
| `--kimg=5000` | plenty for this dataset size; stop earlier if `fakes*.png` stop improving |

Resume from a snapshot with `--resume=runs/<id>/network-snapshot-XXXXXX.pkl`.

What good looks like in `fakes*.png`: streets that connect into a network, blocks with
a consistent style per city row, height variation visible inside the grey, no tiles
that are all-red or all-black.

## 4. Fit the sliders (Phase 2)

    python fit_directions.py --repo stylegan3 --network runs/<id>/network-snapshot-005000.pkl ^
        --lut ../dataset128/height_lut.json --out directions --n 20000

Prints one line per metric with the regression R^2 and the measured slope, and writes
`directions.npz`, which the viewer consumes. It carries both the orthogonalised slider
directions and the raw Jacobian of the linear fit, which is what lets the viewer solve
for absolute targets ("FAR 2.5") rather than only nudge relatively.

Metrics fitted by default, in slider priority order: `far_proxy, coverage_pct, green_pct,
street_pct, mean_height_m, street_length_m, intersections, building_count, mean_block_m2`.
Change the set or order with `--metrics a,b,c`; any key from `tile_codec.METRIC_INFO`
works, including road length, intersections, dead ends, block count and size, and the
absolute areas in m². Earlier metrics keep their direction exactly; later ones are made
orthogonal to everything before them, so put the sliders you care about most first.

Run it on the GPU (a few minutes for 20k samples); on CPU use `--n 5000` for a first
look. It can run while training continues.

Read `directions.csv` first:

* `r2` above roughly 0.3 means the metric is linearly controllable in W. Coverage,
  street and green percent usually are. Building count often is not, because it is a
  texture property rather than a global one, and that is fine.
* `monotonic_frac` near 1.0 means dragging the slider moves the measured metric in one
  direction the whole way. Below 0.7 the direction is not trustworthy.
* `walk_<metric>.png` shows one tile walked along each direction so you can see what
  the slider actually does.

Using a direction:

    d = np.load("directions/directions.npz")
    k = list(d["names"]).index("green_pct")
    w_new = w + alpha * float(d["w_scale"]) * d["directions"][k]     # alpha in about -3..3

## Not covered yet

Phase 4, the browser UI: export `G_ema` to ONNX (`torch.onnx.export` on
`G.synthesis` with a fixed `w` input), load it with ONNX Runtime Web, sliders write
`w`, the page renders the four channels back to the palette with `channels_to_rgb`
ported to JS. The `height_lut.json` inverse table is all the JS needs for that.
