"""
Does the request carry information the surroundings do not?

run02 learned to fill beautifully and to ignore the six numbers: `delivered` sat at 0.04 to
0.10 for twenty-six thousand steps. The suspected reason is redundancy. During training the
numbers always describe the very hole the model is being asked to fill, so as the network
gets better at reading the fabric around that hole it needs them less, and eventually not at
all.

If that is the whole story then the size of the hole should matter. A small hole in a
regular block is almost entirely determined by what surrounds it, and there the condition
genuinely has nothing to add. A large one is not, and there a model that had learned to use
the condition would show it.

So this sweeps the hole from a few per cent of the window to most of it, holding the tiles,
the seeds and the noise fixed, and reports `delivered` at each size: how far the fills
actually move apart when a metric is asked for low and then high, as a fraction of how far
they were asked to move.

Reading it:
  rises with hole size   the condition is wired but starved. Bigger holes in the training
                         mix would help, and the fix is cheap.
  flat and near zero     the condition is not reaching the output at any size. Training
                         longer will not change that; the objective has to change
                         (train.py --metric-weight and --ask-frac).

  python probe_masks.py --cache C:\\opengan\\cache_v3 --ckpt C:\\opengan\\inpaint\\run02\\latest.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase0"))
from data import CityCache, CropSampler, METRICS, random_masks  # noqa: E402
from diffusion import Schedule  # noqa: E402
from sample import load_checkpoint  # noqa: E402
from train import conditioning_response  # noqa: E402


def centred_masks(n: int, size: int, frac: float, device, margin: int = 6) -> torch.Tensor:
    """
    One square hole per tile, `frac` of the window, centred, never touching the edge.

    Centred and square on purpose: the point of the sweep is to change one thing, the amount
    of fabric the model has to invent. A mask mix that also changed shape and position would
    confound the answer with those.
    """
    side = int(round(math.sqrt(max(1e-6, frac)) * size))
    side = max(4, min(side, size - 2 * margin))
    o = (size - side) // 2
    m = torch.zeros(n, 1, size, size, device=device)
    m[:, :, o:o + side, o:o + side] = 1.0
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="the city cache from prepare.py")
    ap.add_argument("--ckpt", required=True, help="the checkpoint to probe")
    ap.add_argument("--metrics", default="far,coverage,green,height_m",
                    help="which metrics to push on, comma separated")
    ap.add_argument("--fracs", default="0.02,0.05,0.10,0.20,0.35,0.55,0.75",
                    help="hole area as a fraction of the window")
    ap.add_argument("--n", type=int, default=6, help="tiles per variant; more is steadier and slower")
    ap.add_argument("--steps", type=int, default=24, help="denoising steps per fill")
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--delta", type=float, default=1.5, help="how far the request is moved, in sigmas")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--batch", type=int, default=24, help="cap on tiles in flight, for VRAM")
    ap.add_argument("--mixed", action="store_true",
                    help="also run the real training mask mix, as the reference point")
    ap.add_argument("--out", default=None, help="write the table as json here too")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, lut, norm, ck = load_checkpoint(args.ckpt, dev)
    size = int(ck.get("size", 128))
    res_m = float(ck.get("res_m_per_px", 3.125))
    print(f"{Path(args.ckpt).name}: step {ck.get('step', 0):,}, {size} px windows at {res_m} m/px, device {dev.type}")

    cache = CityCache(args.cache)
    sampler = CropSampler(cache, size, args.scale, args.scale, seed=args.seed, device=dev)
    sched = Schedule(device=dev)
    wanted = [m for m in args.metrics.split(",") if m.strip() in METRICS]

    rows, hdr = [], f"{'hole':>7} {'side':>8}" + "".join(f"{m:>12}" for m in wanted)
    print("\ndelivered: 1.0 is the request arriving in full, 0 is the model ignoring it")
    print(hdr)
    print("-" * len(hdr))

    jobs = [(f, centred_masks(args.n, size, f, dev)) for f in
            [float(x) for x in args.fracs.split(",") if x.strip()]]
    if args.mixed:
        g = torch.Generator(device=dev).manual_seed(args.seed)
        x0, _ = sampler.fixed(args.n, args.scale, seed=args.seed)
        jobs.append((None, random_masks(x0, g)))

    for frac, mask in jobs:
        resp = conditioning_response(model, sched, sampler, norm, cache, dev, wanted,
                                     n=args.n, steps=args.steps, cfg=args.cfg, scale=args.scale,
                                     delta=args.delta, seed=args.seed, max_batch=args.batch, mask=mask)
        by = {r["metric"]: r for r in resp}
        real = float(mask[:, 0].mean())
        side = math.sqrt(real) * size * res_m * args.scale
        label = "mixed" if frac is None else f"{100 * real:.0f} %"
        print(f"{label:>7} {side:>6.0f} m" + "".join(f"{by[m]['delivered']:>12.3f}" for m in wanted))
        rows.append({"frac": round(real, 4), "side_m": round(side), "kind": "mixed" if frac is None else "square",
                     "delivered": {m: by[m]["delivered"] for m in wanted},
                     "detail": resp})

    sq = [r for r in rows if r["kind"] == "square"]
    if len(sq) >= 2:
        first = sum(sq[0]["delivered"].values()) / len(wanted)
        last = sum(sq[-1]["delivered"].values()) / len(wanted)
        rise = last / max(first, 1e-6)
        print(f"\nmean delivered: {first:.3f} at the smallest hole, {last:.3f} at the largest, "
              f"a factor of {rise:.1f}")
        # Two separate questions, and they have separate answers. Does the request reach the
        # output at all, which the rise answers, and does enough of it arrive to be a control,
        # which only the absolute number answers. Treating them as one gives a wrong headline.
        wired = rise >= 2.5
        usable = last >= 0.5
        if wired and usable:
            print("Wired and merely starved: the request arrives once the hole is big enough that\n"
                  "the surroundings stop determining the answer. Shift the training mask mix toward\n"
                  "larger holes; the objective does not need changing.")
        elif wired:
            print("Wired but weak. Delivery scales with how much the model has to invent, which is\n"
                  "the redundancy showing plainly: where the context determines the answer the\n"
                  "numbers add nothing and are ignored. But even at the largest hole it tops out\n"
                  "well short of a usable control, so bigger holes alone will not get there.\n"
                  "Train with --metric-weight 0.05 --ask-frac 0.3, and note that the asked samples\n"
                  "want room: --ask-min-frac keeps them off the holes that have none.")
        else:
            print("Not wired: the request does not reach the output at any hole size, so this is\n"
                  "not a matter of starving it. Train with --metric-weight 0.05 --ask-frac 0.3.")

    if args.out:
        Path(args.out).write_text(json.dumps({"ckpt": str(args.ckpt), "step": ck.get("step", 0),
                                              "scale": args.scale, "rows": rows}, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
