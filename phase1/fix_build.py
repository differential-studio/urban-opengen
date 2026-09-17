"""
fix_build.py

The StyleGAN CUDA kernels fail to compile on some Windows toolchains. Rather than
guessing, this tries a series of compiler-flag combinations in fresh subprocesses
and reports which one works, then writes the winner to

    <stylegan3>/torch_utils/ops/build_flags.json

which the patched bias_act.py / upfirdn2d.py pick up automatically (patch 9 in
apply_patches.py). Run it once; after that, training just works.

    cd C:\\opengan\\stylegan3
    python <path-to-phase1>\\fix_build.py

Add --keep to skip clearing the extension cache between attempts (faster, but a
stale successful build can mask a failure).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PLUGIN = "bias_act_plugin"
SOURCES = ["bias_act.cpp", "bias_act.cu"]
HEADERS = ["bias_act.h"]
BASE_CUDA = ["--use_fast_math", "--allow-unsupported-compiler"]

# Ordered cheapest / most likely first. extra_cflags land after torch's own
# /std:c++20 and extra_cuda_cflags after its -std=c++20, so a later /std wins.
# torch 2.14 headers require C++20 (designated initializers, std::strong_ordering),
# so the C++17 route is impossible. CUDA 13.4's driver_types.h in turn breaks under
# MSVC's traditional preprocessor at C++20. The conforming preprocessor is the
# narrowest fix; /permissive- is next; /std:c++latest changes the language version
# torch itself was built against, so it goes last.
CANDIDATES = [
    ("C++20 + /Zc:preprocessor", ["/Zc:preprocessor"], []),
    ("C++20 + /permissive-", ["/permissive-"], []),
    ("C++20 + /Zc:preprocessor, also on the nvcc host pass",
     ["/Zc:preprocessor"], ["-Xcompiler", "/Zc:preprocessor"]),
    ("baseline (torch default, /std:c++20)", [], []),
    ("/std:c++latest", ["/std:c++latest"], []),
]

WORKER_OK = "___BUILD_OK___"


def worker(cfg: dict) -> int:
    repo = Path(cfg["repo"])
    sys.path.insert(0, str(repo))
    import torch  # noqa
    from torch_utils import custom_ops

    custom_ops.verbosity = "full"
    try:
        mod = custom_ops.get_plugin(
            module_name=PLUGIN,
            sources=SOURCES,
            headers=HEADERS,
            source_dir=str(repo / "torch_utils" / "ops"),
            extra_cflags=cfg["extra_cflags"],
            extra_cuda_cflags=BASE_CUDA + cfg["extra_cuda_cflags"],
        )
    except Exception as e:
        print(f"worker failed: {type(e).__name__}: {e}")
        return 1
    # A compile that produces no usable module is not a success.
    if mod is None:
        print("worker failed: get_plugin returned None")
        return 1
    if not hasattr(mod, "bias_act"):
        print(f"worker failed: built module has no bias_act(); attrs={dir(mod)[:20]}")
        return 1
    print(f"built module: {getattr(mod, '__file__', '?')}")
    print(WORKER_OK)
    return 0


def clear_cache():
    try:
        import torch.utils.cpp_extension as ce
        d = Path(ce._get_build_directory(PLUGIN, verbose=False))
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        print(f"  (could not clear cache: {e})")


def list_msvc():
    import glob
    pats = [
        "C:/Program Files*/Microsoft Visual Studio/*/*/VC/Tools/MSVC/*",
        "C:/Program Files (x86)/Microsoft Visual Studio/*/*/VC/Tools/MSVC/*",
    ]
    found = set()
    for p in pats:
        for m in glob.glob(p):
            if os.path.isdir(m):
                found.add(m)
    return sorted(found)


def main():
    if "--worker" in sys.argv:
        sys.exit(worker(json.loads(sys.argv[sys.argv.index("--worker") + 1])))

    repo = Path.cwd()
    if not (repo / "torch_utils" / "custom_ops.py").exists():
        sys.exit("run this from inside the stylegan3 folder")
    keep = "--keep" in sys.argv

    print("MSVC toolsets found:")
    for m in list_msvc():
        print(f"  {m}")
    print()

    results = []
    winner = None
    for name, cflags, cuflags in CANDIDATES:
        print("=" * 70)
        print(f"trying: {name}")
        print(f"  extra_cflags      {cflags}")
        print(f"  extra_cuda_cflags {cuflags}")
        print("=" * 70, flush=True)
        if not keep:
            clear_cache()
        cfg = {"repo": str(repo), "extra_cflags": cflags, "extra_cuda_cflags": cuflags}
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(cfg)],
            capture_output=True, text=True, errors="replace",
        )
        out = proc.stdout + proc.stderr
        ok = WORKER_OK in out
        results.append((name, ok))
        if ok:
            print("  -> SUCCESS")
            winner = (name, cflags, cuflags)
            break
        # show the first real compiler error line, not the whole wall of text
        interesting = [ln for ln in out.splitlines()
                       if "error" in ln.lower() and "warning" not in ln.lower()]
        for ln in interesting[:4]:
            print("  " + ln.strip()[:200])
        if not interesting:
            print("  -> failed, no error line matched; see full output below")
            print(out[-1500:])
        print("  -> FAILED\n", flush=True)

    print("=" * 70)
    print("summary")
    print("=" * 70)
    for name, ok in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")

    if winner is None:
        print("\nNothing worked. Training still runs on the reference implementation,")
        print("just slower. Send this whole output back and we'll pick another route.")
        return

    name, cflags, cuflags = winner
    out_path = repo / "torch_utils" / "ops" / "build_flags.json"
    out_path.write_text(json.dumps(
        {"_winner": name, "extra_cflags": cflags, "extra_cuda_cflags": cuflags}, indent=1))
    print(f"\nwinner: {name}")
    print(f"written to {out_path}")
    print("bias_act.py and upfirdn2d.py will pick this up automatically.")
    print("Re-run apply_patches.py first if you have not applied patch 9 yet.")


if __name__ == "__main__":
    main()
