"""
diagnose_build.py

Print the FULL compiler output when the StyleGAN CUDA kernels fail to build.
The training script only shows a truncated reason, which is rarely enough.

    cd C:\\opengan\\stylegan3
    python <path-to-phase1>\\diagnose_build.py

Writes everything to build_log.txt in the current directory as well as stdout.
"""
import os
import sys
import traceback
from pathlib import Path


def main():
    repo = Path.cwd()
    if not (repo / "torch_utils" / "custom_ops.py").exists():
        sys.exit("run this from inside the stylegan3 folder")
    sys.path.insert(0, str(repo))

    log_path = repo / "build_log.txt"
    log = open(log_path, "w", encoding="utf-8", errors="replace")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)
                st.flush()

        def flush(self):
            for st in self.streams:
                st.flush()

    sys.stdout = Tee(sys.__stdout__, log)
    sys.stderr = Tee(sys.__stderr__, log)

    import torch
    from torch.utils.cpp_extension import CUDA_HOME
    from torch_utils import custom_ops

    print("=" * 70)
    print("environment")
    print("=" * 70)
    print(f"python            {sys.version.split()[0]}")
    print(f"torch             {torch.__version__}")
    print(f"torch cuda        {torch.version.cuda}")
    print(f"cuda available    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device            {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"compute cap       {cap[0]}.{cap[1]}  (sm_{cap[0]}{cap[1]})")
    print(f"CUDA_HOME         {CUDA_HOME}")
    print(f"CUDA_PATH env     {os.environ.get('CUDA_PATH')}")
    print(f"TORCH_CUDA_ARCH_LIST {os.environ.get('TORCH_CUDA_ARCH_LIST')}")
    print(f"compiler bindir   {custom_ops._find_compiler_bindir()}")
    try:
        import ninja
        print(f"ninja             {ninja.__file__}")
    except ImportError:
        print("ninja             NOT INSTALLED")

    cache = Path(os.environ.get("TORCH_EXTENSIONS_DIR", Path.home() / "AppData/Local/torch_extensions"))
    print(f"extensions cache  {cache}  (exists={cache.exists()})")

    print()
    print("=" * 70)
    print("building bias_act_plugin with full verbosity")
    print("=" * 70)
    custom_ops.verbosity = "full"
    try:
        custom_ops.get_plugin(
            module_name="bias_act_plugin",
            sources=["bias_act.cpp", "bias_act.cu"],
            headers=["bias_act.h"],
            source_dir=str(repo / "torch_utils" / "ops"),
            extra_cuda_cflags=["--use_fast_math", "--allow-unsupported-compiler"],
        )
        print("\nBUILD SUCCEEDED")
    except Exception:
        print("\nBUILD FAILED, full traceback follows:\n")
        traceback.print_exc()

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log.close()
    print(f"\nfull log written to {log_path}")


if __name__ == "__main__":
    main()
