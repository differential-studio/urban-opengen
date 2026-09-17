"""
apply_patches.py

Patch a fresh clone of https://github.com/NVlabs/stylegan3 so it trains on the
Urban OpenGen 4-channel tiles.npz. Idempotent: safe to run twice.

    git clone https://github.com/NVlabs/stylegan3
    python apply_patches.py stylegan3

What it changes:
  1. copies npz_dataset.py into stylegan3/training/
  2. train.py: --data ending in .npz uses NpzDataset
  3. train.py: new --augpipe and --dihedral options (default 'bg', geometry only, which is what
     4-channel semantic tiles need; the repo's hardcoded 'bgc' colour ops assume RGB)
  4. training_loop.py: save_image_grid renders 4-channel tiles to the palette so
     reals.png and fakesNNNN.png stay viewable
  5. conv2d_gradfix.py / grid_sample_gradfix.py: use packaging.version instead of
     pkg_resources, which setuptools 81+ no longer ships
  6. torch_utils/misc.py: InfiniteSampler works on torch >= 2.2
  7. bias_act.py / upfirdn2d.py: a failed CUDA kernel build warns and falls back
     to the reference implementation instead of crashing (stylegan2-ada behaviour)
  8. training_loop.py: every fakes*.png also gets a fakes*_ch.png with the four raw
     channels side by side, so sub-threshold output is visible during early training
  9. bias_act.py / upfirdn2d.py: read optional torch_utils/ops/build_flags.json so
     fix_build.py can pin working compiler flags without further edits
 10. custom_ops.py: use the module returned by torch.utils.cpp_extension.load()
     instead of re-importing it by name, which torch >= 2.0 no longer supports
"""
import shutil
import sys
from pathlib import Path

MARK = "# [urban-opengen patch]"
OLD_MARK = "# [urban-opengan patch]"   # clones patched under the old name stay idempotent


def patch(path: Path, old: str, new: str, must=True, sentinel=None):
    src = path.read_text(encoding="utf-8")
    probe = sentinel if sentinel is not None else (new if MARK in new else None)
    if probe is not None and (probe in src or probe.replace(MARK, OLD_MARK) in src):
        print(f"  already patched: {path.name}")
        return
    if old not in src:
        if must:
            sys.exit(f"could not find anchor in {path}:\n{old}")
        return
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    print(f"  patched: {path.name}")


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python apply_patches.py <path-to-stylegan3-clone>")
    repo = Path(sys.argv[1])
    if not (repo / "train.py").exists():
        sys.exit(f"{repo} does not look like the stylegan3 repo")

    print("1. dataset class")
    shutil.copy(Path(__file__).with_name("npz_dataset.py"), repo / "training" / "npz_dataset.py")

    print("2. --data .npz support")
    patch(repo / "train.py",
        "        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)",
        "        class_name = 'training.npz_dataset.NpzDataset' if data.endswith('.npz') else 'training.dataset.ImageFolderDataset'  " + MARK + "\n"
        "        dataset_kwargs = dnnlib.EasyDict(class_name=class_name, path=data, use_labels=True, max_size=None, xflip=False)")

    print("3. --augpipe option")
    patch(repo / "train.py",
        "@click.option('--aug',          help='Augmentation mode',",
        "@click.option('--augpipe',      help='Augmentation pipeline: blit|geom|bg|bgc (bg = geometry only, use for 4-channel tiles)', type=click.Choice(['blit', 'geom', 'bg', 'bgc']), default='bg', show_default=True)  " + MARK + "\n"
        "@click.option('--aug',          help='Augmentation mode',")
    patch(repo / "train.py",
        "        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)",
        "        augpipe_specs = {  " + MARK + "\n"
        "            'blit': dict(xflip=1, rotate90=1, xint=1),\n"
        "            'geom': dict(scale=1, rotate=1, aniso=1, xfrac=1),\n"
        "            'bg':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),\n"
        "            'bgc':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),\n"
        "        }\n"
        "        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[opts.augpipe])")

    print("3b. --dihedral option (8-way dataset augmentation, NpzDataset only)")
    patch(repo / "train.py",
        "@click.option('--mirror',       help='Enable dataset x-flips',",
        "@click.option('--dihedral',     help='8-way rotation+flip dataset augmentation (npz datasets only, subsumes --mirror)', metavar='BOOL', type=bool, default=False, show_default=True)  " + MARK + "\n"
        "@click.option('--mirror',       help='Enable dataset x-flips',")
    patch(repo / "train.py",
        "    c.training_set_kwargs.xflip = opts.mirror",
        "    c.training_set_kwargs.xflip = opts.mirror\n"
        "    if opts.dihedral:  " + MARK + "\n"
        "        if not opts.data.endswith('.npz'):\n"
        "            raise click.ClickException('--dihedral needs an .npz dataset')\n"
        "        c.training_set_kwargs.dihedral = True\n"
        "        c.training_set_kwargs.xflip = False")
    print("4. 4-channel preview grids")
    patch(repo / "training" / "training_loop.py",
        "    assert C in [1, 3]\n"
        "    if C == 1:",
        "    if C == 4:  " + MARK + "\n"
        "        # Urban OpenGen tiles: footprint, height, street, green -> palette RGB\n"
        "        fp = img[:, :, 0] >= 128\n"
        "        rgb = np.zeros(img.shape[:2] + (3,), dtype=np.uint8)\n"
        "        h = np.maximum(img[:, :, 1], 100)\n"
        "        rgb[fp] = np.stack([h[fp]] * 3, axis=-1)\n"
        "        rgb[img[:, :, 3] >= 128] = (128, 255, 0)\n"
        "        rgb[img[:, :, 2] >= 128] = (255, 0, 0)\n"
        "        PIL.Image.fromarray(rgb, 'RGB').save(fname)\n"
        "        return\n"
        "    assert C in [1, 3]\n"
        "    if C == 1:")
    print("5. pkg_resources removed in setuptools 81+, use packaging instead")
    for rel in ("torch_utils/ops/conv2d_gradfix.py", "torch_utils/ops/grid_sample_gradfix.py"):
        patch(repo / rel,
            "from pkg_resources import parse_version",
            "try:  " + MARK + "\n"
            "    from packaging.version import parse as parse_version\n"
            "except ImportError:\n"
            "    from pkg_resources import parse_version")
    print("6. torch>=2.2 Sampler.__init__ no longer takes data_source")
    patch(repo / "torch_utils" / "misc.py",
        "        super().__init__(dataset)",
        "        try:  " + MARK + "\n"
        "            super().__init__(dataset)\n"
        "        except TypeError:\n"
        "            super().__init__()")
    print("7. fall back to reference ops when the CUDA kernels cannot be compiled")
    for mod in ("bias_act", "upfirdn2d"):
        patch(repo / "torch_utils" / "ops" / f"{mod}.py",
            "def _init():\n"
            "    global _plugin\n"
            "    if _plugin is None:\n"
            "        _plugin = custom_ops.get_plugin(\n"
            f"            module_name='{mod}_plugin',",
            "_plugin_failed = False  " + MARK + "\n"
            "\n"
            "def _init():\n"
            "    global _plugin, _plugin_failed\n"
            "    if _plugin_failed:\n"
            "        return False\n"
            "    if _plugin is None:\n"
            "        try:\n"
            "            _plugin = _build_plugin()\n"
            "        except Exception as e:\n"
            "            _plugin_failed = True\n"
            "            import warnings\n"
            f"            warnings.warn('Failed to build CUDA kernels for {mod}, falling back to the slow reference implementation. '\n"
            "                          'Install ninja, Visual Studio Build Tools (C++ workload) and the CUDA toolkit matching torch to fix. '\n"
            "                          f'Reason: {e!s:.2000}')\n"
            "            return False\n"
            "    return True\n"
            "\n"
            "def _build_plugin():\n"
            "    if True:\n"
            "        return custom_ops.get_plugin(\n"
            f"            module_name='{mod}_plugin',",
            sentinel="_plugin_failed = False  " + MARK)
        # the original body ended with `        )\n    return True` — the second return is now dead code
        # inside _build_plugin after our `return custom_ops.get_plugin(...)`; harmless, leave it.
    print("8. also save the raw 4 channels next to every preview grid (thresholded palette can hide early training)")
    patch(repo / "training" / "training_loop.py",
        "        PIL.Image.fromarray(rgb, 'RGB').save(fname)\n"
        "        return",
        "        PIL.Image.fromarray(rgb, 'RGB').save(fname)\n"
        "        # raw channels side by side: footprint | height | street | green, no threshold  " + MARK + "\n"
        "        raw = np.concatenate([img[:, :, i] for i in range(4)], axis=1)\n"
        "        PIL.Image.fromarray(raw, 'L').save(fname.replace('.png', '_ch.png'))\n"
        "        return")
    print("7b. widen the truncated failure reason from an earlier run of this script")
    for mod in ("bias_act", "upfirdn2d"):
        patch(repo / "torch_utils" / "ops" / f"{mod}.py",
            "f'Reason: {e!s:.300}')",
            "f'Reason: {e!s:.2000}')",
            must=False, sentinel="{e!s:.2000}")

    print("9. let build_flags.json override the compiler flags (written by fix_build.py)")
    for mod in ("bias_act", "upfirdn2d"):
        patch(repo / "torch_utils" / "ops" / f"{mod}.py",
            "_plugin_failed = False  " + MARK,
            "def _extra_build_flags(base_cuda):\n"
            "    \"\"\"Merge optional overrides written by phase1/fix_build.py.\n"
            "\n"
            "    torch puts its own /std:c++20 before extra_cflags and -std=c++20 before\n"
            "    extra_cuda_cflags, so anything here wins.\n"
            "    \"\"\"\n"
            "    import json\n"
            "    cfg = {}\n"
            "    p = os.path.join(os.path.dirname(__file__), 'build_flags.json')\n"
            "    if os.path.isfile(p):\n"
            "        try:\n"
            "            with open(p) as f:\n"
            "                cfg = json.load(f)\n"
            "        except Exception:\n"
            "            cfg = {}\n"
            "    return dict(\n"
            "        extra_cflags=list(cfg.get('extra_cflags', [])),\n"
            "        extra_cuda_cflags=list(base_cuda) + list(cfg.get('extra_cuda_cflags', [])),\n"
            "    )\n"
            "\n"
            "_plugin_failed = False  " + MARK,
            sentinel="def _extra_build_flags")
        patch(repo / "torch_utils" / "ops" / f"{mod}.py",
            "            source_dir=os.path.dirname(__file__),\n"
            "            extra_cuda_cflags=['--use_fast_math', '--allow-unsupported-compiler'],",
            "            source_dir=os.path.dirname(__file__),\n"
            "            **_extra_build_flags(['--use_fast_math', '--allow-unsupported-compiler']),  " + MARK,
            sentinel="**_extra_build_flags(")
    print("10. torch >= 2.0 returns the built module from load() and no longer registers it")
    patch(repo / "torch_utils" / "custom_ops.py",
        "            torch.utils.cpp_extension.load(name=module_name, build_directory=cached_build_dir,\n"
        "                verbose=verbose_build, sources=cached_sources, **build_kwargs)\n"
        "        else:\n"
        "            torch.utils.cpp_extension.load(name=module_name, verbose=verbose_build, sources=sources, **build_kwargs)\n"
        "\n"
        "        # Load.\n"
        "        module = importlib.import_module(module_name)",
        "            module = torch.utils.cpp_extension.load(name=module_name, build_directory=cached_build_dir,  " + MARK + "\n"
        "                verbose=verbose_build, sources=cached_sources, **build_kwargs)\n"
        "        else:\n"
        "            module = torch.utils.cpp_extension.load(name=module_name, verbose=verbose_build, sources=sources, **build_kwargs)\n"
        "\n"
        "        # Load. torch >= 2.0 hands the module back from load() and does NOT put it\n"
        "        # in sys.modules, so the original importlib.import_module() always raised\n"
        "        # ModuleNotFoundError even when the compile had succeeded.\n"
        "        if module is None:\n"
        "            module = importlib.import_module(module_name)",
        sentinel="module = torch.utils.cpp_extension.load(name=module_name, build_directory=cached_build_dir,  " + MARK)
    print("done")


if __name__ == "__main__":
    main()
