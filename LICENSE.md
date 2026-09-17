# Licences

Urban OpenGen is released as open source. Different parts of the repository are different
kinds of work and carry the licence that fits each; all of them require credit.

| part | what it covers | licence | credit line |
|---|---|---|---|
| **Code** | `phase0/` data and the OpenStreetMap renderer, `phase1/` training patches and direction fitting, `phase3/` evaluation, `phase4/` the configurator (viewer, regulariser, inference server), `phase6/` the diffusion inpainter | [Apache License 2.0](LICENSE) | keep the `NOTICE` file with any copy; "Urban OpenGen, Copyright 2026 Differential" |
| **Diffusion inpainter weights** | checkpoints trained with `phase6/` | [Creative Commons Attribution 4.0](LICENSE-MODELS.md) | "Urban OpenGen inpainter by Differential, CC BY 4.0" |
| **GAN generator weights** | `network-snapshot-*.pkl` from the StyleGAN2-ADA runs | non-commercial research use only, see [LICENSE-MODELS.md](LICENSE-MODELS.md) | "Urban OpenGen generator by Differential, trained with NVIDIA StyleGAN2-ADA" |
| **Dataset v2** | city rasters and tiles rendered from OpenStreetMap, and the field caches built from them | [Open Database License 1.0](LICENSE-DATA.md) | "© OpenStreetMap contributors, rendered by Differential for Urban OpenGen" |
| **Dataset v1** | the original Grasshopper tiles, 100 px, what the GAN was trained on | [Open Database License 1.0](LICENSE-DATA.md) | "© OpenStreetMap contributors, rendered by Differential for Urban OpenGen" |
| **Third-party** | three.js vendored in `phase4/vendor/` (MIT); NVIDIA's `stylegan3` is not redistributed, `phase1/apply_patches.py` modifies a clone you make yourself under NVIDIA's own licence | their own | see `NOTICE` |

How to credit us in one line, anywhere the work is used or shown:

    Urban OpenGen by Differential (differential.studio), github.com/differential-studio/urban-opengen

`CITATION.cff` at the root gives the same in a form GitHub and Zenodo read.

What is and is not in the public repository is listed in `RELEASE.md`. This file is a summary; the licence texts govern.
