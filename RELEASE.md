# What is published

This repository is the pipeline and the configurator: everything needed to build the
data, train both models, evaluate them, and run the interactive tool. The models and the
data themselves are downloads, because weights and rasters do not belong in git. See
`DOWNLOADS.md`.

## In the repository

    README.md, LICENSE, LICENSE.md, LICENSE-MODELS.md, LICENSE-DATA.md, NOTICE,
    CITATION.cff, RELEASE.md, DOWNLOADS.md, requirements.txt
    assets/     the decoding metadata the viewer needs: height LUT, city labels, slider directions
    phase0/     tile codec, OpenStreetMap renderer, city lists, batch renderer, audits, re-encoder
    phase1/     StyleGAN2-ADA patches, dataset class, direction fitting, Windows build helpers
    phase3/     evaluation against the real corpus
    phase4/     the configurator: viewer (index.html, regular.js, vendor/) and inference server (serve.py)
    phase6/     the diffusion inpainter: prepare, data, unet, diffusion, train, sample, cloud recipe
    phase7/     analysis and the design space: sun, views, density, and the option grid

The phases are numbered by the technical plan, so there is no `phase2/` and no `phase5/`
here: phase 2 is the slider fitting and lives in `phase1/` because it runs against the
training repo, and phase 5 is the film.

## Released as downloads

The inpainter checkpoints (CC BY 4.0), the GAN snapshots (non-commercial research use),
dataset v1 tiles and dataset v2 rasters (both ODbL). Both datasets go out: v1 is what the
GAN was trained on, v2 is what the inpainter was trained on, and neither model is
reproducible without its own. See the licence files before using either model.

What is **not** a download is `assets/`. Those three files are the height lookup table,
the city label order and the fitted latent directions: a few dozen kilobytes, identical
for every released snapshot, and the viewer does not run without them. They are in the
repository so the download is only ever the weights.

## Not in the repository

    the film          the page that renders it, its narration, its data and the videos
    the archive       the original DCGAN attempts and the Grasshopper definition
    working folders   audits, caches, checkpoints, run directories, rendered tiles

The film is a communication piece rather than a tool. It needs assets and data that make
no sense to a user of the models, and it is © Differential.

## Building this repository from the working folder

`make_release.ps1` on the working drive assembles the published tree, excludes the weights
and the caches, and then checks what is left for machine paths and key-shaped strings. It
is a build script for us and is not itself published.
