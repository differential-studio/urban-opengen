# Model weights

## Diffusion inpainter (`phase6/`)

Checkpoints trained with the code in `phase6/` (`latest.pt`, `ckpt_*.pt`) are released
under the **Creative Commons Attribution 4.0 International** licence (CC BY 4.0).

You may share and adapt them, including commercially, on one condition: credit.
Attribution must name the work, the licensor and the licence, and link to the source,
for example:

    Urban OpenGen inpainter, Differential (differential.studio), CC BY 4.0,
    github.com/differential-studio/urban-opengen

Full text: https://creativecommons.org/licenses/by/4.0/legalcode
Summary:   https://creativecommons.org/licenses/by/4.0/

The inpainter is trained from scratch with our own code and has no dependency on any
third-party model licence. Its training data is dataset v2 (OpenStreetMap, see
`LICENSE-DATA.md`); weights are not a database and do not carry the ODbL, but OpenStreetMap
deserves the credit line above alongside ours.

## GAN generator (StyleGAN2-ADA runs, `network-snapshot-*.pkl`)

These weights were produced with NVIDIA's `stylegan3` repository, which is released under
the NVIDIA Source Code License for StyleGAN3 and permits **non-commercial research use
only** of the software and works derived from it. We read that as covering models trained
with it. The GAN generator snapshots are therefore released for **non-commercial research
and educational use only**, with credit to both Differential and NVIDIA:

    Urban OpenGen generator, Differential (differential.studio), trained with
    NVIDIA StyleGAN2-ADA (github.com/NVlabs/stylegan3), non-commercial research use only

If you need a commercially usable generator, train the diffusion model instead, or contact
Differential. This is our reading of NVIDIA's licence, not legal advice; check it against
the licence text for your own case.

## Outputs

Tiles, geometry and images generated with either model are yours. We ask, and CC BY
requires for the inpainter, that published results credit Urban OpenGen.
