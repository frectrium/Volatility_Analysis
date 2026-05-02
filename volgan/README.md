# VolGAN (vendored copy, adapted for this project)

This directory is a **third-party copy** of the reference implementation that
accompanies

> Vuletić, Lassance & Cont (2024–25), *VolGAN: a generative model for
> arbitrage-free implied volatility surfaces*,
> *Applied Mathematical Finance* —
> <https://www.tandfonline.com/doi/full/10.1080/1350486X.2025.2471317>

The code (`VolGAN.py`, `VolGAN-example.py`, `datacleaning.py`) was authored
by the paper's authors and was obtained from their public repository. It is
**not original work of this project**. We have vendored it here, unmodified
in substance, so that the Phase-2 forecasting experiments in
`code_files/shared_eval/volgan_adapter.py` are reproducible without an
external clone step.

## Purpose in this project

`code_files/shared_eval/volgan_adapter.py` imports the training utilities
from `VolGAN.py` and trains one VolGAN per pipeline (A / B / C / D) on
that pipeline's surfaces. The adapter is the only file in `code_files/`
that touches this directory; nothing here was rewritten for our
experiments.

## Licensing and use

The upstream repository did not ship a `LICENSE` file at the time it was
copied here. Accordingly, **all rights to the contents of this directory
remain with the original authors**. This snapshot is included strictly
for **non-commercial academic use** in the context of the BTech
mini-project that owns this repository — i.e. as a citation-grade
reproduction artefact under the customary fair-use / fair-dealing
allowance for research, teaching, and scholarship. If you are reading
this and intend to use VolGAN beyond reproducing the experiments in this
project, please obtain it directly from the authors and observe whatever
licence terms they have since published.

We claim no authorship and no copyright over any file in this directory.
The original paper is the canonical citation; please cite it rather than
this repository if you build on the VolGAN code.

## Files

| File | Origin | Modified here? |
|---|---|---|
| `VolGAN.py` | upstream | no |
| `VolGAN-example.py` | upstream | no |
| `datacleaning.py` | upstream | no |
| `README.md` | this file | written for this project |

The original README content (data-format notes from the upstream authors)
is preserved below for reference.

---

## Original upstream notes (verbatim)

VolGAN.py contains the necessary functions to train VolGAN, alongside
arbitrage penalty calculation functions. VolGAN-example.py is an example
of how to use the file and check the arbitrage penalties in the simulations.

datacleaning.py contains functions used for cleaning up and extracting
implied volatility data from the Option Prices file downloaded from
OptionMetrics. It also contains smoothing (Nadaraya-Watson and
vega-weighted Nadaraya-Watson) functions (and interpolation functions).

Description of the .csv files:
- `datapath` contains `data.csv`, the data file downloaded from
  OptionMetrics Implied Volatility Surface File.
- `surfacepath` contains `surfaces_transform.csv`, which has daily
  implied volatility surfaces on a pre-defined `(m, tau)` grid, in vector
  form ('flattened', use the `detangle_kt` and `entangle_kt` functions
  for this).

The original authors note that they cannot share the pre-processed data;
all data is downloaded from OptionMetrics. Two types of implied
volatility data sets can be downloaded from OptionMetrics, and both can
be utilised for VolGAN (option prices or implied vol surface, which is
pre-smoothed). Processing the raw data to reach a suitable format is
relatively straightforward.

Advice on data processing (from the original authors):
- If opting for the Option Prices file, make sure to use options which
  have a non-zero volume traded.
- The repo has `.py` files which clean and prepare the data from the
  Option Prices file.
- Implied Volatility Surface file has implied vols on a fixed
  `(delta, tau)` grid and is pre-smoothed, so fewer manipulations are
  required.
- To reach a fixed `(m, tau)` grid, some interpolation/extrapolation is
  necessary.
