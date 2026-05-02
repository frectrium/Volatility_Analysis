# Hard Price-Domain Constraints vs. Native Volatility-Domain Models for the IV Surface

A BTech mini-project by **Parshv Joshi**, mentored by **Prof. Abhishek Tilva**.

An empirical comparison of four families of S&P 500 implied-volatility (IV) surface
models — ISNN, HyperIV, HyperISNN (ours) and SSVI — under one preprocessing
pipeline, one arbitrage filter, one train/val/test split, and one Phase-1 / Phase-2
evaluation protocol. The full write-up is `report/main.tex`.

## Pipelines

| Code | Name | Family | What it does |
|------|------|--------|--------------|
| **A** | ISNN-2 | hard-constrained price-domain net | per-day input-specified network on normalised call prices; butterfly + calendar arbitrage-free by construction |
| **B** | HyperIV | volatility-domain hyper-network | Set-Transformer encodes the day's quotes; a hyper-net emits the weights of a tiny IV-MLP. Trained once. |
| **C** | HyperISNN *(novel)* | price-domain hyper-network | residual hyper-net that emits the strictly-positive weights of an ISNN-2 |
| **D** | SSVI | parametric arbitrage-free surface | Gatheral–Jacquier 5-parameter surface, L-BFGS-B daily, 10 restarts |

For each pipeline we run:
- **Phase 1** — same-day fit at the *exact* (k, T) of every test-day market quote, in IV and dollar-price space.
- **Phase 2** — train one VolGAN per pipeline on its training-day surfaces, then forecast next-day (12, 11) IV grids.

## Headline result (test, 2019)

SSVI is the most accurate (test IV RMSE 0.00774) but pays a daily L-BFGS fit;
HyperIV is a close second (0.0131) and is roughly **500× faster at inference**
than per-day SSVI calibration. The two price-domain pipelines (ISNN, HyperISNN)
are arbitrage-free by construction but materially less accurate, even in their
*native* price domain — inverting through Black–Scholes to IV and back to price
gives smaller dollar errors than reading prices off the network directly.
HyperISNN is less accurate than ISNN but two orders of magnitude faster,
illustrating the general amortisation argument for hyper-networks.

## Repository layout

```
main_project/
├── code_files/
│   ├── shared/                  # preprocessing + Moussa arbitrage filter
│   ├── pipeline_A_isnn/         # A — ISNN-2 (per-day, price domain)
│   ├── pipeline_B_hyperiv/      # B — HyperIV (Set-Transformer + hyper-MLP)
│   ├── pipeline_C_hyperisnn/    # C — HyperISNN (ours)
│   ├── pipeline_D_ssvi/         # D — SSVI parametric surface
│   └── shared_eval/             # unified dataset + cross-pipeline runner + plots
├── data/                        # derived caches only — see data/README.md
│   ├── README.md                # ← read this for the data contract
│   └── unified/                 # built on first run of run_all.ipynb
├── volgan/                      # Cont–Vuletić VolGAN (Phase-2 forecaster)
├── papers/                      # the four reference papers
├── report/                      # LaTeX write-up + slides
├── plots/                       # every figure used in the report
├── uml_diagrams/                # PlantUML diagrams of each pipeline
└── run_all.ipynb                # ← single entry point: train + evaluate + plot
```

## Data

Raw CSVs live **outside** the project tree, in `../data_csv/` (relative to
`main_project/`). See [`data/README.md`](data/README.md) for the schema and the
exact filenames the loader expects. Everything under `data/unified/` is fully
reproducible and is gitignored in spirit — delete and rerun.

## Running

From `main_project/`:

```bash
jupyter notebook run_all.ipynb
```

Run all cells. The notebook will:

1. read CSVs from `../data_csv/`,
2. build the unified Moussa-filtered + BS-inverted quote set,
3. train (or load from cache) pipelines A/B/C/D,
4. score Phase 1 and Phase 2,
5. write `data/unified/results/summary.md` and every figure into `plots/`.

Caching is automatic: re-running picks up where the previous run left off.
For a fast smoke test, set `QUICK = True` in the configuration cell — caches
land under `data/unified/quick/` and don't clobber the full-run artefacts.

## References

- Moussa (2025), *Arbitrage filtering of option prices: a simple real-time approach* — `papers/Moussa, K. (2025) - ...pdf`
- Gatheral & Jacquier (2014), *Arbitrage-free SVI volatility surfaces*
- Jadoon et al. (2025), *Input-Specified Neural Networks (ISNN-2)* — `papers/2503.00268v1.pdf`
- Yang & Jacquier (2025), *HyperIV* — `papers/271_HyperIV_Real_time_Implied_.pdf`
- Cont & Vuletić (2025), *VolGAN* — `papers/2411.12854v1.pdf`
