# Data layout

The pipelines train on the SPX option-chain dataset of Ning Ning et al.
The raw CSVs are kept **outside** the project tree (they are large and
non-redistributable) and are read from `../data_csv/` relative to the
project root, i.e.:

```
Volatility_Analysis/
├── data_csv/                ← raw CSVs live here
│   ├── op_df.csv            ~1.6 GB   per-quote OptionMetrics records
│   ├── St_df.csv            ~50 KB    daily SPX spot
│   ├── fwd_df.csv           ~280 KB   forward prices
│   ├── discount_df.csv      ~280 KB   risk-free discount factors
│   ├── list_exp.csv         tiny      grid of maturities (days)
│   └── list_mny.csv         tiny      grid of moneyness levels
└── main_project/
    └── data/                ← this folder, holds derived caches only
        └── unified/         ← built on first notebook run
            ├── quotes.pkl       arb-filtered + BS-inverted quote dict
            ├── models/          per-pipeline trained weights
            ├── surfaces/        materialised (12,11) IV grids per pipe
            ├── volgan_ckpts/    one VolGAN per pipeline
            └── results/         phase1/phase2 metric JSON + summary.md
```

## Required schemas (`../data_csv/`)

| File | Index | Columns | Notes |
|---|---|---|---|
| `op_df.csv` | row id | `date, exdate, strike, cp_flag, midP, volume, ...` | per-option daily quotes; `cp_flag = 'C'` for calls |
| `St_df.csv` | date | `Spot` (or first numeric col) | underlying close |
| `fwd_df.csv` | date | one column per maturity | forward prices |
| `discount_df.csv` | date | one column per maturity | discount factors |
| `list_exp.csv` | row id | `Days` | maturities used as the grid axis |
| `list_mny.csv` | row id | `Moneyness` | moneyness used as the grid axis |

The header on `St_df`, `fwd_df`, `discount_df` is a date in `YYYY-MM-DD`
format and is parsed with `parse_dates=True`.

## How the pipelines find these files

`code_files/shared_eval/unified_dataset.py` defines

```python
csv_dir = "../data_csv"   # relative to main_project/
```

Override at call time:

```python
from shared_eval.unified_dataset import load_unified_dataset
payload = load_unified_dataset(
    rebuild=True,
    csv_dir="/absolute/path/to/your/csvs",
)
```

## Regenerating `data/unified/`

Everything under `data/unified/` is **fully reproducible** by running
`run_all.ipynb` from the project root. Delete it any time and re-run the
notebook to rebuild. Date splits are fixed: train 2013–2017, val 2018,
test 2019.
