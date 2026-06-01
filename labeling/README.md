# FinBERT Aspect Classifier

Multi-label aspect classifier for financial text. Trained on 3,450 sentences with probabilistic labels from a Snorkel pipeline.

## Aspects (6-class multi-label)
| Aspect | Covers |
|---|---|
| Monetary & Financial | Interest rates, central banks, liquidity, credit, markets |
| Inflation & Prices | CPI, PPI, wages, commodity-driven price pressures |
| Real Economic Activity | GDP, recession, manufacturing, investment, productivity |
| Labor & Consumption | Employment, wages, retail sales, consumer spending |
| Fiscal & Government | Taxes, deficits, spending, regulation, stimulus |
| External Sector | Trade, exports/imports, FX, tariffs, current account |

## Model
- **Fine-tune:** `beethogedeon/Modern-FinBERT-large` (ModernBERT-large + financial corpus)
- **Baseline:** `ProsusAI/finbert` (BERT-base + financial corpus)
- Both use a 6-way sigmoid head with masked soft BCE loss

## Repo structure
```
├── config.py              # all hyperparameters
├── train.py               # entry point
├── colab_runner.ipynb     # Colab launcher notebook
├── requirements.txt
├── data/
│   └── dataset.py         # cleaning, splits, Dataset, DataLoaders
├── models/
│   └── model.py           # AspectModel (shared), freeze helpers
└── utils/
    ├── loss.py            # MaskedSoftBCE with pos_weight
    ├── trainer.py         # 3-phase loop, early stopping, checkpointing
    ├── metrics.py         # per-aspect F1, threshold tuning, comparison table
    └── plot.py            # training curves + comparison bar charts
```

## Running on Colab
1. Open `colab_runner.ipynb` via your GitHub connection in Colab
2. Set your repo URL in Cell 1
3. Runtime → Change runtime type → **T4 GPU**
4. Run all cells top to bottom

## Running locally
```bash
pip install -r requirements.txt
python train.py --data path/to/labels.csv --output_dir outputs/
python utils/plot.py --output_dir outputs/
```

## Data format
CSV with a `text` column and exactly these 6 columns (Snorkel probabilities in [0, 1]):
```
text, Monetary_Financial, Inflation_Prices, Real_Economic_Activity,
      Labor_Consumption, Fiscal_Government, External_Sector
```

## Key design decisions
- **Soft labels** — BCE treats Snorkel probabilities as fractional targets, preserving label uncertainty
- **Abstain masking** — labels in [0.35, 0.55] are excluded from loss (Snorkel prior zone)
- **Fiscal pos_weight = 8** — hard prevalence only 6.5%; capped from raw ~14.4
- **3-phase training** — head warmup → top-4 layers → optional full fine-tune
- **Per-aspect threshold tuning** — each aspect gets its own sigmoid threshold optimised on val set
