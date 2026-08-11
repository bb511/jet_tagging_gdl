# Jet Tagging with Geometric Deep Learning

Studies how well different architectures (DeepSets, Transformer, MLP, PELICAN, LorentzNet, ...)
learn to be invariant to SO(2)/SO(3)/Lorentz symmetries on the HLS4ML jet-tagging dataset,
and compares two ways of teaching them that invariance: data **augmentation** vs. a
**symmetry-consistency loss (SCL)**.

## Framework

- [PyTorch Lightning](https://lightning.ai/) — training loop / `LightningModule` per model
- [Hydra](https://hydra.cc/) — config composition and CLI overrides
- Built from the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

## File structure

```
configs/
├── data/hls4ml.yaml         <- dataset config (path, features, nconstituents, method, ...)
├── model/                   <- one config per architecture (deepsets_8/16/32, jetformer,
│                                transformer, mlp_8/16/32, pelican*, lorentz_net, ...)
├── callbacks/                  results_table.yaml -> writes results/*/results_table_*.csv
│                                invariance_metric.yaml -> Wasserstein/flip/stable-fraction metrics
├── experiment/
│   ├── baseline/             <- method=none, per nconstituents
│   ├── augmentation/         <- method=augmentation, per nconstituents x {so2,so3,lorentz}
│   ├── symm_loss/            <- method=symm_loss (SCL), per nconstituents x {so2,so3,lorentz}
│   └── permutation/          <- permutation-equivariance check
└── train.yaml                <- main config, composes all of the above

src/
├── train.py                  <- entry point: trains + tests one model/experiment
├── data/hls4ml_datamodule.py <- HLS4ML datamodule (K-fold, on-the-fly augmentation)
├── models/*_module.py        <- one LightningModule per architecture
├── callbacks/
│   ├── invariance_metric_callback.py  <- measures f(x) vs f(T(x)) at test time
│   └── results_table_callback.py      <- appends a row per run to results_table_*.csv
└── utils/
    ├── augmentation.py       <- SO(2)/SO(3)/Lorentz transforms
    ├── symmetry_loss.py      <- SCL loss (L_cls + weight * symmetry_distance)
    ├── add_pvalues.py        <- post-processing: deltas vs. baseline + bootstrap CI/p-values
    └── make_figures.py       <- final comparison figures from the _pvalues.csv tables

scripts/                      <- SLURM launch scripts (Euler / Clariden clusters)
results/<features>/           <- results_table_*.csv, *_pvalues.csv, fig_*.svg per feature set
```

## Setup

```bash
conda env create -f environment.yaml -n jet_tagging
conda activate jet_tagging
# or: pip install -r requirements.txt
```

Set the dataset path in `configs/data/hls4ml.yaml` (`data_dir`), or override on the CLI.

## Order of running things

**1. Train a model.** Pick one of the experiment configs (these set the model, `nconstituents`,
training method, and symmetry group for you):

```bash
python src/train.py experiment=baseline/hls4ml_nconst16_none model=deepsets_16
python src/train.py experiment=augmentation/hls4ml_nconst16_aug_lorentz model=deepsets_16
python src/train.py experiment=symm_loss/hls4ml_symm_loss_nconst16_lorentz model=deepsets_16
```

Any parameter can be overridden from the CLI, e.g. `data.features=[pt,etarel,phirel]`. To launch
the full sweep (all models x all nconstituents x all groups x both methods) on a SLURM cluster,
use `scripts/run_euler_array.sh` (or `run_clariden_array.sh`).

Each run appends one row to `results/<features>/results_table_{baseline,augmentation,symm_loss}.csv`
(via `results_table_callback.py`), with per-fold accuracy and per-group invariance metrics
(Wasserstein distance, stable fraction, recovery rate) already computed at test time.

**2. Add bootstrap statistics.** Once all runs for a feature set are done:

```bash
python src/utils/add_pvalues.py results/<features>/results_table_*.csv
```

This adds bootstrap CIs/p-values and delta columns (trained vs. `none` baseline, e.g.
`delta_wasserstein_mean`, `delta_accuracy_transformed_mean`) and writes `*_pvalues.csv` + `.tex`
next to each input table.

**3. Generate figures.**

```bash
python src/utils/make_figures.py --features <features>
```

Reads `results/<features>/results_table_*_pvalues.csv` and writes
`fig_augmentation_<features>.svg` / `fig_symm_loss_<features>.svg` — one delta-bar row each for
accuracy (clean and under-transform), Wasserstein invariance, and self-consistency, per architecture.
