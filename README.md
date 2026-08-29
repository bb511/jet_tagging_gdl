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

**1. Install dependencies** — Python 3.11 venv (matches the `HYDRA` venv the SLURM launcher
activates, see `configs/hydra/launcher/submitit_slurm_euler.yaml`):

```bash
python3.11 -m venv HYDRA
source HYDRA/bin/activate
pip install -r requirements.txt
```

(`requirements.txt` pins CUDA 12.6 PyTorch wheels — swap the `--extra-index-url` line at the top
if your cluster needs a different CUDA version.)

**2. Set the dataset path** in `configs/data/hls4ml.yaml` (`data_dir`), or override on the CLI
with `data.data_dir=...` / `$HLS4ML_DATA_DIR` as used in the SLURM scripts.

## Data

The [hls4ml LHC jet dataset](https://doi.org/10.5281/zenodo.3602260) (150 particles per jet),
Zenodo record 3602260, CC-BY-4.0. Two routes read it and end at the same tensors, with one
known exception: the legacy route normalises its test split with the scales as they come back
off `norm_params.npz` (rounded to float32), the record with the scales as fitted, so the test
`x` tensors differ by one float32 ulp in some entries (max |diff| 1.2e-7). Pick a route with
the `data=` group.

**`data=hls4ml` (legacy).** Downloads the two Zenodo tarballs into `data.data_dir`, reads the
`jetImage_*_150p_*.h5` files, and caches the pT-sorted, normalised arrays under
`<data_dir>/processed/`:

```bash
python src/train.py data=hls4ml data.data_dir="$HLS4ML_DATA_DIR"
```

**`data=hls4ml_hf` (published record).** Reads `fastmachinelearning/hls4ml_lhc_jets_150p` through the loader
shipped inside the record, so it needs no data path. Point `HF_HUB_CACHE` at somewhere with room:
the first run downloads the record (a few GB) and builds its caches under `data/hf/`, later runs
reuse both.

```bash
export HF_HUB_CACHE=/scratch/$USER/hf_cache
python src/train.py data=hls4ml_hf
```

Setting `data.repo_id` to a local directory reads a mirror of the record without going through
the hub, and `data.revision` pins a record revision.

**Parity.** `scripts/analysis/check_hf_parity.py` surveys both routes in separate child processes
and compares the train/val/test `x` and `y` tensors by sha256 over shape, dtype and bytes, along
with `feature_names`, `norm_params` and the split sizes, for three configurations (single split,
5-fold, `symm_loss`). It also prints the wall clock and peak RSS of each route.

```bash
python scripts/analysis/check_hf_parity.py \
    --legacy-dir "$HLS4ML_DATA_DIR" \
    --repo fastmachinelearning/hls4ml_lhc_jets_150p \
    --hf-cache data/hf \
    --work /tmp/parity_work
```

## Order of running things

**1. Train — submit jobs to SLURM.** Every run goes through Hydra's
submitit launcher (`hydra/launcher=submitit_slurm_euler` or `..._clariden`), which submits an
`sbatch` job and returns immediately — the `python src/train.py ...` call is just the *submission*
command, the actual training happens later on a GPU node.

Edit `scripts/run_euler_array.sh` (or `run_clariden_array.sh` on Clariden) to uncomment the
experiment blocks you want — one block per model x `nconstituents` x symmetry group x method —
then submit them all from a login node:

```bash
bash scripts/run_euler_array.sh
```

Each block looks like this (model, `nconstituents`, features, and log dir change per run):

```bash
python src/train.py -m \
    hydra/launcher=submitit_slurm_euler \
    experiment=symm_loss/hls4ml_symm_loss_nconst16_lorentz \
    model=deepsets_16 \
    trainer=gpu \
    data.data_dir="$HLS4ML_DATA_DIR" \
    hydra.sweep.dir="logs/slurm/pt_etarel_phirel/deepsets/symm_loss_lorentz_nconst16" \
    hydra.job.name="symm_loss_lorentz_deepsets_nconst16"
```

SLURM settings (partition, GPUs, timeout, venv activation) live in
`configs/hydra/launcher/submitit_slurm_euler.yaml`. Check queued/running jobs with
`squeue -u $USER`.

Each finished run appends one row to `results/<features>/results_table_{baseline,augmentation,symm_loss}.csv`
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
