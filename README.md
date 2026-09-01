# Jet tagging with geometric deep learning

Are symmetry-based inductive biases useful for jet tagging?
This repository contains a series of experiments attempting to answer this question.
Three different ways of teaching symmetries to a neural network are studied: data augmentation, symmetry-consistency loss (SCL),
and embedded symmetries at the level of the architecture.
Data augmentation transforms the jets the network sees during training.
The symmetry-consistency loss (SCL) penalises the change in the logits under a transformation of the input.
The embedded symmetries make the networks that possess them naturally invariant under a transformation, by construction.
Three architectures are used to study the first two ways of learning symmetries: an MLP, a DeepSets, and a JetFormer.
The embedded symmetries are studied separately, through implementations of LorentzNet (add other architectures when implemented).

All of the above architectures are trained on the hls4ml LHC jet data set at 16, 50 and 150 constituents.
The augmentation and SCL methods are applied under three groups (SO(2) rotations about the beam axis, SO(3) rotations, and Lorentz boosts in a random
direction with rapidity up to 2).
The invariance each acquires is measured on the test data set.
The boosts are deliberately not restricted to the beam axis: an isotropic boost is not a
symmetry of a pp collision, so it probes a property of the model rather than of the data.

## Setup

Dependencies are managed with [Poetry](https://python-poetry.org/).
From the repository root:

```bash
poetry install --no-root
```

Alternatively, `pip install -r requirements.txt` installs the same pinned set for anyone not using Poetry.
Both routes take `torch` from the CUDA 12.6 index declared in `pyproject.toml`.

Next, create the working directories and the `.env` file:

```bash
bash scripts/setup.sh
```

The `configs/paths/default.yaml` configuration file resolves `${oc.env:PROJECT_ROOT}`  out of `.env`.
The file also has `HF_HUB_CACHE`, which sets where the record snapshot is downloaded, and `MLFLOW_ALLOW_FILE_STORE=true`, without which
MLflow 3 refuses the file-based tracking store the experiments log to.
Re-running the script never overwrites an existing `.env`.
To keep `data/`, `logs/`, `outputs/` and `results/` on another filesystem, edit `RES_DIR` in `.env` and run `bash scripts/symbolink.sh`,
which replaces those four directories with symlinks.
That step deletes any real directory in the way, so run it before the first training.

## Data

We use the [hls4ml LHC jet data set](https://doi.org/10.5281/zenodo.3602260) (Zenodo record 3602260,
CC-BY-4.0) HuggingFace mirror and loader [`fastmachinelearning/hls4ml_lhc_jets_150p`](https://huggingface.co/datasets/fastmachinelearning/hls4ml_lhc_jets_150p).
A run needs `HF_HUB_CACHE` pointing somewhere with room for the 3 GB snapshot.

Because the record carries executable code, `data.revision` pins the commit that a run reads.
Setting it to `null` follows the head of the record instead, so what a run trained on then depends on when it was launched.
The loader restricts each jet to `data.nconstituents` leading constituents by pT and normalises the features.
The result is cached under `data/hf/nconst_<N>/`.

The `data.nconstituents` parameter is set to 16, 50 or 150 in the experiments.
The `data.features` parameter takes any subset of the record's 16 columns and defaults to `[px, py, pz, e]`.
LorentzNet requires exactly those four, in that order.

## Running

The entry nodes of the code are in `src/train.py` for the training and `src/eval.py` for the testing.
These are configured through an experiment config:

```bash
python src/train.py experiment=hls4ml/deepsets_16 trainer=gpu trainer.devices=[0]
```

The method and the symmetry group are command-line overrides in the scripts.
One experiment config covers all seven regimes over which the first two classes, data
augmentation and the SCL, are compared:

```bash
# Baseline: nothing is transformed anywhere.
python src/train.py experiment=hls4ml/deepsets_16
# Augmentation: training batches are transformed online, the loss is unchanged.
python src/train.py experiment=hls4ml/deepsets_16 data.augmentation_scheme=so2
# SCL: batches are untransformed, the loss gains a symmetry-consistency term.
python src/train.py experiment=hls4ml/deepsets_16 \
    data.method=symm_loss data.augmentation_scheme=so3
```

`data.method` is `augmentation` or `symm_loss` and `data.augmentation_scheme` is `none`,
`so2`, `so3` or `lorentz`.
The algorithm configs interpolate both.
A single override configures the data and the model together and they cannot disagree.

The seven regimes are run for the MLP, the DeepSets and the JetFormer.
LorentzNet constitutes the symmetry embedded class so it is trained in the baseline regime only, at 16, 50 and 150 constituents.
The same test-time invariance probes run on those baseline runs, which is what puts the embedded class
on the same axis as the other two, and its rows land in `results_table_baseline.csv` beside
every other unaugmented run.

`data.n_folds` sets the cross-validation. The experiment configs use 5, the searches and the
debug configs 1. With `n_folds >= 2`, `train.py` fits the main run on the record's own split
first and then one run per fold, on a pool of the record's train and validation splits; the
test split is never pooled. Each run writes its checkpoints under
`<hydra run dir>/checkpoints/{main,fold_<i>}/` and its figures under the matching
`<hydra run dir>/plots/{main,fold_<i>}/`, and the run as a whole writes
`checkpoint_manifest.json` beside them, listing every checkpoint with its `val/acc_best`.
The number `train.py` returns, and therefore the number an Optuna study maximises, is the
main run's `val/acc_best`: the clean accuracy, never an invariance metric.

`eval.py` reads that manifest and tests every checkpoint it names, re-seeding before each so
that the folds see the same transform draws:

```bash
python src/eval.py experiment=hls4ml/deepsets_16 \
    checkpoint_manifest_path=logs/train/hls4ml/deepsets/nconst16/baseline/checkpoint_manifest.json
```

A single checkpoint can be tested with `ckpt_path=...` instead. Either way the run appends
one row to `results/<features>/results_table_{baseline,augmentation,symm_loss}.csv`, where
the directory is the feature list (`px_py_pz_e` by default) and the file is the regime. The
row holds the test accuracy over the folds, the invariance metrics per group (the Wasserstein
distance between clean and transformed logit distributions, the flip rate, and its
recovered/stable/degraded breakdown) and the model's forward FLOP count. That count is taken
on a fully occupied jet, since the models are handed a fixed-width tensor; for LorentzNet,
whose cost follows the number of constituent pairs actually present, it is therefore an upper
bound, and a loose one at 150 constituents.

## Hyperparameter search

There is one TPE study per (architecture, constituent count, regime), 66 in all: the MLP, the
DeepSets and the JetFormer over three constituent counts and seven regimes, plus the three
LorentzNet baseline studies. Every study has the same space and the same budget, so that the
regimes stay comparable:

```bash
python src/train.py -m \
    experiment=hls4ml/deepsets_16 \
    hparams_search=hls4ml/deepsets_scl_optuna \
    data.method=symm_loss data.augmentation_scheme=so2 \
    data.n_folds=1 callbacks=search logger=none
```

The study name is derived from the regime inside the search config, so nothing names it on
the command line, and the databases are written to `logs/optuna/hls4ml/<model>.db`. The
`_scl_optuna` variant of a config adds `algorithm.symm_loss_weight` to the space and serves
the three SCL regimes; the other four regimes, and every LorentzNet study, use the plain
`<model>_optuna`. Under `callbacks=search` a trial draws no plots and never touches
`results/`.

The best trial of a finished study is read back as Hydra overrides and as a YAML block to
paste into an experiment config:

```bash
python scripts/optuna/fetch_best_params.py \
    --db logs/optuna/hls4ml/deepsets.db --study deepsets_nconst16_symm_loss_so2
```

## Scripts

`scripts/` holds the exact commands behind every result, one catalogue of commented,
copy-pasteable invocations per architecture and stage, plus the tooling that generates them,
submits them and harvests the studies. See [`scripts/README.md`](scripts/README.md).

## Analysis

Once every run of a feature set has been evaluated:

```bash
python scripts/analysis/add_pvalues.py results/<features>/results_table_*.csv
python scripts/analysis/make_figures.py --features <features>
```

The first adds the deltas against the baseline, with bootstrap confidence intervals and
p-values, writing `*_pvalues.csv` beside each table. The second turns those into the
comparison figures.

## Layout

```
configs/          Hydra configs; one group per subdirectory, train.yaml and eval.yaml at the root
src/
├── train.py      fit the main run and the folds, write the checkpoint manifest
├── eval.py       test every checkpoint the manifest names
├── algorithms/   JetTaggingModule and its four subclasses, their components and losses
├── data/         the datamodule, its dataset and the augmentation transforms
├── callbacks/    invariance probes, results table, plots, throughput
├── plot/         the figure families the plotting callback draws
└── utils/        resolvers, checkpoint manifest, instantiation and logging helpers
scripts/          run catalogues, cluster submitters, Optuna tooling, analysis scripts
tests/
results/          results tables and figures, untracked
```

## Licence

The code is MIT; see [`LICENSE`](LICENSE). The data set is distributed separately under
CC-BY-4.0.
