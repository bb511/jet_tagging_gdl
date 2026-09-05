# Jet tagging with geometric deep learning

Are symmetry-based inductive biases useful for fast jet tagging?
This repository contains a series of experiments providing insight into this conundrum.
We investigate three different ways of symmetry learning in a neural network: data augmentation, symmetry-consistency loss (SCL),
and embedding a symmetry into the architecture of the network.
Data augmentation involves generating additional copies of the jets by transforming them under a symmetry group; the network can thus understand a symmetry by learnint to map the original and transformed jets to the same output.
The symmetry-consistency loss (SCL) method consists of penalising the change in the logits under a transformation of the input.
The embedded symmetries approach involves designing the neural network architecture such that it is invariant to transformed data by construction.

The data augmentation and SCL methodologies are applicable to any deep learning model.
Hence, we constrain our study to three typical model architectures ubiquitous in jet tagging:
a multi-layer perceptron, a [deep sets](https://arxiv.org/abs/1703.06114) network, and a [JetFormer](https://arxiv.org/abs/2601.17215).
We study three symmetry groups: SO(2) rotations about the beam axis, isotropic SO(3) rotations, and Lorentz boosts along the beam axis.
The first and last are symmetries of the CMS and ATLAS experiments at the LHC, while the middle one is not and is used as a baseline.
Meanwhile, the embedded symmetry approach is studied by implementing the following architectures:
[LorentzNet](https://arxiv.org/abs/2201.08187), [Nano-PELICAN](https://arxiv.org/abs/2211.00454), [LGATr](https://arxiv.org/abs/2405.14806), (maybe Heidelberg new paper).
The invariance of each trained model is quantified under their respective symmetry transformation.

We perform our investigation on three different data sets:

- [hls4ml LHC jet data set](https://doi.org/10.5281/zenodo.3602260) at 16, 50 and 150 constituents.
- [top tagging data set](https://doi.org/10.5281/zenodo.2603256).
- [JetClass data set](https://doi.org/10.5281/zenodo.6619768).

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

The `configs/paths/default.yaml` configuration file resolves `${oc.env:PROJECT_ROOT}` out of `.env`.
The file also has `HF_HUB_CACHE`, which sets where the HuggingFace data is downloaded.
Additionally, `MLFLOW_ALLOW_FILE_STORE=true` is set there, without which MLflow 3 refuses the file-based tracking store that the experiments log to.
Configure where `data/`, `logs/`, `outputs/` and `results/` go by editing `RES_DIR` in `.env` and then run `bash scripts/symbolink.sh`.

## Data

We use the [hls4ml LHC jet data set](https://doi.org/10.5281/zenodo.3602260) (Zenodo record 3602260,
CC-BY-4.0) HuggingFace mirror and loader [`fastmachinelearning/hls4ml_lhc_jets_150p`](https://huggingface.co/datasets/fastmachinelearning/hls4ml_lhc_jets_150p).

HuggingFace mirrors TBD for the other two data sets.

The `data.revision` config attribute pins the HuggingFace commit for the data that is stored there; setting it to `null` gives the most recent commit.

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
`so2`, `so3`, `lorentz` or `lorentz_iso`; the scripted grid uses the first four.
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

The deltas against the baseline are computed by the evaluation, not afterwards. An
evaluation of a run trained under augmentation or the SCL takes a second manifest, the
baseline run of the same architecture and constituent count:

```bash
python src/eval.py experiment=hls4ml/deepsets_16 \
    data.method=augmentation data.augmentation_scheme=so2 \
    checkpoint_manifest_path=logs/train/hls4ml/deepsets/nconst16/aug_so2/checkpoint_manifest.json \
    baseline_checkpoint_manifest_path=logs/train/hls4ml/deepsets/nconst16/baseline/checkpoint_manifest.json
```

Both models then see the same transformed test batches and the results table gains the
delta, bootstrap confidence-interval and p-value columns. A baseline run, and LorentzNet,
which is trained in the baseline regime alone, name no baseline and hold NaN there.

Once every run of a feature set has been evaluated:

```bash
python scripts/analysis/make_figures.py --features <features>
python scripts/analysis/make_latex_table.py results/<features>/results_table_*.csv
```

The first draws the comparison figures. The second is optional and purely presentational:
it renders whatever columns a results table already holds as a LaTeX table beside it.

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
