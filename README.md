# Jet tagging with geometric deep learning

Two ways of teaching a jet tagger a symmetry are compared here. Data augmentation transforms
the jets the network sees during training and leaves the loss alone; a symmetry-consistency
loss (SCL) leaves the batches alone and penalises the change in the logits under a
transformation of the input. Four architectures (DeepSets, MLP, JetFormer and LorentzNet)
are trained on the hls4ml LHC jet data set at 16, 50 and 150 constituents, under three
groups (SO(2) rotations about the beam axis, SO(3) rotations, and Lorentz boosts in a random
direction with rapidity up to 2), and the invariance each acquires is measured on the test set.
The boosts are deliberately not restricted to the beam axis: an isotropic boost is not a
symmetry of a pp collision, so it probes a property of the model rather than of the data.

The stack is [Lightning](https://lightning.ai/) for the training loop and
[Hydra](https://hydra.cc/) for config composition, laid out as in the
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).

Architectures studied earlier and removed from this codebase, with the recipe for putting
them back, are recorded in [`ARCHITECTURES.md`](ARCHITECTURES.md).

## Setup

Dependencies are managed with [Poetry](https://python-poetry.org/). From the repository
root:

```bash
poetry install --no-root
```

`pip install -r requirements.txt` installs the same pinned set for anyone not using Poetry.
That file is generated, not edited by hand:

```bash
poetry self add poetry-plugin-export    # once, if poetry export is not available
poetry export -f requirements.txt --without-hashes -o requirements.txt
```

Both routes take `torch` from the CUDA 12.6 index declared in `pyproject.toml`.

Next, create the working directories and the `.env` file:

```bash
bash scripts/setup.sh
```

Nothing composes without it, since `configs/paths/default.yaml` resolves
`${oc.env:PROJECT_ROOT}` out of `.env`. The file also carries `HF_HUB_CACHE`, which sets
where the record snapshot is downloaded, and `MLFLOW_ALLOW_FILE_STORE=true`, without which
MLflow 3 refuses the file-based tracking store the experiments log to. Re-running the script never overwrites an existing
`.env`. To keep `data/`, `logs/`, `outputs/` and `results/` on another filesystem, edit
`RES_DIR` in `.env` and run `bash scripts/symbolink.sh`, which replaces those four
directories with symlinks. That step deletes any real directory in the way, so run it before
the first training rather than after; it asks for confirmation unless given `--force`.

## Data

The [hls4ml LHC jet data set](https://doi.org/10.5281/zenodo.3602260) (Zenodo record 3602260,
CC-BY-4.0) holds jets of five classes with up to 150 constituents each, 620 000 of them for
training and validation and 260 000 for testing. They are read through the record
[`fastmachinelearning/hls4ml_lhc_jets_150p`](https://huggingface.co/datasets/fastmachinelearning/hls4ml_lhc_jets_150p),
which is the only route this repository supports. The record ships its own loader, and a run
executes that loader, so no data path is configured anywhere; what a run needs instead is
`HF_HUB_CACHE` pointing somewhere with room for the 3 GB snapshot.

Because the record carries executable code, `data.revision` pins the commit that a run
reads. Setting it to `null` follows the head of the record instead, so what a run trained on
then depends on when it was launched.

The loader restricts each jet to `data.nconstituents` leading constituents by pT, normalises
the features, and caches the result under `data/hf/nconst_<N>/`. It tests for that cache by
existence alone, so a changed `data.revision` rebuilds nothing: point `data.cache_dir` at a
fresh directory when the record itself changes.

`data.nconstituents` takes 16, 50 or 150 in the experiments (150 is every constituent the
record carries). `data.features` takes any subset of the record's 16 columns and defaults to
`[px, py, pz, e]`; LorentzNet requires exactly those four, in that order. The test split
keeps all 16 columns whatever the model reads, so the invariance probes can find the momenta
they transform.

## Running

`src/train.py` fits and `src/eval.py` tests. Both are driven by an experiment config, one
per (architecture, constituent count) pair:

```bash
python src/train.py experiment=hls4ml/deepsets_16 trainer=gpu trainer.devices=[0]
```

The method and the symmetry group stay command-line overrides, so one experiment config
covers all seven regimes of the comparison:

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
`so2`, `so3` or `lorentz`. The algorithm configs interpolate both, so a single override
configures the data and the model together and they cannot disagree.

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

There is one TPE study per (architecture, constituent count, regime), 84 in all, each with
the same space and the same budget so that the regimes stay comparable:

```bash
python src/train.py -m \
    experiment=hls4ml/deepsets_16 \
    hparams_search=hls4ml/deepsets_scl_optuna \
    data.method=symm_loss data.augmentation_scheme=so2 \
    data.n_folds=1 callbacks=search logger=none
```

The study name is derived from the regime inside the search config, so nothing names it on
the command line, and the databases are written to `logs/optuna/hls4ml/<model>.db`. The
`_scl_optuna` variant of each config adds `algorithm.symm_loss_weight` to the space and
serves the three SCL regimes; the other four use the plain `<model>_optuna`. Under
`callbacks=search` a trial draws no plots and never touches `results/`.

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
