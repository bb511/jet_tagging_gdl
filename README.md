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

We use the [hls4ml LHC jet data set](https://doi.org/10.5281/zenodo.3602260) (Zenodo record 3602260, CC-BY-4.0) HuggingFace mirror
and loader [`fastmachinelearning/hls4ml_lhc_jets_150p`](https://huggingface.co/datasets/fastmachinelearning/hls4ml_lhc_jets_150p).
The `data.revision` config attribute pins the HuggingFace commit for the data that is stored there; setting it to `null` gives the most recent commit.

For the top tagging and JetClass datasets, we use the Zenodo records, since an official HuggingFace mirror was not available at the moment of conducting our study.

## Training

The entry nodes of the code are in `src/train.py` for the training and `src/eval.py` for the testing.
These are configured through an experiment config:

```bash
python src/train.py experiment=hls4ml/deepsets_16 trainer=gpu trainer.devices=[0]
```

The methodology and the symmetry group are command-line overrides, as follows.
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

The following options can be given the the method:

```bash
data.method=null/augmentation/symm_loss
```

When the method is set to `null`, there's no augmentation or symmetry loss being implemented; this is how the embedded symmetry architectures are ran.
When the method is set to `augmentation`, then `data.augmentation_scheme` needs to also be set to one of the following: `none`, `so2`, `so3`, `lorentz` or `lorentz_iso`.
The experiment config establishes what the data and the model are, because a certain data configuration determines the configuration of the model.

The `data.n_folds` argument enables cross-validation.
When `n_folds >= 2`, the training and validation data sets are merged and resplit following a conventional k-folding procedure.
The test split is never touched during this process.

The checkpoints of each run are written under `logs/<hydra run dir>/checkpoints/{main,fold_<i>}/`.
The training figures, such as the loss curve, are written under `<hydra run dir>/plots/{main,fold_<i>}/`.
A checkpoint manifest is also written in the run directory: `checkpoint_manifest.json`.
This latter `json` files is used at evaluation time to compute the invariance metrics on the checkpointed models.
The training script returns the validation accuracy over the jet classes.

## Evaluation

The evaluation entry point is the `eval.py` script.
This script reads the checkpoint manifest and tests every checkpoint (the same experiment as for the training needs to be specified):

```bash
python src/eval.py experiment=hls4ml/deepsets_16 \
    checkpoint_manifest_path=logs/train/hls4ml/deepsets/nconst16/baseline/checkpoint_manifest.json
```

If you'd rather test a single checkpoint, specify `ckpt_path=...` instead.
After the evaluation is complete for a run, one row is appended to a csv table in `results/<features>/results_table_{baseline,augmentation,symm_loss}.csv`.
For the embedded architectures, the results are funneled to `results/embedded/results_table_baseline.csv` instead.
One row in the produced .csv contains the following information:

- the test accuracy over the folds
- the invariance metrics per group
- the model's forward FLOP count.

The evaluation script can also take a baseline model corresponding to an augmented/SCL model, to compute meaningful statistics:

```bash
python src/eval.py experiment=hls4ml/deepsets_16 \
    data.method=augmentation data.augmentation_scheme=so2 \
    checkpoint_manifest_path=logs/train/hls4ml/deepsets/nconst16/aug_so2/checkpoint_manifest.json \
```

Deltas against a reference are a separate step:

```bash
python scripts/analysis/compare_to_baseline.py <run table> <baseline table>.
```

After the evaluation is completed, you can make figures of the results by using:

```bash
python scripts/analysis/make_figures.py --features <features>
python scripts/analysis/make_latex_table.py results/<features>/results_table_*.csv
```

## Hyperparameter search

Every hyperparameter optimisation employed in this study optimises for maximum validation accuracy.
Every SCL and augmentation study have the same hyperparameter space for the model and the same budget, to ensure a fair comparison:

```bash
python src/train.py -m \
    experiment=hls4ml/deepsets_16 \
    hparams_search=hls4ml/deepsets_scl_optuna \
    data.method=symm_loss data.augmentation_scheme=so2 \
    data.n_folds=1 callbacks=search logger=none
```

The embedded symmetry architectures are optimised independently, since these architectures are fundamentally different from the ones used in SCL and augmentation studies.
The databases are written to `logs/optuna/hls4ml/<model>.db`.
The `_scl_optuna` variant of a config adds `algorithm.symm_loss_weight`.
The other four regimes use the plain `<model>_optuna`.
The `callbacks=search` is set such that a trial does not run any plotting and does not output anything to `results/`.

The best trial in a study can be fetches with the following script:

```bash
python scripts/optuna/fetch_best_params.py \
    --db logs/optuna/hls4ml/deepsets.db --study deepsets_nconst16_symm_loss_so2
```

## Licence

The code is MIT; see [`LICENSE`](LICENSE). The data set is distributed separately under
CC-BY-4.0.
