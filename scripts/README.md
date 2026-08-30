# `scripts/`

Everything outside `src/` that produces the results: the exact commands each
experiment was run with, the tooling that generates and submits them, and the
analysis that turns finished runs into tables and figures.

Run everything from the repository root.

| Directory                 | What lives there                                                                                                        |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `hls4ml/`                 | The **command catalogues**, one triple of files per architecture. Not runnable scripts; see below.                      |
| `cluster/`                | Parsing those catalogues and running them: on a local multi-GPU host, or through the submitit launcher on a slurm site. |
| `optuna/`                 | Reading the study databases: best parameters out, train and eval catalogues in.                                         |
| `analysis/`               | Turning the results tables into p-values and figures.                                                                   |
| `setup.sh` `symbolink.sh` | One-time bootstrap. Run `setup.sh` before anything else.                                                                |

## The catalogues are documentation, not programs

Every `run<model>_{search,train,eval}.sh` is entirely commented out. A file is a
list of `python3 src/train.py` or `python3 src/eval.py` blocks, each
self-contained and copy-pasteable, covering the full grid of three constituent
counts (16, 50, 150) crossed with seven regimes: the unaugmented `baseline`,
`aug_so2`, `aug_so3` and `aug_lorentz` for data augmentation, and `symm_so2`,
`symm_so3` and `symm_lorentz` for the symmetry-consistency loss. That is 21
blocks per file, in that order, and 12 files.

- `*_search.sh` holds one Optuna driver per grid cell. Hand-maintained: no tool
  regenerates them, and they are the canonical record of what each sweep asked
  for. The study name is never on the command line, since
  `configs/hparams_search/hls4ml/<model>_optuna.yaml` derives it from the
  regime as `<model>_nconst<N>_<method>_<scheme>`.
- `*_train.sh` and `*_eval.sh` are **generated** by
  `optuna/make_train_catalogues.py` from the study databases, each block
  carrying the best trial of its own study. Regenerate rather than edit by hand.

The submitters in `cluster/` parse these blocks and run them; `cluster/lib.sh`
holds the parser, which is the one thing that must stay in step with the
generator's output layout.

### Block grammar

A block opens at a line matching `^# python3 src/(train|eval)\.py` and runs to
the first line without a trailing `\`. Continuation lines carry one override
each, behind a `#` and five spaces. Sections are separated by 72-character
`=` banners, blocks by 72-character `-` banners. Nothing else in the file is
parsed, so free prose between blocks is safe.

Every block carries `experiment=hls4ml/<model>_<N>`, `data.method`,
`data.augmentation_scheme`, `trainer=gpu` and `trainer.devices=[0]`. The two
`data.*` overrides are what `lib.sh::block_label` reads to rebuild a block's
label, so `--only symm_so2` and `--only nconst150` both select the obvious
thing; a block that omits them cannot be selected by name.

### Run directories and checkpoint manifests

A training block writes to `hydra.run.dir=logs/train/hls4ml/<model>/nconst<N>/<variant>`
and drops a `checkpoint_manifest.json` there listing the main run's checkpoint
and one per fold. Its evaluation block reads exactly that path, so the two stay
paired by construction, and `src/eval.py` tests every checkpoint the manifest
names.

Under `--cluster` the convention needs one adjustment. The submitit launcher
only runs multiruns, so `submit_train.sh` rewrites `hydra.run.dir=X` to
`hydra.sweep.dir=X`, and hydra then writes the single job's output to `X/0`.
`submit_eval.sh` applies the same rewrite and additionally moves the manifest
path to `X/0/checkpoint_manifest.json`. Checkpoints trained on a cluster must
therefore be evaluated with `--cluster` as well, or the manifest is looked for
one directory too high. Both rewrites live in `lib.sh::to_sweep_dir`.

## The pipeline

```
run<model>_search.sh
  |  cluster/submit_search.sh
  v
logs/optuna/hls4ml/<model>.db              the sweeps' only durable record
  |  optuna/fetch_best_params.py           -> configs/experiment/hls4ml/<model>_<N>.yaml
  |  optuna/make_train_catalogues.py
  v
run<model>_train.sh
  |  cluster/submit_train.sh
  v
logs/train/hls4ml/<model>/nconst<N>/<variant>/checkpoint_manifest.json
  |  cluster/submit_eval.sh  (run<model>_eval.sh)
  v
results/<features>/results_table_{baseline,augmentation,symm_loss}.csv
  |  analysis/add_pvalues.py
  v
results/<features>/results_table_*_pvalues.csv
  |  analysis/make_figures.py
  v
results/<features>/fig_{augmentation,symm_loss}.svg
```

`fetch_best_params.py` prints a study's best trial twice: as hydra override
tokens for a one-off rerun, and as a YAML block for the experiment config. Make
the values the config's own and the next regeneration of the catalogues picks
them up.

## Records, not bugs

Two things about the checked-in catalogues look wrong and are not.

Every block says `trainer.devices=[0]`, on a host with four GPUs. The device is
a property of the queue, not of the experiment, so the submitters rewrite it
from `--gpu`, and under `--cluster` each slurm job sees one GPU anyway.

Every train and eval block currently carries no `algorithm.*` overrides and
says `transferred from experiment config`. No study has been run yet, so the
grid trains at the experiment configs' own values. That state is itself the
transferred-hyperparameter ablation, which is kept as a comparison once the
searches have replaced the values, and `make_train_catalogues.py --transferred`
reproduces it deliberately into `*_transferred.sh` files.

## Running on olgpu

No scheduler, four V100s, so concurrency is one sequential queue per GPU. Each
invocation of `submit_train.sh` writes its selection to a queue script and runs
it under a single `nohup`, one training at a time:

```bash
bash scripts/setup.sh                 # once: creates data logs outputs results, writes .env
bash scripts/cluster/submit_train.sh --list scripts/hls4ml/rundeepsets_train.sh

bash scripts/cluster/submit_train.sh --only baseline,aug_so2,aug_so3,aug_lorentz \
    --gpu 0 scripts/hls4ml/rundeepsets_train.sh
bash scripts/cluster/submit_train.sh --only symm_so2,symm_so3,symm_lorentz \
    --gpu 1 scripts/hls4ml/rundeepsets_train.sh
```

The two selections are disjoint, so the queues never contend for a GPU. Their
queue scripts and logs land under `logs/submit/hls4ml_rundeepsets_train/`, and
the queue script is the exact list of commands that will run, worth reading
before walking away. Evaluate the same way with `submit_eval.sh` once the
manifests exist; `--list` marks the blocks whose manifest is still missing.

Searches are long-lived drivers rather than a queue, one per grid cell, so
select a few and pin them:

```bash
bash scripts/cluster/submit_search.sh --only nconst16 --gpu 0 \
    scripts/hls4ml/rundeepsets_search.sh
python3 scripts/cluster/sweep_status.py
```

`--dry-run` prints the assembled commands and touches nothing, on any machine.
Both it and `--list` work without the data, the environment or a GPU, which
makes them the way to check a catalogue before committing a batch to it.

## File by file

### `cluster/`

| File               |                                                                                                                                                                                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `submit_search.sh` | Launch a `run*_search.sh` file's Optuna drivers. Blocks already carry `-m` and their sweeper settings, so this adds the GPU pinning, a private sweep directory and, with `--cluster`, the launcher. Pre-creates each sqlite study serially to dodge the alembic bootstrap race. |
| `submit_train.sh`  | Run a `run*_train.sh` file. Locally the selected blocks go into one nohup'd queue on one GPU; with `--cluster` each block becomes its own submitit job.                                                                                                                         |
| `submit_eval.sh`   | The same for `run*_eval.sh`, plus the manifest rewrite of the section above and a warning for each block whose training has not finished.                                                                                                                                       |
| `sweep_status.py`  | Every study's COMPLETE trials against its target, whether a driver is still alive (read from the driver log, not from the trial counts: a dead driver leaves its last batch RUNNING forever), and the `submit_search.sh` command to finish it.                                  |
| `lib.sh`           | Sourced by the three submitters: block parser, label reconstruction, `--only` matcher, `.env` loading, GPU pinning, the sweep-dir rewrite and the repo-root and `HF_HUB_CACHE` guards.                                                                                          |

### `optuna/`

| File                       |                                                                                                                                                                                                                                                                                               |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fetch_best_params.py`     | A study's best trial as hydra overrides and as a YAML block. `--all` reports every study in the database, `--csv` dumps the trials.                                                                                                                                                           |
| `make_train_catalogues.py` | The study databases to the `run*_{train,eval}.sh` catalogues. `--dry-run` reports which files a regeneration would change; `--transferred` writes the no-overrides ablation. Feeds its own output back through `lib.sh`'s parser and refuses to write a file that does not yield 21 commands. |

### `analysis/`

| File              |                                                                                                                                                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `add_pvalues.py`  | Paired bootstrap p-values and confidence intervals on the results tables, against the `augmentation_scheme=none` row of the same (model, constituent count). Writes `*_pvalues.csv` and a LaTeX table beside each input. |
| `make_figures.py` | The `*_pvalues.csv` files to one figure per regime: accuracy, invariance metric and stable fraction, each as a delta against that model's unaugmented baseline.                                                          |

`make_figures.py` loads `add_pvalues.py` by path, since `scripts/` is not a
package; keep the two as siblings.
