# `scripts/`

Everything outside `src/` that produces the results lives here: the exact
commands each experiment was run with, the tooling that generates and submits
them, and the analysis that turns finished runs into tables and figures.

Run everything from the repository root.

| Directory                 | What lives there                                                                                                                                            |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hls4ml/`                 | The **command catalogues**, one triple of files per architecture. Not runnable scripts; see below.                                                          |
| `cluster/`                | Parsing those catalogues and running them: on a local multi-GPU host, through the submitit launcher on a slurm site, or as a pool of workers on an NGT pod. |
| `optuna/`                 | Reading the study databases: best parameters out, train and eval catalogues in.                                                                             |
| `analysis/`               | Turning the results tables into the comparison figures and the LaTeX tables.                                                                                |
| `setup.sh` `symbolink.sh` | One-time bootstrap. Run `setup.sh` before anything else.                                                                                                    |

## The catalogues are documentation, not programs

Every `run<model>_{search,train,eval}.sh` is entirely commented out. A file is a
list of `python3 src/train.py` or `python3 src/eval.py` blocks, each
self-contained and copy-pasteable. Together they cover the full grid, three
constituent counts (16, 50, 150) crossed with seven regimes: the unaugmented
`baseline`, `aug_so2`, `aug_so3`, and `aug_lorentz` for data augmentation, and
`symm_so2`, `symm_so3`, and `symm_lorentz` for the symmetry-consistency loss.
That is 21 blocks per file, in that order, for the `deepsets`, `mlp`, and
`jetformer` triples. The `lorentznet` triple holds the three `baseline` blocks
alone, because LorentzNet, the symmetry-embedded class, is trained in the
baseline regime only.

- `*_search.sh` holds one Optuna driver per grid cell. These files are
  hand-maintained: no tool regenerates them, and they are the canonical record
  of what was asked of each sweep. The study name never appears on the command
  line, since `configs/hparams_search/hls4ml/<model>_optuna.yaml` derives it
  from the regime as `<model>_nconst<N>_<method>_<scheme>`.
- `*_train.sh` and `*_eval.sh` are **generated** by
  `optuna/make_train_catalogues.py` from the study databases, each block
  carrying the best trial of its own study. Regenerate rather than edit by hand.

The submitters in `cluster/` parse these blocks and run them. The parser itself
lives in `cluster/lib.sh`, the one thing that must stay in step with the
generator's output layout.

### Block grammar

A block opens at a line matching `^# python3 src/(train|eval)\.py` and runs to
the first line without a trailing `\`. Continuation lines carry one override
each, behind a `#` and five spaces. Sections are separated by 72-character
`=` banners, blocks by 72-character `-` banners. Nothing else in the file is
parsed, so free prose between blocks is safe.

Every block carries `experiment=hls4ml/<model>_<N>`, `data.method`,
`data.augmentation_scheme`, `trainer=gpu`, and `trainer.devices=[0]`. The two
`data.*` overrides are read by `lib.sh::block_label` to rebuild a block's label,
so `--only symm_so2` and `--only nconst150` both select the obvious thing; a
block that omits them cannot be selected by name.

### Run directories and checkpoint manifests

A training block writes to `hydra.run.dir=logs/train/hls4ml/<model>/nconst<N>/<variant>`
and drops a `checkpoint_manifest.json` there, listing the main run's checkpoint
and one per fold. Its evaluation block reads exactly that path, so the two stay
paired by construction, and every checkpoint the manifest names is tested by
`src/eval.py`.

Under `--cluster` the convention takes one adjustment. The submitit launcher
only runs multiruns, so `submit_train.sh` rewrites `hydra.run.dir=X` to
`hydra.sweep.dir=X`, and hydra then writes the single job's output to `X/0`.
The same rewrite is applied by `submit_eval.sh`, which additionally moves both
manifest paths of an eval block, `checkpoint_manifest_path` and
`baseline_checkpoint_manifest_path`, to the `X/0/checkpoint_manifest.json` of the
run each names. Checkpoints trained on a cluster must therefore be evaluated with
`--cluster` as well, or the manifests are looked for one directory too high. Both
rewrites live in `lib.sh::to_sweep_dir`.

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
  |  analysis/make_figures.py --features <features>
  v
results/<features>/fig_{augmentation,symm_loss}.svg
```

`fetch_best_params.py` prints a study's best trial twice: as hydra override
tokens for a one-off rerun, and as a YAML block for the experiment config. Move
the values into the config and the next regeneration of the catalogues picks
them up.

## Records, not bugs

Two things about the checked-in catalogues look wrong and are not.

Every block carries `trainer.devices=[0]`, on a host with four GPUs. The device is
a property of the queue, not of the experiment, so the submitters rewrite it
from `--gpu`, and under `--cluster` each slurm job sees one GPU anyway.

No train or eval block carries an `algorithm.*` override, and every banner reads
`transferred from experiment config`. No study has been run yet, so the grid
trains at the experiment configs' own values. That state is itself the
transferred-hyperparameter ablation, kept as a comparison once the searches have
replaced the values; `make_train_catalogues.py --transferred` reproduces it
deliberately into `*_transferred.sh` files.

## Running on olgpu

olgpu carries four V100s and no scheduler, so concurrency is one sequential
queue per GPU. Each call to `submit_train.sh` writes its selection to a queue
script and runs that script under a single `nohup`, one training at a time:

```bash
bash scripts/setup.sh                 # once: creates data logs outputs results, writes .env
bash scripts/cluster/submit_train.sh --list scripts/hls4ml/rundeepsets_train.sh

bash scripts/cluster/submit_train.sh --only baseline,aug_so2,aug_so3,aug_lorentz \
    --gpu 0 scripts/hls4ml/rundeepsets_train.sh
bash scripts/cluster/submit_train.sh --only symm_so2,symm_so3,symm_lorentz \
    --gpu 1 scripts/hls4ml/rundeepsets_train.sh
```

The two selections are disjoint, so the queues never contend for a GPU. Their
queue scripts and logs land under `logs/submit/hls4ml_rundeepsets_train/`; the
queue script is the exact list of commands that will run, worth reading before
walking away. Evaluate the same way with `submit_eval.sh` once the manifests
exist; `--list` marks the blocks whose manifest is still missing.

The folds of one block can run side by side instead. `--folds k --gpus 1,2,3`
launches k+1 detached processes per selected block rather than a queue entry:
the main run with `data.n_folds=1`, and one `fold=<i> data.n_folds=<k>` process
per fold writing to `<run dir>/fold_<i>`. The devices of `--gpus` are handed out
in turn, the count running on across blocks. An integer token goes into
`trainer.devices`; a `MIG-...` or `GPU-...` UUID is exported as
`CUDA_VISIBLE_DEVICES`, which is how the same flag serves an NGT pod. Every
process is `setsid nohup`'d, with a log at
`logs/submit/<catalogue>/<i>_<label>_{main,fold_<j>}.log`. Each process leaves a
manifest of its own, and `src/eval.py` merges them when `checkpoint_manifest_path`
names the run directory rather than a file: evaluate such a block with
`checkpoint_manifest_path=logs/train/hls4ml/<model>/nconst<N>/<variant>` in place
of the catalogue's `.../checkpoint_manifest.json`. `--folds` is local-only, so it
cannot be combined with `--cluster`.

Searches are long-lived drivers rather than a queue, one per grid cell, so
select a few and pin them:

```bash
bash scripts/cluster/submit_search.sh --only nconst16 --gpu 0 \
    scripts/hls4ml/rundeepsets_search.sh
python3 scripts/cluster/sweep_status.py
```

`--dry-run` prints the assembled commands and touches nothing, on any machine.
Both it and `--list` work without the data, the environment, or a GPU, which
makes them the way to check a catalogue before committing a batch to it.

## Running on NGT

The NGT sessions are interactive kubernetes pods rather than a cluster, so the
pod is the machine and, as on olgpu, a driver is started by hand. Two things
differ. A pod's GPUs are listed in `NVIDIA_VISIBLE_DEVICES` as UUIDs, one per
MIG slice, and CUDA enumerates only one slice per parent card, so
`trainer.devices=[i]` cannot reach the second slice of a card; a worker is
pinned with `CUDA_VISIBLE_DEVICES=<uuid>` instead and keeps the catalogue's
`trainer.devices=[0]`. And a process tree started under `kubectl exec` dies with
the exec session, so every worker is detached with `setsid nohup`.

`submit_search_ngt.sh` does both and starts several drivers of one study at
once. That works because the Optuna sweeper opens its study with
`load_if_exists`: drivers agreeing on the study name and the storage share it,
their trials accumulate, and `n_startup_trials` is counted over the study rather
than per driver. The trial budget is therefore split between the workers
(`--trials 100` over four of them is 25 each, the remainder going to the ones
that start first), and the sampler seed is not, since workers drawing the same
start-up points would turn four searches into one search done four times.

```bash
# a MIG pod: four slices, four workers, 100 trials between them
bash scripts/cluster/submit_search_ngt.sh --only nconst150 --workers 4 \
    --trials 100 --seed 1234 --tag s1 scripts/hls4ml/runlorentznet_search.sh

# a full-card pod: one device, time-shared by two workers
bash scripts/cluster/submit_search_ngt.sh --only nconst150 --workers 2 \
    --trials 50 --seed 1238 --tag s6 scripts/hls4ml/runlorentznet_search.sh
```

`--only` selects one block, because a pool of workers sweeps one study, and
`--tag` names them, defaulting to the hostname. Worker `j` gets seed
`--seed` + `j`, the tag `<tag>_w<j>`, a launch log in
`logs/searches/ngt/<tag>_w<j>.log`, and a driver log where `submit_search.sh`
always puts it, under `logs/searches/hls4ml_run<model>_search/`, now suffixed
with that same tag. The suffix is the whole point: `logs/` on the pods is one
NFS volume shared by every pod, so without it the second worker would rotate the
first one's live log, and two workers starting in the same second would share a
multirun directory.

The shared volume is also why `python3 scripts/cluster/sweep_status.py` reports
the entire campaign from any pod. It reads every worker log of a block, counts
them in the `workers` column, and calls a study alive while any one of them is.
To replace a worker that died, take the command it prints, which covers exactly
the trials still missing, and give it a `--tag` no live worker is using and a
`hydra.sweeper.sampler.seed=` of its own.

## File by file

### `cluster/`

| File                   |                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `submit_search.sh`     | Launch a `run*_search.sh` file's Optuna drivers. Blocks already carry `-m` and their sweeper settings, so this adds the GPU pinning, a private sweep directory and, with `--cluster`, the launcher. Pre-creates each sqlite study serially to dodge the alembic bootstrap race. `--tag` suffixes the driver log and the sweep directory, which is what lets several drivers share one study. |
| `submit_search_ngt.sh` | Several workers of one study on an NGT pod: one per MIG UUID through `CUDA_VISIBLE_DEVICES`, the trial budget split between them, a seed and a tag each, all `setsid nohup`'d. Calls `submit_search.sh` once per worker.                                                                                                                                                                     |
| `submit_train.sh`      | Run a `run*_train.sh` file. Locally the selected blocks go into one nohup'd queue on one GPU; with `--folds k --gpus <list>` each block instead fans out into its main run and k `fold=<i>` processes over those devices; with `--cluster` each block becomes its own submitit job.                                                                                                          |
| `submit_eval.sh`       | The same for `run*_eval.sh`, plus the manifest rewrite of the section above and a warning for each block whose training has not finished.                                                                                                                                                                                                                                                    |
| `sweep_status.py`      | Every study's COMPLETE trials against its target, how many workers it has and whether any is still alive (read from the driver logs, not from the trial counts: a dead driver leaves its last batch RUNNING forever), and the `submit_search.sh` command to finish it.                                                                                                                       |
| `lib.sh`               | Sourced by the four submitters: block parser, label reconstruction, `--only` matcher, `.env` loading, GPU pinning, the sweep-dir rewrite and the repo-root and `HF_HUB_CACHE` guards.                                                                                                                                                                                                        |

### `optuna/`

| File                       |                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fetch_best_params.py`     | A study's best trial as hydra overrides and as a YAML block. `--all` reports every study in the database, `--csv` dumps the trials.                                                                                                                                                                                                                                   |
| `make_train_catalogues.py` | The study databases to the `run*_{train,eval}.sh` catalogues. `--dry-run` reports which files a regeneration would change; `--transferred` writes the no-overrides ablation. Feeds its own output back through `lib.sh`'s parser and will not write a file that does not yield its model's block count, 21 for `deepsets`, `mlp` and `jetformer`, 3 for `lorentznet`. |

### `analysis/`

| File                  |                                                                                                                                                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `make_figures.py`     | One results table per regime to one figure: accuracy, invariance metric and stable fraction, each as a delta against that model's unaugmented baseline. `--features` names the one `results/<features>/` bucket to read. |
| `make_latex_table.py` | The same tables to a `.tex` beside each of them. Formatting only: every delta, confidence interval and p-value it prints was written by the evaluation.                                                                  |

The deltas themselves are no longer computed here. An evaluation block carries
`baseline_checkpoint_manifest_path=<the baseline run's manifest>` alongside its
own, so `src/eval.py` runs both models over the same transformed test batches,
and the delta, confidence-interval, and p-value columns are written by the
results table callback. A baseline block and the whole LorentzNet catalogue name
no baseline and hold NaN in those columns.

`make_figures.py` loads `make_latex_table.py` by path, since `scripts/` is not a
package; keep the two as siblings. `make_baseline_figure.py` loads
`make_figures.py` the same way and draws the baseline-only counterpart from the
same tables, for a model trained in the baseline regime alone (LorentzNet). Its
rows hold the accuracy change under the transformation (with the clean accuracy
printed on the panel), the invariance metric, and the flip rate, all in absolute
terms, one column per model and one bar cluster per constituent count.
