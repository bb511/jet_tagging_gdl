# Removed architectures

This repository carried eight architecture modules before the refactor to the `adl1t` layout
and carries seven now: DeepSets, MLP, JetFormer, LorentzNet, PELICAN (back since 2026-09-13 as
a clean rewrite), L-GATr-slim and LLoCa. The last two were added on 2026-09-28: L-GATr-slim
runs in two rows, `lgatr_slim` with the spurion tokens and `lgatr_slim_lorentz` without, and
LLoCa in four, one per frames class (`lloca_lorentz`, `lloca_z`, `lloca_so2` and `lloca_so3`).
PELICAN, a plain Transformer encoder, a Lipschitz-constrained MLP and an IRC-safe DeepSets were
removed at the refactor; nano-PELICAN, which existed as configuration only, went with them, and
so did the permutation check. This file records what each was, why it went, and what putting it
back involves.

The last commit that holds all of it is `a667b346`. Any file named below is recovered with
`git show a667b346:<path>`.

Two reasons for removal are shared. Each architecture was a `LightningModule` carrying its
own copy of the training loop, the metric wiring, the optimiser plumbing and the symmetry
branch, so a change to the comparison had to be made eight times; by `a667b346` the copies
had drifted far enough that `lorentz_net_module.py` misspelled `utilities` in its
`pytorch_lightning` import and could not be loaded at all. And none of the removed four
contributed a single row to a results table: every CSV under `results/` at that commit holds
DeepSets, JetFormer and MLP alone.

The refactor replaced those copies with one base class,
`src/algorithms/__init__.py::JetTaggingModule`, which owns all of it. An architecture now
consists of an `nn.Module` and a subclass that builds it, which is why removing four of them
costs little: what was deleted is mostly the duplicated loop, and what a port has to supply
is the network.

## Putting one back

The base class owns `forward`, `model_step`, the metrics, the optimiser and scheduler
construction, the feature resolution in `setup()` and the FLOP count. A subclass supplies
`build_net(**net_kwargs) -> nn.Module` and, where the architecture is picky about its inputs,
two hooks. The steps:

- Write the network as an `nn.Module` under `src/algorithms/components/<name>.py`. It takes
  `(B, N, F)` float32, with padded constituents as all-zero rows, and returns `(B, 5)`
  logits. The masking convention the kept components use is `x.abs().sum(-1) > 0`.
- Write `src/algorithms/<name>.py`, a `JetTaggingModule` subclass whose only required member
  is `build_net`. Do not give it an `__init__`: `save_hyperparameters` walks the caller frame
  to collect `**net_kwargs`, and an intervening constructor that does not forward every
  argument loses them.
- If the architecture needs particular columns, set the class attribute
  `required_features` (LorentzNet's is `["px", "py", "pz", "e"]`). `forward` then reduces
  whatever width it is handed to those columns, in that order.
- If it needs them de-normalised or reordered, override `_prepare_inputs(x)`, and fill any
  buffer it needs in `_on_features_resolved()`, which runs at the end of `setup()` once
  `norm_scales` and the feature names are known. `src/algorithms/lorentznet.py` is the worked
  example of both hooks.
- Add `configs/algorithm/<name>.yaml` on the pattern of `deepsets.yaml`: `_target_`,
  `_convert_: all`, a short `name`, `nbr_feats: ${len:${data.features}}`, the
  `method`/`group` interpolations from the data config, the architecture arguments, and an
  optimiser and scheduler as Hydra partials.
- Add `configs/experiment/hls4ml/<name>_{16,50,150}.yaml`, the search configs
  `configs/hparams_search/hls4ml/<name>_optuna.yaml` and `<name>_scl_optuna.yaml`, and the
  three catalogues `scripts/hls4ml/run<name>_{search,train,eval}.sh`. An architecture with an
  embedded symmetry follows LorentzNet and PELICAN instead: the baseline regime alone, so no
  `_scl_optuna` config and three blocks per catalogue rather than 21.
- Extend `tests/test_algorithms.py`, which exercises every algorithm config under both
  methods.

Every one of the removed modules will need two things on the way in. They import
`pytorch_lightning.utilities.memory.garbage_collection_cuda`; the `pytorch_lightning`
namespace is not used anywhere in this repository any more, and the base class handles memory
without it. They also accept `method="equivariance"`, which no longer exists: the supported
methods are `augmentation` and `symm_loss`.

## PELICAN

**Back since 2026-09-13, as a clean rewrite**, not as the vendored copy below:
`src/algorithms/components/pelican.py` and `src/algorithms/pelican.py`, with
`configs/algorithm/pelican.yaml`, `configs/experiment/hls4ml/pelican_{16,50,150}.yaml` and
`configs/hparams_search/hls4ml/pelican_optuna.yaml`. The rewrite carries the general classifier
of arXiv:2307.16506 alone: the 15 aggregators of the rank-2 equivariant layer built from compact
sources rather than the reference's `(B, C, 15, N, N)` stack, one masked BatchNorm per block, and
none of the dict inputs, config strings, IRC weighting, factorisation or spurion machinery. Two
beam four-vectors are appended inside the network, so the exact symmetry is SO(2) about the beam;
the other groups are measured by the probe. It is the second symmetry-embedded architecture
beside LorentzNet, run in the baseline regime alone. Since 2026-09-27 it is run twice, with the
beams as `pelican` and without them as `pelican_lorentz`: the same module and network, switched by
`configs/algorithm/pelican_lorentz.yaml` (a standalone copy of `pelican.yaml` with `add_beams:
false`, kept in step by `tests/test_configs.py`), with its own experiment configs, search config
and catalogues; the 150-constituent row is not searched and its experiment config carries the
beams-on optimum. The rest of this section is the record of
the vendored copy that was removed at the refactor, kept as history.

Permutation-invariant, Lorentz-invariant aggregator network: pairwise Lorentz invariants of
the constituent four-momenta are fed through equivariant rank-2 layers and aggregated to a
class score, so the architecture is invariant by construction rather than by training. It was
also the largest thing here to carry: a vendored copy of the upstream implementation, five
configs spanning the published 6k to 208k parameter points, and its own masked normalisation
and permutation-equivariant layer libraries.

Files:

```
src/models/pelican_module.py                       PELICANClassifier and its PELICANModule wrapper
src/models/components/pelican_layers/              generic_layers, masked_batchnorm,
                                                   masked_instancenorm, perm_equiv_layers,
                                                   perm_equiv_models
src/utils/lorentz_metric.py                        Minkowski helpers, CATree, SDMultiplicity
configs/model/pelican.yaml                         base config, so13 stabilizer, no spurions
configs/model/pelican_6k.yaml                      upstream 6k point, so2 stabilizer + spurions
configs/model/pelican_48k.yaml                     upstream 48k point, so2 + spurions
configs/model/hls4ml_pelican.yaml                  the 132/78 multi-class point, about 208k
configs/model/hls4ml_pelican_11k.yaml              the 25/15 multi-class point, about 11k
```

Why removed: the four kept architectures span the axis the study needs (no built-in
symmetry, through a set model, to a Lorentz-equivariant graph network), and PELICAN's
invariance is structural, so it answers a different question from "does augmentation or a
consistency loss teach invariance better".

Porting notes, superseded by the 2026-09-13 rewrite, which takes a `(B, N, F)` tensor and
builds its own masks and beams; they describe what the vendored copy needed.
`PELICANClassifier` consumes a dict (`Pmu`, `particle_mask`, `edge_mask`,
`Nobj`, optional `scalars`), not a tensor, so `build_net` should return a wrapper that
assembles that dict from `(B, N, F)`. The old wrapper rebuilt the energy from the momenta
under a massless assumption; with the current default features the energy column is present,
so a port should set `required_features = ["px", "py", "pz", "e"]` and reorder in
`_prepare_inputs` as LorentzNet does. Hydra hands nested lists as `ListConfig`, which
`expand_var_list` rejects, hence `_convert_: all` in the algorithm config. The config's
`spurion_method` key exists to avoid the clash with `method`, which in this repository means
the symmetry-handling method throughout.

## L-GATr-slim

Added on 2026-09-28 as the third symmetry-embedded architecture, and the first whose network is
not written here: L-GATr-slim (arXiv:2512.17011), the scalars-and-vectors variant of L-GATr
(arXiv:2405.14806), comes as `lgatr.LGATrSlim` from the pip package `lgatr` 2.0.0. Our own code
is the wrapper `src/algorithms/components/lgatr_slim.py` and the `JetTaggingModule` subclass
`src/algorithms/lgatr_slim.py`, which reads `[px, py, pz, e]` and hands the network
(E, px, py, pz) in GeV. Around them sit `configs/algorithm/lgatr_slim.yaml` and
`lgatr_slim_lorentz.yaml`, the six experiment configs
`configs/experiment/hls4ml/lgatr_slim{,_lorentz}_{16,50,150}.yaml`, the two search configs
`configs/hparams_search/hls4ml/lgatr_slim{,_lorentz}_optuna.yaml` and the six catalogues
`scripts/hls4ml/runlgatr_slim{,_lorentz}_{search,train,eval}.sh`. The wrapper divides the
constituents by `units` (20 GeV), prepends a fixed global token (a zero four-vector carrying a
single scalar flag) and the spurion tokens, passes a dense boolean key-padding mask, and reads
the logits from the global token's scalar outputs. The constituents enter as the null vectors
they are, with no mass regulator, and the experiments run in fp32, since the package's
documentation advises against mixed precision for the slim network.

The spurion tokens set the exact group, and the switches `add_beams` and `add_time` set the
spurions. With neither, the logits are invariant under the full Lorentz group: this is the
`lgatr_slim_lorentz` row. The time vector (1, 0, 0, 0) alone leaves SO(3), a setting the
switches allow but no catalogue scripts. The `lgatr_slim` row, the published tagging setting,
adds the two lightlike beams (1, 0, 0, ±1) to the time vector and leaves SO(2) about the beam,
because a boost along the beam rescales the null beam vectors rather than fixing them.
The reflection p_z -> -p_z only swaps the two beam tokens, and attention does not depend on the
order of its tokens, so the reflection is exact too. Both rows are one module and one network:
`lgatr_slim_lorentz.yaml` is a standalone copy of `lgatr_slim.yaml` with the two switches off,
kept in step by `tests/test_configs.py`, as for PELICAN.

## LLoCa

Added on 2026-09-28 as the fourth symmetry-embedded architecture, and the second whose network is
not written here: LLoCa, Lorentz local canonicalisation (arXiv:2505.20280 for the method,
arXiv:2508.14898 for its use in particle physics), comes from the pip package `lloca` 2.0.0, which
supplies the frames-net, the frames and the LLoCa-Transformer. Our own code is the wrapper
`src/algorithms/components/lloca.py` and the `JetTaggingModule` subclass
`src/algorithms/lloca.py`, which reads `[px, py, pz, e]` and hands the network (E, px, py, pz) in
GeV. The four rows are named `lloca_<row>`, with `<row>` one of `lorentz`, `z`, `so2` and `so3`.
Around the code sit the four algorithm configs `configs/algorithm/lloca_<row>.yaml`, the twelve
experiment configs `configs/experiment/hls4ml/lloca_<row>_{16,50,150}.yaml`, the four search
configs `configs/hparams_search/hls4ml/lloca_<row>_optuna.yaml` and the twelve catalogues
`scripts/hls4ml/runlloca_<row>_{search,train,eval}.sh`. The wrapper divides the constituents by
`units` (20 GeV) and runs the frames-net over the real constituents alone, in the package's packed
layout, because its edge convolution takes no padding mask. Each constituent's four-momentum is
then canonicalised into its local frame and handed to the LLoCa-Transformer with a dense boolean
key-padding mask and the jet momentum as the reference of `preserve_variance`; the logits are the
mean of its outputs over the real constituents. A mass regulator lifts every constituent lighter
than `mass_reg` (2.5e-4 in those units, the references' 5 MeV) to that mass inside the frames-net
alone; the lift does not commute with a boost, so the boost-exact rows keep a residual of relative
size m²/(2p²), below 1e-4 for every constituent above 0.35 GeV. The frames-net receives the packed
constituents in float64, as the references feed it: in float32 the mass floor sits below the
rounding of |p|^2 and the massless constituents' pair masses fall to zero, where the frames-net's
sqrt has no finite gradient, which turned the first smoke run's logits to NaN. The backbone and
the experiments run in fp32, as the references train without mixed precision.

The frames class sets the exact group, and the switch `frames` picks the class, one per row.
`LearnedPDFrames`, the class of the `lloca_lorentz` row, predicts a general Lorentz
transformation for every constituent, so every probe group is exact for it. The other three
classes restrict the frames to a subgroup, and the probe measures every group outside it.
`LearnedZFrames` (`lloca_z`) allows a rotation about the beam composed with a boost along it,
SO(2)×SO(1,1), so SO(3) and the isotropic boost are measured. `LearnedSO3Frames` (`lloca_so3`)
allows rotations alone, so the boost along the beam and the isotropic boost are measured, and
`LearnedSO2Frames` (`lloca_so2`) allows rotations about the beam alone, so SO(3) and both boosts
are. The logits are the mean over the constituents, not the output of a global token as in
L-GATr-slim: a token with a fixed identity frame does not transform with the jet, so the
tensorial attention would treat it as a spurion and the row would lose its exact group. The four
rows are one module and one network: `lloca_z.yaml`, `lloca_so2.yaml` and `lloca_so3.yaml` are
standalone copies of `lloca_lorentz.yaml` that differ in `name` and `frames` alone, kept in step
by `tests/test_configs.py`. A jet of one or two constituents gives the frames-net too few
independent vectors, and the package regularises such frames with random noise, so these jets
may leave a small floor on the flip rates even under an exact group.

## nano-PELICAN

A small single-block PELICAN, configured but never implemented: the config pointed at
`src.models.nano_pelican.PELICANModule`, a module that was never written, so
`model=nano_pelican` could never be instantiated. A search config sat beside it regardless, with its
study database hard-coded to another user's home directory.

Files: `configs/model/nano_pelican.yaml`, `configs/hparams_search/nano_pelican_optuna.yaml`.

Why removed: dead configuration. If the idea is wanted, it is a narrow instance of the
PELICAN port above and should start there. It is one now: the corner `algorithm.n_blocks=1`,
`algorithm.n_message=0` of the PELICAN search space, at the smallest width, rather than a model
of its own.

## Transformer

A plain transformer encoder over the constituents: a linear lifting layer to `d_models`, four
`nn.TransformerEncoderLayer` blocks with eight heads, and a linear classifier, trained with
AdamW and a cosine schedule with warmup.

Files: `src/models/transformer_module.py`, `configs/model/transformer.yaml`.

Why removed: JetFormer covers the attention family and is the variant every result was
produced with, so keeping both meant maintaining two transformer paths for one column of the
comparison.

Porting notes. The config's scheduler was `src.utils.utils.get_cosine_schedule_with_warmup`,
which the refactor retired along with `get_onecycle_schedule`; use
`torch.optim.lr_scheduler.OneCycleLR` with `interval: step`, as `configs/algorithm/jetformer.yaml`
does; `_set_up_scheduler` injects `total_steps` from
`trainer.estimated_stepping_batches`. The module ran with `use_mask: false`, so padded
constituents were attended to like any other token; that is a deliberate choice of the
reference architecture, worth revisiting at 150 constituents where most slots are empty.

## Lipschitz MLP

An MLP built from `deel.torchlip.SpectralLinear` layers and GroupSort activations, globally
1-Lipschitz by construction through Bjorck orthonormalisation. The point was a bounded
sensitivity to input perturbations, which puts a ceiling on how far the logits can move under
a symmetry transformation of bounded size.

Files: `src/models/lipschitz_module.py` (`LipschitzMLP` and `LipschitzMLPClassifier`),
`configs/model/lipschitz_{8,16,32}.yaml`. External dependency: `deel-torchlip`, dropped from
`pyproject.toml` at the refactor and needed again for any port; it constrains the torch
version, which is why it is not carried speculatively.

Why removed: the bound is a different mechanism from the two the study compares, so it
belongs to a follow-up rather than to the augmentation-against-SCL grid.

Porting notes. `groupsort2` and `maxmin` require even layer widths, which a search space over
`hidden_layers` has to respect.

## IRC-safe DeepSets

DeepSets restricted to be infrared and collinear safe: only the angular coordinates
(`etarel`, `phirel`) enter the per-constituent network, and each output is weighted by that
constituent's pT fraction `z_i = pt_i / sum_j pt_j` before aggregation, so soft emissions
contribute proportionally to their energy and collinear splittings sum back to the parent.

Files: `src/models/irc_deepsets_module.py` (`DeepSetsInvariantIRC`),
`configs/model/irc_deepsets_{8,16,32}.yaml`.

Why removed: IRC safety is a property of the observable, not of the symmetry the study
teaches.

Porting notes. It needs three named columns rather than four momenta, so
`required_features = ["pt", "etarel", "phirel"]` and `data.features` must contain them; with
the default `[px, py, pz, e]` the pT fraction cannot be formed. The augmentation transforms
act on the momenta, so an angular-input model under `symm_loss` sees the full 16-column
batch and reduces it in `forward` like any other.

## The permutation check

Not an architecture: a diagnostic that shuffled the constituents of a jet and measured how
far the logits moved, run as `data.method=equivariance` with
`data.augmentation_scheme=permute`. It was retired at the same commit, together with
`permute_constituents` in the augmentation module and the three
`configs/experiment/permutation/*.yaml`. Every architecture in the study is permutation
invariant or equivariant by construction except the MLP, where the answer is known, so the
check confirmed what the code already guaranteed. Reviving it means adding a scheme to
`src/data/components/augmentation.py::make_transform` and a method to `SUPPORTED_METHODS`;
the invariance callback would then report it beside the three physical groups.
