# Removed architectures

This repository carried eight architecture modules before the refactor to the `adl1t` layout and
carries eight now: DeepSets, MLP, JetFormer, LorentzNet, PELICAN (back since 2026-09-13 as a clean
rewrite), L-GATr-slim, LLoCa and the vector neurons. The last three were added on 2026-09-28 and
2026-09-29: L-GATr-slim runs in two rows, `lgatr_slim` with the spurion tokens and
`lgatr_slim_lorentz` without, LLoCa in four, one per frames class (`lloca_lorentz`, `lloca_z`,
`lloca_so2` and `lloca_so3`), and the vector neurons in one, `vn`, on the lifts of (p_T, η, φ),
which on 2026-10-03 replaced their two rows on the four-momentum, `vn_z` and `vn_so2`. A ninth
architecture, JEDI-linear, joined on 2026-09-30 as a second class in the DeepSets module; it runs in
one row, `jedi_linear`, the comparator from the FPGA literature, which since 2026-10-02 also runs
the seven regimes of the grid. A tenth architecture, the Linformer, joined on 2026-10-01 in a module
of its own, the ninth, and runs in one row, `linformer`, over the seven regimes. Three rows joined
on 2026-10-05 in the DeepSets module, with no architecture or module of their own: DeepSets and
JEDI-linear reading the jet moments of the vector neurons, `deepsets_moments` and
`jedi_linear_moments`, and the control on the moments alone, `moments_only`. Two rows joined on
2026-10-06 with no network of their own, in place of a standalone script of the day before: the
MLP on the record's jet-level observables, `jet_level_all53` and `jet_level_noeta52`, a ceiling
for the networks (see "Jet-level ceiling"). Two more joined that ceiling on 2026-10-07,
gradient-boosted trees on the same observables, `jet_level_bdt_all53` and `jet_level_bdt_noeta52`,
in a module of their own, `src/algorithms/bdt.py`, which wraps scikit-learn rather than a network.
A quantised twin of the vector neurons, `vn_hgq`, joined the same day in modules of its own, which
train the `vn` network quantisation-aware with HGQ2 on Keras for synthesis with alkaid (see "Vector
neurons trained quantisation-aware"). On 2026-10-09 a replica of 18 rows on a second data set, the
top-quark tagging reference set, joined with no architecture of its own (see "Top-tagging
replica").
PELICAN, a plain
Transformer encoder, a Lipschitz-constrained MLP and an IRC-safe DeepSets were removed at the
refactor; nano-PELICAN, which existed as configuration only, went with them, and so did the
permutation check. The two four-momentum rows of the vector neurons went on 2026-10-03. This file
records what each was, why it went, and what putting it back involves.

The last commit that holds all of the refactor's removals is `a667b346`, and any file named in
their sections is recovered with `git show a667b346:<path>`; the four-momentum vector neurons
are recovered from `b54eef5a`.

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
construction, the feature resolution in `setup()` and the FLOP and eBOP counts. A subclass
supplies `build_net(**net_kwargs) -> nn.Module` and, where the architecture is picky about its
inputs, two hooks. The steps:

- Write the network as an `nn.Module` in `src/algorithms/<name>.py`. It takes
  `(B, N, F)` float32, with padded constituents as all-zero rows, and returns `(B, n_class)`
  logits, `n_class` being the algorithm's `output_dim` (5 on hls4ml, 2 on top tagging), which
  `setup()` checks against the datamodule's `class_names`. The masking convention the kept
  networks use is `x.abs().sum(-1) > 0`. Only a block that several networks share goes in
  `src/algorithms/components/`.
- In the same file, write a `JetTaggingModule` subclass whose only required member
  is `build_net`. Do not give it an `__init__`: `save_hyperparameters` walks the caller frame
  to collect `**net_kwargs`, and an intervening constructor that does not forward every
  argument loses them.
- If the architecture needs particular columns, set the class attribute
  `required_features` (LorentzNet's is `["px", "py", "pz", "e"]`). `forward` then reduces
  whatever width it is handed to those columns, in that order.
- If it needs them de-normalised or reordered, override `_prepare_inputs(x)`, and fill any
  buffer it needs in `_on_features_resolved()`, which runs at the end of `setup()` once
  `norm_scales` and the feature names are known. `DenormalisedModule` and `FourMomentumModule`
  in `src/algorithms/__init__.py` are the worked example of both hooks: a row that reads the
  four-momentum in GeV subclasses `FourMomentumModule` and calls `self._register_denorm_scales()`
  in `build_net`, as `src/algorithms/lorentznet.py` does.
- Add `configs/algorithm/<name>.yaml` on the pattern of `deepsets.yaml`: `_target_`,
  `_convert_: all`, a short `name`, `nbr_feats: ${len:${data.features}}`, the
  `method`/`group` interpolations from the data config, the architecture arguments, and an
  optimiser and scheduler as Hydra partials.
- Add `configs/experiment/hls4ml/<name>_{16,50,150}.yaml`, the search configs
  `configs/hparams_search/hls4ml/<name>_optuna.yaml`, which composes the shared sweeper
  `configs/hparams_search/optuna.yaml`, and `<name>_scl_optuna.yaml`, which composes the first,
  and the three catalogues `run<name>_{search,train,eval}.sh`, under
  `scripts/experiments/hls4ml/px_py_pz_e/` for a grid row and at the root of
  `scripts/experiments/hls4ml/` for any other. An architecture
  with an embedded symmetry follows LorentzNet and PELICAN instead: the baseline regime alone, so
  no `_scl_optuna` config and three blocks per catalogue rather than 21. A third case, a row that
  runs the baseline regime alone but reads `data.features` (JEDI-linear until 2026-10-02), goes in
  `BASELINE_ONLY`, not `EMBEDDED`, in both `scripts/experiments/make_train_catalogues.py` and
  `tests/conftest.py`, has the `_optuna` config alone, and needs a search catalogue under every
  feature-set directory, `px_py_pz_e/`, `pt_eta_phi/` and `pt_etarel_phirel/` of
  `scripts/experiments/hls4ml/`.
  A row of the grid that reads `data.features`, as the Linformer does, needs both search configs
  and a 21-block search catalogue under every feature-set directory. A variant of a grid row pinned
  to columns of its own, as the moment rows are, gets a subclass of its own that sets
  `required_features`, so that the grid row keeps reading `data.features`, and an algorithm config
  that sets `nbr_feats` to a number, since `tests/conftest.py::algorithm` builds the rows of the
  tests with the four-momentum as `data.features` by default.
- Extend `tests/test_algorithms.py`, which steps every row of `MODELS` in `tests/conftest.py` and
  reads its columns out of a full-width batch, and add the row's state-dict digests to `LAYOUTS`
  in `tests/test_configs.py`, which builds every experiment and holds its layout to them.
- To join the top-tagging replica, a row needs `configs/experiment/top_tagging/<name>_100.yaml` on
  the pattern of the 18 there (`output_dim: 2`, the `top_tagging` data and results-table
  overrides, the trainer precision and TF32 setting of its hls4ml experiment at N = 150) and its
  name in `TOP_TAGGING` in both `scripts/experiments/make_train_catalogues.py` and
  `tests/conftest.py`.
  A row on a published recipe also goes in `TOP_TAGGING_RECIPES` and carries the recipe in its
  experiment; a searched row gets `configs/hparams_search/top_tagging/<name>_optuna.yaml` and the
  hand-written `run<name>_search.sh` in `scripts/experiments/top_tagging/`, under the directory of
  each feature set for a grid row. The generator then writes its train and
  eval catalogues under `--dataset top_tagging`.

Every one of the removed modules will need two things on the way in. They import
`pytorch_lightning.utilities.memory.garbage_collection_cuda`; the `pytorch_lightning`
namespace is not used anywhere in this repository any more, and the base class handles memory
without it. They also accept `method="equivariance"`, which no longer exists: the supported
methods are `augmentation` and `symm_loss`.

## LorentzNet

Kept at the refactor, so it needs no porting notes; this section holds the measurements behind its
batch sizes. Its edges join real constituents alone, so memory follows the squared multiplicity,
whose mean is 1 800 at N = 50 and 2 800 at N = 150. On a 32 GB V100, 1024 jets take 22 GiB at
N = 50 and 1536 do not fit, and 512 jets take 17 GiB at N = 150, so the experiments set half the
usual batch of 2048 at N = 50 and a quarter at N = 150. A larger card would take more. In the
search, `n_hidden` and `n_layers` bracket the paper's (72, 6) LGEB stack from below, and
`c_weight`, which scales the Minkowski-coordinate update against the scalar one in each LGEB,
runs log-scaled up to 1: under the tanh gate the per-edge step saturates at |x_diff|, so at 1 a
node moves by a whole mean edge difference per block.

## PELICAN

**Back since 2026-09-13, as a clean rewrite**, not as the vendored copy below:
`src/algorithms/pelican.py`, with
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
and catalogues; the 150-constituent row is not searched (to spare the H100 week) and its
experiment config carries the beams-on optimum, at the corner of the space where that study's top
eight trials all sit, so a search of the beams-off twin would land there too.

The pair tensor costs B N² per channel, and every slot is built whether or not it holds a
constituent, so memory follows the squared slot count (N + 2 with the beams) rather than the
multiplicity. The experiments of both rows therefore cut the usual batch of 2048 jets to a quarter
at N = 50 and to a sixteenth at N = 150. A V100 pre-flight at the largest search point (5 blocks,
35/35) at those batches peaked at 3.7 GB and 0.12 s per step at N = 16, at 7.3 GB and 0.21 s at
N = 50, which fits an 11 GB slice, and at 15.2 GB, 0.42 s and 27 min per epoch at N = 150
(2026-09-13). N = 150 needs a full card or a third of an H100, never an 11 GB slice, and its study
was the costliest of the three.

`nbar` (`configs/algorithm/pelican.yaml`) normalises the aggregators' (N/nbar)^α scaling: the
training jets' mean real multiplicity at the truncation, 15.98, 41.43 and 49.44 at N = 16, 50 and
150 (record cache, 2026-09-13), plus the two beams when they are on. `n_chunks` sorts the batch
by multiplicity and cuts it into that many chunks, each trimmed to its own widest jet, which drops
the padded pair slots of all but the widest chunk; the logits, the gradients and the pooled
normalisation statistics are those of the whole batch, exact up to the order of the summations,
and the gain follows the padding, none at N = 16. It is two at N = 150, where the pair tensor is
dense over the 152 slots while the mean real multiplicity is 49.4, so that 89 % of it is padding.
On one V100 at a batch of 128 (2026-09-15), a step took, in ms:

| chunks                                 | default point, 5 × 25/15 | largest search point, 5 × 35/35 |
| -------------------------------------- | ------------------------ | ------------------------------- |
| none, dense over the 152 slots         | 192                      | 346                             |
| one, trimmed to the batch's widest jet | 123                      | 223                             |
| two                                    | 83                       | 142                             |
| four                                   | 127                      | 155                             |
| eight                                  | 235                      | 238                             |

Past two chunks the kernel launches of the small chunks dominate a dedicated card, but three
drivers time-sharing one card hide them: three drivers at four chunks reach 13.9 steps per second
in aggregate, against 11.8 at two, so four is worth trying on a shared card
(`algorithm.n_chunks=4`). Peak memory falls from 8.4 GB dense to 4.6 GB at two chunks and 2.8 GB
at four (15.2, 8.3 and 5.1 GB at the largest point), so a trial fits a 1g slice.

The rest of this section is the record of the vendored copy that was removed at the refactor,
kept as history.

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
is the wrapper `LGATrSlimClassifier` and the `JetTaggingModule` subclass `LGATrSlim` in
`src/algorithms/lgatr_slim.py`, which reads `[px, py, pz, e]` and hands the network
(E, px, py, pz) in GeV. Around them sit `configs/algorithm/lgatr_slim.yaml` and
`lgatr_slim_lorentz.yaml`, the six experiment configs
`configs/experiment/hls4ml/lgatr_slim{,_lorentz}_{16,50,150}.yaml`, the two search configs
`configs/hparams_search/hls4ml/lgatr_slim{,_lorentz}_optuna.yaml` and the six catalogues
`scripts/experiments/hls4ml/runlgatr_slim{,_lorentz}_{search,train,eval}.sh`. The wrapper
divides the
constituents by `units` (20 GeV), prepends a fixed global token (a zero four-vector carrying a
single scalar flag) and the spurion tokens, passes a dense boolean key-padding mask, and reads
the logits from the global token's scalar outputs. The constituents enter as the null vectors
they are, with no mass regulator, and the experiments run in fp32, since the package's
documentation advises against mixed precision for the slim network. Compiled, a training step
runs 1.7 to 2.1 times faster than eager on a V100 (2026-09-29), with logits within 4e-7 of the
eager network's, after a one-off compile of 15 to 55 s per fit.

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

A V100 pre-flight at the largest search point (12 blocks, 32/96 channels, 8 heads, `mlp_ratio` 4,
2.0 M parameters), on a shared card on 2026-09-28, took 12.9 GiB and 0.33 s per step at N = 16 with
the usual 2048 jets, which fits a full card but not a 1g MIG slice, and 18.1 GiB and 0.44 s at
N = 50 with 1024 jets. At N = 150 the attention scores cost B × heads × 154² per block: 256 jets
took 14.3 GiB and 0.29 s, and 512 reach 28.5 GiB, too close to the 32 GB of the card. The
experiments of both rows set these batches.

## LLoCa

Added on 2026-09-28 as the fourth symmetry-embedded architecture, and the second whose network is
not written here: LLoCa, Lorentz local canonicalisation (arXiv:2505.20280 for the method,
arXiv:2508.14898 for its use in particle physics), comes from the pip package `lloca` 2.0.0, which
supplies the frames-net, the frames and the LLoCa-Transformer. Our own code is the wrapper
`LLoCaClassifier` and the `JetTaggingModule` subclass `LLoCa` in
`src/algorithms/lloca.py`, which reads `[px, py, pz, e]` and hands the network (E, px, py, pz) in
GeV. The four rows are named `lloca_<row>`, with `<row>` one of `lorentz`, `z`, `so2` and `so3`.
Around the code sit the four algorithm configs `configs/algorithm/lloca_<row>.yaml`, the twelve
experiment configs `configs/experiment/hls4ml/lloca_<row>_{16,50,150}.yaml`, the four search
configs `configs/hparams_search/hls4ml/lloca_<row>_optuna.yaml` and the twelve catalogues
`scripts/experiments/hls4ml/runlloca_<row>_{search,train,eval}.sh`. The rows train compiled
(`compile: true`),
as their searches at 50 and 150 constituents ran (those at 16 ran eager): a step takes 25 to 35 %
less time and about 10 % less peak memory than eager on a V100 (2026-09-30), with logits within
about 1e-6 of the eager network's, after a one-off compile of 1 to 1.5 min per fit. The wrapper
divides the constituents by `units` (20 GeV) and runs the frames-net over the real constituents
alone, in the package's packed layout, because its edge convolution takes no padding mask. Each
constituent's four-momentum is
then canonicalised into its local frame and handed to the LLoCa-Transformer with a dense boolean
key-padding mask and the jet momentum as the reference of `preserve_variance`; the logits are the
mean of its outputs over the real constituents. A mass regulator lifts every constituent lighter
than `mass_reg` (2.5e-4 in those units, the references' 5 MeV) to that mass inside the frames-net
alone; the lift does not commute with a boost, so the boost-exact rows keep a residual of relative
size m²/(2p²), below 1e-4 for every constituent above 0.35 GeV. The frames-net receives the packed
constituents in float64, as the references feed it: in float32 the mass floor sits below the
rounding of |p|^2 and the massless constituents' pair masses fall to zero, where the frames-net's
sqrt has no finite gradient, which turned the first smoke run's logits to NaN. The backbone and
the experiments run in fp32, as the references train without mixed precision. Both cost counters
run the network on the CPU at construction, where the LLoCa-Transformer's
`F.scaled_dot_product_attention` dispatches `aten._scaled_dot_product_flash_attention_for_cpu`, a
kernel torch's FLOP counter does not know. `utils/flops.py` prices it through a custom mapping,
which restores the attention's 524 288 multiply-adds per jet at N = 16 (about 3 % of the row's
FLOPs), and `utils/bitops.py` prices it as the math path would compute it. The wrapper's
`framesnet_dropout`, 0 and without a layer in every hls4ml row, serves the top-tagging recipe (see
"Top-tagging replica").

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

A V100 pre-flight at the largest search point (12 blocks, 8 heads of 8x0n+2x1n, `mlp_factor` 4,
frames-net hidden 64, 2.4 M parameters; 2026-09-28) reserved 5.5 GiB and took 0.15 s per step at
N = 16 with the usual 2048 jets, which fits a 1g MIG slice, 9.6 GiB and 0.23 s at N = 50 with 1024
jets, which fits a full card and, with little margin, a 1g slice, and 12.3 GiB and 0.39 s at
N = 150 with 512 jets, which fits a full card but not a 1g slice. The experiments of the four rows
set these batches.

## Vector neurons

Added on 2026-10-03 in one row, `vn`, which replaced the two four-momentum rows `vn_z` and `vn_so2`
of 2026-09-29 (see "Vector neurons on the four-momentum" below). Unlike those of L-GATr-slim and
LLoCa, its network is written here, so it adds no dependency. Vector neurons (Deng et al.,
arXiv:2104.12229) carry lists of vectors as hidden features and mix them with a bias-free linear
layer that acts on the channel index alone, so that the layer commutes with the group; here the
vectors are lifts of the constituents' (p_T, η, φ), and the group is the Euclidean group of the
(η, φ) plane. Our code is the network `VectorNeuronsClassifier` and the `DenormalisedModule`
subclass `VectorNeurons`
in `src/algorithms/vector_neurons.py`, which pins `required_features` to `[pt, eta, phi]` and whose
inherited `_prepare_inputs` returns the batch to GeV and radians. Around them sit
`configs/algorithm/vn.yaml`, the three experiment configs
`configs/experiment/hls4ml/vn_{16,50,150}.yaml`, the search config
`configs/hparams_search/hls4ml/vn_optuna.yaml` and the catalogues
`scripts/experiments/hls4ml/runvn_{search,train,eval}.sh`. It is the one embedded row that does not
read the
four-momentum: `setup()` refuses any list but `required_features`, so its experiment configs set
`data.features: [pt, eta, phi]`. Its catalogues and run directories sit in the default layout all
the same, and its results land in `results/embedded/`.

The row is meant for the Level-1 trigger of the CMS Phase-2 upgrade, which delivers every
particle-flow and PUPPI candidate as (p_T, η, φ): p_T in 16 bits and η and φ in 10 bits each, with
no Cartesian components (CMS-TDR-021, Sec. 3.5). A network on the four-momentum, as every other
embedded row is, would first need trigonometric and hyperbolic lookup tables. This one holds only
multiply-adds, ReLU and comparisons, with no lookup table, logarithm, square root or per-jet
division, the rule the FPGA rows have followed since 2026-10-01. It reads the constituents by p_T
rank, an order the trigger is designed to supply: in the TDR the Layer-2 algorithms "assume that
within a single region the inputs are sorted by pT", and "the sorting is currently not implemented
in Layer-1, but it is expected to not take a considerable amount of resources"; the CMS DeepSets
tagger of arXiv:2509.24371 labels its inputs "16 x PUPPI Candidates pT sorted". The order the TDR
describes is per region and per particle type, so one merge gives the order across types.

Each constituent is lifted to u = p (1, η, φ, η² + φ²), where p is its p_T over `units` (20 GeV,
the scale of the L-GATr-slim and LLoCa rows) and the angles are measured from the leading
constituent, φ wrapped into the range from −π to π. A shift of the plane acts on the lift by a
fixed linear map T, and a rotation about the origin by an orthogonal map on its (η, φ) components.
Both preserve the form B(u, v) = uᵀ G v, with G₀₃ = G₃₀ = 1, G₁₁ = G₂₂ = −2 and every other entry
zero, and both leave the first component, p, alone. On two constituents
B(u_i, u_j) = p_i p_j ΔR_ij², and B(u, u) = 0 for every lift. A change of reference is itself a
shift, so the logits do not depend on it; the leader is taken because it moves with the jet and
keeps the angles small, so that the wrap of φ stays away from the jet and B is spared the
cancellation between large terms that absolute angles would cost in float32.

The network sorts the constituents of each jet by decreasing p_T and lifts them. The first block
reads the N sorted lifts as its vector channels, one weight per rank, and as its scalars, for every
rank, p_i and B(u_i, U), where U = Σ_i u_i is the lift of the jet, and for the jet, Σ_i p_i and
B(U, U), together with the stream and the tensor invariants described below. From there on the
state belongs to the jet, and every block runs once per jet. A block mixes the vector channels with
`ChannelLinear`, one bias-free weight matrix shared by the four components of the lift. It then reads
the forms B(L_k u, R_k u) of pairs of learned combinations of the mixed channels (16 by default)
and the first component of every mixed channel, a weighted sum of p, and updates the scalars with a
BatchNorm and a two-layer ReLU MLP of the previous scalars, those invariants and the context. Last,
it gates each mixed channel, its four components alike, by a hard sigmoid of the new scalars. The
forms and the p sums are the only invariants the network reads, and a gate scales a whole channel,
so every operation on a vector commutes with the group. The gates start at exactly 1/2, their
linear layer being initialised to zero, because a gate started near zero would stay there: the
first block has no residual, and the forms after it are quadratic in the vectors, so their gradient
vanishes with the vectors. Every block after the first adds its input to both states. The context,
the number of real constituents over the constant N, reaches every block and the head, an MLP
behind a BatchNorm that reads the final scalars, the forms and p sums of the final vectors and the
context. Every BatchNorm folds into the dense layer after it at inference. One block is the
default, since three gave the same accuracy (below) at about twice the FLOPs.

Two branches feed the first block. The stream is the per-constituent network of JEDI-linear
(arXiv:2508.15468) with one context layer, run over the two scalars of each rank, p_i and
B(u_i, U): a dense ReLU layer embeds them as h_i, and the stream returns
Σ_i ReLU(W_1 h_i + W_2 Σ_j h_j + w) over the real ranks (16 channels by default). Both of its pools
are sums, which need no per-jet division. The tensor invariants are three-point correlators. With
n_i the lift at unit p_T and t_i = u_i n_iᵀ, each of K learned weightings of the ranks forms the
vectors L = Σ_i a_i u_i and R = Σ_i b_i u_i and the tensors M, M′ and M″ = Σ_i (c_i, d_i, f_i) t_i,
and from them

```
chain     Lᵀ G M G R         = Σ_ijl a_i c_j b_l p_i p_j p_l ΔR_ij² ΔR_jl²
pair4     tr(G M G M′)       = Σ_ij  c_i d_j p_i p_j ΔR_ij⁴
triangle  tr(G M G M′ G M″)  = Σ_ijl c_i d_j f_l p_i p_j p_l ΔR_ij² ΔR_jl² ΔR_li²
```

A shift or a rotation S maps t_i to S t_i Sᵀ, and Sᵀ G S = G, so all three are invariant. They are
the β = 2 energy flow polynomials (arXiv:1712.07124) of the two-edge path, the doubled edge and the
triangle, with learned weights per rank in place of the energy fractions, and they cost O(N), as
energy flow moments do (arXiv:1911.04491), where the sums over pairs and triples would cost O(N²)
and O(N³). The first block reads K = 2 of each kind by default.

The logits are exact under shifts of η and of φ, across the wrap of φ, under the reflections
η → −η and φ → −φ, under rotations of the plane about any point and under any permutation of the
constituents. A rotation about the beam shifts every φ by one angle, and a boost along the beam
shifts every rapidity by one amount, which for a massless constituent is a shift of η, so the probe
groups `so2` and `lorentz` are exact. `so3` and `lorentz_iso` change every p_T and move the
constituents in the plane by different amounts, and are measured. The rotation of the plane has no
probe of its own. To second order in ΔR the lift is the constituent's null four-momentum in
coordinates local to the jet and B is 2 p_i · p_j, so this rotation is the rotation about the jet
axis; to the same order a boost along the jet axis rescales the components of the lift and
preserves B. That boost is left out on purpose: the network reads p itself, in its p sums and in
the unit lifts of its tensor invariants, and the boost rescales p. In single runs at N = 16 and 50
(2026-10-01, three blocks, no tensor invariants), reading no p_T sum cost 7.7 and 4.2 points of
validation accuracy, and reading the jet's Σp alone cost 1.5 and 0.8.

In float32 the probe's own arithmetic sets the floor. The transform recomputes p_T, η and φ from
the moved momenta, so a boost along the beam rounds η afresh, and a rotation about the beam
recomputes p_T from the rotated px and py, so two constituents whose p_T agree to rounding can swap
ranks. On a three-epoch smoke model of the committed row (2026-10-03, test set, N = 16, test
accuracy 0.709), the flip rates were 0.000 % under `so2`, 0.01 % under the beam boost, 41.0 % under
`so3` and 41.6 % under `lorentz_iso`; the version without tensor invariants gave 0.000, 0.013, 40.2
and 40.8 % on 2026-10-01. The experiments run in
fp32, since the forms of the learned channels cancel to a small fraction of their terms, which fp16
cannot resolve, and the tensor invariants, cubic in p_T, widen the range further; fixed point on an
FPGA will need range handling for the same reason, which is open. They also leave
`extras.matmul_precision` unset, which `tests/test_configs.py` checks: TF32 rounds the inputs of
every float32 matmul to about 5e-4, and the linear layers here act on the vector components
themselves, so a transformed jet would be rounded differently and the embedded group would hold to
that precision only.

The optimiser is AdamW with OneCycleLR, the block of L-GATr-slim and LLoCa, at a default learning
rate of 2e-3 and a weight decay of 1e-2, and the experiments train for 200 epochs on batches of 2048
with 5 folds. The search, `vn_optuna.yaml`, runs one study per constituent count,
`vn_nconst<N>_augmentation_none`, in `logs/optuna/hls4ml/vn.db`, each of 200 trials of 50 epochs,
none of them launched yet (2026-10-03). It draws the blocks from {1, 2, 3}, the vector channels from
{8, 16, 32}, the scalar channels from {32, 64, 128}, the forms from {8, 16, 32}, the tensor
invariants of each kind from {1, 2, 4}, the stream width from {8, 16, 32}, the head from {[32],
[64, 32], [128, 64]}, the dropout from {0, 0.1, 0.2}, the learning rate log-uniformly from 1e-4 to
5e-3 and the weight decay from {0, 1e-4, 1e-3, 1e-2, 5e-2}, so the default lies inside the space;
`units` is never searched. The train and eval catalogues hold three blocks each, all still
"transferred from experiment config (no study yet)".

The design was measured on olgpu between 2026-09-30 and 2026-10-02 with a scratch copy of the
network whose switches, set to this row's choices, compute the committed network (to 9e-16 with
copied weights, checked on 2026-10-03); the run records are in
`~/Research/local/claude/scratchpad/jet_tagging_gdl/vnx/rounds/` on the Mac. Each number below is
the best validation accuracy of a 50-epoch run on the record's own split (`data.n_folds=1`) at the
row's default sizes, with the number of blocks stated, from one seed unless marked; two seeds are
given as mean ± standard deviation, and seeds differed by up to 0.0011. At N = 16, 50 and 150 the
row reached 0.7203 and 0.8195 with one block (N = 150 was not run with one) and 0.7202 ± 0.0004,
0.8199 ± 0.0004 (two seeds each) and 0.8250 with three. With three blocks and no tensor invariants
it reached 0.7155 ± 0.0001, 0.8168 ± 0.0008 (two seeds) and 0.8222, so the tensor invariants are
worth 0.5, 0.3 and 0.3 points; without the tensor invariants and the vector channels, the stream and
the scalars alone reached 0.7118 and 0.8106 (two seeds), so the vectors are worth 0.4 and 0.6 points
there. One, two, four and eight tensor invariants of each kind lie within 0.05 points of one another
at N = 16 and 0.15 at N = 50. Forms of every rank against learned axes, read by the stream, gained
0.30 and 0.17 points (two seeds) without the tensor invariants and nothing with them. On a precursor
that lifted η and φ separately, to p (1, η, η²) and p (1, φ, φ²) with weights of their own (three
blocks, no stream, 2026-09-30), ψ compression of the invariants (the signed log of the LorentzNet
component), sigmoid gates and a log2 on the invariants each moved the accuracy by less than 0.1
points, so none is used. One split beat the two by 0.05 and 0.25 points (two seeds), with a
train-validation gap of 0.6 and 0.7 points against 0.9 and 1.3. The |η| of the leading constituent
as an extra input gained 0.28 and 0.19 points (two seeds) but is not used: it gives up the beam
boost and fits the rapidity distribution of the sample.

For comparison, tuned study bests (best of the first 200 trials, 50 epochs) at N = 16, 50 and 150
are 0.705, 0.802 and 0.813 for DeepSets and 0.720, 0.823 and 0.832 for JetFormer on
`[pt, etarel, phirel]`, and 0.7305, 0.836 and 0.843 for `lgatr_slim` and 0.733, 0.838 and 0.845 for
`pelican` on the four-momentum; `jedi_linear` reached 0.7114, 0.8033 and 0.8125 on
`[pt, etarel, phirel]` in its first campaign (2026-09-30, sum and ReLU fixed, 40 to 90 trials per
study). Untuned, the row matches JetFormer at N = 16 and sits 0.3 and 0.7 points below it at 50 and
150, and 1.0 and 1.3 points below `lgatr_slim` and `pelican` at N = 16. The design was chosen on
these validation accuracies, over about a hundred runs, so the studies and the test folds, not
these numbers, will settle the comparison.

At the defaults the row has 21.9 k, 27.3 k and 43.1 k parameters and costs 0.064, 0.105 and
0.223 MFLOP per jet at N = 16, 50 and 150, by the FLOP counter, which counts matrix products
alone, at two FLOPs per multiply-add, and so leaves out elementwise products such as the last step
of every form and the gates (measured on 2026-10-03). At the corner of the search space (3 blocks,
32 vector and 128 scalar channels, 32 forms, 4 tensor invariants of each kind, a 32-wide stream,
head [128, 64]) it has 178 k, 189 k and 220 k parameters and costs 0.46, 0.58 and 0.92 MFLOP. The
cost grows with N through the first block's rank inputs, vector and scalar, the stream and the
tensor invariants alone. A V100 pre-flight at the corner, in fp32 on batches of 2048 fully occupied
jets, reserved 0.09, 0.14 and 0.28 GiB and took 11 ms per step at every N (7.4 ms at the
defaults), so every study fits a 1g MIG slice. At N = 16, `jedi_linear` costs 0.29 MFLOP at its
published point and PELICAN 4.5 MFLOP.

A search on 2026-10-03, over the web and the INSPIRE citation lists of the closest papers, placed
the row in the literature. The tensor invariants are the energy flow moments of Komiske, Metodiev
and Thaler (arXiv:1911.04491) with learned weights. Their Sec. IV notes: "While we use the notation
of Lorentz indices and the Minkowski metric here, the conclusions hold for any inner product among
vectors using any choice of metric", and their Sec. VI C: "While most current LHC analyses focus on
β = 1 observables, comparable (or better) performance can be obtained using β = 2". The three
contractions are the two-edge path, the doubled edge and the triangle of their Table II, with
learned per-rank weights in place of the energy fractions; equal weights give the unnormalised
β = 2 energy flow polynomials of arXiv:1712.07124, and per-rank weights are not collinear safe.
PELICAN (arXiv:2307.16506, App. B) cannot form the triangle with its default layers: its
single-index aggregators generate only "starfish" graphs, and "even something as simple as the
triangle graph can never be generated due to the fact that it requires three free indices at an
intermediate step"; adding a matrix-product aggregator gave "no performance improvements". The form
is PELICAN's small-angle relation read backwards: "the Lorentz-invariant version of the polynomial
can then be obtained by replacing each θ_ij² with 2 d̂_ij". Exact E(2) in the (y, φ) plane by
architecture exists for offline IRC-safe message passing (arXiv:2402.12449, arXiv:2509.22059), with
pairwise MLP messages and fixed p_T fractions, and BIP (arXiv:2207.08272) builds O(N) polynomial
invariants in the frame of the jet axis with logarithms and exponentials. The gated non-linearity is
that of Weiler et al. (arXiv:1807.02547); the vector-neuron projection of Deng et al.
(arXiv:2104.12229) divides by the length of a learned direction, which an indefinite form such as B
may send to zero. The search found no prior use of the lift (1, η, φ, η² + φ²) with this form for
jets or for FPGA networks, and no energy flow moments with learned weights inside a trainable
equivariant network; that is an absence after a search, not a proof.

`tests/test_algorithms.py` checks, on the network in float64 and eval mode with its gates drawn at
random, that through `forward`, on massless record jets, the probe's `so2` and `lorentz` transforms
move the logits by less than 1e-6 (9e-8 and 2e-8, the float32 rounding in `forward`), and `so3` and
`lorentz_iso` by more than a thousand times the `so2` shift. It also checks the form and the tensor
invariants against their closed forms, that a padded slot changes nothing (an eight-slot network
against a nine-slot copy) and that the batch is de-normalised. `tests/test_configs.py` checks that
the experiments leave `extras.matmul_precision` unset.

## Vector neurons trained quantisation-aware (`vn_hgq`)

Added on 2026-10-07 in one row, `vn_hgq`, on a plan agreed with Patrick on 2026-10-06 and
2026-10-07 (`~/.claude/plans/the-vector-neuron-networks-drifting-meadow.md` on the Mac). The row
measures what the `vn` network costs in a Level-1 trigger: its latency and resources on the VU13P
(`xcvu13p-flga2577-2-e`) at a clock of 5 ns, the part and clock of Odagiu et al. 2024
(arXiv:2402.01876). The network is trained quantisation-aware with HGQ2 and compiled to Verilog by
alkaid, the successor of da4ml, whose flow JEDI-linear (arXiv:2508.15468) used for its published
designs. The two alternatives, read on 2026-10-05, do not fit the port. hls4ml 1.3 has no handler
for HGQ2's `QSum`, confines the two-input Einsum to the Latency strategy and maps a hard sigmoid to
a slope of 0.2, where Keras 3 and torch use 1/6 (read in its source, not tried). TriggerFlow 0.3.11
supports the uGT alone and pins Keras 3.11, below the 3.15 that hgq2 0.2.0 needs. Our code is the
port and `VectorNeuronsHGQ` in `src/algorithms/vector_neurons_hgq.py` and the weight copy
`src/algorithms/utils/vector_neurons_port.py`. The port is built on the shared helpers of
`components/hgq_port.py`, which sets Keras's backend to JAX before it imports Keras and which every
module of a port therefore imports first, and `VectorNeuronsHGQ` is a subclass of
`KerasQuantisedModule`
(`src/algorithms/keras_base.py`, the Keras machinery shared with `jedi_linear_hgq`, whose
section below describes the split) and of `VectorNeurons`. Around them sit
`configs/algorithm/vn_hgq.yaml`, a standalone copy of `vn.yaml` that `tests/test_configs.py` keeps
equal to it up to `units`, the optimiser, the scheduler and the keys of the port and its training,
`configs/callbacks/quantised.yaml`, the experiment configs
`configs/experiment/hls4ml/vn_hgq_{16,50,150}.yaml`, the hand-written catalogues
`scripts/experiments/hls4ml/runvn_hgq_{train,eval}.sh`, `scripts/synthesis/synthesise_vn.py`
with its
`requirements.txt`, and `scripts/analysis/compare_outputs.py`. The dependencies form the optional
Poetry group `hgq`, installed with `poetry install --no-root --with hgq` and absent from the NGT
image: Keras 3.15, hgq2 0.2.0, alkaid 0.8.1, JAX 0.10.2 with its CUDA 12 plugin, and nvcc 12.6.85,
pinned so that ptxas matches torch's CUDA runtime. In `scripts/experiments/make_train_catalogues.py`
and
`tests/conftest.py` the row forms `QUANTISED`, outside `MODELS`: it has no study, and no catalogue
of it is generated.

Lightning keeps the loop, the data, validation, testing and the invariance probe; Keras 3.15 on
JAX owns the network, its optimiser and its training step. Patrick ruled out Keras's torch
backend. `net` is a `KerasNetwork`, an `nn.Module` without torch parameters whose `forward` hands
the prepared batch to Keras's `predict_on_batch` and returns the logits as a torch tensor, so that
`model_step`, validation, the test, the probe, `eval.py` and `write_results_row` run unchanged.
Only training is overridden. Under manual optimisation each batch is one call of Keras's
`train_on_batch`, which in Keras 3.15.1 is the jitted step of `fit`, the layers' EBOPs losses
included (read in `keras/src/backend/jax/trainer.py`). The optimiser is Keras's AdamW, with a
weight decay of 0.05 that spares the quantisers' variables, `global_clipnorm` 1 and a cosine decay
from 1e-3 to 1e-5 stepped per batch. `configure_optimizers` compiles it into the model and returns
None, so Lightning puts its mock optimiser in place of a torch one, and `training_step` calls that
optimiser's `step` once per batch: the step does nothing but advance `global_step`, without which
`ModelCheckpoint` never saves. The experiments therefore run at `precision: 32`, since 16-mixed
would pass the step through a gradient scaler with no gradient to check, and at
`gradient_clip_val: null`, which manual optimisation requires. The Keras weights are stored in the
Lightning checkpoint (`on_save_checkpoint`, and `on_load_checkpoint`, which casts them to the
port's float type); the Keras optimiser state is not, so a run cannot resume mid-way. β, the
weight of the EBOPs in the loss, is assigned to every quantised layer at the start of each epoch,
as HGQ2's `BetaScheduler` does, constant or from HGQ2's `PieceWiseSchedule` over `beta_schedule`.
The epoch means of `train/loss` (the cross entropy), `train/acc` (Keras's micro accuracy, not the
macro accuracy of the other rows) and `train/loss_total` (with the EBOPs and bit penalties) are
logged, and per epoch `train/ebops`, `train/beta` and `train/lr`. `setup()` refuses the SCL, which
the Keras step does not compute. Three edits to shared code serve the row and change nothing for
the others: `JetTaggingModule.__init__` prices a model through the hook `_record_costs`, `forward`
reads a batch through `_network_inputs` (the cast, the width guard and `_prepare_inputs`), and
`HLS4MLDataModule.load_split` returns one split of the record at full width, rebased as `setup()`
rebases it.

The port starts at the rows [p_T in GeV, Δη, Δφ] of the constituents sorted by decreasing p_T,
the angles measured from the leading constituent and zero on a padded row, which `_prepare_inputs`
computes in torch. The sort, the leader subtraction and the wrap of φ are integer pre-processing of
the trigger words, outside the synthesised block, so the latencies and costs below leave them out.
`_prepare_inputs` also clamps each column to the range of its word, because the words wrap: alkaid
0.8.1 traces no input quantiser but a WRAP one. Vectors are stored as (jets, 4, channels),
components first, so that every channel mixing is a dense layer on the last axis. In the order of
`VectorNeuronsClassifier.forward`:

| stage                    | layers                                                        | quantisers                    |
| ------------------------ | ------------------------------------------------------------- | ----------------------------- |
| words                    | `Quantizer` `pt_word` and `angle_word`                        | fixed kif, WRAP               |
| p, keep, count n         | p_T / 16, min(4 p_T, 1), `QSum` `count`                       | exact; n one fixed bit        |
| lifts, jet, rank forms   | `QEinsum`, `QSum`                                             | datalanes                     |
| stream                   | `QBatchNormalization`, `QDense`, `QMultiply`, `QSum`, `QAdd`  | datalanes, weights            |
| tensor invariants        | `QEinsum`, bias-free `QEinsumDense`                           | datalanes, weights            |
| channel mixing           | `QEinsumDense` (first block), bias-free `QDense` (later ones) | datalanes, weights            |
| forms                    | two bias-free `QDense`, `QEinsum`                             | datalanes, weights            |
| scalar update, residuals | `QBatchNormalization`, two `QDense`, `QAdd`                   | datalanes, weights            |
| gates                    | `QDense`, `Quantizer` of `HardSigmoidConfig`, `QEinsum`       | gate fixed at 8 fraction bits |
| head                     | `QBatchNormalization`, `QDense` layers, `head_last`           | logits fixed kif (1, 6, 8)    |

The p_T word is the trigger's `pt_t` (CMSSW `DataFormats/L1TParticleFlow/interface/datatypes.h`),
`ap_ufixed<14,12>` or kif (0, 12, 2): [0, 4096) GeV at an LSB of 0.25 GeV. The angle words, kif
(1, 1, 8), hold [−2, 2) rad at 2⁻⁸ rad in 10 bits, the width of the trigger's local angles, whose
LSB of π/720 they approximate with a power of two. The dry run of 2026-10-07 found p_T up to
1634 GeV and |Δη| and |Δφ| up to 1.77 and 0.83 on the record. The word rounds to nearest, so a
constituent below 0.125 GeV becomes a zero word and is read as padding. p is counted in units of
16 GeV, a power of two, so that it is an exact shift of the word, and the multiplicity enters as
the count n rather than n/N; the copy folds the float row's 20 GeV and the 1/N into the batch
normalisations. The metric G is a slice, a factor −2 and a concatenation, (x₃, −2 x₁, −2 x₂, x₀),
exact in Keras and in alkaid. The keep flag is min(4 p_T, 1), exact because a real p_T is a
multiple of the LSB; the float network's flag is a cast of p ≠ 0, and alkaid has no handler for a
cast of a boolean. The gate is HGQ2's `HardSigmoidConfig` quantiser applied to (2/3) z. It returns
q/4 + 1/2, q being the rounded input saturated on [−2, 2), which equals torch's relu6(z + 3)/6 to
within 2^−(f+2) for the gate's f = 8 fraction bits and costs a clamp, a shift and an add; Keras's
`hard_sigmoid` would become a lookup table in alkaid. The gates start at exactly 1/2, as in torch.
Each batch normalisation is a `QBatchNormalization` followed by a `QDense`, built with torch's
epsilon 1e-5 and momentum 0.9 (torch's 0.1), and alkaid replays it as one quantised scale and
offset per channel; the fused layers do not reproduce torch (see the caveats below). A dropout
layer, absent at the configs' rate of 0, would be replayed by alkaid as the identity.

`copy_float_weights` copies a trained `vn` into the port, which in eval mode and in `float_mode`
(every quantiser but the gate's clamp the identity) then reproduces its logits. Every channel
mixing takes the transposed weight, the stacked rank weights of the tensor invariants, (3, K, N),
become an (N, 3K) kernel, and the gate's weight and bias are scaled by 2/3. A batch normalisation
that reads a channel scaled by s keeps its output if its running mean becomes s·mean and its
running variance s²(var + 1e-5) − 1e-5, where s is λ^d for a channel of degree d in p, λ = 20/16
being the ratio of the units, and N for the count; `_first_segments`, `_read_segments` and
`_copy_stream` list the degrees. The first block's normalisation and the kernel rows of the first
layer of its MLP are de-interleaved, because torch's `r.flatten(1)` reads p₁, f₁, p₂, f₂, …, where
the port reads the N values of p before the N forms f. `algorithm.init_from` names the float run, as
a run directory, its manifest or a `.ckpt`, and `setup("fit")` raises unless its architecture keys
equal the port's, `units` aside. On the main checkpoint of the two-block `vn` finals at N = 16, the
float-mode port reproduced the torch logits to 3.6e-8 relative (dry run on olgpu, 2026-10-07).

Every input of a product, a sum or an affine map is a datalane: an HGQ2 kif quantiser with WRAP
overflow and rounding to nearest, per element, whose fraction bits are trained and whose integer
bits are traced. Kernels and batch-normalisation scales are kbi quantisers with SAT_SYM overflow,
and biases and offsets kbi quantisers with WRAP; the words, the keep bit, the count, the gates and
the logits are fixed. The model is built inside `LayerConfigScope(beta0=0.0, enable_ebops=True)`.
Before training, `setup("fit")` copies the float network, then `seed_weight_bits` gives every
weight quantiser `weight_bits` bits and the integer bits its copied value needs, so that no weight
saturates, but never fewer than `WEIGHT_I_MIN` (-12), which also bounds them in training
(HGQ2's `Min` constraint); and `seed_datalane_bits` gives every datalane i + f = `data_bits`,
with i traced by HGQ2's `trace_minmax` on the first `calibration_jets` (100 000) training jets.
The floor was added on 2026-10-10: a copied weight near zero was seeded at i = -23, hence 31
fraction bits, and alkaid 0.8.1 emulated such a layer wrongly (20 458 of the 20 480 logits of a
one-epoch export of the first `jedi_linear_hgq` port differed); with i floored at -16, -12 or -8
the same layer was exact. Every export before the floor whose emulation was checked was exact,
since training prunes those weights. It traces twice: the
first trace reads each lane through upstream lanes still at HGQ2's default of six fraction bits,
and a lane whose range the second trace widens would otherwise wrap. The defaults, 14 data and 8
weight bits, come from a table of seeded widths on the two-block finals at N = 16 (dry run,
2026-10-07): 14 and 8 bits cost 0.10 points of validation accuracy at 2.85 M EBOPs, 12 and 10
bits, the plan's first choice, 0.87 points at 2.99 M, and 8 data bits collapsed the network. The
largest traced integer width was 10 bits. The float logits reached 15.1 in magnitude, so the output
word, kif (1, 6, 8) with saturation, keeps two integer bits spare.

Under `quantisation.trained_words` (false by default, the port of every earlier checkpoint) HGQ2
trains the p_T, angle and offset words from the trigger's widths. Each is a WRAP kif quantiser
(`Quantisation.word`) with one width per slot, the two angles of a slot sharing theirs and the two
offset words one, so that a rotation of the (η, φ) plane acts on words of equal precision; a
standardisation per feature, as in the published JEDI-linear, would shear the plane, so the words
stay in GeV and radians. Their fraction bits, under the published L1 weight of 1e-8, lie in
[0, 12] (`WORD_F_RANGE`), so that a real p_T is at least 2⁻¹² and the keep flag, then
min(p_T · 2¹², 1), stays exact; their integer bits are traced as a datalane's.
`seed_datalane_bits` gives the words the datalanes' width, so `seed_word_bits` then resets their
fraction bits to the trigger's and traces afresh. In the trained-word D2 of block 42, near epoch
300, HGQ2 had made the angle words finer, 10 to 12 fraction bits where the trigger has 8, and left
the leader's slot, whose relative angles are zero, at 0. `tests/test_quantised.py` checks the
seeded widths and their shapes, and that a trained-word export emulates bit for bit and
co-simulates.

At test time `on_test_start` traces the integer bits of every WRAP quantiser afresh on the record's
whole training split, the range the hardware is built for, and stores the EBOPs of the traced port
as `ebops_total`, NaN until then. The split comes from `load_split("train")`, because
`trainer.test` runs `setup("test")` alone and `setup("fit")` would serve fold 1 of the pooled
splits; at N = 16 the trace took 106 s for its 496 k jets. The callbacks' hooks run first and read
the feature layout alone, so every test batch, the probe's included, meets the traced port. The
FLOPs are counted on the float network of the same architecture (`_record_costs`), so `flops_mean`
equals that of `vn`, while `ebops_mean` holds HGQ2's EBOPs of the traced port at its trained widths,
not the uniform 8-bit count of every other row. The rows therefore go to `results/quantised/`
(`results_table.subdir`), beside the float reference, whose `ebops_mean` is the 8-bit count. In
the dry run on the two-block port, 31 clean test jets left a range traced on the training split,
against 11 to 15 thousand jets under the probe's `so3` and `lorentz_iso` transforms; their lanes
wrap, and the flip rates of those groups measure the wrapped port.

The evaluation runs Keras in float64 (`floatx`, the default, which the eval catalogue repeats) with
JAX on the CPU, which `on_test_start` enforces. In float64 every value of the port is the
fixed-point value of the hardware, so the probe's `logits_clean` are the bit-exact reference of
the design; in float32 the products of quantised values exceed the 24-bit mantissa. Two numerical
faults stood in the way, both found in the dry run: Keras 3.15 computes floor and ceil in float32
on JAX, which the component patches for float64 tensors at import, and XLA's float64 `2**f` on the
V100 is off by one ulp for some f, which no patch reaches and which made about 2 % of the jets
round otherwise than alkaid. Under `eval.py` the module also writes `outputs/<label>_hgq/` beside
the probe's `outputs/<label>.npz`: `model.keras`, the traced port; `inputs.npz`, whose
`inputs_clean` holds the (jets, N, 3) words in the probe's jet order, exact in float32; and
`export.json`, the run behind the export (row, N, β, EBOPs, units, quantisation, float type, test
count, checkpoint, test accuracy and versions).

The protocol fine-tunes a float reference at one constant β per run and takes the last epoch as
the model. The reference is a one-block `vn` main run (`fold=null`) at the settings of the N = 16
study's best (16 vector and 32 scalar channels, 8 forms, 4 tensor invariants of each kind, a stream
of 8, head [64, 32], learning rate 4.99e-3, weight decay 0.05), trained by block 1 of
`runvn_hgq_train.sh` into `logs/train/hls4ml/vn/nconst16/baseline_1block`, which
`algorithm.init_from` of blocks 2 to 5 names. One block replaced the two of the finals on
2026-10-07, when Patrick decided that the paper shows the one-block scan alone: it comes within
0.1 points of the two-block best at about half the depth and cost. Blocks 2 to 5 run β ∈ {1e-8,
3e-8, 1e-7, 3e-7} for 100 epochs on batches of 2048 under `callbacks=quantised`, whose
`ModelCheckpoint` has `monitor: null` and `save_top_k: 1`, so that each epoch's file replaces the
last and the manifest points at the last epoch; nothing stops a run early, and no
`LearningRateMonitor` runs, since it reads a torch optimiser. The grid was fixed on the seeded
two-block finals, where β·E₀ over the initial cross entropy (E₀ = 2.85 M EBOPs, cross entropy
0.761) ran from 0.037 to 1.1; 1e-6 was dropped. The four β blocks share the label
`vn_hgq_nconst16_baseline`, so `--only` selects them by index. The first epoch, at a learning rate
of 1e-3, takes the copied solution from 0.72 to 0.66 in accuracy before it recovers; a lower rate
for the fine-tuning is untried. `val/acc` during training does not measure the model: validation
runs in eval mode on the integer bits that training re-traced on its last batch, so some
validation jets wrap, and in the scan `val/acc` swung between 0.45 and 0.71 while `train/acc`
stayed at 0.723. The `val_acc_best` of these runs' manifests is therefore meaningless, and the
calibrated evaluation is the measure. The experiment configs at N = 50 and 150 carry the
three-block architectures of the float finals; nothing has been trained at those counts.

Blocks 37 and 38 train D2 and A2 (blocks 19 and 24) with the recipe of the published JEDI-linear,
read in `configs/sweep-n16-f3.yaml` and `src/train.py` of its code, so that both can stand beside
its Tables I and II on equal terms. They start from the same float runs and keep the seeded widths,
the folded normalisations and the truncating datalanes. Training runs for 7000 epochs on batches
of 2790 with Adam, without weight decay or clipping. The learning rate falls along a cosine from
3e-3 to 1e-6 over nine tenths of every 100 epochs and stays at 1e-6 for the rest
(`CosineRestarts` in `utils/keras_schedules.py`), stepped per batch where theirs is set
per epoch. β rises linearly from 2e-8 to 3e-7 over 2000 epochs, then geometrically to 3e-6 at the
last epoch. JEDI-linear's published designs are points of a front of accuracy against EBOPs, not
last epochs, so these blocks keep a checkpoint every 50 epochs (`every_n_epochs` 50,
`save_top_k` −1). Every checkpoint is evaluated on the test split. The front points implemented are
those whose stage count, traced with JEDI-linear's pipelining, lies nearest the 23 and 19 stages of
its two tables. Picking by a hardware target leaves the test accuracy unselected. Their pipelining,
"every two adders" in da4ml 0.3, counts adders, where alkaid 0.8.1's cutoff counts its modelled
nanoseconds: two adders weigh about 2.7 there, at a clock of 2 ns with no uncertainty. Their Vivado
script is alkaid's but for the `-directive AreaOptimized_High` of `synth_design`, which those
projects drop.

Blocks 40 to 43 repeat blocks 37 and 38 with a checkpoint every epoch and the selection of the
JEDI-linear reproduction, the front of validation accuracy against EBOPs (next section): with the
fixed words in blocks 40 and 41 (`baseline_<tag>_foldtrn_jedirecipe_everyepoch`) and with trained
words in blocks 42 and 43 (`baseline_<tag>_foldtrn_trainedwords_jedirecipe_everyepoch`). All four
collapsed between about epoch 300 and epoch 1930, their training accuracy falling to 0.20, and
were stopped. The training accuracy first fell below 0.5 at epoch 1101 for D2, at 1136 for A2,
which recovered and kept front points to epoch 1928 at about 101 k EBOPs, at 304 for D2 with
trained words and at 1303 for A2 with trained words. The sudden collapses came four or five
epochs after a restart of the learning rate at 3e-3 (both trained-word runs, at 305 and 1304) or
30 to 36 epochs into a cycle, at a high rate (A2 at 1136 and 1930); D2 with fixed words declined
gradually. The trained-word D2 failed downstream of intact inputs: across its collapse its p_T
words and widths did not change, nearly every real constituent was kept, and class 0 won every jet
on near-constant logits. Whether trained words make a collapse likelier cannot be told from one
run each. The picks of A2 with fixed words (epoch 1900, 100 k EBOPs, 71.96 %; 73.8 ns and 53.0 k
LUTs at 5 ns, 79.7 ns and 73.4 k at 2 ns) and of D2 with trained words (epoch 293, 249 k, 71.43 %;
89.4 ns and 100.6 k, 85.5 ns and 118.4 k) stand outside the head-to-head with JEDI-linear, whose
table in the next section holds the other two.

`scripts/synthesis/synthesise_vn.py` takes the `outputs/<label>_hgq/` directory and the probe's
`.npz` of one evaluation. It imports nothing from `src` and runs wherever
`scripts/synthesis/requirements.txt` is installed (alkaid 0.8.1, hgq2 0.2.0, Keras 3.15.1 and JAX
0.10.2 on the CPU, on Python 3.11 or later). It loads the model with Keras on JAX's CPU backend at
the export's float type, refusing any other backend and applying the same floor patch, and checks
Keras's prediction of the stored words against `logits_clean`. alkaid then traces the model into
combinational logic, as `alkaid convert` does, and emulates it bit for bit; both predictions are
written to `keras_outputs.npz` and `alkaid_outputs.npz` in the probe's layout, and the emulation
must match `logits_clean` exactly (`--max-mismatch` 0). The trace skips alkaid's pass
`canonicalize_msb_wraps` (`_disable_msb_wrap_pass`), which in alkaid 0.8.1 zeroes the shift of a
one-bit wrap whose output step equals its source's MSB weight, comparing a step measured after the
shift with an MSB measured before it, so that the op reads another bit; the emulation and the RTL
share the error, which only the check against Keras sees. The trace also computes directly the
product of a matrix row whose elements meeting a nonzero weight are all constant
(`_keep_constant_products`), on which alkaid 0.8.1 and 0.8.2 raise `StopIteration`; such a row
arose in four checkpoints of A2 with trained words, in a dense layer that the EBOPs penalty had
pruned to two weights. alkaid's `RTLModel` writes the Verilog
project `prj/` for the VU13P at 5 ns, with a latency cutoff of 5 ns per stage. Verilator
co-simulates 1000 jets, which must match the emulation exactly, and Vivado implements the project
out of context through alkaid's `build_vivado_prj.tcl`, with its reports under
`prj/output_vn_hgq/reports/`. `alkaid report` then reads the project, and `check.json` and the
report make one row of `synthesis.csv`: latency, resources, WNS, Fmax, alkaid's cost and register
bits, the accuracies, the mismatch counts and the versions, the Vivado columns NaN until Vivado has
run. `--no-cosim` and `--no-vivado` skip either step. A synthesis pod with Vivado alone takes the
split mode: olgpu writes `prj/` with `--no-vivado`, the pod runs Vivado in `prj/`, and
`--report-only` on olgpu parses the returned reports into the row.

`scripts/analysis/compare_outputs.py` compares two outputs files of the same jets figure by figure:
the accuracy, the AUC and the three working points of `src.symmetry.WORKING_POINTS`, each read at
thresholds set on its own scores. Its verdict passes when every macro working point lies within a
two-sided relative tolerance of 1 %, and per class it reports, without judging, the largest
vertical gap between the two ROC curves at an FPR of 1e-3 or more. It reads an evaluation's `.npz`
and the synthesis script's two files alike. Patrick decided on 2026-10-07 that its tolerance
applies between the HGQ2 model of a run and the alkaid design; the float network enters the paper
as the comparison of accuracy against cost.

The scan ran at N = 16 on olgpu on 2026-10-07, in the clone's `logs/`, and its chain left the
projects, `synthesis.csv` and the fidelity table in
`/data/deodagiu/claude/jet_tagging_gdl/hgq_scan/`. The float reference reached a `val/acc_best` of
0.7214, and the seeded port started at 1 985 251 EBOPs. On the full test split, with the macro
figures of the probe (the rejection 1/ε_B at ε_S = 30 % and the TPR at FPR = 1 % and 10 %) and
alkaid's latency at 5 ns:

| run             | test acc | AUC    | rejection | TPR 1 % | TPR 10 % | EBOPs   | latency     | alkaid cost | register bits |
| --------------- | -------- | ------ | --------- | ------- | -------- | ------- | ----------- | ----------- | ------------- |
| float reference | 0.7195   | 0.9235 | 272.4     | 0.3777  | 0.7778   |         |             |             |               |
| β = 1e-8        | 0.7191   | 0.9233 | 271.4     | 0.3766  | 0.7771   | 840 850 | 34 (170 ns) | 321 501     | 232 951       |
| β = 3e-8        | 0.7193   | 0.9232 | 271.7     | 0.3760  | 0.7768   | 602 569 | 32 (160 ns) | 262 025     | 180 790       |
| β = 1e-7        | 0.7181   | 0.9229 | 266.8     | 0.3750  | 0.7754   | 374 093 | 29 (145 ns) | 195 432     | 126 604       |
| β = 3e-7        | 0.7158   | 0.9218 | 255.3     | 0.3686  | 0.7730   | 225 163 | 27 (135 ns) | 139 931     | 89 860        |

Every design was bit-exact: Keras and alkaid differed from `logits_clean` on no logit, the
Verilator co-simulation matched, and `compare_outputs.py` passed every working point with a ROC gap
of 0. Every design accepts a jet per clock (an initiation interval of 1). Against the reference, β
of 1e-8 and 3e-8 keep the three macro working points within 0.5 % (relative), while 1e-7 loses
2.1 % of the rejection and 3e-7 loses 6.3 %, with 2.4 % of the TPR at 1 %. The untrained two-block
port at the seeded 12 and 10 bits had taken 60 cycles (300 ns) and a cost of 803 317 in the dry
run. Vivado has not yet run, so no LUT, FF or timing figure exists: alkaid's cost is its own
estimate, and the latencies assume that the 5 ns clock is met. At N = 16 the comparators are
JEDI-linear, 136 k LUT at 75 ns and 73.6 % (on the untruncated jet's axis, see "JEDI-linear"), and
the DeepSets of Odagiu et al. 2024, 747 k LUT and 555 DSPs at 115 ns.

`tests/test_quantised.py` checks the port and the module on the CPU on a stub datamodule. A float
`vn` of random weights, saved as a run, is copied through `init_from` into a port in float mode,
which then reproduces it in float64 to 1e-5 of the largest logit (eight slots, some padded, two
blocks), for the network of the finals and for four variants of the latency ablation, the batch
normalisations folded in two of them; a float run of another architecture is refused. In two
epochs of training the module steps Keras and the mock optimiser once per batch, gives every
quantised layer the β of its epoch under a schedule and checkpoints the last epoch with the Keras
weights, which a fresh port restores in its float type. The evaluation of that checkpoint traces
the training split alone and exports a model whose prediction of the stored words equals
`logits_clean`, with one EBOPs figure in `export.json`, `test/ebops` and `ebops_total`, and JAX off
the CPU is refused at test time. With alkaid installed, `scripts/synthesis/synthesise_vn.py` runs
on that export through `--no-cosim --no-vivado`: neither Keras nor alkaid differs from
`logits_clean` on any logit, both prediction files are written, the CSV row has its frozen columns
with NaN Vivado figures, the floor patch keeps float64, and a CSV of another header is refused;
where Verilator exists, the co-simulation matches exactly. `tests/test_scripts.py` checks that
`compare_outputs.py` passes an identical candidate with no difference and a ROC gap of 0, fails a
noisy one, scores the macro accuracy as torchmetrics does and appends its rows, that `QUANTISED`
agrees in both modules and that neither the default nor the `[pt, eta, phi]` layout generates its
catalogues. `tests/test_configs.py` checks the experiments (fp32, no clipping, the unranked
checkpoint, no early stopping, `num_workers` 0, positive FLOPs beside NaN EBOPs before the trace)
and `vn_hgq.yaml` against `vn.yaml`, and `tests/test_data.py` that `load_split("train")` at
`n_folds` 1 is the training split at full width. Every module that imports Keras sets
`KERAS_BACKEND=jax` and `JAX_PLATFORMS=cpu` first and skips without `hgq` (or `alkaid`), so
`make test` passes without the group.

The port was shaped by the following behaviours of hgq2 0.2.0, alkaid 0.8.1, Keras 3.15.1 and
JAX 0.10.2, each read in their sources or met on olgpu. `QBatchNormDense` trains with the offset
−mean·kernel, without the scale and the shift of the normalisation, and `QEinsumDenseBatchnorm`
normalises its output with Keras's global `epsilon()` in place of torch's 1e-5, so neither fused
layer reproduces a torch BatchNorm followed by a Linear. alkaid's `mirror_quantizer` drops the
`scaler` of a `Quantizer`, which Keras multiplies back, so no quantiser carries one. alkaid maps
Keras's `hard_sigmoid` to a full-width lookup table, hence the gate's `HardSigmoidConfig`, and
traces WRAP input quantisers alone, hence the clamp in torch. Keras 3.15 casts a float64 tensor to
float32 in floor and ceil on JAX, so HGQ2's rounding floor(x 2^f + 1/2) misrounded values within
float32 resolution of a tie: 39 of 12 288 test jets of the dry run disagreed with alkaid until the
component replaced both for float64 tensors at import (`_float64_kept`, copied into the synthesis
script). XLA's float64 `2**f` on the V100 is off by one ulp for some f, which only running the
evaluation on the CPU avoids. `trace_minmax` stages its data on `cpu:0`, so a GPU run keeps JAX's
CPU backend beside CUDA, and it recompiles for every batch shape, about 35 s at its first call.
Keras marks the product of a tensor and a Python scalar as sparse, which the float32 build fails to
trace, so p is p_T divided by the units. alkaid has no handler for a cast of a boolean,
`keras.ops.relu` or `Stack`, and drops a batch axis of size 1 in `Reshape` and `Flatten`; the port
uses none of them. On olgpu `~/.keras/keras.json` selects tensorflow, so `KERAS_BACKEND` must be
`jax` before Keras is first imported: the component sets it by default and raises if Keras came up
on another backend. Verilator 5.052 from conda-forge (alkaid warns of miscompiled RTL below 5.048)
sits in a scratch environment on olgpu,
`/data/deodagiu/claude/jet_tagging_gdl/hgq_impl/verilator_env`,
and needs `CXX=x86_64-conda-linux-gnu-c++` there.

## JEDI-linear trained quantisation-aware (`jedi_linear_hgq`)

Rebuilt on 2026-10-11 and 2026-10-12 on a plan agreed with Patrick on 2026-10-11
(`~/.claude/plans/jedi-linear-table2-reproduction.md` on the Mac). The row reproduces the
published JEDI-linear (arXiv:2508.15468) as the FPGA comparator of the quantised vector neurons;
the float `jedi_linear` stays a row of the grid. The target is its Table II design at N = 16 on
three features, sorted by p_T with widths per slot: 71.95 % at 57 237 EBOPs, 19 stages, 53.6 ns and
43.8 k LUTs, the latency being the stage count times the achieved period, 2.82 ns, since the 2 ns
constraint was missed. Network, quantisers and recipe are ported to HGQ2 and trained from scratch;
the published code is never run, so every figure below comes from this port, alkaid 0.8.1 and
Vivado 2025.2. The row's first design, a port of `jedi_linear_moments` fine-tuned from float runs in
twenty catalogue blocks, is in git at 3813d4cd and its rows in `results/quantised/`. The code is
`JEDILinearHGQ` in `src/algorithms/jedi_linear_hgq.py`, with
`configs/algorithm/jedi_linear_hgq.yaml` (equal to `jedi_linear.yaml` up to the keys of the port
and its training, which `tests/test_configs.py` checks), the experiments
`configs/experiment/hls4ml/jedi_linear_hgq_{16,50,150}.yaml` and the hand-written catalogues
`scripts/experiments/hls4ml/runjedi_linear_hgq_{train,eval}.sh`; the row is in `QUANTISED`, beside
`vn_hgq`.

The two rows share the Keras machinery, moved out of `vn_hgq` on 2026-10-10 without a change of
behaviour (the `vn_hgq` layout digests were unchanged, and three rebuilt exports, D2 and A2 among
them, predicted their `logits_clean` bit for bit). `components/hgq_port.py` holds what does not
depend on the network: the JAX backend and the float64 floor and ceil patch, `Quantisation`,
`KerasNetwork`, the bit seeding, the calibration, β, the EBOPs and the export.
`keras_base.py::KerasQuantisedModule`, a `DenormalisedModule`, holds the module: the refusal of the
SCL, `_initialise` in `setup("fit")` (by default the copy of `init_from` and the seeding), the
Keras step and the mock optimiser, β and the log line of each epoch, the checkpoint hooks, the
trace at test time and the export, whose `export.json` names its port (`port`); and the stages that
ports share, the builders of the quantised layers, the word quantisers and their clamp, `_p_keep`
and `_lifts`. A row supplies `_build_port`, `_float_twin` (the float network that prices its
FLOPs), `_prepare_inputs` and, to start from a float run, `_copy_float_network`, and may bound the
batch of HGQ2's trace (`_calibration_batch`). The base class defines no `__init__`, since
`save_hyperparameters` reads the row's own frame, and no `required_features`. The vn port rebuilds
its lift from `_p_keep`, the count and `_lifts` in the original order of operations, because the
order and names of the layers fix the order of the Keras weights in its checkpoints.

`_build_port` wires `get_gnn` of the published code (github.com/calad0i/JEDI-linear at 48032802,
`configs/sweep-n16-f3.yaml`) layer for layer, each layer a `QEinsumDenseBatchnorm`, a dense layer
with its batch normalisation folded in: `embed`, `local` and `shared` of width 64 on each slot;
`context` on `pool0`, a `QSum` scaled by 2^−round(log₂ N), added to `local` by `context_add`;
`pool1`, a `QSum` scaled by 1/16; and the head 64-32-16 with ReLU and `head_last`, which has no
activation, no output quantiser and no softmax. Nothing masks a padded row. The quantiser scopes
are those of their `get_model` and `get_gnn`: kbi weights and biases from 7 bits and no integer
bit, SAT_SYM; kif datalanes from 7 fraction bits, WRAP, their integer bits traced within [0, 12];
an L1 weight of 1e-8 on every bit count, β₀ = 0 and HGQ2's rounding to nearest. Datalane widths are
per slot and feature (`per_slot: true`, Table II) or shared by the slots (`false`, their `uq1`).
The one deviation is the floor `WEIGHT_I_MIN` (−12) on the integer bits of the weights and biases
(see `vn_hgq`): their untrained `get_gnn`, built with their `src/model.py`, differed from the port
in those 16 constraints alone and, under `words`, in the word layers. `_initialise` copies and
seeds nothing, so HGQ2 traces the integer bits in training. The FLOPs are priced on `JEDILinearNet`
with sum pooling and no moments, and a trace batch holds max(1024, 16 384 · 16 // N) jets, since the
trace keeps (jets, N, 64) intermediates in float64.

The switch `inputs` sets what the port reads from the rows, which `_prepare_inputs` de-normalises
to GeV and radians and sorts by p_T. Under `standardised`, the default and the published input, it
standardises each column by its mean and standard deviation over every jet and slot of the pooled
training and validation splits, padded rows included (they enter as −μ/σ), computed in float64 at
`setup("fit")` and kept in the checkpoint as `input_mean` and `input_std`. The first layer's
trainable input quantiser, one width per slot and feature, is then the only input quantisation
(7 to 11 bits per element in the shipped Table II design), and the stored test inputs are its
output, on which alkaid's trace of it is the identity. Under `words` the port reads the fixed
trigger words of `vn_hgq`, p_T in units of 16 GeV and padded rows zero. `setup()` accepts
`[pt, etarel, phirel]` and `[pt, eta, phi]` alone, the latter under `words` only with an angle word
that holds ±π (`angle_kif: [1, 2, 8]`). Keras runs in float64. `export.json` adds the columns, the
input mode, `per_slot`, the word widths or the standardisation and the rounding; its `beta` is null
beside `beta_schedule`, so `synthesis.csv` carries no β.

The configs carry the published recipe (`configs/sweep-n16-f3.yaml` and `src/train.py`): 7000
epochs on batches of 2790, Adam at 3e-3 without weight decay or clipping under `CosineRestarts`, as
in blocks 37 and 38 of `vn_hgq`, and β rising linearly from 2e-8 to 3e-7 over 2000 epochs, then
geometrically to 3e-6 at epoch 7000. Four differences remain: the learning rate is stepped per
batch, where theirs is set once per epoch; validation uses the record's split, where they hold out
10 % of the shuffled training files (their test set is the record's test split); Keras runs in
float64, where they used TF32 products on float16 inputs; and the results rows hold the macro
accuracy, where the paper quotes the micro, as the figures below do. Their Table II run stopped on
a NaN at epoch 5120, and Table II is its epoch 4350, where our first run of the reproduction read a
training accuracy of 0.7205 at 64.7 k logged EBOPs against their 0.7190 at 60.4 k.

Blocks 1 to 4 of each catalogue train one variant each, with a checkpoint every epoch, in
`logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_<variant>_allepochs`; they differ in their
inputs alone. `axis_trained`, the reproduction, reads `etarel` and `phirel` from the full jet's
axis (`data.full_axis=true`) as standardised rows. `kept_words` reads them from the kept
constituents' axis as fixed trigger words, the like-for-like row against the vector neurons.
`abs_words` reads the absolute `[pt, eta, phi]` as fixed words, the paper's text taken literally
("only pT, eta, and phi are used"). `kept_trained` reads the kept axis as standardised rows, so
that `axis_trained` minus `kept_trained` measures the kept axis under the published recipe and
`kept_trained` minus `kept_words` the fixed words. The evaluations write the rows
`jedi_linear_hgq_<variant>` to `results/quantised/`, the probe off for `axis_trained`, whose
invariance figures would mean nothing, since a transform rebases `etarel` and `phirel` on the kept
constituents.

Their `src/train.py` keeps every epoch on HGQ2's `ParetoFront` of validation accuracy (above 0.5)
against EBOPs, so Table II was selected on validation. The row takes the same rule on calibrated
figures (2026-10-14; a pick on the training plateau preceded it until 2026-10-12, and one on the
logged figures until 2026-10-14). An epoch's validation accuracy is that of its port tested on the
validation jets of its own training once its integer bits are traced on the training split, as at
the test, and its EBOPs are those of the trace; the design is the front point whose EBOPs lie
nearest a target, 57 237 for Table II. The validation accuracy that `train.log` logs runs on the
integer bits traced on an epoch's last batch, past which the vector neurons' inputs wrap (see
`vn_hgq`), whereas JEDI-linear's tracks the calibrated one within 0.002. The test accuracy plays
no part. `scripts/experiments/quantised_picks.py` implements the rule: `candidates <run dir>`
prints the epochs worth evaluating, those with a checkpoint on the front of the logged validation
accuracy or on that of the logged training accuracy, which does not wrap, against the logged
EBOPs. A guard their selection lacks leaves out every epoch whose logged training accuracy lies
below 95 % of the run's best (`TRAIN_FRACTION`, 2026-10-14): a vector-neuron run collapsing under
the recipe still validates above 0.5 for an epoch or two, and its front's cheapest point, the pick
when no front point reaches the target, then sat inside the collapse (A2 with trained words, folds
1 and 5 at N = 16: training accuracy 0.52, calibrated validation 0.71 and 0.55, against 0.73 two
epochs earlier). No JEDI-linear pick at N = 16 changes under it. Each is tested by `src/eval.py` with `data.test_split=validation` and `fold=<i>` (null for a
main run), which serve the validation jets of that run's training and write no results row. Then
`pick <run dir> <evaluations> <target> <picks dir>` reads each candidate's micro accuracy off
`<evaluations>/epoch_<k>/outputs/main.npz` and its traced EBOPs off `main_hgq/export.json`, takes
their front, and writes the picked checkpoint's manifest to `<picks dir>`, or `<picks dir>/fold_<i>`
for a fold, the layout `src/eval.py` merges; the eval blocks point `checkpoint_manifest_path` at
`<run dir>/picks/<tag>`, `target57237` for Table II's EBOPs and `matched_a2_trained` and
`matched_d2` for the points at the vector neurons' EBOPs below.
A training that `chain.sh` of the campaign stopped after its collapse wrote no manifest, since
`train.py` writes it last; it was written afterwards with the last checkpoint and the run's MLflow
run, as `train.py` would have. Checkpoints are kept every
epoch because the cadence matters: a first run that kept one every 50 epochs put the reproduction
1.0 point under Table II (70.89 % at 54 182 EBOPs), although the per-epoch fronts of their history
and of that run coincide within 0.001 (the best validation accuracy at no more than 57.2 k logged
EBOPs is 0.7166 and 0.7174); on every 50th epoch alone ours read 0.7098, about 0.75 points lower.

The runs ended on 2026-10-12, `abs_words` stopped once its EBOPs fell below 1000 near epoch 5160.
`scripts/synthesis/synthesise_vn.py` implemented each pick, its top module and its Vivado reports'
directory named after the export's `port` (or `--project`), in both flows of blocks 37 and 38 of
`vn_hgq`: 5 ns with a latency cutoff of 4.2 ns, and the published pipelining, 2 ns with a cutoff
of 2.7 ns. The picks at N = 16 (main runs, micro test accuracy):

| design         | epoch | EBOPs  | test acc | 5 ns: latency, LUTs | 2 ns: latency, LUTs |
| -------------- | ----- | ------ | -------- | ------------------- | ------------------- |
| Table II       | 4350  | 57 237 | 71.95 %  |                     | 53.6 ns, 43.8 k     |
| `axis_trained` | 4498  | 55 363 | 71.48 %  | 56.3 ns, 31.1 k     | 48.1 ns, 38.6 k     |
| `kept_trained` | 3336  | 57 976 | 70.93 %  | 57.0 ns, 33.7 k     | 55.4 ns, 40.6 k     |
| `kept_words`   | 3576  | 55 798 | 70.83 %  | 57.9 ns, 30.5 k     | 51.7 ns, 38.6 k     |
| `abs_words`    | 3279  | 57 574 | 53.25 %  | 56.2 ns, 30.0 k     | 48.2 ns, 37.5 k     |

The reproduction lies 0.47 points under Table II (71.40 % macro), and two runs of the recipe
differ by about 0.3 points of validation accuracy at fixed EBOPs. It takes 12 cycles at 5 ns and
18 stages at the published pipelining, against Table II's 19, and is faster and smaller there than
the published design. The kept axis costs about 0.55 points and the fixed words about 0.1.
`abs_words` reflects the network on absolute coordinates, not the port: the searched float
`jedi_linear` reaches 62.7 % on `[pt, eta, phi]` at N = 16 against 71.6 % on
`[pt, etarel, phirel]`.

alkaid's pass `canonicalize_msb_wraps` (0.8.1, unchanged on `master` at 7cfabca) broke five
checkpoints of the first run: in each it rewrote exactly one op, a one-bit unsigned wrap with a
shift of +1 to +3, so that the op read its source's MSB, and the emulation differed from Keras on
13 to 5204 of 1.3 M test logits. Without the pass all five were exact in Keras and Verilator at the
same op count and cost, so every design is traced without it; da4ml 0.3, which built the published
designs, emits no shift on a quantise op. Every design here was exact and co-simulated exactly.

Blocks 40 to 43 of `runvn_hgq_train.sh` fine-tune D2 and A2 under the same recipe and checkpoints
(see `vn_hgq`). The agreed pairs read the same information: the full axis with trained words,
`axis_trained` against A2 with trained words (one width per slot for both angles, unstandardised,
where JEDI-linear's are per element on standardised rows); and trigger information with fixed
words, `kept_words` against D2, which reads no axis. The five folds of these four designs at
N = 16, and of both pairs' designs at N = 50 and 150, trained on NGT on 2026-10-13 and 14. Under
the recipe's peak learning rate of 3e-3 every VN run collapses, between about 100 k and 250 k
EBOPs at N = 16 and from about 200 k at N = 50, so that no VN front reaches JEDI-linear's 57 237
EBOPs and the pick nearest it is each run's cheapest front point, on D2 inside its degradation.

Each architecture is therefore reported under its own protocol (decided 2026-10-14). JEDI-linear
is picked as its paper picks it, nearest the EBOPs of its published design at the nearest
constituent count, read off its repository's result reports: 57 237 at N = 16, 102 664 at
N = 64 for our 50 and 144 427 at N = 128 for our 150. The vector neurons are picked on hardware,
since EBOPs price their products of two variables as bit operations, whereas Vivado maps them to
DSPs and JEDI-linear has none: the most accurate front point whose latency at 5 ns stays under
100 ns. JEDI-linear's most accurate design under 100 ns is kept for a referee; at N = 16 it is
the more accurate on the full axis and ties on trigger information.

`scripts/synthesis/synthesise_vn.py --estimate-only` estimates a checkpoint's hardware without
Vivado: it traces the export under every CMVM method at hard_dc 0 and 1, keeps the setting of
fewest pipeline stages at the 4.2 ns cutoff, as the implementations were chosen, and writes
`estimate.json` with its cycles, their latency at 5 ns, and the DSPs and LUTs Vivado would use.
A product of two variables whose narrower operand has more than 5 bits lands in a DSP, and the
LUTs are 0.92 of alkaid's cost without those products. On the 13 designs implemented at 5 ns by
2026-10-14 the DSPs came within 2 % and the LUTs within 0.89 to 0.95 of that cost, for both
architectures; at 2 ns Vivado maps fewer products to DSPs, which the estimate does not model.
`quantised_picks.py front` lists a run's calibrated front, each of whose epochs is estimated, and
`pick-hw <run dir> <evaluations> 100 <picks dir>` writes the pick (`picks/best100ns`). Its
`--match` instead picks the point nearest another design's utilisation, the larger of its LUT and
DSP fractions of the VU13P; it is unused, since the VN's cheapest healthy designs take 2.3 to 5 %
of the device, set by their DSPs, against 1.8 % for JEDI-linear's 57 k picks.

A pilot at N = 50 (D2's analogue, `cheap_h32`, main runs) tested the learning rate. The cheapest
epoch whose training accuracy stayed at 0.78 or above lay at about 206 k EBOPs at 3e-3, at 97 k
at 1e-3 and at 106 k at 3e-4, each run collapsing below it: lowering the peak from 3e-3 to 1e-3
halves the cheapest healthy cost, and lowering it further gains nothing. The vector neurons are
therefore retrained at a peak of 1e-3, JEDI-linear's recipe otherwise unchanged, for both designs
at N = 16, 50 and 150 with their five folds (run directories `..._jedirecipe_lr1e-3_everyepoch`),
which replace the runs at 3e-3.

`tests/test_quantised.py` checks, on the CPU and the record stub of the vn tests, that the port
builds the published layers and quantisers under both `inputs` and both `per_slot`, that it refuses
other columns, a narrow angle word for absolute angles and an unknown `inputs`, and that
`CosineRestarts` follows the published `cosine_decay_restarts` at every epoch start. Two epochs
from scratch set the standardisation of the pooled splits, padded rows included, and round-trip it
in the checkpoint. The evaluation of that checkpoint, and that of an untrained port on absolute
words shared by the slots, export a model that predicts `logits_clean` from the stored inputs,
which `synthesise_vn.py` emulates bit for bit and, where Verilator exists, co-simulates exactly;
the estimate's cycles and cost equal those of the project its chosen setting writes.
`tests/test_configs.py` holds the three experiments to layout digests, and
`tests/test_quantised_picks.py` checks the candidates on a synthetic `train.log`, that the picks
of a main run and a fold, chosen on the calibrated front, merge as `src/eval.py` merges them, and
that the hardware pick keeps the latency budget and lets the DSP fraction bind.

## JEDI-linear

Added on 2026-09-30 as a comparator from the literature, at first in the baseline regime alone, and
made a row of the grid on 2026-10-02, when Patrick decided to run it under augmentation and the SCL
as well. JEDI-linear (Que, Sun et al., arXiv:2508.15468, Field Programmable Technology 2025) is a
jet tagger built for FPGAs; it is judged on accuracy and cost, against the vector neurons in
particular. Its published accuracy, 73.6 % at 16 constituents on three features, is measured on the
record's `etarel` and `phirel`, which are taken from the axis of the untruncated jet. This
repository rebases them on the kept constituents, which is worth about 2 points, so a comparison
needs the network trained on this preparation. Our code is the network `JEDILinearNet` and the
`JetTaggingModule` subclass `JEDILinear` in `src/algorithms/jedi_linear.py`. Around them sit
`configs/algorithm/jedi_linear.yaml`, the three experiment
configs `configs/experiment/hls4ml/jedi_linear_{16,50,150}.yaml`, the search config
`configs/hparams_search/hls4ml/jedi_linear_optuna.yaml` with its companion
`jedi_linear_scl_optuna.yaml`, the three search catalogues
`scripts/experiments/hls4ml/{px_py_pz_e,pt_eta_phi,pt_etarel_phirel}/runjedi_linear_search.sh` and
the generated `scripts/experiments/hls4ml/px_py_pz_e/runjedi_linear_{train,eval}.sh`.

The reference is the paper and `src/model.py::get_gnn` of its code, github.com/calad0i/JEDI-linear,
both read on 2026-09-30. In float it is one network for every number of constituents and every
feature set. A dense layer of width 64 projects each constituent to h_i. A dense layer of h_i and a
dense layer of the pooled Σ_j h_j, both of width 64, are added, the second by broadcast over the
constituents; a shared dense layer of width 64 follows; the result is pooled over the jet and read
by a head 64-32-16-5. Every dense layer carries a BatchNorm, and every one but the last a ReLU. Both
poolings are sums with a constant scale, 2^-round(log2 N) at N slots for the context and 1/16 for
the readout, never per-jet means, and nothing masks the padded rows, which the reference's
standardisation turns into the constant -μ/σ. This is DeepSets whose per-constituent network carries
one permutation-equivariant layer of Zaheer et al. (Deep Sets, arXiv:1703.06114), f(X) = σ(XΛ +
(1/N) 1 1ᵀ X Γ), which the paper rederives by linearising the edge function of JEDI-net (its eq. 4).
What the paper adds is the FPGA co-design, per-weight bitwidths and distributed arithmetic, not the
function class; our paper says so in a sentence. The paper's variant on p_T-sorted constituents,
described as not permutation-invariant, differs only in its per-slot quantisation bitwidths: in
float there is one architecture, and it is permutation-invariant, so there is one row.

The reference reads three features, record columns 5, 8 and 11, which are `pt`, `etarel` and
`phirel` (the paper writes p_T, η and φ), or all 16 columns. The 16-feature variant is not run
here: no transform rewrites `erel`, `etarot`, `phirot`, `costheta` or `costhetarel`, so the probes
would feed it stale values. The row is searched instead on the three feature sets of the
repository. The paper's cut at 2 GeV on the constituents is not in the released code and is not
applied. The reference trains with Adam at 3e-3, a cosine decay restarted every 100 epochs and
batches of 2790, for 7000 epochs of quantisation-aware training under HGQ with a ramped penalty on
the EBOPs cost. That training is a scan along the Pareto front of accuracy against cost and does not
transfer to a float network. The reference's histories peak at a validation accuracy of 0.738 at
16 constituents on three features, before the quantisation starts to cost accuracy. With the 2
points of the jet axis removed, the row should land near 0.715 at 16 constituents on
`[pt, etarel, phirel]`, between tuned DeepSets (0.705) and JetFormer (0.720). A calibration on olgpu
on 2026-09-30, one fold at 16 constituents on that set, gave a validation accuracy of 0.707 after
50 epochs at the reference's learning rate of 3e-3 (0.696 at the config's 1e-3), while DeepSets
and the row reached 0.668 and 0.670 under the same 20-epoch protocol at 1e-3; the search sets the
learning rate.

The port departs from the released code in three ways, each by choice. It has no BatchNorm: at
inference a BatchNorm folds into the dense layer before it, and in training it served the
quantiser, so without it the row differs from DeepSets in the context layer alone, which keeps the
comparison fair. Its two pools use one aggregation over the real constituents, which the search
picks among the sum, the maximum and the sum divided by the constant N (`sum_over_n`), the DeepSets
choices; the published point is the sum. A padded slot is an all-zero row but embeds to the
activation of the first bias, so the mask (`x.abs().sum(-1) > 0`, the convention of the other
components) keeps it out of both pools; for the maximum it enters as minus infinity. The sum and
`sum_over_n` carry the
multiplicity to the head, which then reads it from the real constituents rather than from the
padded rows. A constant scale in front of a dense
layer is absorbed by its weights, so dropping it changes nothing the network can express. And the
network is float only, with no quantisation. With σ the activation and P the pool over the real
constituents, it computes h_i = σ(W_0 x_i + b_0), g_i = σ(W_1 h_i + b_1) + σ(W_2 P_j h_j + b_2) and
z = P_i σ(W_3 g_i + b_3), and the head reads z. One width, `hidden` (64 by default, the published
point), serves the projection, both context branches and the shared layer; the head is
`[64, 32, 16]`, the activation is ReLU, which the search leaves alone, and there is no dropout.
The state dict does not depend on the number of constituents.

The optimiser and scheduler are the DeepSets block verbatim, Adam at 1e-3 with weight decay 1e-7
and ReduceLROnPlateau on `val/acc`: `jedi_linear.yaml` is a standalone copy of `deepsets.yaml`
that differs in `_target_`, `name` and the architecture keys, and `tests/test_configs.py` keeps the
rest in step. The experiments keep the `16-mixed` precision of `trainer/gpu.yaml`, as DeepSets, MLP
and JetFormer do. As a row of the grid that reads `data.features`
(`nbr_feats: ${len:${data.features}}`), `jedi_linear` is in `MODELS` of
`scripts/experiments/make_train_catalogues.py` and `tests/conftest.py` but in neither `EMBEDDED` nor
`BASELINE_ONLY`, so its results land in `results/<label>/results_table_<regime>.csv` and
`results_table.subdir` stays null. Until 2026-10-02 it ran the baseline alone and was the one row
of `BASELINE_ONLY` outside `EMBEDDED`; the set now equals the embedded class.

The search, `jedi_linear_optuna.yaml`, runs one study per constituent count and regime,
`jedi_linear_nconst<N>_<method>_<scheme>`, in `logs/optuna/hls4ml/jedi_linear.db` on the
four-momentum and in `logs/optuna/hls4ml/<label>/jedi_linear.db` on the other sets, each of 200
trials of 50 epochs; the SCL regimes select `jedi_linear_scl_optuna.yaml`, which adds the
consistency-loss weight over the range of `deepsets_scl_optuna.yaml`. It draws `hidden` from {16,
32, 64, 128}, the head from the `rho_layers` choices of DeepSets plus [64, 32, 16], the pooling from
{sum, max, sum_over_n}, the learning rate log-uniformly from 1e-4 to 1e-2 and the weight decay from
{0, 1e-7, 1e-6, 1e-5, 1e-4}, so the published point lies inside the space. No activation is
searched, so every trial takes the network's ReLU. The pooling, the learning rate and the weight
decay are searched over the values of the DeepSets search, which `tests/test_configs.py` checks, so
that the two rows differ in the network alone. Both spaces hold only operations an FPGA computes
without a lookup table or a per-jet division, the rule set on 2026-10-01: tanh needs a lookup table
and the mean a division by each jet's multiplicity. 63 studies, three feature sets times three
counts times seven regimes, are run.

The pooling and activation choices came from a first campaign, launched on 2026-09-30 with the sum
and ReLU fixed and stopped after 40 to 90 trials per study. It matched tuned DeepSets on
`[pt, etarel, phirel]` (0.711, 0.803 and 0.813 at 16, 50 and 150 constituents, against 0.705, 0.802
and 0.813), where the best DeepSets trials also pooled with the sum. On `[px, py, pz, e]` and
`[pt, eta, phi]` it fell 8 to 14 points behind wherever the best DeepSets trial took the maximum,
and agreed where it took the sum (0.513 against 0.514 on the four-momentum at 150). Its three
databases were backed up to `/shared/claude/jet_tagging_gdl/searches_2026-09-30_jedi/stale_dbs/`
and removed. A second campaign, launched the same afternoon with the activation searched over
{relu, tanh} and the pooling over {sum, mean, max}, completed its nine studies. Its best trials took
tanh with the maximum on the absolute-coordinate sets and ReLU on `[pt, etarel, phirel]` (0.713,
0.804 and 0.813), and the FPGA rule of 2026-10-01 superseded it.

At the published point the network has 64F + 19 397 parameters on F features, 19 589 on three and
19 653 on four, at every number of constituents; at the corner of the search space (hidden 128,
head [64, 32, 16]) it has 128F + 60 613, 61 125 on four. It costs 0.290, 0.860 and 2.537 MFLOP per
jet at 16, 50 and 150 constituents on three features and 0.292, 0.866 and 2.556 on four, counting
two FLOPs per multiply-add and leaving the poolings out, as the FLOP counter does; the corner costs
1.12, 3.37 and 10.0 MFLOP on three features. Nearly all of the cost is per constituent,
128F + 16 384 FLOPs per slot against 21 664 per jet at the published point, so it grows linearly
with N. The vector neurons run every block once per jet, and `vn` costs 0.064, 0.105 and
0.223 MFLOP at its defaults (2026-10-03); tuned DeepSets costs about 0.28, 0.64 and 0.98. A V100
pre-flight at the corner, with batches of 2048 under fp16 autocast, reserved 0.12, 0.32 and
0.77 GiB and took 3.3, 3.9 and 6.3 ms per step at 16, 50 and 150 constituents, so every study
fits a 1g MIG slice. These figures were measured on olgpu on 2026-09-30.

`tests/test_algorithms.py` checks that a padded slot changes no logit, the network at eight slots
against a copy at nine that sees one more padded slot, and that `_masked_pool`, which the row shares
with DeepSets, reads the real slots alone under each of the three pools, to 1e-12 in float64. The
pool is tested on a hole among a jet's slots and on padded slots flooded with 1e6, which would win
every channel of an unmasked maximum and swamp either sum. Last, it checks the parameter count at
the published point on four features, 19 653, and the FLOPs and eBOPs per jet at N = 16 against a
hand count. `tests/test_configs.py` keeps `jedi_linear.yaml` a copy of `deepsets.yaml`, the shared
search choices, the SCL weight included, equal to those of DeepSets and both spaces free of an
activation, and `tests/test_scripts.py` checks that the generator's registries equal those of the
tests and that a feature-set layout writes catalogues for `jedi_linear` beside the other grid rows
and none for the embedded rows.

## Linformer

Added on 2026-10-01 as the fourth row of the acquired-symmetry class, beside DeepSets, MLP and
JetFormer, and as a further candidate for an FPGA beside `jedi_linear` and the vector neurons. The
row, `linformer`, is the single-head Linformer transformer of Laatu et al. (arXiv:2510.24784),
trained in float on this repository's preparation. Unlike `jedi_linear`, it runs all seven regimes
on the three feature sets, 21 studies per set and 63 in all, none of them launched yet
(2026-10-01); Patrick took the decisions recorded below that day. The network, `LinformerNet`, and
its attention, `LinformerAttention`, are in `src/algorithms/linformer.py`, as is the subclass
`Linformer`, which supplies `build_net` alone. The
configs (`linformer.yaml`, `linformer_{16,50,150}.yaml`, `linformer_optuna.yaml` and
`linformer_scl_optuna.yaml`) and the catalogues `runlinformer_{search,train,eval}.sh` sit where
those of the other grid rows do, with a search catalogue under every feature-set directory. Each
catalogue holds 21 blocks, and every train and eval block is still "transferred from experiment
config (no study yet)".

The reference is Laatu, Sun, Cox, Gandrakota, Maier, Ngadiuba, Que, Luk, Spiropulu and Tapper,
"Sub-microsecond Transformers for Jet Tagging on FPGAs" (arXiv:2510.24784, NeurIPS 2025 ML4PS
workshop), read on 2026-10-01. Its model is an encoder-only transformer with a single attention
head and no positional encoding, fed the 8, 16, 32 or 64 hardest constituents, sorted by p_T, with
three features (p_T, η, φ). The attention comes in two variants: vanilla multi-head attention
(MHA), O(N²), and the linear attention of the Linformer (Wang et al., arXiv:2006.04768), which
projects the keys and values onto k < N slots and costs O(kN); a figure legend reads "Linformer
proj dim 2". Both are trained quantisation-aware with HGQ at a fixed target of 350 000 EBOPs;
da4ml optimises the constant matrix-vector products, and the firmware is synthesised with Vitis
HLS. Table 1 of the paper, test accuracy after quantisation at 8, 16, 32 and 64 particles, gives
66.3, 72.8, 78.4 and 79.8 % for the Linformer, 66.3, 72.3, 77.0 and 77.9 % for MHA and 64.7, 70.1,
77.4 and 79.4 % for DeepSets under HGQ. MHA falls behind DeepSets at 32 particles, and its
attention block collapsed in several trainings at 64, which is why Patrick chose the Linformer;
MHA with one head and one block would also nearly duplicate a corner of `jetformer_optuna.yaml`
(one block, one head, width 16 or 32). None of these accuracies is comparable with ours. They are
test accuracies after quantisation at a fixed budget, and the example loader described below reads
the record's `etarel` and `phirel`, measured from the axis of the untruncated jet, which this
repository rebases on the kept constituents; at 16 constituents the difference is worth about 2
points (see JEDI-linear above).

The paper gives no widths, no block layout and no readout, and no training script is published:
"code is available as part of the HGQ and hls4ml packages". The layers are `QMultiHeadAttention`
and `QLinformerAttention` in github.com/calad0i/HGQ2 (`src/hgq/layers/attn/mha.py` and
`linformer.py`), which hls4ml parses (pull request 1360). In `QLinformerAttention`, two learned,
bias-free (N, k) matrices, separate by default (`share_kv_proj=False`), project the key and value
inputs over the constituent axis before their dense layers; the key dense layer has no bias, and
the softmax runs over the k slots with scale 1/√(key width). The nearest model definitions are in
github.com/calad0i/HGQ2-examples, `jsc150/model.py`, which Chang Sun, a co-author, added in March
and July 2026, after the paper. `get_llformer` has one head, k = 4, width 16, LUT-based dense
layers with BatchNorm, an attention and a feed-forward sub-layer each with a residual, a global
average pool over all N slots and a head of width 16; `get_transformer` is the MHA variant, with
2 heads. Neither matches the paper's description: `get_llformer` has k = 4 against the 2 of the
legend, and `get_transformer` two heads against one. The row is therefore the Linformer of the
paper with the block layout of its co-author's example, not a replica of an unpublished model. The
example's loader, `jsc150/data.py`, reads record columns 5, 8 and 11 (`pt`, `etarel` and
`phirel`), zeroes the constituents below 2 GeV, standardises over all slots and applies no mask.
Training there uses Adam at 3e-3 with cosine restarts and batches of 2790, for 7000 epochs under
HGQ.

With d = `embed_dim`, k = `kv_proj_dim`, f = `ff_dim` and real the mask `x.abs().sum(-1) > 0`, the
network computes

```
h   = ReLU(BN(W_e x + b_e))
q   = W_q h + b_q
K   = W_k (E_kᵀ (h·real))            E_k: (N, k), no bias in W_k
V   = W_v (E_vᵀ (h·real)) + b_v      E_v: (N, k)
A   = softmax(q Kᵀ / √d) over the k slots
h   = h + W_o (A V) + b_o
h   = h + W_2 ReLU(BN(W_1 h + b_1)) + b_2
z   = pool of h over the real constituents
out = head(z): Linear, BN, ReLU per width of head_layers, then Linear to 5
```

with one block, one head, no positional encoding, no dropout and no LayerNorm; ReLU is the only
activation besides the softmax. E_k and E_v are initialised Glorot-uniform, as in the reference.
The port departs from the reference code in four ways, each by choice. It is float only, with plain
dense layers where the example has LUT-based ones. It masks the padding: padded embeddings are
zeroed before both projections and kept out of the readout. A padded row is all zeros, but its
embedding is a function of the first bias, so the mask, and not the zero input, keeps it out. The
queries of padded slots are computed all the same; with BatchNorm on, they enter the batch
statistics of the feed-forward sub-layer and nothing else. The readout pool is searched among
`sum`, `max` and `sum_over_n`, the masked sum divided by the constant N, which is the reference's
average pool restricted to the real constituents; all three come from `_masked_pool` of
`components/pooling.py`. The sum and `sum_over_n` express the same functions, since the head's
first dense layer absorbs 1/N, and differ only in their conditioning at initialisation; both stay
in the search at Patrick's request. All three are cheap on an FPGA: an adder tree, a comparator
tree and a constant scale. Last, BatchNorm is a switch, `batchnorm`, searched on or off as the
MLP's is, since it folds into the dense layers at inference. Its statistics include the padded
slots, as in the reference and in JetFormer's batch option.

The network is not permutation-invariant. E_k and E_v carry one weight per slot, which is one per
p_T rank because the record is sorted by p_T; the parameter count grows with N by 2Nk, and the
state dict depends on N. The first block of the vector neurons also has one weight per rank, but
that network sorts each jet afresh; this one reads the slots in the record's order, with no sort,
as the MLP does. No transform re-sorts a jet. SO(2) about the beam and boosts along it keep every
p_T, so the slots stay in p_T order. SO(3) and `lorentz_iso` change the p_T, and the network then
sees the slots in the p_T order of the untransformed jet, exactly as the MLP row does.

The config defaults in `configs/algorithm/linformer.yaml` are the example's point in float:
`embed_dim: 16`, `kv_proj_dim: 4`, `ff_dim: 16`, `head_layers: [16]`, `aggregation: sum_over_n` and
`batchnorm: true`. The optimiser and scheduler are the DeepSets block, Adam at 1e-3 with weight
decay 1e-7 and ReduceLROnPlateau on `val/acc`. The experiments keep the `16-mixed` precision of
`trainer/gpu.yaml` and leave `extras.matmul_precision` unset, as the DeepSets, MLP, JetFormer and
`jedi_linear` experiments do. They train for 200 epochs on batches of 2048, with 5 folds, and the
150 file sets `data.num_workers: 16` for the online transform. As a row of the grid that reads
`data.features`, `linformer` is in `MODELS` of `scripts/experiments/make_train_catalogues.py` and
`tests/conftest.py` but in neither `EMBEDDED` nor `BASELINE_ONLY`, so its results land in
`results/<label>/results_table_<regime>.csv` and `results_table.subdir` stays null.

The search, `linformer_optuna.yaml`, names its studies `linformer_nconst<N>_<method>_<scheme>` and
keeps them in `logs/optuna/hls4ml/linformer.db` on the four-momentum and in
`logs/optuna/hls4ml/<label>/linformer.db` on the other sets, each of 200 trials of 50 epochs;
`linformer_scl_optuna.yaml` adds `algorithm.symm_loss_weight` for the SCL regimes. With one block
and one head fixed, it draws `embed_dim` from {16, 24, 32, 48, 64}, `kv_proj_dim` from {2, 4, 8},
`ff_dim` from {16, 32, 64, 128}, the head from {[16], [32], [32, 16], [32, 32, 16], [64, 32],
[32, 32, 32]} (the `rho_layers` choices of DeepSets plus [32, 32, 32]), the pool from
{sum, max, sum_over_n}, `batchnorm` from {true, false}, the learning rate log-uniformly from 1e-4
to 1e-2 and the weight decay from {0, 1e-7, 1e-6, 1e-5, 1e-4}. The choices of k cover the 2 of
the paper's legend and the 4 of the example, and the example's point lies inside the space.

The figures below were measured on olgpu on 2026-10-01, with the helper scripts and logs in
`/data/deodagiu/claude/jet_tagging_gdl/linformer/`. With BatchNorm on, the network has

```
P = (F d + 3d) + (4d² + 3d + 2Nk) + (2df + 3f + d) + Σ_i (h_{i-1} h_i + 3h_i) + 5h_L + 5
```

parameters on F features, the terms being the embedding, the attention, the feed-forward sub-layer
and the head, with h_0 = d and h_L the last width of the head. At the defaults that comes to
16F + 8N + 2085: 2261, 2533 and 3333 at N = 16, 50 and 150 on three features and 2277, 2549 and
3349 on four. At the corner of the search space (width 64, k = 8, ff 128, head [64, 32]) it is
40 645, 41 189 and 42 789 on three features and 40 709, 41 253 and 42 853 on four. Counting two
FLOPs per multiply-add and leaving out the BatchNorm, the softmax and the pooling, as the FLOP
counter does, a jet costs 2[N(F d + 2d² + 4kd + 2df) + 2kd² + head], where head stands for the
multiply-adds of the head; the counter and this count agree to every digit. At the defaults that
is 0.0473, 0.1376 and 0.4032 MFLOP at N = 16, 50 and 150 on three features (0.0478, 0.1392 and
0.4080 on four), and at the corner 1.0018, 2.8253 and 8.1885 (1.0038, 2.8317 and 8.2077 on four).
For comparison, the JEDI-linear section gives 0.290, 0.860 and 2.537 MFLOP on three features for
`jedi_linear` at its published point, 0.064, 0.105 and 0.223 for `vn` at its defaults and about
0.28, 0.64 and 0.98 for tuned DeepSets. A V100 pre-flight at the corner with BatchNorm on, on
batches of 2048 fully occupied random jets with four features under fp16 autocast, reserved 0.11,
0.30 and 0.97 GiB and took 5.7, 6.2 and 13.1 ms per step at N = 16, 50 and 150, with finite
losses throughout, so every study fits a 1g MIG slice.

A calibration on one fold at 16 constituents, 20 epochs at a learning rate of 1e-3, recorded the
validation accuracy of the best epoch. At a mid point of the space (width 32, k = 4, ff 64, head
[32, 16], `sum_over_n`) the row reached 0.687 with BatchNorm and 0.652 without on
`[pt, etarel, phirel]`, and 0.431 with and 0.407 without on `[px, py, pz, e]`; with the maximum as
pool and BatchNorm on, it reached 0.561 on `[px, py, pz, e]`. At the config defaults (2261
parameters) the accuracy was 0.661 on `[pt, etarel, phirel]`. Under the same 20-epoch protocol on
that set, DeepSets reached 0.668 and `jedi_linear` 0.670 on 2026-09-30, and their tuned studies
(50 epochs, best of 200 trials) have 0.705 and 0.713 there and 0.548 and 0.576 on
`[px, py, pz, e]`. So the block trains with and without BatchNorm, BatchNorm adds 3.5 points on
`[pt, etarel, phirel]` and 2.4 on the four-momentum at this length of training, and the maximum is
worth 13 points over the scaled sum on the four-momentum, as it was for `jedi_linear`. These are
single short runs, not tuned results. Smoke runs at N = 16, a fit and an evaluation on each
feature set, one SCL fit (so3) and one augmented fit (lorentz), went end to end under `16-mixed`,
with the results rows written to a scratch directory, not to `results/`.

`tests/test_algorithms.py` runs the network in eval mode, since BatchNorm in training mode would
couple the jets of a batch. With the sum and BatchNorm on, a copy at nine slots that sees one more
padded slot, whose rows of E_k and E_v keep their own random draw, must give the logits of the
network at eight; `sum_over_n` is left out there, since it divides by a slot count that differs by
design. `_masked_pool` must read the real constituents alone, with `sum_over_n` equal to the sum
divided by the slot count. In `tests/test_scripts.py`, a feature-set layout must generate train and
eval catalogues for the row.

## Jet moments

Added on 2026-10-05 in three rows, after a scratch test of the same day, to find where the
advantage of the vector neurons over DeepSets and JEDI-linear comes from. The vector neurons
multiply the shift-invariant angular moments of a jet, sums of p_T products weighted by powers of
ΔR², inside their first block (the tensor invariants of "Vector neurons" above). `deepsets_moments`
and `jedi_linear_moments` hand those moments to the head of DeepSets and of JEDI-linear as
jet-level inputs beside the pooled vector, and the control, `moments_only`, reads the moments and
the multiplicity alone, with no per-constituent network. All three read `[pt, etarel, phirel]`,
the feature set of the scratch test, and are filed with the embedded rows: the baseline regime
alone, results in `results/embedded/` and catalogues in the default layout alone. Our code is
`JetMoments` in `src/algorithms/components/moments.py`, which reuses
`_form_matrix`, `_sort_by_pt` and `_unit_lift` of `components/geometry.py`, and the subclasses
`DeepSetsMoments` and `JEDILinearMoments` of `_MomentsRow`, in `src/algorithms/deepsets.py` and
`jedi_linear.py`. Around them sit
`configs/algorithm/{deepsets_moments,jedi_linear_moments,moments_only}.yaml`, the nine experiment
configs `configs/experiment/hls4ml/<row>_{16,50,150}.yaml`, the three search configs
`configs/hparams_search/hls4ml/<row>_optuna.yaml`, the hand-written search catalogues
`scripts/experiments/hls4ml/run<row>_search.sh` and the generated
`scripts/experiments/hls4ml/run<row>_{train,eval}.sh`.

The scratch test ran on olgpu on 2026-10-05, with the moments behind environment switches in a
copy of the repository; its records are in
`~/Research/local/claude/scratchpad/jet_tagging_gdl/moments_2026-10-05/` on the Mac. Each parent was
trained at the best of the first 200 trials of its baseline study on `[pt, etarel, phirel]`, and
the control at the DeepSets one without phi and the pooling, for 50 epochs on the record's own
split, with two seeds per cell. The table gives the best validation accuracy as the mean ± the
half-range of the two seeds, without the moments (`none`), with the five uniform moments and with
the 25 rank-cut ones (defined below).

| row         | moments | N = 16          | N = 50          | N = 150         |
| ----------- | ------- | --------------- | --------------- | --------------- |
| DeepSets    | none    | 0.7019 ± 0.0012 | 0.7991 ± 0.0010 | 0.8090 ± 0.0001 |
| DeepSets    | uniform | 0.7126 ± 0.0000 | 0.8108 ± 0.0009 | 0.8197 ± 0.0004 |
| DeepSets    | rankcut | 0.7039 ± 0.0007 | 0.8065 ± 0.0004 | 0.8204 ± 0.0005 |
| JEDI-linear | none    | 0.7088 ± 0.0004 | 0.8073 ± 0.0010 | 0.8147 ± 0.0010 |
| JEDI-linear | uniform | 0.7140 ± 0.0008 | 0.8168 ± 0.0008 | 0.8221 ± 0.0003 |
| JEDI-linear | rankcut | 0.7062 ± 0.0004 | 0.8165 ± 0.0002 | 0.8227 ± 0.0002 |
| control     | uniform | 0.6669 ± 0.0002 | 0.7891 ± 0.0001 | 0.8107 ± 0.0001 |
| control     | rankcut | 0.7015 ± 0.0002 | 0.8025 ± 0.0001 | 0.8173 ± 0.0002 |

The uniform moments raised DeepSets by 1.1, 1.2 and 1.1 points and JEDI-linear by 0.5, 1.0 and
0.7 points at N = 16, 50 and 150. The rank-cut moments did as much at N = 50 and 150 and less at
N = 16, where they added nothing to JEDI-linear; why is not understood. The control matched
DeepSets at N = 16 with the rank-cut moments (−0.04 points) and beat it at N = 50 and 150 (+0.3
and +0.8). The vector neurons' study bests are 0.7217, 0.8229 and 0.8269, so the moments close 40
to 66 % of the gap to them; the study bests carry a selection bias that the re-runs do not, of the
size by which the re-runs without moments sit below their own rows' study bests, 0.27 to 0.41
points. The inference, not a measurement, is that the vector neurons' advantage over the set
networks is carried by the moment products and that a set network handed them recovers most of
it; the studies and the 200-epoch finals of these rows test it on the pipeline every other row
uses.

With the rows of a jet sorted by decreasing p_T, p_i = p_T,i / 20 GeV, the lift of its `etarel`
and `phirel` n_i = (1, η_i, φ_i, η_i² + φ_i²), u_i = p_i n_i and G the form of the vector neurons,
so that n_iᵀ G n_j = ΔR_ij², each weighting w of the ranks forms U = Σ_i w_i u_i and
M = Σ_i w_i u_i n_iᵀ and from them five moments in O(N):

```
sum_p     Σ_i w_i p_i
pair      Uᵀ G U          = Σ_ij  w_i w_j p_i p_j ΔR_ij²
chain     Uᵀ G M G U      = Σ_ijl w_i w_j w_l p_i p_j p_l ΔR_ij² ΔR_jl²
pair4     tr(G M G M)     = Σ_ij  w_i w_j p_i p_j ΔR_ij⁴
triangle  tr(G M G M G M) = Σ_ijl w_i w_j w_l p_i p_j p_l ΔR_ij² ΔR_jl² ΔR_li²
```

each invariant under shifts and rotations of the (η, φ) plane and under permutations of the
constituents. They are the tensor invariants of the vector neurons with fixed weights in place of
learned ones, and the uniform weights give the unnormalised β = 2 energy flow polynomials
(arXiv:1712.07124) of the edge, the two-edge path, the doubled edge and the triangle. The
`uniform` set weighs every real constituent by one, five moments; the `rankcut` set adds the
moments of the leading 1, 2, 4 and 8 constituents by p_T, 25 in all. Five of the 25 vanish
identically, every moment but sum_p under the top-1 cut and the triangle under the top-2 cut, and
are masked to exact zeros, because in float32 they come out as roundoff of terms of order p³,
which the batch normalisation would scale up to unit variance. A BatchNorm1d normalises the
moments before the head reads them and folds into the head's first layer at inference. The batch
arrives normalised; `_MomentsRow._on_features_resolved` fills
`JetMoments.denorm_scales` from the datamodule's `norm_scales`, as the vector neurons fill theirs.
The control's head reads the moments and the number of real constituents over the constant slot
count N.

`DeepSetsNet` and `JEDILinearNet` take `moments` in {none, uniform, rankcut}, and `DeepSetsNet`
also `moments_only`, under which it builds no phi and requires `phi_layers=None`. Under `none`,
which the parents' modules pass by default, `self.moments` is a plain attribute set to None: the
networks register no module, draw no random number and keep their state-dict keys, so the DeepSets
and JEDI-linear studies, finals and checkpoints are untouched and `eval.py` still loads the finals.
The parent modules `DeepSets` and `JEDILinear` are unchanged and keep `required_features = None`;
the pin to `[pt, etarel, phirel]` is a class attribute of `_MomentsRow`, whose subclasses refuse
`moments="none"`, and `JEDILinearMoments` has no `moments_only`, so Hydra refuses the key.

The algorithm configs are standalone copies of `deepsets.yaml` and `jedi_linear.yaml` that differ
in `_target_`, `name`, `nbr_feats` and `moments`, the control also in `moments_only: true` and in
having no `phi_layers` or `aggregation`; `tests/test_configs.py` keeps the rest in step. As in
`vn.yaml`, `nbr_feats` is 3 rather than `${len:${data.features}}`, because
`tests/conftest.py::algorithm` builds the rows of the tests with the four-momentum as
`data.features` by default. The default set is `uniform`. The experiments train for 200 epochs on
batches of 2048 with five folds at the algorithm defaults and set
`data.features: [pt, etarel, phirel]`. Each search config is its parent's space plus
`algorithm.moments: choice(uniform, rankcut)`, the control's being the DeepSets space without
`phi_layers` and `aggregation`, so every space holds only the FPGA-cheap pools and ReLU. They run
one study per constituent count, `<row>_nconst<N>_augmentation_none` in
`logs/optuna/hls4ml/<row>.db`: nine studies of 200 trials of 50 epochs, none launched yet
(2026-10-05). The scratch runs took 1.9 to 3.1 minutes each on a V100, which puts a study at 6 to
10 V100-hours. The train and eval catalogues hold three blocks each, all "transferred from
experiment config (no study yet)".

The experiments keep the `16-mixed` precision of the parents and leave `extras.matmul_precision`
unset. `JetMoments.raw` leaves autocast and computes the moments in float32: they are cubic in p,
which reaches about 50, so fp16 would overflow, and pair, chain and triangle cancel between large
terms, which bf16, or TF32's rounding of the matmul inputs to about 5e-4, would not resolve. The
normalised moments are of order one, and the head reads them under autocast like its other inputs.

The rows embed SO(2) through their columns: a rotation about the beam leaves `pt`, `etarel` and
`phirel` unchanged. A boost along the beam shifts the η of every massless constituent by its
rapidity and that of the massive jet axis by another amount, so it moves every `etarel` of a jet by
one amount, which the moments, built from differences, do not see and the per-constituent networks
do. `so2` is therefore exact for all three rows and `lorentz` for the control alone, for massless
constituents as for `vn`; `so3` and `lorentz_iso` change every p_T and are measured. Through the
probe on massless record jets, with the networks in float64, the `so2` shift of the logits is
1.6e-8 to 2.5e-8 of the largest logit and the control's `lorentz` shift 2.0e-8, the float32
rounding of the transform matrices, while the broken groups move the logits by 7e-3 to 1.6 of it
(2026-10-05). Evaluated on 4096 test jets, smoke models trained on five batches at N = 16, with
either moment set, flipped no jet under `so2` for any row and none under `lorentz` for the control.

At the defaults (DeepSets' phi [64, 64, 32] and rho [32, 32, 16], JEDI-linear's published point,
the control's rho [32, 32, 16], all with the uniform moments) and at the corner of each search
space (phi [128, 64, 32] and rho [64, 32]; hidden 128 and head [64, 32, 16]; rho [64, 32]; all
with the rank-cut moments), the rows have these parameters and costs per jet at N = 16, 50 and
150, measured on olgpu on 2026-10-05:

| row                   | point   | parameters | MFLOP                | M eBOPs             |
| --------------------- | ------- | ---------- | -------------------- | ------------------- |
| `deepsets_moments`    | default | 9391       | 0.209, 0.642, 1.913  | 6.80, 20.8, 62.1    |
| `deepsets_moments`    | corner  | 16 855     | 0.357, 1.086, 3.231  | 11.5, 35.2, 104.6   |
| `jedi_linear_moments` | default | 19 919     | 0.292, 0.863, 2.544  | 9.56, 28.3, 83.5    |
| `jedi_linear_moments` | corner  | 62 647     | 1.124, 3.385, 10.035 | 36.4, 109.7, 325.3  |
| `moments_only`        | default | 1903       | 0.005, 0.006, 0.010  | 0.166, 0.242, 0.466 |
| `moments_only`        | corner  | 4023       | 0.013, 0.020, 0.040  | 0.449, 0.734, 1.573 |

The moments add w inputs to the head's first layer and 2w parameters in their batch
normalisation, w = 5 or 25, and cost 749, 1905 and 5305 multiplications per jet at N = 16, 50 and
150 for the uniform set and 3105, 7525 and 20 525 for the rank-cut one. The FLOP counter prices
matrix products alone and misses the elementwise products among them, which matters only for the
control: its eBOPs exceed 32 × FLOPs by 12, 26 and 46 % at the defaults, where every other row
sits within 5 %. The unpriced share, the sort and the batch normalisation of the moments, is
0.02 %, 0.01 % and 0.8 % at N = 16. A V100 pre-flight at the corner, with batches of 2048 record
jets under fp16 autocast and a gradient scaler, reserved 0.10, 0.21 and 0.43 GiB and took 4.3,
4.4 and 4.9 ms per step for `deepsets_moments`, 0.12, 0.28 and 0.57 GiB and 5.1, 5.4 and 7.9 ms
for `jedi_linear_moments` and 0.05, 0.06 and 0.11 GiB and 3.1 ms for the control at N = 16, 50 and
150 (olgpu, 2026-10-05), so every study fits a 1g MIG slice.

`tests/test_algorithms.py` checks the moments against their explicit sums over pairs and triples at
every rank cut, the cut keeping the leading constituents, on jets whose padded slots sit among the
real ones and on jets of one and two constituents, and their dtype and accuracy under bf16 autocast
on the CPU and fp16 autocast on a GPU (both fail without the guard). Against stacks rebuilt from the
same seed in the committed order, it checks that the two networks without moments keep their
state-dict keys, draw the same weights and compute the same logits, bit for bit, under every
pooling. It builds the three rows with the other rows of `MODELS`, checks that `moments_only` fills
its scales and that `setup()` refuses a longer column list for `deepsets_moments` and a reordered
one for `jedi_linear_moments`, and holds the rows to their exact groups at 1e-6 and to a thousand
times the `so2` shift under the others, calling the network directly in float64 with its batch
normalisation calibrated on the test jets, since at its initial statistics the raw moments, about
1e10 there, would drown every other input of the head. `tests/test_configs.py` keeps the configs
copies of their parents, the search spaces the parents' plus the moment set and the experiments at
the trainer's mixed precision with no `matmul_precision`; it also holds the state-dict layout of
every experiment to a digest of the committed one.

## Jet-level ceiling

Added on 2026-10-06 in two rows, after a scratch run of 2026-10-05, to measure how much the
constituent networks know beyond fixed substructure observables. Beside its constituents, every
jet of the record carries 53 observables computed upstream from the whole jet, the `j_*` columns
of its parquet shards: its p_T, rapidity, mass and multiplicity, N-subjettiness, energy correlation
functions and groomed masses. `jet_level_all53` reads all of them and `jet_level_noeta52` all but
the jet rapidity `j_eta`, which a row exact under the beam boost cannot see; it is the ceiling for
those rows. `j_ptfrac` is 1 on every jet, so 52 and 51 of the columns carry information. The rows
are the MLP of `src/algorithms/mlp.py` on one slot of observables per jet, with no network or
module of their own. Around it sit `src/data/hls4ml_jet_level_datamodule.py`,
`configs/data/jet_level.yaml`,
`configs/algorithm/jet_level_{all53,noeta52}.yaml`, the experiment configs
`configs/experiment/hls4ml/jet_level_{all53,noeta52}_150.yaml`, the search configs
`configs/hparams_search/hls4ml/jet_level_{all53,noeta52}_optuna.yaml`, the hand-written search
catalogues `scripts/experiments/hls4ml/runjet_level_{all53,noeta52}_search.sh` and the generated
`scripts/experiments/hls4ml/runjet_level_{all53,noeta52}_{train,eval}.sh`.

The scratch run fitted an MLP and gradient-boosted trees on the record's own split, over a grid of
eight MLP points (widths [64, 32, 32] or [256, 128, 64], dropout 0 or 0.1, learning rate 1e-3 or
3e-4, batches of 1024, up to 100 epochs with early stopping at a patience of 12) and six tree
points, with three seeds at the selected MLP point. Its records are in
`~/Research/local/claude/scratchpad/jet_tagging_gdl/ceiling_2026-10-05/` on the Mac. The selected
MLP, widths [256, 128, 64] without dropout at a learning rate of 1e-3, reached a validation and
test accuracy of 0.8319 ± 0.0002 and 0.8316 ± 0.0004 on all 53 observables and 0.8308 ± 0.0005 and
0.8311 ± 0.0004 without `j_eta`, the ± being the sample deviation over the seeds. The trees landed
0.0 to 0.5 points below; the 16 groomed observables of the original hls4ml tagger gave 0.774, and
the mass, p_T, rapidity and multiplicity alone 0.786. Against the validation accuracy of the N = 150
finals' main runs, only L-GATr-slim and PELICAN with the beams, 0.840 and 0.838, beat the 52
observables, by 0.9 and 0.7 points; LLoCa's SO(2) × SO(1,1) row matched them and every FPGA-oriented
row sat below. The inference, not a measurement, is that the constituent networks carry at most
about one point of information beyond these observables on this data set. The rows re-measure the
ceiling under the protocol of the networks: their search space, their trial budget, their folds and
their test-time metrics.

`JetLevelDataModule` reads the `j_*` columns and the `label` of the snapshot's
`{train,validation,test}-*.parquet` shards with pyarrow, in sorted order, which is the order in
which the record's loader reads its jets. It splits them as `HLS4MLDataModule` does: the record's
own split at `n_folds` 1, otherwise fold `fold_idx` of the pooled training then validation jets
through `fold_indices`, with the `kfold_seed` of `configs/data/hls4ml.yaml`, so that fold i of a
jet-level row is fitted and validated on the jets of fold i of every network. Every split is
standardised with the float64 mean and standard deviation of the record's own training split, a zero
spread replaced by one: fixed constants, the same for every fold and at test time, as the record's
loader normalises the constituents of every fold with constants fitted on its training split.
Per-fold statistics would be wrong at evaluation, where `eval.py` tests the checkpoints of the five
folds with one datamodule. The batches are `(x, y)` with `x` of shape (jets, 1, F) in float32 and
`y` one-hot, served by `HLS4MLDataset` with its pre-batching, so that the base class and the
callbacks see the usual contract: training and validation drop the trailing partial batch and the
test split keeps every jet. The loaders run in the main process, since the splits sit in memory and
nothing transforms them. The datamodule exposes `features` (every observable but those of
`data.exclude`), `feature_names` (all 53, in the shards' order), `norm_scales` (None) and
`keep_full_features_at_test` (false); `JetTaggingModule.setup` then reads the batches unreduced.
The constructor reads nothing, as `tests/test_configs.py` needs when it instantiates every
experiment's data node, and refuses every regime but augmentation/none; the first access to the
shards refuses an `exclude` that names no column. `configs/data/jet_level.yaml`
pins the revision of `configs/data/hls4ml.yaml` and sets `nconstituents: 150`, the label of the rows:
the observables come from the whole jet, which N = 150 never truncates, so the rows pair with the
networks' N = 150 rows under `scripts/analysis/compare_to_baseline.py --baseline-model`. The data
config has no `features` key, so `write_results_row` reads it with `get`.

The algorithm configs are standalone copies of `mlp.yaml` that differ in `name`, in
`nconstituents: 1`, the one slot, and in `nbr_feats`, 53 or 52, a number because the observables
come from the shards at run time; `tests/test_configs.py` keeps the rest in step. The MLP network
and its state-dict layout are those of the constituent MLP, whose finals' checkpoints still load.
The experiments train the MLP's recipe (Adam, `ReduceLROnPlateau` at a patience of 80, early
stopping at 100) for 200 epochs on batches of 2048 with five folds under the trainer's 16-mixed,
set `results_table.subdir: jet_level` and `callbacks.invariance_metric.probe: false`, and
`jet_level_noeta52_150` sets `data.exclude: [j_eta]`. Each search config is the MLP's space, its 200
trials of 50 epochs and its sampler (TPE, seed 1234, 20 start-up trials), which
`tests/test_configs.py` keeps equal, with the study `<row>_nconst150_augmentation_none` in
`logs/optuna/hls4ml/<row>.db`: two studies, none launched yet (2026-10-06). The space's widest
point, [256, 128, 64], is the one the scratch run selected. In
`scripts/experiments/make_train_catalogues.py` and `tests/conftest.py` the rows form `JET_LEVEL`,
inside
`BASELINE_ONLY` and outside `MODELS`, which `tests/test_algorithms.py` builds on constituent
batches; the generator gives them the constituent count 150 alone, a header note of their own and
catalogues in the default layout alone, one block each, "transferred from experiment config (no
study yet)".

No group acts on the observables, so the invariance probe would report a perfect invariance that is
never tested. Under `probe: false` `InvarianceMetricCallback` transforms nothing: it keeps the clean
logits, recomputed in fp32 as for every row, and the labels, logs the 24 groupless
`performance/*` keys of `clean_metrics` at the end of the test epoch, and under `eval.py` writes
`outputs/<label>.npz` with `logits_clean` and `labels` alone. Every other row keeps the default,
`probe: true`. The results row has the networks' header: every invariance cell holds NaN, through
the missing keys, and `selected_inv_key` reads `invariance/mean_tv`. `PlottingCallback` reads the
step's logits and the labels alone and draws its ROC curves and confusion matrices unchanged.

The base class counts the FLOPs and eBOPs of the MLP alone, on its 53 or 52 inputs: they leave out
the upstream computation of the observables (N-subjettiness, energy correlation functions,
grooming), which an FPGA would have to carry too, so they understate what the tagger costs as a
whole and compare with no network's. At the algorithm default, widths [128, 64, 32], the counts per
jet are 34 368 and 34 112 FLOPs and 1.103 and 1.095 M eBOPs for `jet_level_all53` and
`jet_level_noeta52` (olgpu, 2026-10-06); at the widest point of the space, [256, 128, 64], the layer
widths give 109 696 and 109 184 FLOPs and 3.52 and 3.50 M eBOPs. A smoke training on the record took
2.1 s per epoch on a V100 at batches of 2048, which puts a 50-epoch trial near 2 minutes, a study
near 7 V100-hours and a 200-epoch process near 7 minutes; two epochs reached a test accuracy of
0.814 and 0.813.

`tests/conftest.py` writes synthetic shards in the record's layout (a constituent list column, 53
`j_*` columns with a constant `j_ptfrac`, and `label`; two shards per split, each split at its own
offset), on which `tests/test_data.py` checks the column order and `exclude`, that every split and
fold is the standardisation of its jets by the training split's constants, bit for bit, with the
constant column centred to zero, that a fold holds the jets `fold_indices` selects from the pooled
jets, the batch shape and the one-hot labels, and the refusal of every regime but the baseline.
`tests/test_entry_points.py` trains `jet_level_noeta52` on the CPU, its main run and fold 1 of 2,
evaluates both checkpoints with `eval.py` and checks the one row: the networks' header,
`nconstituents` 150, the fold alone, NaN in every invariance cell of the folds, positive costs, and
an `.npz` per checkpoint with the two keys and every test jet, the trailing partial batch included.
Under `slow` and with the record, `tests/test_data.py` checks that the datamodule's labels of every
split equal those of the loader's cache at N = 16 and that the shards' constituent lists, capped at
16, hold the cache's real-constituent counts, jet by jet. `tests/test_evaluation.py` checks the
probe-less mode against `clean_metrics`, `tests/test_configs.py` the copies, the search spaces and
the experiments, `tests/test_data.py` the shared revision and fold seed, and `tests/test_scripts.py`
the catalogues.

Two rows of gradient-boosted trees joined on 2026-10-07, `jet_level_bdt_all53` and
`jet_level_bdt_noeta52`, on the same observables through the same datamodule, so that the ceiling
rests on a second model family and not on the MLP alone. In the scratch run of 2026-10-05 the best
of six tree points reached a validation and test accuracy of 0.8302 and 0.8298 on all 53
observables, against the MLP's 0.8319 and 0.8316: close enough that either family could set the
ceiling once both are searched alike. The rows run scikit-learn's `HistGradientBoostingClassifier`
through the networks' pipeline: `src/train.py` with the default callbacks, the Optuna sweeper,
`submit_train.sh --folds 5`, `src/eval.py`, the generated catalogues and the results tables. The
module is `src/algorithms/bdt.py::BoostedTreesClassifier` and the wrapper
`src/algorithms/bdt.py::BoostedTrees`; around them sit
`configs/algorithm/jet_level_bdt_{all53,noeta52}.yaml`, the experiment configs
`configs/experiment/hls4ml/jet_level_bdt_{all53,noeta52}_150.yaml`, the search configs
`configs/hparams_search/hls4ml/jet_level_bdt_{all53,noeta52}_optuna.yaml`, the hand-written
`scripts/experiments/hls4ml/runjet_level_bdt_{all53,noeta52}_search.sh` and the generated
`scripts/experiments/hls4ml/runjet_level_bdt_{all53,noeta52}_{train,eval}.sh`.

Boosting is mapped onto epochs. One estimator is kept for the whole run, with `warm_start=True` and
`early_stopping=False`; before each fit its `max_iter` is raised by `trees_per_epoch`, 10 and never
searched, and each iteration grows one tree per class. An epoch therefore adds ten iterations and
`val/acc` is logged per epoch, so the networks' callbacks apply unchanged: the checkpoint keeps the
best epoch by `val/acc`, early stopping waits 100 epochs, and Optuna reads `val/acc_best`. A trial
of 50 epochs grows up to 500 iterations and a training of 200 epochs up to 2000. Lightning
validates an epoch before it calls `on_train_epoch_end`, so the fit happens inside `training_step`
on the epoch's last batch (`trainer.is_last_batch`), and the validation of epoch k is scored on
(k + 1) × 10 iterations. The fit reads the datamodule's whole training split, not the epoch's
batches, which are a fresh permutation less a random partial batch. scikit-learn refits its bin
edges on the jets of every fit and, inside `fit`, scores the earlier trees on the binned jets, so
only identical jets keep the earlier trees reading as they were grown; on identical jets three fits
of ten iterations give the trees of one fit of thirty. Training is manual
(`automatic_optimization = False`): `configure_optimizers` returns None, for which Lightning puts in
its mock optimiser, and each batch takes one mock step, without which `global_step` would stay at
zero and `ModelCheckpoint` would save nothing. The training loss and accuracy are the base class's,
computed once per epoch on the epoch's last batch after the fit: a pass over the whole split would
score every tree on every jet once more, a cost that grows with the trees, for figures that select
nothing.

The estimator holds no tensor. It travels in the state dict as `pickle.dumps(estimator)` bytes,
through `get_extra_state` and `set_extra_state` under the key `net._extra_state`, so Lightning's
checkpoints carry it and `eval.py`, which loads them with `weights_only=False`, restores it.
Validation and test run the base class's steps on the log-probabilities
`log(clip(predict_proba, 1e-7, 1))`, zeros before the first fit. The estimator's draws (the
subsample its bin edges are fitted on, the feature subsampling) are seeded from
`torch.initial_seed()`, the run's seed, so that the folds share them as the networks share their
initialisation. The trees run in scikit-learn, outside the torch operations the counters trace, so
the module overrides `_record_costs`, the base class's hook for the counts, to set `flops_total`,
`ebops_total` and `ebops_unpriced` to NaN, and the results row holds `nan` in `flops_mean` and
`ebops_mean`. The experiments select `/trainer: cpu` with `precision: 32` and
`gradient_clip_val: null`, since Lightning refuses clipping under manual optimisation. The
generator lists the two rows in `CPU_ROWS` as well as `JET_LEVEL`, and their blocks carry
`trainer=cpu` and no `trainer.devices`, which the submitters' `--gpu` and `--gpus` would rewrite:
a block runs unchanged on a CPU host. The
algorithm configs are copies of each other up to `name` and `nbr_feats`, set `optimizer` and
`scheduler` to null and leave the trees at scikit-learn's defaults (learning rate 0.1, 31 leaves, no
depth limit, 20 jets per leaf, no L2 term, every feature at each split). The two search configs hold
one space, with 200 trials and TPE at seed 1234 with 20 start-up trials: a log-uniform learning rate
on [0.01, 0.3], `max_leaf_nodes` in {15, 31, 63, 127, 255}, `max_depth` in {0, 6, 12},
`min_samples_leaf` in {20, 100, 500}, `l2_regularization` in {0, 0.1, 1} and `max_features` in
{0.5, 1}. A depth of 0 means no limit: the sweeper writes a None choice as the string 'None', which
Hydra keeps as a string, so the search spells it 0 and the wrapper maps 0 and None to None.

On olgpu's CPUs at 8 threads (2026-10-07), an epoch of ten iterations on the 496 000 training jets
took 3.2 s at the start and about 0.63 s longer for every ten iterations already grown, 15.4 s at
epoch 19: every fit, the training metrics and the validation evaluate all the earlier trees again.
The process peaked at 1.8 GB. A linear extrapolation, not a measurement, puts a 50-epoch trial at
the defaults near 16 minutes, a 200-trial study near 50 hours on one driver, and a 200-epoch process
near 3.7 hours unless early stopping ends it. At the defaults the validation accuracy peaked at
0.8268 after 80 iterations, and three epochs gave a test accuracy of 0.8215 and 0.8210 for the two
rows. A checkpoint holds about 5 kB per tree at 31 leaves (0.78 MB at 30 iterations), so one of
2000 iterations at 255 leaves would hold about 330 MB.

`tests/test_algorithms.py` checks the wrapper (ten iterations per fit, the state-dict round trip,
the depth mapping) and that epoch k is validated on (k + 1) × 10 iterations with one mock step per
batch. `tests/test_entry_points.py` runs `jet_level_bdt_noeta52` end to end on the synthetic shards
with the default callbacks: a main run and fold 1 of 2, two epochs each, then `eval.py` on the
directory. The best checkpoint of each run holds (epoch + 1) × 10 iterations, and the one row has
the networks' header, `nconstituents` 150, NaN costs and invariance cells, with a two-key `.npz` per
checkpoint. `tests/test_configs.py` checks the copies, the shared search space and the experiments'
fp32, absent clipping, disabled probe and subdirectory, and exempts the two rows from the cost
inequality, and `tests/test_scripts.py` checks `CPU_ROWS` against `tests/conftest.py` and that the
blocks of the two rows carry `trainer=cpu` without a device, in the default layout alone.

## Top-tagging replica

Added on 2026-10-09: 18 of the rows on a second data set, the top-quark tagging reference set of
arXiv:1902.09914 (Zenodo 2603256), on which LorentzNet, PELICAN, L-GATr-slim and LLoCa published
their figures. The nine rows of those four architectures train on the published recipes, so that
their figures test the ports against the papers'; `vn`, the five grid rows and the three moment rows
are searched around their hls4ml optimum. Every row runs at N = 100 in the baseline regime with two
classes, `qcd` and `top`. CLAUDE.md ("Top-tagging replica") describes the record, its conversion,
the datamodule, the configs and the catalogues; this section holds the recipes and their sources,
where the ports depart from the references, the measured costs and the published figures. The
literature verification behind the recipes and the figures (2026-10-08), with verbatim quotes, is
`~/Research/local/claude/scratchpad/jet_tagging_gdl/toptagging_2026-10-08/literature_toptagging.md`
on the Mac. Around the existing modules sit `src/data/top_tagging_datamodule.py`, which holds
the conversion and the datamodule, `src/algorithms/utils/schedules.py`,
`configs/data/top_tagging.yaml`,
`configs/results_table/top_tagging.yaml`, the 18 experiment configs
`configs/experiment/top_tagging/<row>_100.yaml`, the 14 search configs
`configs/hparams_search/top_tagging/<row>_optuna.yaml` and `<row>_scl_optuna.yaml`, the catalogues
under `scripts/experiments/top_tagging/`,
`scripts/analysis/top_tagging_summary.py` and `tests/test_top_tagging.py`.

The training jets hold 49.05 constituents on average and 160 at most, so N = 100 truncates 6006 of
the 1 211 000 (0.5 %) and leaves 49.01 on average; the references read every constituent, except
the 208k PELICAN paper, which caps the jets at 80. The classes are balanced to within 172 jets per
split (605 523 QCD and 605 477 top jets in training). The conversion's stable sort reordered 7
training jets, p_T ties at float32 precision (2026-10-09).

The recipes, as the experiment configs set them:

| row                                | network                                                                                             | optimiser                                      | schedule                      | batch, epochs | clipping |
| ---------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ----------------------------- | ------------- | -------- |
| `lorentznet`                       | 6 blocks of width 72, c = 5e-3, dropout 0.2                                                         | AdamW, 1e-3, weight decay 0.01                 | `warmup_restarts_decay`       | 128, 35       | none     |
| `pelican`, `pelican_lorentz`       | 5 blocks, `n_message` 60, `n_hidden` 35, dropout 0.01, `nbar` 51 and 49                             | AdamW, 2.5e-3, weight decay 0.005              | `warmup_restarts_decay`       | 100, 35       | norm 1   |
| `lgatr_slim`, `lgatr_slim_lorentz` | the algorithm's 12 blocks, 32 vector and 96 scalar channels, 8 heads, `mlp_ratio` 4                 | Lion, 3e-5, weight decay 2, betas (0.9, 0.99)  | `cosine`                      | 128, 21       | norm 1   |
| the four `lloca_*`                 | 12 blocks, 8 heads of 12 scalars and 1 vector, `mlp_factor` 2, frames-net 128 wide with dropout 0.1 | Lion, 3e-5, weight decay 2, betas (0.9, 0.999) | `cosine_restarts`, two cycles | 128, 32       | norm 1   |

The sources are arXiv:2201.08187 §3.3 and the LorentzNet-release script `top_tagging.py` for
LorentzNet; arXiv:2211.00454 §4, arXiv:2307.16506 Table 2 and the PELICAN repository's
`config/48k.yaml` and `slurm/` commands for PELICAN; arXiv:2512.17011 App. A and the
tagger-quantization repository's config for L-GATr-slim; and arXiv:2508.14898 App. A with the
lloca-experiments files `config/model/tag_top_transformer.yaml` and
`config/training/tag_default.yaml` for LLoCa. The twins without spurions (`pelican_lorentz`,
`lgatr_slim_lorentz`) and the three LLoCa rows of another frames class take their partner's recipe.

Where the sources disagree or are silent, the replica chooses as follows. LorentzNet warms up for
the paper's 4 epochs, where the repository's README and its shipped `args.json` say 5; its batch of
128 is the paper's 32 per GPU on four V100s, and its experiment sets
`trainer.gradient_clip_val: null`, since the reference script clips no gradient. PELICAN takes the
learning rate of 2.5e-3 and the dropout of 0.01 of the 45k paper (arXiv:2211.00454), which the
repository's top-tagging commands share, against the 1e-3 and 0.025 of the 208k paper, the weight
decay of 0.005 of arXiv:2307.16506 and the repository, and the widths of the 48k row
(`config/48k.yaml`: `num_channels_m` 60 for the messaging MLP, `num_channels_2to2` 35 for the pair
tensor). No source states a gradient clipping, so the trainer's norm of 1 stays. The 208k paper
caps the jets at 80 constituents and fills every batch with equal numbers of signal and background
jets; the replica does neither. The port has 179 023 parameters with the beams and 179 027 without,
against the reference's 48k, because the rewrite does not factorise the aggregation weights (see
"PELICAN"); its `nbar` is the training jets' mean multiplicity at N = 100, 49.01, plus the two
beams, 51 for `pelican` and 49 for `pelican_lorentz`. The L-GATr-slim paper states no Lion betas,
so the experiments write out the package's default (0.9, 0.99); 200 000 steps at a batch of 128
are 21 epochs of the training split. The LLoCa paper states its betas, (0.9, 0.999), and a dropout
of 0.1 in the frames-net alone, which the wrapper's `framesnet_dropout` carries while the
backbone's `dropout` stays 0. At 0, the default of every hls4ml row, the package builds no dropout
layer; a nonzero value inserts layers that renumber the later linear layers of the frames-net in
the state dict, so the hls4ml checkpoints load at 0 alone. 300 000 steps at a batch of 128 are 32
epochs, and the restart after 150 000 falls at epoch 16. The paper puts its weight decay of 2 on
"the main network"; the replica's one optimiser applies it to every parameter, the frames-net's
included.

`warmup_restarts_decay` multiplies the initial rate by

```
f(t) = t / 4                          0 ≤ t < 4
f(t) = ½ (1 + cos π u)                4 ≤ t < 32
f(t) = f(31) · 0.5^(⌊t − 32⌋ + 1)     32 ≤ t < 35
```

with t the epoch, fractional within an epoch, and u the fraction of the current cosine cycle run
through, the cycles lasting 4, 8 and 16 epochs from epoch 4 (`t0` 4, `t_mult` 2) as those of
`CosineAnnealingWarmRestarts` do. The warm-up and the cosine are evaluated at every step. The decay
is anchored at f(31) ≈ 0.0096, the factor at the start of the last cosine epoch, and halves it once
per epoch: the LorentzNet reference steps its scheduler once per epoch, so its last cosine rate is
that of epoch 31. `cosine` anneals to zero over the run, and `cosine_restarts` does so `cycles`
times, restarting at the initial rate. Each is a `LambdaLR` stepped per batch (`interval: step`),
its `total_steps` injected by `_set_up_scheduler`; `warmup_restarts_decay` also takes
`epochs: ${trainer.max_epochs}`, which splits the steps into epochs. An experiment cannot replace
the algorithm's scheduler by merging into `algorithm.scheduler`, since the keys of the algorithm
config's `ReduceLROnPlateau` or `OneCycleLR` (`factor`, `patience`, `max_lr`, `pct_start`) would
survive the merge and reach a factory that takes none of them. The recipe experiments therefore
define the schedule as a top-level `scheduler:` node and assign it whole with
`algorithm.scheduler: ${scheduler}`, and `tests/test_configs.py` checks that the recipe rows, and
they alone, build a `LambdaLR`.

The replica copies the published widths and optimisation; the inputs stay those of the ports. The
LorentzNet reference appends the two beams as particles and gives every node a one-hot beam flag
times |m| as its scalars (arXiv:2201.08187 §3.1 and §4.1; `top/collate.py` of its repository). The
port reads the constituents alone, with ψ(m²) as the one scalar and a tanh gate in place of the
reference's per-component clamp of the coordinate update at ±100 GeV (see
`src/algorithms/lorentznet.py`), so it embeds the whole Lorentz group, which the
reference's beam particles reduce to rotations about the beam. The
L-GATr-slim and LLoCa references boost every jet to its rest frame and give each constituent seven
ParT-style scalar features (`boost_jet` and `tagging_features` in their configs); the ports read the
four-momenta in units of 20 GeV alone, L-GATr-slim with one scalar flag on its global token and
LLoCa with no scalar in its frames-net. The LLoCa reference's frames-net also reads the beam and
time spurions, so its exact group is SO(2) about the beam, an inference of the literature
verification rather than a statement of the paper, where `lloca_lorentz` holds the Lorentz group.
PELICAN's beams (1, 0, 0, ±1) and L-GATr-slim's time and lightlike beam tokens are those of the
references.

Smokes on a V100 of olgpu (2026-10-09), each one epoch cut to the training batches below and five
validation batches at the recipe's batch, gave the step rates below. The peak memory is
nvidia-smi's, the CUDA context included, and the parameters were counted on the built modules. The
minutes per epoch and the hours per recipe extrapolate the training steps alone (9461 per epoch at
a batch of 128, 12 110 at 100) and leave out the validation of 403 000 jets after every epoch, the
compilation of L-GATr-slim and LLoCa (15 s to 1.5 min per fit), and the five folds, each a training
of its own. LorentzNet's rate comes from 20 batches and is the roughest of the four.

| row                       | batches | steps per s | min per epoch | h per recipe | peak memory | parameters |
| ------------------------- | ------- | ----------- | ------------- | ------------ | ----------- | ---------- |
| `lorentznet`              | 20      | 8.0         | 20            | 11.5         | 12.6 GiB    | 223 568    |
| `pelican`, two chunks     | 220     | 7.9         | 26            | 15           | 7.3 GiB     | 179 023    |
| `lgatr_slim`, compiled    | 220     | 15.9        | 10            | 3.5          | 4.6 GiB     | 2 013 826  |
| `lloca_lorentz`, compiled | 220     | 13.9        | 11            | 6.0          | 6.3 GiB     | 1 601 669  |

The counts of LorentzNet, L-GATr-slim and LLoCa match the papers' 224k, 2.0M and 1.6M. The
published figures the replica is measured against, as `scripts/analysis/top_tagging_summary.py`
prints them:

| tagger             | source                                 | accuracy        | AUC             | 1/ε_B at ε_S = 30 % | 1/ε_B at ε_S = 50 % | parameters |
| ------------------ | -------------------------------------- | --------------- | --------------- | ------------------- | ------------------- | ---------- |
| LorentzNet         | arXiv:2201.08187, Table 1, 6 runs      | 0.942           | 0.9868          | 2195 ± 173          | 498 ± 18            | 224k       |
| PELICAN, 60/35     | arXiv:2307.16506, Table 2, 5 seeds     | 0.9424 ± 0.0001 | 0.9868 ± 0.0001 | 2148 ± 125          | not reported        | 48k        |
| L-GATr-slim        | arXiv:2512.17011, Table 1, 5 trainings | 0.9420          | 0.9869 ± 0.0001 | 2264 ± 93           | 546 ± 7             | 2.0M       |
| LLoCa-Transformer  | arXiv:2508.14898, Table 5, 5 seeds     | 0.9416 ± 0.0001 | 0.9866 ± 0.0001 | 2150 ± 130          | 492 ± 15            | 1.6M       |
| ParT, from scratch | arXiv:2202.03772                       | 0.940           | 0.9858          | 1602 ± 81           | 413 ± 16            | 2.1M       |

The papers' uncertainties are spreads over repeated trainings on the record's split, whereas the
folds of this repository share their seed and differ in their split, so the summary's fold spread
measures something else. The record carries no jet-level observables, so the replica
computes no ceiling of its own as "Jet-level ceiling" does on hls4ml, and the literature supplies
it: the taggers on expert variables reach accuracies of 0.892 to 0.932 and rejections at
ε_S = 30 % of 151 to 1249, namely the multi-body N-subjettiness DNNs (0.922 and 0.929, 792 ± 18
and 867 ± 15), LDA (0.892, 151) and the linear EFP discriminant (0.932, 384) of arXiv:1902.09914
Table 1, and BIP (0.929 and 0.931, 600 ± 47 and 853 ± 68) and DisCo-FFS (AUC 0.982, 1249 ± 43) as
arXiv:2307.16506 Table 1 quotes them. The summary script prints these too.

`tests/test_top_tagging.py` converts a synthetic record of three HDFStore files at N = 8, with two
jets planted out of p_T order, and checks the sort, the truncation, the scales and the labels of
the cache and its `meta.json`; that the datamodule serves training batches at the width of
`data.features` and test batches at nine columns, with the cache's `class_names`, scales and axis
columns and the folds of `fold_indices`; that a cache is built once and that a stale
`format_version`, an md5 mismatch and `full_axis` are refused; the LorentzNet schedule over 35
epochs; and `deepsets_100` and `lorentznet_100` end to end on the CPU, a main run and fold 1 of 2
trained and evaluated into `results/top_tagging/{pt_etarel_phirel,embedded}/` with two-column
logits. Under `slow`, with `TOP_TAGGING_RAW` naming the directory of the raw files, it reads the
first 10 000 jets of `val.h5` and checks their labels and balance, E ≥ |p|, the jet p_T and η
windows and the p_T order. `tests/test_configs.py` builds every top-tagging experiment and search
config, `tests/test_algorithms.py` steps every row at `output_dim=2` and checks the refusal of a
class-count mismatch, `tests/test_evaluation.py` checks the two-class probe keys and plots, and
`tests/test_scripts.py` the generator's top-tagging layout, the submitters' reading of its
catalogues and the summary script against the probe. The fast suite passed at 403 tests on olgpu
(2026-10-09).

## Vector neurons on the four-momentum

The rows `vn_z` and `vn_so2`, added on 2026-09-29 in bb236e70 and replaced on 2026-10-03 by `vn`
(above). They ran vector neurons on the four-momentum (E, px, py, pz) in units of 20 GeV, divided
into splits with weights of their own under the Minkowski metric: (px, py) and (E, p_z) for `vn_z`,
exact under rotations about the beam, boosts along it and the reflections φ → −φ and p_z → −p_z,
and so under the probe's `so2` and `lorentz`; and E, p_z and (px, py) for `vn_so2`, exact under
`so2` alone. A block mixed the vector channels split by split, read learned Minkowski products
compressed by the signed log `psi` of the LorentzNet component, and gated each split of each
channel with a sigmoid. The first block also read fixed rank invariants, the Minkowski squares of
every sorted constituent and its Cauchy-Schwarz deficits against the jet on each split, without
which `vn_z` reached a validation accuracy of 0.66 rather than 0.71 at N = 16. The default was
three blocks of 16 vector and 64 scalar channels with 16 invariants.

Files, recovered with `git show b54eef5a:<path>`: `configs/algorithm/vn_{z,so2}.yaml`,
`configs/experiment/hls4ml/vn_{z,so2}_{16,50,150}.yaml`,
`configs/hparams_search/hls4ml/vn_{z,so2}_optuna.yaml` and
`scripts/hls4ml/runvn_{z,so2}_{search,train,eval}.sh`, all added in bb236e70, and the versions of
`src/algorithms/components/vector_neurons.py`, `src/algorithms/vector_neurons.py`,
`tests/test_algorithms.py` and `tests/test_configs.py` at b54eef5a; the two modules were rewritten
in place for `vn`. The section "Vector neurons" of this file at b54eef5a holds the design and its
reasons. The two N = 16 study databases, `logs/optuna/hls4ml/vn_z.db` and `vn_so2.db` (200 trials
each, 2026-09-29), stay.

Why removed: the Phase-2 L1 trigger delivers (p_T, η, φ), so these rows would need trigonometric
and hyperbolic tables in front of their first layer, and lookup tables for their logarithms and
sigmoids, while the lifted row needs none and equals or beats them: 0.7203 against 0.7105 at
N = 16, both untuned, with under half the parameters and 0.6 of the FLOPs.

The numbers are best validation accuracies of single 50-epoch runs at the defaults, measured on
olgpu on 2026-09-29 and 2026-09-30; reruns and second seeds spread by up to 0.0007. `vn_z` reached
0.7105, 0.8107 and 0.8164 at N = 16, 50 and 150, and `vn_so2` 0.7115, 0.8110 and 0.8164; the
N = 16 study bests (200 trials each, 2026-09-29) are 0.7138 and 0.7149. On the test set, `vn_z` at
N = 16 flipped 0.01 % of the jets under `so2` and under the beam boost, 39.8 % under `so3` and
40.0 % under `lorentz_iso`; `vn_so2`, which does not embed the boost, flipped 3.1 to 5.0 % under it
at the three counts. At the defaults `vn_z` had 48 805, 60 773 and 95 973 parameters and cost
0.108, 0.134 and 0.211 MFLOP per jet at N = 16, 50 and 150, and `vn_so2` 58 965, 80 181 and
142 581 parameters and 0.122, 0.166 and 0.294 MFLOP (2026-09-29).

Porting notes. The partition was set by the switch `symmetry`, not `group`, which
`JetTaggingModule.__init__` owns for the SCL. A port restores `required_features` to the
four-momentum, `vn_z` and `vn_so2` to `MODELS` and `EMBEDDED` in both
`scripts/experiments/make_train_catalogues.py` and `tests/conftest.py`, and their display names to
`scripts/analysis/common.py`; the TF32 exclusion holds for them as for `vn`.

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
`src/data/components/augmentation.py::make_transform` and a method to the datamodule's
`SUPPORTED_METHODS`, which `_validate_run` alone checks;
the invariance callback would then report it beside the three physical groups.
