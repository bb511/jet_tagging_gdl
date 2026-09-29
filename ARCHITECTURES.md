# Removed architectures

This repository carried eight architecture modules before the refactor to the `adl1t` layout
and carries eight now: DeepSets, MLP, JetFormer, LorentzNet, PELICAN (back since 2026-09-13 as
a clean rewrite), L-GATr-slim, LLoCa and the vector neurons. The last three were added on
2026-09-28 and 2026-09-29: L-GATr-slim runs in two rows, `lgatr_slim` with the spurion tokens and
`lgatr_slim_lorentz` without, LLoCa in four, one per frames class (`lloca_lorentz`, `lloca_z`,
`lloca_so2` and `lloca_so3`), and the vector neurons in two, one per partition of the
four-momentum (`vn_z` and `vn_so2`). PELICAN, a plain Transformer encoder, a
Lipschitz-constrained MLP and an IRC-safe DeepSets were removed at the refactor; nano-PELICAN,
which existed as configuration only, went with them, and so did the permutation check. This file
records what each was, why it went, and what putting it back involves.

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

## Vector neurons

Added on 2026-09-29 as the fifth symmetry-embedded architecture. Unlike those of L-GATr-slim and
LLoCa, its network is written here, so it adds no dependency. Vector neurons (arXiv:2104.12229)
carry lists of vectors as hidden features and mix them with a bias-free linear layer that acts on
the channel index alone, so that the layer commutes with the group; here the vectors are
four-momenta and the group is a subgroup of the Lorentz group. Our code is the network
`src/algorithms/components/vector_neurons.py` and the `JetTaggingModule` subclass
`src/algorithms/vector_neurons.py`, which reads `[px, py, pz, e]` and hands the network
(E, px, py, pz) in GeV. Around them sit the two algorithm configs `configs/algorithm/vn_z.yaml` and
`vn_so2.yaml`, the six experiment configs `configs/experiment/hls4ml/vn_{z,so2}_{16,50,150}.yaml`,
the two search configs `configs/hparams_search/hls4ml/vn_{z,so2}_optuna.yaml` and the six
catalogues `scripts/hls4ml/runvn_{z,so2}_{search,train,eval}.sh`. The network divides the
constituents by `units` (20 GeV), sorts them by p_T and runs a stack of blocks, the first of which
reads the sorted constituents as its vector channels and their rank invariants, described below,
as its scalars. Each block mixes the vector channels with a linear layer, reads invariants off the
mixed vectors, updates the scalars with a two-layer MLP of the scalars, the invariants and the
context, and gates the vectors with a sigmoid of the new scalars. A learned invariant is the
Minkowski product of two learned combinations of the vectors, compressed by the signed log `psi`
of the LorentzNet component. The gate replaces the projection of the reference, which would divide
by the Minkowski norm of a learned direction, zero for a lightlike one, so the network is vector
neurons with a gated non-linearity. The gates start at exactly 1/2, their last layer being
initialised to zero, because a gate started near zero would stay there: the learned invariants are
quadratic in the vectors, so their gradient vanishes with them. Every block but the first, whose
input is the constituents and their rank invariants rather than a state, adds its input to both
states. The context, psi(n) of the number n of real constituents, reaches every block and the
head, since a missing constituent and a vanishingly soft one give a bias-free layer the same
output; the head is an MLP of the final scalars, the invariants of the final vectors and the
context. No layer normalises the states: on a boost-exact row no invariant divisor can bound the
components of a vector. The optimiser is AdamW with OneCycleLR, the block of L-GATr-slim and
LLoCa, at a default learning rate of 2e-3 in place of their 3e-4. The vector-neuron models the
repository carried before the refactor (`network/vector_neurons_lorentz.py` at `14c1c788`,
`network_new/vector_neurons_so2.py` at `87ae3ab6`) also gave their first layer one weight per
constituent slot, but had no sort, no padding mask and no partition of the four-vector; only their
idea is reused.

The partition of the four-momentum sets the exact group. The components (E, px, py, pz) are
divided into splits, and the metric is the Minkowski one, (+, -, -, -), which every split inherits.
Every linear layer shares its weights inside a split and keeps them free across splits, the gate
scales each split of each vector channel as a whole, and every product, learned or among the rank
invariants, is taken on a split or a union of splits, so a transformation that acts within every
split and preserves the metric on it leaves the logits alone. The sign the metric puts on a whole
split is absorbed by that split's weights and leaves the group unchanged. The switch `symmetry`
picks the partition, one per row; it is not called `group` because `JetTaggingModule.__init__`
already owns that name for the group of the symmetry-consistency loss. The `vn_z` row splits the
components into (px, py) and (E, p_z) and is exact under O(2)×O(1,1): rotations about the beam,
boosts along it, and the reflections φ -> -φ and p_z -> -p_z. The probe groups `so2` and `lorentz`
are therefore exact for it, and `so3` and `lorentz_iso` are measured. The `vn_so2` row has three
splits, E, p_z and (px, py), and is exact under O(2) about the beam and p_z -> -p_z, so `so2` alone
is exact and the three other groups are measured. With one weight for all four components the
linear layer would be the Lorentz-equivariant one of L-GATr-slim; the separate weights are what cut
the group down. The `vn_so2` networks contain the `vn_z` ones: a `vn_z` network is the `vn_so2`
network whose E and p_z weights are tied and whose first scalar layer gives no weight to the rank
invariants of E and p_z taken apart, which `vn_z` lacks; `tests/test_algorithms.py` checks this on
whole networks. The two rows are one module and one network: `vn_so2.yaml` is a standalone copy of
`vn_z.yaml` that differs in `name` and `symmetry` alone, kept in step by `tests/test_configs.py`,
as for PELICAN, L-GATr-slim and LLoCa.

The network reads the constituents by rank. It sorts those of each jet by decreasing p_T and feeds
them to the first block as its input vector channels, one weight per rank. From the first block on,
the state belongs to the jet, and every layer runs once per jet, not once per constituent, which
is what makes the network cheap. Rotations about the beam and boosts along it leave every p_T
unchanged, so the rank is an invariant label under both groups and under no larger one: that is
why a network that reads by rank stays exact here, and why no Lorentz or SO(3) row exists. The
sort also makes the logits invariant under any permutation of the constituents and sends the
padded rows, whose p_T is zero, to the end, where they contribute nothing to the bias-free first
layer. The record is already sorted by p_T (checked on the first 50 000 training jets at 16, 50
and 150 constituents on 2026-09-29), so on it the sort is the identity, and in a trigger the sort
is already done. The network sorts what it is given, so a jet transformed by the probe is ranked
afresh, as a trigger would rank it; the `mlp` row, the other network that reads by rank, is probed
in the record's order.

The rank invariants, the first block's input scalars, are fixed functions of the sorted
constituents. With J the sum of the constituents (the jet), they are, for every rank i and every
block b of components, the Minkowski square of p_i on b and the Cauchy-Schwarz deficit of p_i
against J on b; for every rank, the product ⟨p_i, J⟩ over all four components; and for the jet,
its square on every block and its mass squared, all compressed by `psi`. For N constituents
and n_b blocks their width is N(2 n_b + 1) + n_b + 1, 83 for `vn_z` and 149 for `vn_so2` at 16
constituents. The blocks of `vn_z` are its splits, (px, py) and (E, p_z), on which the entries of
rank i are -p_T^2, m_T^2, p_T p_TJ (1 - cos Δφ), m_T m_TJ (cosh Δy - 1) and ⟨p_i, J⟩. Those of
`vn_so2`, listed in `RANK_BLOCKS`, are its splits E, p_z and (px, py) and the union (E, p_z) that
`vn_z` uses, so that the inputs of `vn_so2` contain those of `vn_z`. The invariants carry no
parameters, every entry of a padded rank is exactly zero, and every entry is a product within a
block that the row's group preserves, so both rows stay exact under the same groups. They are
there because every constituent of a jet is nearly parallel to the jet: an inner product is then
dominated by the product of the norms, and the information sits in the deficit, of relative size
ΔR^2, about 1e-2, which the signed log of the raw product buries and which the learned products
reach only through a fine cancellation between the weights of two splits. In single runs of fifty
epochs at the default size and 16 constituents, on olgpu on 2026-09-29, `vn_z` reached a
validation accuracy of 0.66 without them and 0.71 with them. The deficit
equals |p||J| - p·J on a definite block and ⟨p, J⟩ - |p||J| on (E, p_z), but is computed as half
the squared norm of the difference of the two unit vectors, times the two norms, which subtracts
no two nearly equal numbers. The products ⟨p_i, J⟩ and the jet mass squared, by contrast,
are raw sums that cancel to about 1e-2 of their terms, so in float32 they carry a rounding that
differs between a jet and its transformed copy, as the Minkowski products of LorentzNet, PELICAN
and L-GATr-slim do.

The experiments run in fp32, since the metric products cancel to a small fraction of their terms,
which fp16 cannot resolve. They alone among the embedded experiments also run without TF32, with
`extras.matmul_precision` unset, which `tests/test_configs.py` checks: TF32 rounds the inputs of
every float32 matmul to about 5e-4, and the linear layers here act on the vector components
themselves, so a transformed jet would be rounded differently and the embedded group would hold to
that precision only. Even so, px and py are rounded afresh after a rotation about the beam, so two
constituents whose p_T agree to rounding can swap ranks and leave a small flip rate under `so2`; a
boost along the beam leaves px and py untouched. At the defaults (3 blocks, 16 vector and 64
scalar channels, 16 invariants, head [64, 32]) `vn_z` has 48 805, 60 773 and 95 973 parameters and
costs 0.108, 0.134 and 0.211 MFLOP per jet at 16, 50 and 150 constituents, and `vn_so2` has
58 965, 80 181 and 142 581 parameters and costs 0.122, 0.166 and 0.294 MFLOP. At the corner of the
search space (6 blocks, 64 vector and 128 scalar channels, 32 invariants, head [128, 64]) `vn_z`
has 0.45 to 0.55 M parameters and costs 1.09 to 1.33 MFLOP, and `vn_so2` 0.55 to 0.73 M and 1.20
to 1.58 MFLOP. These figures were measured on olgpu on 2026-09-29. The cost grows with the number
of constituents through the first block alone, mostly through the first layer of its scalar MLP,
which reads the rank invariants; the invariants themselves are elementwise and so are left out of
the FLOP count, which covers matmul-type operations alone. At 16 constituents PELICAN costs
4.5 MFLOP (33 k parameters), LLoCa 34 (1.06 M) and L-GATr-slim 97 to 114 (2.0 M).

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
