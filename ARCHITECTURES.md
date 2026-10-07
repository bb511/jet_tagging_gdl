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
neurons trained quantisation-aware").
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
  `_scl_optuna` config and three blocks per catalogue rather than 21. A third case, a row that
  runs the baseline regime alone but reads `data.features` (JEDI-linear until 2026-10-02), goes in
  `BASELINE_ONLY`, not `EMBEDDED`, in both `scripts/optuna/make_train_catalogues.py` and
  `tests/conftest.py`, has the `_optuna` config alone, and needs a search catalogue under every
  feature-set directory, `scripts/hls4ml/pt_eta_phi/` and `scripts/hls4ml/pt_etarel_phirel/` as
  well as `scripts/hls4ml/`.
  A row of the grid that reads `data.features`, as the Linformer does, needs both search configs
  and a 21-block search catalogue under every feature-set directory. A variant of a grid row pinned
  to columns of its own, as the moment rows are, gets a subclass of its own that sets
  `required_features`, so that the grid row keeps reading `data.features`, and an algorithm config
  that sets `nbr_feats` to a number, since `tests/test_algorithms.py` builds every row with the
  four-momentum as `data.features`.
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
the experiments run in fp32, as the references train without mixed precision. Both cost counters
run the network on the CPU at construction, where the LLoCa-Transformer's
`F.scaled_dot_product_attention` dispatches `aten._scaled_dot_product_flash_attention_for_cpu`, a
kernel torch's FLOP counter does not know. `utils/flops.py` prices it through a custom mapping,
which restores the attention's 524 288 multiply-adds per jet at N = 16 (about 3 % of the row's
FLOPs), and `utils/bitops.py` prices it as the math path would compute it.

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

Added on 2026-10-03 in one row, `vn`, which replaced the two four-momentum rows `vn_z` and `vn_so2`
of 2026-09-29 (see "Vector neurons on the four-momentum" below). Unlike those of L-GATr-slim and
LLoCa, its network is written here, so it adds no dependency. Vector neurons (Deng et al.,
arXiv:2104.12229) carry lists of vectors as hidden features and mix them with a bias-free linear
layer that acts on the channel index alone, so that the layer commutes with the group; here the
vectors are lifts of the constituents' (p_T, η, φ), and the group is the Euclidean group of the
(η, φ) plane. Our code is the network `VectorNeuronsClassifier` in
`src/algorithms/components/vector_neurons.py` and the `JetTaggingModule` subclass `VectorNeurons` in
`src/algorithms/vector_neurons.py`, which pins `required_features` to `[pt, eta, phi]` and returns
the batch to GeV and radians in `_prepare_inputs`. Around them sit `configs/algorithm/vn.yaml`, the
three experiment configs `configs/experiment/hls4ml/vn_{16,50,150}.yaml`, the search config
`configs/hparams_search/hls4ml/vn_optuna.yaml` and the catalogues
`scripts/hls4ml/runvn_{search,train,eval}.sh`. It is the one embedded row that does not read the
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

`tests/test_algorithms.py` checks, on the network in float64 and eval mode, that the logits are
invariant to 1e-9 under shifts of η and φ across the wrap, under rotations of the plane about the
leader and about the fourth-hardest constituent and under both reflections, and to 1e-12 under a
permutation. Through `forward`, on massless record jets, the probe's `so2` and `lorentz` transforms
must move the logits by less than 1e-6 (9e-8 and 2e-8, the float32 rounding in `forward`), and
`so3` and `lorentz_iso` by more than a thousand times the `so2` shift. It also checks the form and
the tensor invariants against their closed forms, that a padded slot changes nothing (an eight-slot
network against a nine-slot copy), that the batch is de-normalised, that removing a constituent
moves the logits through the multiplicity, that every gate starts at 1/2 and that the logits are
finite on the FLOP counter's input. `tests/test_configs.py` checks that the experiments set
`data.features` to `required_features` and leave `extras.matmul_precision` unset.

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
port `src/algorithms/components/vector_neurons_hgq.py`, which sets Keras's backend to JAX before
it imports Keras and which the subclass therefore imports first, the weight copy
`src/algorithms/components/vector_neurons_port.py` and the `VectorNeurons` subclass
`VectorNeuronsHGQ` in `src/algorithms/vector_neurons_hgq.py`. Around them sit
`configs/algorithm/vn_hgq.yaml`, a standalone copy of `vn.yaml` that `tests/test_configs.py` keeps
equal to it up to `units`, the optimiser, the scheduler and the keys of the port and its training,
`configs/callbacks/quantised.yaml`, the experiment configs
`configs/experiment/hls4ml/vn_hgq_{16,50,150}.yaml`, the hand-written catalogues
`scripts/hls4ml/runvn_hgq_{train,eval}.sh`, `scripts/synthesis/synthesise_vn.py` with its
`requirements.txt`, and `scripts/analysis/compare_outputs.py`. The dependencies form the optional
Poetry group `hgq`, installed with `poetry install --no-root --with hgq` and absent from the NGT
image: Keras 3.15, hgq2 0.2.0, alkaid 0.8.1, JAX 0.10.2 with its CUDA 12 plugin, and nvcc 12.6.85,
pinned so that ptxas matches torch's CUDA runtime. In `scripts/optuna/make_train_catalogues.py` and
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
saturates, and `seed_datalane_bits` gives every datalane i + f = `data_bits`, with i traced by
HGQ2's `trace_minmax` on the first `calibration_jets` (100 000) training jets. It traces twice: the
first trace reads each lane through upstream lanes still at HGQ2's default of six fraction bits,
and a lane whose range the second trace widens would otherwise wrap. The defaults, 14 data and 8
weight bits, come from a table of seeded widths on the two-block finals at N = 16 (dry run,
2026-10-07): 14 and 8 bits cost 0.10 points of validation accuracy at 2.85 M EBOPs, 12 and 10
bits, the plan's first choice, 0.87 points at 2.99 M, and 8 data bits collapsed the network. The
largest traced integer width was 10 bits. The float logits reached 15.1 in magnitude, so the output
word, kif (1, 6, 8) with saturation, keeps two integer bits spare.

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

`scripts/synthesis/synthesise_vn.py` takes the `outputs/<label>_hgq/` directory and the probe's
`.npz` of one evaluation. It imports nothing from `src` and runs wherever
`scripts/synthesis/requirements.txt` is installed (alkaid 0.8.1, hgq2 0.2.0, Keras 3.15.1 and JAX
0.10.2 on the CPU, on Python 3.11 or later). It loads the model with Keras on JAX's CPU backend at
the export's float type, refusing any other backend and applying the same floor patch, and checks
Keras's prediction of the stored words against `logits_clean`. alkaid then traces the model into
combinational logic, as `alkaid convert` does, and emulates it bit for bit; both predictions are
written to `keras_outputs.npz` and `alkaid_outputs.npz` in the probe's layout, and the emulation
must match `logits_clean` exactly (`--max-mismatch` 0). alkaid's `RTLModel` writes the Verilog
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

`tests/test_vector_neurons_hgq.py` checks the port: that in float mode it reproduces the torch
network in float64 to 1e-6 relative (eight slots, some padded, two blocks, random gates and running
statistics), that the fold leaves an eval batch normalisation unchanged, that the first block's
scalars follow torch's flattening and units, that the gate is torch's hard sigmoid within 2⁻¹⁰ and
exactly 1/2 at zero, that floor and ceil keep float64 and the p_T word rounds a near tie as integer
arithmetic does, that keep is 1 on a real word and 0 on padding (0.12 GeV is padding), that seeding
sets the trainable widths and leaves the fixed quantisers alone, that the quantised port trains two
steps, traces and survives a `.keras` round trip, and, with alkaid installed, that alkaid
reproduces it bit for bit. `tests/test_vn_hgq_module.py` runs the module on the CPU on a stub
datamodule: training steps Keras per batch, advances `global_step` and checkpoints the last epoch
with the Keras weights; a checkpoint restores them in the port's float type; β reaches every
quantised layer, constant or scheduled; the evaluation traces the training split and exports a
model whose prediction of the stored words equals `logits_clean`; JAX off the CPU is refused at
test time; a GPU run keeps the CPU backend; the float twin prices the FLOPs while the EBOPs wait for
the trace; and `init_from` copies a float run of the same architecture alone.
`tests/test_synthesise_vn.py` loads the script by path and checks the CSV row with and without
Vivado, its header, the macro accuracy against torchmetrics, a tiny export end to end through
`--no-cosim --no-vivado` and `--report-only` (bit-exact, both prediction files written), the
co-simulation where Verilator exists, the refusal of JAX off the CPU and the floor patch.
`tests/test_compare_outputs.py` checks that an identical candidate passes with no difference, that
shuffling the logits of one class fails its working points, that noise within the tolerance passes
and beyond it fails, the macro accuracy against torchmetrics, the ROC gap against the step
functions, the CSV rows, the refusals of a bad floor, a foreign header and other labels, and the
figure. `tests/test_configs.py` checks the experiments (fp32, no clipping, the unranked checkpoint,
no early stopping, `num_workers` 0, the features, JAX following the trainer) and `vn_hgq.yaml`
against `vn.yaml`, `tests/test_make_train_catalogues.py` that `QUANTISED` agrees in both modules
and that no layout generates its catalogues, and `tests/test_datamodule.py` that
`load_split("train")` at `n_folds` 1 is the training split at full width. Every module that imports
Keras sets `KERAS_BACKEND=jax` and `JAX_PLATFORMS=cpu` first and skips without `hgq` (or
`alkaid`), so `make test` passes without the group.

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

## JEDI-linear

Added on 2026-09-30 as a comparator from the literature, at first in the baseline regime alone, and
made a row of the grid on 2026-10-02, when Patrick decided to run it under augmentation and the SCL
as well. JEDI-linear (Que, Sun et al., arXiv:2508.15468, Field Programmable Technology 2025) is a
jet tagger built for FPGAs; it is judged on accuracy and cost, against the vector neurons in
particular. Its published accuracy, 73.6 % at 16 constituents on three features, is measured on the
record's `etarel` and `phirel`, which are taken from the axis of the untruncated jet. This
repository rebases them on the kept constituents, which is worth about 2 points, so a comparison
needs the network trained on this preparation. Our code is a second class in each DeepSets file: the
network `JEDILinearNet` beside `DeepSetsNet` in `src/algorithms/components/deepsets.py` and the
`JetTaggingModule` subclass `JEDILinear` beside `DeepSets` in `src/algorithms/deepsets.py`, the two
older classes untouched. Around them sit `configs/algorithm/jedi_linear.yaml`, the three experiment
configs `configs/experiment/hls4ml/jedi_linear_{16,50,150}.yaml`, the search config
`configs/hparams_search/hls4ml/jedi_linear_optuna.yaml` with its companion
`jedi_linear_scl_optuna.yaml`, the three search catalogues
`scripts/hls4ml{,/pt_eta_phi,/pt_etarel_phirel}/runjedi_linear_search.sh` and the generated
`scripts/hls4ml/runjedi_linear_{train,eval}.sh`.

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
`scripts/optuna/make_train_catalogues.py` and `tests/conftest.py` but in neither `EMBEDDED` nor
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

`tests/test_algorithms.py` checks, with the maximum, that the logits are invariant under padding,
to 1e-10 in float64; it calls the network directly, because `forward` casts every batch to float32.
Padding is tested by a hole among a jet's slots and by a zero slot appended to a full jet. The
appended slot is the jet's only padded one, so a maximum that let its embedding in unmasked would
fail wherever that embedding exceeds the real values.
Last, it checks the parameter count. `tests/test_configs.py` keeps `jedi_linear.yaml` a copy of
`deepsets.yaml`, the shared search choices, the SCL weight included, equal to those of DeepSets and
both spaces free of an activation, and `tests/test_make_train_catalogues.py` checks that the
generator's registries equal those of the tests and that a feature-set layout writes catalogues for
`jedi_linear` beside the other grid rows and none for the embedded rows.

## Linformer

Added on 2026-10-01 as the fourth row of the acquired-symmetry class, beside DeepSets, MLP and
JetFormer, and as a further candidate for an FPGA beside `jedi_linear` and the vector neurons. The
row, `linformer`, is the single-head Linformer transformer of Laatu et al. (arXiv:2510.24784),
trained in float on this repository's preparation. Unlike `jedi_linear`, it runs all seven regimes
on the three feature sets, 21 studies per set and 63 in all, none of them launched yet
(2026-10-01); Patrick took the decisions recorded below that day. The network, `LinformerNet`, and
its attention, `LinformerAttention`, are in `src/algorithms/components/linformer.py`, and the
subclass `Linformer`, which supplies `build_net` alone, is in `src/algorithms/linformer.py`. The
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
average pool restricted to the real constituents; the sum and the maximum come from `_masked_pool`
of the DeepSets component. The sum and `sum_over_n` express the same functions, since the head's
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
`data.features`, `linformer` is in `MODELS` of `scripts/optuna/make_train_catalogues.py` and
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

`tests/test_algorithms.py` calls the network directly, in float64 and in eval mode, since `forward`
casts every batch to float32 and BatchNorm in training mode would couple the jets of a batch. With
`sum_over_n` and BatchNorm on, overwriting
the rows of E_k and E_v that belong to padded slots must leave the logits unchanged, while a change
to a real slot's row must move them, so that the check cannot pass vacuously. The pool must read
the real constituents alone, with `sum_over_n` equal to the sum divided by the slot count, and the
parameter count must match the formula above.
In `tests/test_make_train_catalogues.py`, a feature-set layout must generate train and eval
catalogues for the row.

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
`JetMoments` beside the two networks in `src/algorithms/components/deepsets.py`, which reuses
`_form_matrix`, `_sort_by_pt` and `_unit_lift` of the vector neurons, and the subclasses
`DeepSetsMoments` and `JEDILinearMoments` of `_MomentsRow` in `src/algorithms/deepsets.py`. Around
them sit
`configs/algorithm/{deepsets_moments,jedi_linear_moments,moments_only}.yaml`, the nine experiment
configs `configs/experiment/hls4ml/<row>_{16,50,150}.yaml`, the three search configs
`configs/hparams_search/hls4ml/<row>_optuna.yaml`, the hand-written search catalogues
`scripts/hls4ml/run<row>_search.sh` and the generated `scripts/hls4ml/run<row>_{train,eval}.sh`.

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
`tests/test_algorithms.py` builds every row with the four-momentum as `data.features`. The default
set is `uniform`. The experiments train for 200 epochs on batches of 2048 with five folds at the
algorithm defaults and set `data.features: [pt, etarel, phirel]`. Each search config is its
parent's space plus
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

`tests/test_moments.py` checks the moments against their explicit sums over pairs and triples, the
exact zeros of the rank cuts, their invariance under shifts and rotations of the plane and under
permutations, the rank cuts against moments of the leading constituents alone, their dtype and
accuracy under bf16 autocast on the CPU and fp16 autocast on a GPU (both fail without the guard),
the widths of the heads, that a DeepSets without phi reads the moments and the multiplicity alone
and refuses inconsistent options, and that the moment rows refuse `none`. Against literal lists of
the committed state-dict keys and stacks rebuilt from the same seed in the committed order, it
checks that the two networks without moments draw the same weights and compute the same logits,
bit for bit, under every pooling. `tests/test_algorithms.py` builds the three rows with the other
algorithms, checks that each fills its scales and refuses a reordered or a longer column list,
that the moments widen the head and add their batch normalisation alone, that padded slots change
nothing, and holds the rows to their exact groups at 1e-6 and to a thousand times the
`so2` shift under the others, calling the network directly in float64 with its batch normalisation
calibrated on the test jets, since at its initial statistics the raw moments, about 1e10 there,
would drown every other input of the head. `tests/test_configs.py` keeps the configs copies of
their parents, the search spaces the parents' plus the moment set, and the experiments pinned to
their columns with mixed precision and no `matmul_precision`.

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
catalogues `scripts/hls4ml/runjet_level_{all53,noeta52}_search.sh` and the generated
`scripts/hls4ml/runjet_level_{all53,noeta52}_{train,eval}.sh`.

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
`data.exclude`), `feature_names` (all 53, in the shards' order), `feature_indices()`,
`norm_scales` (None) and `keep_full_features_at_test` (false); `JetTaggingModule.setup` then reads
the batches unreduced. The constructor reads nothing, as `tests/test_configs.py` needs when it
instantiates every experiment's data node, and refuses every regime but augmentation/none; the
first access to the shards refuses an `exclude` that names no column. `configs/data/jet_level.yaml`
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
`scripts/optuna/make_train_catalogues.py` and `tests/conftest.py` the rows form `JET_LEVEL`, inside
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

`tests/test_jet_level.py` writes synthetic shards in the record's layout (a constituent list
column, 53 `j_*` columns with a constant `j_ptfrac`, and `label`; two shards per split, each split
at its own offset) and checks the column order and `exclude`, that every split and fold is the
standardisation of its jets by the training split's constants, bit for bit, with the constant
column centred to zero, that a fold holds the jets `fold_indices` selects from the pooled jets,
the batch shapes and the one-hot labels, that the test split keeps its trailing jets, and the
refusals. It trains `jet_level_noeta52` on the CPU, its main run and two folds, evaluates the three
checkpoints with `eval.py` and checks the one row under `jet_level/`: the networks' header,
`nconstituents` 150, NaN in every invariance cell, positive costs, and an `.npz` per checkpoint
with the two keys. Under `slow` and with the record, it checks that the datamodule's labels of
every split equal those of the loader's cache at N = 16 and that the shards' constituent lists,
capped at 16, hold the cache's real-constituent counts, jet by jet. `tests/test_invariance.py`
checks the probe-less mode against `clean_metrics`, `tests/test_configs.py` the copies, the search
spaces, the shared revision and fold seed and the experiments, and
`tests/test_make_train_catalogues.py` the catalogues.

Two rows of gradient-boosted trees joined on 2026-10-07, `jet_level_bdt_all53` and
`jet_level_bdt_noeta52`, on the same observables through the same datamodule, so that the ceiling
rests on a second model family and not on the MLP alone. In the scratch run of 2026-10-05 the best
of six tree points reached a validation and test accuracy of 0.8302 and 0.8298 on all 53
observables, against the MLP's 0.8319 and 0.8316: close enough that either family could set the
ceiling once both are searched alike. The rows run scikit-learn's `HistGradientBoostingClassifier`
through the networks' pipeline: `src/train.py` with the default callbacks, the Optuna sweeper,
`submit_train.sh --folds 5`, `src/eval.py`, the generated catalogues and the results tables. The
module is `src/algorithms/bdt.py::BoostedTreesClassifier` and the wrapper
`src/algorithms/components/bdt.py::BoostedTrees`; around them sit
`configs/algorithm/jet_level_bdt_{all53,noeta52}.yaml`, the experiment configs
`configs/experiment/hls4ml/jet_level_bdt_{all53,noeta52}_150.yaml`, the search configs
`configs/hparams_search/hls4ml/jet_level_bdt_{all53,noeta52}_optuna.yaml`, the hand-written
`scripts/hls4ml/runjet_level_bdt_{all53,noeta52}_search.sh` and the generated
`scripts/hls4ml/runjet_level_bdt_{all53,noeta52}_{train,eval}.sh`.

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

`tests/test_bdt.py` checks the wrapper (zeros before the first fit, ten iterations per fit,
finite log-probabilities that sum to one up to the floor, three fits of the same jets equal to one,
the state-dict round trip, the depth mapping), that epoch k is validated on (k + 1) × 10 iterations
with one mock step per batch, and an end-to-end run of `jet_level_bdt_noeta52` on the synthetic
shards with the default callbacks and an MLflow logger: a main run and fold 1 of 3, three epochs
each, then `eval.py` on the directory. That run ends at 30 iterations, the best checkpoint holds
(epoch + 1) × 10, and the one row under `jet_level/` has the networks' header, `nconstituents` 150,
NaN costs and invariance cells, with a two-key `.npz` per checkpoint. The test also checks the
copies, the shared search space and the experiments' CPU trainer, fp32, absent clipping, disabled
probe and subdirectory. `tests/test_configs.py` exempts the two rows from the cost inequality and
keeps its MLP-space check to the MLP rows, and `tests/test_make_train_catalogues.py` checks
`CPU_ROWS` against `tests/conftest.py` and that each catalogue of the two rows holds one block,
`trainer=cpu` without a device, in the default layout alone.

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
`scripts/optuna/make_train_catalogues.py` and `tests/conftest.py`, and their display names to
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
