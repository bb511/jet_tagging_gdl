import os

# Before anything imports keras: the olgpu ~/.keras/keras.json names tensorflow.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("KERAS_BACKEND", "jax")

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("hgq")

import keras
from hgq.layers import QBatchNormalization, Quantizer
from hgq.layers.core import QLayerBase
from hgq.quantizer.internal import (
    FixedPointQuantizerBase,
    FixedPointQuantizerKBI,
    FixedPointQuantizerKIF,
)

from src.algorithms.components.vector_neurons import (
    ChannelGate,
    VectorNeuronsClassifier,
    _relative_angles,
    _sort_by_pt,
    _unit_lift,
)
from src.algorithms.components.vector_neurons_hgq import (
    KerasNetwork,
    Quantisation,
    _norm,
    build_port,
    calibrate,
    configure_jax,
    export_port,
    quantiser_variables,
    seed_datalane_bits,
    seed_weight_bits,
    set_beta,
    total_ebops,
)
from src.algorithms.components.vector_neurons_port import _copy_norm, copy_float_weights

N_SLOTS = 8
HEAD = [16]
ARCHITECTURE = {
    "num_blocks": 2,
    "vector_channels": 8,
    "scalar_channels": 16,
    "invariants": 8,
    "tensor_invariants": 2,
    "stream_channels": 8,
}
# (gate, mlp_layers, stream_channels, tensor_invariants): today's network and the
# latency ablation's variants, each switching one more piece off.
VARIANTS = [
    (True, 2, 8, 2),
    (False, 2, 8, 2),
    (False, 1, 8, 2),
    (False, 1, 0, 2),
    (False, 1, 0, 0),
]
# GeV per unit of p_T: the float row's and the port's.
FLOAT_UNITS, PORT_UNITS = 20.0, 16.0
LAMBDA = FLOAT_UNITS / PORT_UNITS
# Today's quantised port and the one of both latency switches.
QUANTISED = {
    "default": Quantisation(),
    "folded_trn": Quantisation(fold_norm=True, round_mode="TRN"),
}


def _jets(n_jets: int = 64, seed: int = 1) -> torch.Tensor:
    # Rows [p_T GeV, eta, phi] in float64 of narrow jets of 1 to N_SLOTS constituents,
    # the padded rows all zero and shuffled among the real ones. Every real p_T is at
    # least 1 GeV, so the keep flag min(4 p_T, 1) is exact without the p_T word too.
    generator = torch.Generator().manual_seed(seed)
    gauss = torch.randn(3, n_jets, N_SLOTS, generator=generator, dtype=torch.float64)
    phi = (0.3 + 0.25 * gauss[2] + math.pi) % (2 * math.pi) - math.pi
    x = torch.stack([1.0 + 40.0 * gauss[0].exp(), 0.4 + 0.25 * gauss[1], phi], dim=-1)
    multiplicity = torch.randint(1, N_SLOTS + 1, (n_jets, 1), generator=generator)
    x[torch.arange(N_SLOTS) >= multiplicity] = 0.0
    return x[:, torch.randperm(N_SLOTS, generator=generator)]


def _torch_net(x: torch.Tensor, **variant: object) -> VectorNeuronsClassifier:
    # float64 and eval mode. The gates get N(0, 1) weights, so that each is a function
    # of the scalars rather than the constant 1/2, the batch normalisations random
    # affine parameters, and one training pass at momentum 1 sets their running
    # statistics to those of the jets, so that every fold is exercised.
    torch.manual_seed(0)
    architecture = {**ARCHITECTURE, **variant}
    net = VectorNeuronsClassifier(N_SLOTS, HEAD, units=FLOAT_UNITS, **architecture)
    net = net.double()
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, ChannelGate):
                module.linear.weight.normal_()
                module.linear.bias.normal_()
            if isinstance(module, torch.nn.BatchNorm1d):
                module.weight.uniform_(0.5, 1.5)
                module.bias.normal_(0.0, 0.1)
                module.momentum = 1.0
        net.train()(x)
    return net.eval()


def _rows(x: torch.Tensor) -> np.ndarray:
    # What the port reads: sorted by p_T, the angles measured from the leader and zero
    # on a padded row.
    x = _sort_by_pt(x)
    eta, phi = _relative_angles(x)
    real = x[..., 0] > 0
    return torch.stack([x[..., 0], eta * real, phi * real], dim=-1).numpy()


def _port(
    quantisation: Quantisation, **variant: object
) -> tuple[keras.Model, keras.Model]:
    configure_jax("cpu", 1, "float64")
    keras.utils.set_random_seed(0)
    architecture = {**ARCHITECTURE, **variant}
    return build_port(
        N_SLOTS, HEAD, units=PORT_UNITS, quantisation=quantisation, **architecture
    )


def _copied(quantisation: Quantisation, x: torch.Tensor, **variant: object) -> tuple:
    net = _torch_net(x, **variant)
    model, words_model = _port(quantisation, **variant)
    copy_float_weights(model, net, PORT_UNITS, fold_norm=quantisation.fold_norm)
    return net, model, words_model


def _seeded(
    x: torch.Tensor, quantisation: Quantisation
) -> tuple[keras.Model, keras.Model, np.ndarray]:
    _, model, words_model = _copied(quantisation, x)
    rows = _rows(x)
    seed_weight_bits(model, 10)
    seed_datalane_bits(model, rows, 12)
    return model, words_model, rows


def _predict(model: keras.Model, rows: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_on_batch(rows))


@pytest.mark.parametrize("fold_norm", [False, True])
@pytest.mark.parametrize(
    "gate, mlp_layers, stream_channels, tensor_invariants", VARIANTS
)
def test_the_float_port_reproduces_the_torch_network(
    gate: bool,
    mlp_layers: int,
    stream_channels: int,
    tensor_invariants: int,
    fold_norm: bool,
) -> None:
    # The criterion of the copy: in float mode only the gate's clamp, at 2^-24, and the
    # float64 rounding of the folds separate the two. Folded, every batch normalisation
    # lives in the dense layer after it: the stream's, both blocks' and the head's.
    variant = dict(
        gate=gate,
        mlp_layers=mlp_layers,
        stream_channels=stream_channels,
        tensor_invariants=tensor_invariants,
    )
    x = _jets()
    q = Quantisation(float_mode=True, fold_norm=fold_norm)
    net, model, _ = _copied(q, x, **variant)
    with torch.no_grad():
        expected = net(x).numpy()

    logits = _predict(model, _rows(x))

    assert logits.dtype == np.float64
    assert np.abs(logits - expected).max() <= 1e-6 * max(1.0, np.abs(expected).max())
    norms = [layer for layer in model.layers if isinstance(layer, QBatchNormalization)]
    assert len(norms) == (0 if fold_norm else 3 + (stream_channels > 0))
    names = {layer.name for layer in model.layers}
    assert all((f"block{b}_gate" in names) == gate for b in range(2))
    assert {f"block0_mlp{j}" for j in range(3)} & names == {
        f"block0_mlp{j}" for j in range(mlp_layers)
    }
    assert ("stream_embed" in names) == (stream_channels > 0)
    tensors = {name for name in names if name.startswith("tensors_")}
    assert bool(tensors) == (tensor_invariants > 0)


def test_the_fold_leaves_an_eval_batch_norm_unchanged() -> None:
    # Channels of degree 1 and 3 in p at lambda = 20/16, and a count at N = 16.
    scales = np.array([LAMBDA, LAMBDA**3, 16.0])
    torch.manual_seed(0)
    norm = torch.nn.BatchNorm1d(3).double().eval()
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
        norm.bias.normal_()
        norm.running_mean.normal_()
        norm.running_var.uniform_(0.01, 2.0)
    configure_jax("cpu", 1, "float64")
    layer = _norm(Quantisation(float_mode=True), "norm")
    layer.build((None, 3))
    _copy_norm(layer, norm, scales)
    x = torch.randn(32, 3, dtype=torch.float64)

    with torch.no_grad():
        expected = norm(x).numpy()
    folded = np.asarray(layer(x.numpy() * scales, training=False))

    assert np.allclose(folded, expected, rtol=1e-12, atol=1e-12)


def test_the_first_scalars_follow_the_flattening_of_torch() -> None:
    # torch's r.flatten(1) interleaves p_1, f_1, p_2, f_2, ...; the port reads the N
    # p before the N f, in units of 16 GeV, so that a scalar of degree d in p is
    # lambda^d times torch's, and n where torch reads n / N.
    x = _jets()
    net, model, _ = _copied(Quantisation(float_mode=True), x)
    inputs = keras.Model(model.input, model.get_layer("block0_norm").input)
    port = _predict(inputs, _rows(x))
    xs = _sort_by_pt(x)
    p = xs[..., 0] / FLOAT_UNITS
    unit = _unit_lift(*_relative_angles(xs))
    keep = (p != 0.0).unsqueeze(-1).double()
    with torch.no_grad():
        first = net._first_scalars(p.unsqueeze(-1) * unit, unit, p, keep).numpy()
    n, k = N_SLOTS, ARCHITECTURE["tensor_invariants"]
    tail = 2 * n + 2 + ARCHITECTURE["stream_channels"]
    expected = np.concatenate(
        [
            LAMBDA * first[:, 0 : 2 * n : 2],
            LAMBDA**2 * first[:, 1 : 2 * n : 2],
            LAMBDA * first[:, [2 * n]],
            LAMBDA**2 * first[:, [2 * n + 1]],
            first[:, 2 * n + 2 : tail],
            LAMBDA**3 * first[:, tail : tail + k],
            LAMBDA**2 * first[:, tail + k : tail + 2 * k],
            LAMBDA**3 * first[:, tail + 2 * k :],
        ],
        axis=1,
    )

    assert np.allclose(port[:, : expected.shape[1]], expected, rtol=1e-9, atol=1e-9)
    assert np.array_equal(port[:, -1], (xs[..., 0] > 0).sum(dim=1).double().numpy())


def test_the_gate_is_the_hard_sigmoid_of_torch_within_its_rounding() -> None:
    # On (2/3) z the gate rounds z to 2^-f and returns q / 4 + 1 / 2, q clamped to
    # [-2, 2): within 2^-(f + 2) of relu6(z + 3) / 6, and exactly 1/2 at z = 0, where
    # the float network initialises every gate.
    configure_jax("cpu", 1, "float64")
    gate = Quantizer(Quantisation(gate_f=8).gate())
    z = np.linspace(-5.0, 5.0, 4001).reshape(1, -1)

    g = np.asarray(gate(2.0 / 3.0 * z))
    expected = F.hardsigmoid(torch.from_numpy(z)).numpy()

    assert np.abs(g - expected).max() <= 2.0**-10
    assert np.all((g >= 0.0) & (g < 1.0))
    assert z[0, 2000] == 0.0 and g[0, 2000] == 0.5


def test_floor_and_ceil_keep_a_float64_tensor_in_float64() -> None:
    # Keras 3.15 computes both in float32 on JAX, where 626 - 2^-15 and 625 + 2^-15 are
    # ties that round to the even 626 and 625; importing the component keeps float64.
    configure_jax("cpu", 1, "float64")
    ties = [626.0 - 2.0**-15, 625.0 + 2.0**-15]
    near = keras.ops.convert_to_tensor(ties, "float64")
    single = keras.ops.convert_to_tensor([1.5], "float32")

    floor, ceil = keras.ops.floor(near), keras.ops.ceil(near)

    dtypes = [keras.backend.standardize_dtype(t.dtype) for t in (floor, ceil)]
    assert dtypes == ["float64", "float64"]
    assert np.asarray(floor).tolist() == [625.0, 625.0]
    assert np.asarray(ceil).tolist() == [626.0, 626.0]
    assert keras.backend.standardize_dtype(keras.ops.floor(single).dtype) == "float32"


def test_the_pt_word_rounds_a_near_tie_as_integer_arithmetic_does() -> None:
    # 625.5 -+ 2^-15 LSB round to 625 and 626 LSB. HGQ2 rounds floor(x 2^f + 1/2), whose
    # argument 626 - 2^-15 a float32 floor would take to 626.
    _, words_model = _port(Quantisation())
    lsb = 2.0 ** -Quantisation().pt_kif[2]
    rows = np.zeros((1, N_SLOTS, 3))
    rows[0, :2, 0] = [(625.5 - 2.0**-15) * lsb, (625.5 + 2.0**-15) * lsb]

    words = _predict(words_model, rows)

    assert words[0, :2, 0].tolist() == [625 * lsb, 626 * lsb]


def test_keep_is_one_on_a_real_word_and_zero_on_padding() -> None:
    # The p_T word has an LSB of 0.25 GeV, rounded to nearest: 0.13 GeV becomes a real
    # constituent of 0.25 GeV, 0.12 GeV a padded one.
    model, words_model = _port(Quantisation())
    rows = np.zeros((2, N_SLOTS, 3))
    rows[0, :, 0] = [4000.0, 600.0, 3.3, 0.25, 0.13, 0.12, 0.0, 0.0]
    rows[0, :6, 1:] = 0.1
    rows[1, :3, 0] = [50.0, 1.0, 0.5]
    count = model.get_layer("count")

    keep = _predict(keras.Model(model.input, count.input), rows)[..., 0]
    n = _predict(keras.Model(model.input, count.output), rows)
    words = _predict(words_model, rows)

    assert np.array_equal(keep, (words[..., 0] > 0).astype(np.float64))
    assert keep[0].tolist() == [1.0] * 5 + [0.0] * 3
    assert n.ravel().tolist() == [5.0, 3.0]


def _fixed_kifs(model: keras.Model) -> dict[str, list[np.ndarray]]:
    return {
        layer.path: [np.asarray(v) for v in layer.kif]
        for layer in model._flatten_layers()
        if isinstance(layer, FixedPointQuantizerBase) and not layer.trainable
    }


def test_seeding_sets_the_trainable_bits_and_leaves_the_fixed_quantisers() -> None:
    x = _jets()
    _, model, _ = _copied(Quantisation(), x)
    before = _fixed_kifs(model)
    seed_weight_bits(model, 10)
    seed_datalane_bits(model, _rows(x), 12)
    lanes = [
        layer
        for layer in model._flatten_layers()
        if isinstance(layer, FixedPointQuantizerKIF)
        and layer.overflow_mode == "WRAP"
        and layer.trainable
    ]
    weights = [
        layer
        for layer in model._flatten_layers()
        if isinstance(layer, FixedPointQuantizerKBI) and layer.trainable
    ]

    assert lanes and weights
    for lane in lanes:
        assert np.all(np.asarray(lane.i + lane.f) == 12)
    for weight in weights:
        assert np.all(np.asarray(weight.b) == 10)
    # The two words, the count, the two keep flags, a gate per block and the logits.
    assert len(before) == 6 + ARCHITECTURE["num_blocks"]
    after = _fixed_kifs(model)
    for path, kif in before.items():
        assert all(np.array_equal(a, b) for a, b in zip(kif, after[path]))


@pytest.mark.parametrize("name", QUANTISED)
def test_the_quantised_port_trains_traces_and_survives_a_round_trip(
    tmp_path: Path, name: str
) -> None:
    x, quantisation = _jets(), QUANTISED[name]
    model, words_model, rows = _seeded(x, quantisation)
    network = KerasNetwork(model, words_model)
    optimizer = keras.optimizers.AdamW(learning_rate=1e-3, global_clipnorm=1.0)
    optimizer.exclude_from_weight_decay(var_list=quantiser_variables(model))
    network.compile_keras(optimizer)
    set_beta(model, 1e-6)
    y = np.eye(5)[np.arange(len(rows)) % 5]

    logs = [network.train_on_batch(rows, y) for _ in range(2)]
    ebops = calibrate(model, rows)
    words = network.quantised_inputs(rows)
    path = export_port(tmp_path / "main_hgq", model, words, {"beta": 1e-6})
    reloaded = keras.saving.load_model(path / "model.keras", compile=False)
    logits = network(torch.from_numpy(rows))

    assert {"loss", "ce", "acc"} <= set(logs[-1])
    assert all(math.isfinite(log["loss"]) for log in logs)
    assert logs[-1]["loss"] > logs[-1]["ce"]
    assert network.learning_rate() == pytest.approx(1e-3)
    betas = [
        float(layer._beta)
        for layer in model._flatten_layers()
        if isinstance(layer, QLayerBase)
    ]
    assert betas and all(beta == pytest.approx(1e-6) for beta in betas)
    assert total_ebops(model) == ebops
    assert isinstance(ebops, int) and ebops > 0
    assert np.array_equal(_predict(model, words), _predict(model, rows))
    assert np.array_equal(_predict(reloaded, words), _predict(model, words))
    assert np.load(path / "inputs.npz")["inputs_clean"].dtype == np.float32
    assert logits.dtype == torch.float64
    assert np.array_equal(logits.numpy(), _predict(model, rows))


@pytest.mark.parametrize("name", QUANTISED)
def test_round_mode_reaches_the_trainable_datalanes_alone(name: str) -> None:
    quantisation = QUANTISED[name]
    model, _ = _port(quantisation)
    kifs = [
        layer
        for layer in model._flatten_layers()
        if isinstance(layer, FixedPointQuantizerKIF)
    ]
    trainable = {layer.round_mode for layer in kifs if layer.trainable}
    fixed = {layer.round_mode for layer in kifs if not layer.trainable}

    assert trainable == {quantisation.round_mode}
    assert fixed == {"RND"}


def test_truncation_floors_a_near_tie_in_float64() -> None:
    # TRN floors x 2^f: at f = 0, -1/4 goes to -1 where RND gives 0, and 626 - 2^-15
    # to 625 where the float32 floor of Keras 3.15 on JAX would give 626.
    configure_jax("cpu", 1, "float64")
    lane = Quantizer(Quantisation(round_mode="TRN").datalane())
    x = np.array([[626.0 - 2.0**-15, -0.25]])
    lane.build(x.shape)
    inner = lane.quantizer
    inner._i.assign(np.full(inner._i.shape, 10.0))
    inner._f.assign(np.zeros(inner._f.shape))

    y = np.asarray(lane(x, training=False))

    assert y.tolist() == [[625.0, -1.0]]


@pytest.mark.parametrize("name", QUANTISED)
def test_alkaid_reproduces_the_port_bit_for_bit(name: str) -> None:
    # alkaid replays the traced port on the words it reads; under float64 every value
    # of Keras is the fixed-point value of the hardware, so the two must agree exactly.
    pytest.importorskip("alkaid")
    from alkaid.converter import trace_model
    from alkaid.trace import HWConfig, trace

    model, words_model, rows = _seeded(_jets(), QUANTISED[name])
    calibrate(model, rows)
    words = _predict(words_model, rows)

    inputs, outputs = trace_model(model, HWConfig(1, 1, -1))
    predicted = trace(inputs, outputs).predict(words.reshape(len(words), -1))

    assert np.array_equal(predicted, _predict(model, words))
