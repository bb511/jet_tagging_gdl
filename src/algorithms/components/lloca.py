"""LLoCa (arXiv:2505.20280), Lorentz local canonicalisation.

The frames-net, the local frames and the tensorial attention of the LLoCa-Transformer
are imported through the ``lloca`` package. This module only wraps that package into
a jet tagger.

The metric is (+, -, -, -), the energy first, and the momenta are in GeV.
Batches must be de-normalised before they reach this module.

Symmetry. Each constituent is read in a frame of its own. The frame is  predicted
equivariantly from the whole jet. The logits are invariant under whichever group the
frames co-transform with. That group is set by the ``frames`` attribute: the full
Lorentz group for ``lorentz`` (a learned boost and rotation), SO(2) about the beam times
boosts along it for ``z``, SO(2) about the beam for ``so2`` and SO(3) for ``so3``.
The readout is the mean over the real constituents.
"""

from functools import partial

import torch
from lloca.backbone.transformer import Transformer
from lloca.equivectors.mlp import MLPVectors
from lloca.framesnet.equi_frames import (
    LearnedPDFrames,
    LearnedSO2Frames,
    LearnedSO3Frames,
    LearnedZFrames,
)
from lloca.framesnet.frames import Frames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from torch import nn

FRAMES = {
    "lorentz": LearnedPDFrames,
    "z": LearnedZFrames,
    "so2": LearnedSO2Frames,
    "so3": LearnedSO3Frames,
}
# The tagging-guide settings: Gram-Schmidt in float64, with eps_norm raised to 1e-10.
ORTHO_KWARGS = {
    key: {
        "use_float64": True,
        "method": "gramschmidt",
        "eps_norm": 1e-10 if key in ("so2", "so3") else 1e-15,
        "eps_reg": 1e-16,
        "eps_reg_lightlike": 1e-16,
    }
    for key in FRAMES
}


def _packed_ptr(real: torch.Tensor) -> torch.Tensor:
    """Offsets ``(jets + 1,)`` of each jet's constituents in the packed ``p[real]``."""
    ptr = real.new_zeros(real.shape[0] + 1, dtype=torch.long)
    ptr[1:] = real.sum(dim=1).cumsum(dim=0)
    return ptr


class LLoCaClassifier(nn.Module):
    """LLoCa-Transformer over a fixed-width batch of jet four-momenta.

    The logits are invariant under any permutation of the constituents and under the
    group of the frames class that ``frames`` specifies.

    :param frames: a key of ``FRAMES``: ``lorentz``, ``z``, ``so2`` or ``so3``.
    :param attn_scalars: scalar channels of each attention head.
    :param attn_vectors: four-vector channels of each attention head.
    :param framesnet_hidden: width of the two hidden layers of the frames-net MLP.
    :param mass_reg: mass floor in ``units``, applied inside the frames-net: a
        lighter constituent is given this mass there, which keeps the pair sums the
        frames-net normalises off the light cone.
    :param units: GeV per unit of everything the package sees, 20 as in the L-GATr-slim
        row; the LLoCa references feed GeV.
    :param n_class: classes, read off the mean of the backbone outputs.
    :raises ValueError: when ``frames`` is not a key of ``FRAMES``.
    """

    def __init__(
        self,
        frames: str = "lorentz",
        num_blocks: int = 8,
        num_heads: int = 8,
        attn_scalars: int = 8,
        attn_vectors: int = 2,
        mlp_factor: int = 2,
        dropout: float = 0.0,
        framesnet_hidden: int = 32,
        mass_reg: float = 2.5e-4,
        units: float = 20.0,
        n_class: int = 5,
    ) -> None:
        super().__init__()
        if frames not in FRAMES:
            raise ValueError(f"frames must be one of {list(FRAMES)}, got {frames!r}.")
        self.units = units
        equivectors = partial(
            MLPVectors,
            num_scalars=0,
            hidden_channels=framesnet_hidden,
            num_layers_mlp=2,
        )
        self.framesnet = FRAMES[frames](
            equivectors=equivectors,
            mass_reg=mass_reg,
            ortho_kwargs=ORTHO_KWARGS[frames],
        )
        self.canonicalise = TensorRepsTransform(TensorReps("1x1n"))
        self.net = Transformer(
            in_channels=4,
            attn_reps=f"{attn_scalars}x0n+{attn_vectors}x1n",
            out_channels=n_class,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_factor=mlp_factor,
            dropout_prob=dropout or None,
        )

    def forward(self, pmu: torch.Tensor) -> torch.Tensor:
        """Return the class logits of a batch of jets.

        :param pmu: (jets, constituents, 4) four-momenta [E, px, py, pz] in GeV, with
            padded constituents as all-zero rows.
        :return: (jets, n_class) logits.
        """
        real = pmu[..., 0] != 0.0
        p = pmu / self.units
        frames = self._frames(p, real)
        local = self.canonicalise(p, frames)
        p_ref = (p * real[..., None]).sum(dim=1)
        out = self.net(local, frames, p_ref=p_ref, attn_mask=real[:, None, None, :])
        # The padded query rows attend to the real keys and give finite values, which
        # the masked mean never reads.
        return (out * real[..., None]).sum(dim=1) / real.sum(dim=1, keepdim=True)

    def _frames(self, p: torch.Tensor, real: torch.Tensor) -> Frames:
        """Predict the frame of every real constituent; a padded slot gets the identity.

        :param p: (jets, constituents, 4) four-momenta [E, px, py, pz] in ``units``.
        :param real: (jets, constituents), True on the real constituents.
        :return: frames of shape (jets, constituents, 4, 4).
        """
        # The edge convolution of the frames-net has no padding mask in the dense
        # layout, hence the packed one: (real constituents, 4) with per-jet offsets.
        # float64, as the references feed it: in float32 the mass floor is below the
        # rounding of |p|^2 for constituents above about 14 GeV, so it lifts nothing and
        # pair masses of the massless constituents fall to zero or below, where the
        # frames-net's sqrt has no finite gradient. The zero-width scalars keep the
        # batch's dtype, which is the parameters' one, so the frames-net's MLP does too.
        packed, ptr = p[real].double(), _packed_ptr(real)
        scalars = packed.new_zeros(packed.shape[0], 0, dtype=p.dtype)
        equivectors = self.framesnet.equivectors
        # The edge standardisation runs once, on the first training batch, into
        # buffers the checkpoint carries; eval-mode forwards before it (the model
        # summary, the sanity check, the tests) read the raw attributes. The FLOP count
        # at construction runs in training mode and so clears the package's one-time
        # assert on edge_inited, whose buffers count_flops then restores.
        if self.training and not equivectors.block.edge_inited:
            equivectors.init_standardization(packed, ptr)
        matrices = torch.eye(4).to(p).repeat(*real.shape, 1, 1)
        frames = self.framesnet(packed, scalars=scalars, ptr=ptr)
        matrices[real] = frames.matrices.to(p.dtype)
        return Frames(matrices)
