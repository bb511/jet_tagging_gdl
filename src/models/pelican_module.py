import torch
import torch.nn as nn

import logging
from typing import Any, Dict, Optional, Tuple

from lightning import LightningModule
from omegaconf import ListConfig, OmegaConf
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.utils.flops import record_flops
from torch.nn.functional import cross_entropy
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from src.utils.symmetry_loss import build_symmetric_batch, combine_with_symmetry_loss, validate_symmetry_group
from ..utils.lorentz_metric import CATree, SDMultiplicity
from .components.pelican_layers.generic_layers import BasicMLP, MessageNet, InputEncoder, GInvariants, MyLinear
from .components.pelican_layers.perm_equiv_models import Net2to2, Eq1to2, Eq2to0
logger = logging.getLogger(__name__)

class PELICANClassifier(nn.Module):
    """
    Permutation Invariant, Lorentz Invariant/Covariant Aggregator Network
    """
    def __init__(self, rank1_dim_multiplier, num_channels_scalar, num_channels_m, num_channels_2to2, num_channels_out, num_channels_m_out, 
                 stabilizer='so13', method='input', num_classes=2,
                 activate_agg_in=False, activate_lin_in=True,
                 activate_agg=False, activate_lin=True, activation='leakyrelu', config='s', config_out='s', average_nobj=49, factorize=False, masked=True,
                 activate_agg_out=True, activate_lin_out=False, mlp_out=True,
                 scale=1, irc_safe=False, dropout = False, drop_rate=0.1, drop_rate_out=0.1, batchnorm=None, dataset='',
                 device=torch.device('cpu'), dtype=None):
        super().__init__()

        logging.info('Initializing network!')

        num_channels_m = expand_var_list(num_channels_m)
        num_channels_2to2 = expand_var_list(num_channels_2to2)
        num_channels_out = expand_var_list(num_channels_out)

        self.device, self.dtype = device, dtype
        self.rank1_width_multiplier = rank1_dim_multiplier
        self.num_channels_scalar = num_channels_scalar
        self.num_channels_m = num_channels_m
        self.num_channels_2to2 = num_channels_2to2
        self.num_channels_m_out = num_channels_m_out
        self.num_channels_out = num_channels_out
        self.stabilizer = stabilizer
        self.method = method
        self.num_classes = num_classes
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.scale = scale
        self.config = config
        self.config_out = config_out
        self.irc_safe = irc_safe
        self.mlp_out = mlp_out
        self.average_nobj = average_nobj
        self.factorize = factorize
        self.masked = masked
        self.dataset = dataset

        if dropout:
            self.dropout_layer = nn.Dropout(drop_rate)
            self.dropout_layer_out = nn.Dropout(drop_rate_out)

        if method.startswith('s'):
            self.ginvariants = GInvariants(stabilizer='so13', irc_safe=irc_safe)
        else:
            self.ginvariants = GInvariants(stabilizer=stabilizer, irc_safe=irc_safe)
        self.rank1_dim = self.ginvariants.rank1_dim
        self.rank2_dim = self.ginvariants.rank2_dim

        self.num_scalars = {'qg': 12, 'jc': 12, '': 0}[dataset]
        if self.method.startswith('s'):
            self.num_scalars += 1 + self.num_spurions()

        if (len(num_channels_m) > 0) and (len(num_channels_m[0]) > 0):
            embedding_dim = self.num_channels_m[0][0]
        else:
            embedding_dim = self.num_channels_2to2[0]

        # rank2_dim_multiplier  =1
        if self.num_scalars > 0 or self.rank1_dim > 0: 
            if self.rank2_dim > 0:
                assert embedding_dim > num_channels_scalar, f"num_channels_m[0][0] has to be at least {num_channels_scalar + 1}, because you have particle scalars, but got {embedding_dim}"
                # rank2_dim_multiplier = (embedding_dim - num_channels_scalar)//self.rank2_dim
                embedding_dim = embedding_dim - num_channels_scalar
        
        # The input stack applies an encoding function
        rank1_dim_multiplier = 1 # each scalar will produce this many channels

        if stabilizer == 'so13' or method.startswith('s'):
            weights = torch.ones((embedding_dim, self.rank2_dim), device=device, dtype=dtype)
        elif stabilizer=='1':
            weights = torch.ones((embedding_dim, self.rank2_dim), device=device, dtype=dtype) - torch.tensor([[0,2,2,2]], device=device, dtype=dtype)
        elif stabilizer=='1_0':
            weights = torch.ones((embedding_dim, self.rank2_dim), device=device, dtype=dtype) - torch.tensor([[0,2,2,2]], device=device, dtype=dtype)
        elif stabilizer in ['so3','so12','se2']:
            weights = torch.zeros((embedding_dim, self.rank2_dim), device=device, dtype=dtype) + torch.tensor([[0,1]], device=device, dtype=dtype)
        elif stabilizer in ['so2','so2_0','R']:
            weights = torch.zeros((embedding_dim, self.rank2_dim), device=device, dtype=dtype) + torch.tensor([[0,0,1]], device=device, dtype=dtype)
        elif stabilizer in ['11','11_0']:
            weights = torch.zeros((embedding_dim, self.rank2_dim), device=device, dtype=dtype) + torch.tensor([[0,0,0,0,1]], device=device, dtype=dtype)
        
        self.linear = MyLinear(self.rank2_dim, embedding_dim, weights, device=device, dtype=dtype)

        mode = 'angle' if self.irc_safe else 'slog'
        self.input_encoder = InputEncoder(rank1_dim_multiplier, embedding_dim, 
                                          rank1_in_dim = self.rank1_dim, rank2_in_dim=self.rank2_dim, 
                                          mode=mode, device = device, dtype = dtype)
        
        # This is the main part of the network -- a sequence of permutation-equivariant 2->2 blocks
        # Each 2->2 block consists of a component-wise messaging layer that mixes channels, followed by the equivariant aggegration over particle indices
        self.net2to2 = Net2to2(num_channels_2to2 + [num_channels_m_out[0]], num_channels_m, 
                               activate_agg=activate_agg, activate_lin=activate_lin, activation = activation, 
                               dropout=dropout, drop_rate=drop_rate, batchnorm = batchnorm, config=config, 
                               average_nobj=average_nobj, factorize=factorize, masked=masked, device = device, dtype = dtype)

        # The final equivariant block is 2->1 and is defined here manually as a messaging layer followed by the 2->1 aggregation layer
        self.msg_2to0 = MessageNet(num_channels_m_out, activation=activation, 
                                   batchnorm=batchnorm, device=device, dtype=dtype)       
        self.agg_2to0 = Eq2to0(num_channels_m_out[-1], num_channels_out[0] if mlp_out else num_classes, 
                               activate_agg=activate_agg_out, activate_lin=activate_lin_out, activation = activation, 
                               config=config_out, factorize=False, average_nobj=average_nobj, device = device, dtype = dtype)
        
        # We have produced a permutation-invariant feature vector, and now we apply an MLP with 2 output channels to it to get the final classification weights
        if mlp_out:
            self.mlp_out = BasicMLP(self.num_channels_out + [num_classes], activation=activation, dropout = False, batchnorm = False, device=device, dtype=dtype)

        self.apply(init_weights)

        # If there are scalars (like beam labels or PIDs) we promote them using an equivariant 1->2 layer and then concatenate them to the embedded dot products
        if self.num_scalars > 0:
            eq1to2_in_dim = self.num_scalars + rank1_dim_multiplier*self.rank1_dim
            # eq2to2_out_dim = (embedding_dim - self.rank2_dim * rank2_dim_multiplier) if self.rank2_dim > 0 else embedding_dim
            self.eq1to2 = Eq1to2(eq1to2_in_dim, num_channels_scalar, 
                                 activate_agg=activate_agg_in, activate_lin=activate_lin_in, 
                                 activation = activation, average_nobj=average_nobj, 
                                 config=config_out, factorize=False, device = device, dtype = dtype)


        logging.info('_________________________\n')
        for n, p in self.named_parameters(): logging.info(f'{"Parameter: " + n:<80} {p.shape}')
        logging.info('Model initialized. Number of parameters: {}'.format(sum(p.nelement() for p in self.parameters())))
        logging.info('_________________________\n')

    def forward(self, data, covariance_test=False):
        """
        Runs a forward pass of the network.
        """
        # Get and prepare the data
        particle_scalars, particle_mask, edge_mask, event_momenta = self.prepare_input(data)
        rank1_inputs, rank2_inputs, irc_weight = self.ginvariants(event_momenta)

        # regular multiplicity
        nobj = particle_mask.sum(-1, keepdim=True)
        # If the aggregation option is set to means, we replace N with the SoftDrop multiplicity (NB: THIS IS EXTREMELY SLOW)
        # TODO: pre-compute SoftDrop Multiplicities before training and safe it into a data file instead of re-computing at each epoch.
        if self.irc_safe:
            if ('m' in self.config or 'M' in self.config) or ('m' in self.config_out or 'M' in self.config_out):
                nobj = SDMultiplicity(CATree(rank2_inputs, nobj, ycut=1000, eps=10e-8)).unsqueeze(-1).to(device=nobj.device)

        # The first nonlinearity is the input encoder, which applies functions of the form ((1+x)^alpha-1)/alpha with trainable alphas.
        # In the C-safe case, this is still fine because inputs depends only on relative angles
        rank2_inputs = self.linear(rank2_inputs)
        rank1_inputs, rank2_inputs = self.input_encoder(rank1_inputs, rank2_inputs, 
                                                        rank1_mask=particle_mask.unsqueeze(-1) ,rank2_mask=edge_mask.unsqueeze(-1))

        inputs = self.apply_eq1to2(particle_scalars, rank1_inputs, rank2_inputs, edge_mask, nobj, irc_weight)

        # Apply the sequence of PELICAN equivariant 2->2 blocks with the IRC weighting.
        act1 = self.net2to2(inputs, mask = edge_mask.unsqueeze(-1), nobj = nobj,
                            irc_weight = irc_weight if self.irc_safe else None)

        # The last equivariant 2->0 block is constructed here by hand: message layer, dropout, and Eq2to0.
        act2 = self.msg_2to0(act1, mask=edge_mask.unsqueeze(-1))
        if self.dropout:
            act2 = self.dropout_layer(act2)
        act3 = self.agg_2to0(act2, nobj = nobj, irc_weight = irc_weight if self.irc_safe else None)

        # The output layer applies dropout and an MLP.
        if self.dropout:
            act3 = self.dropout_layer_out(act3)
        if self.mlp_out:
            prediction = self.mlp_out(act3)
        else:
            prediction = act3

        if covariance_test:
            return {'predict': prediction, 'inputs': inputs, 'act1': act1, 'act2': act2, 'act3': act3}
        else:
            return {'predict': prediction}

    def prepare_input(self, data):
        """
        Extracts input from data class

        Parameters
        ----------
        data : ?????
            Information on the state of the system.

        Returns
        -------
        scalars : :obj:`torch.Tensor`
            Tensor of scalars for each particle.
        particle_mask : :obj:`torch.Tensor`
            Mask used for batching data.
        edge_mask: :obj:`torch.Tensor`
            Mask used for batching data.
        event_momenta: :obj:`torch.Tensor`
            4-momenta of the particles
        """
        device, dtype = self.device, self.dtype

        if self.dataset == "jc":
            data = add_pid_jc(data)
        elif self.dataset == 'qg':
            data = add_pid_qg(data)
        if self.method.startswith('s'): # do this last because spurions need to know the shape of the scalar inputs
            data = self.add_spurions(data)

        event_momenta = data['Pmu'].to(device, dtype)
        # event_momenta.requires_grad_(True)
        particle_mask = data['particle_mask'].to(device, torch.bool)
        edge_mask = data['edge_mask'].to(device, torch.bool)

        if 'scalars' in data.keys():
            scalars = data['scalars'].to(device, dtype)
        else:
            scalars = None
        return scalars, particle_mask, edge_mask, event_momenta
    
    def num_spurions(self):
        if self.method.startswith('i'):
            return 0
        stabilizer = self.stabilizer
        if stabilizer == 'so13':
            return 0
        elif stabilizer in ['so3','so12','se2']:
            return 1
        elif stabilizer in ['so2','so2_0','R']:
            return 2
        elif stabilizer in ['1','11','1_0','11_0']:
            return 4

    def add_spurions(self, data):
        stabilizer = self.stabilizer
        if stabilizer == 'so13':
            return data
        device, dtype = data['Pmu'].device, data['Pmu'].dtype
        batch_size = len(data['Nobj'])
        if stabilizer == 'so3':
            spurions = torch.tensor([[[1,0,0,0]]], dtype=dtype, device=device)
        elif stabilizer == 'so12':
            spurions = torch.tensor([[[0,0,0,1]]], dtype=dtype, device=device)
        elif stabilizer == 'se2':
            spurions = torch.tensor([[[1,0,0,-1]]], dtype=dtype, device=device)
        elif stabilizer == 'so2':
            spurions = torch.tensor([[[1,0,0,0],[0,0,0,1]]], dtype=dtype, device=device)
        elif stabilizer == 'so2_0':
            spurions = torch.tensor([[[1,0,0,1],[1,0,0,-1]]], dtype=dtype, device=device)
        elif stabilizer == 'R':
            spurions = torch.tensor([[[0,1,0,0],[0,0,1,0]]], dtype=dtype, device=device)
        elif stabilizer in ['1','11']:
            spurions = torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]], dtype=dtype, device=device)
        elif stabilizer in ['1_0','11_0']:
            spurions = torch.tensor([[[1,0,0,1],[1,1,0,0],[1,0,1,0],[1,0,0,-1]]], dtype=dtype, device=device)
        spurions = spurions.expand((batch_size, -1, -1))

        num_spurions = spurions.shape[1]
        particle_mask = torch.cat((torch.ones(batch_size, num_spurions).bool().to(device=device), data['particle_mask']),dim=-1)
        edge_mask = particle_mask.unsqueeze(1) * particle_mask.unsqueeze(2)

        data['Pmu'] = torch.cat([spurions, data['Pmu']], 1)
        data['particle_mask'] = particle_mask
        data['edge_mask'] = edge_mask
        labels = 1 - particle_mask.long()
        labels[:, :num_spurions] = torch.arange(1,num_spurions+1)
        spurion_label_onehot = onehot(labels, num_classes=1+num_spurions, mask=particle_mask.unsqueeze(-1))
        if 'scalars' in data.keys():
            data['scalars'] = torch.cat([torch.zeros((batch_size, num_spurions, data['scalars'].shape[2]), device=device, dtype=data['scalars'].dtype), data['scalars']], dim=1)
            data['scalars'] = torch.cat([data['scalars'], spurion_label_onehot], dim=-1)
        else:
            data['scalars'] = spurion_label_onehot
        return data
    
    def apply_eq1to2(self, particle_scalars, rank1_inputs, rank2_inputs, edge_mask, nobj, irc_weight):
        # First concatenate particle scalar inputs with rank 1 momentum features (if any)
        if self.num_scalars > 0:
            if rank1_inputs is None:
                rank1_inputs = particle_scalars
            else:
                rank1_inputs = torch.cat([particle_scalars, rank1_inputs], dim=-1)
        # Now promore all rank 1 data to rank 2 using Eq1to2
        if rank1_inputs is not None:
            rank2_particle_scalars = self.eq1to2(rank1_inputs, mask=edge_mask.unsqueeze(-1), nobj=nobj, irc_weight = irc_weight if self.irc_safe else None)
        else:
            rank2_particle_scalars = None
        # Concatenate all rank 2 data together
        if rank2_inputs is None:
            inputs = rank2_particle_scalars
        elif rank2_particle_scalars is None:
            inputs = rank2_inputs
        else:
            inputs = torch.cat([rank2_inputs, rank2_particle_scalars], dim=-1)
        return inputs
    

def init_weights(m):
    """Xavier-init plain `nn.Linear` layers (e.g. inside BasicMLP/MessageNet/Eq*).

    Deliberately skips `MyLinear`: its weights are constructed by the caller
    to encode the stabilizer's symmetry-breaking pattern (see the `weights`
    tensor built in `PELICANClassifier.__init__`), and re-initializing them
    here would destroy that structure.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def add_pid_jc(data, mask=None):
    # One-hots for the JetClass dataset
    charge_onehot = onehot(data['part_charge']+1,num_classes=3)
    pid_onehot = torch.stack([data['part_isChargedHadron'], data['part_isElectron'], data['part_isMuon'], data['part_isNeutralHadron'], data['part_isPhoton'], data['d0val'], data['d0err'], data['dzval'], data['dzerr']], dim=-1)
    if 'scalars' in data.keys():
        data['scalars'] = torch.cat([data['scalars'],charge_onehot,pid_onehot])
    else:
        data['scalars'] = torch.cat([charge_onehot,pid_onehot],dim=-1)
    return data

def expand_var_list(var):
    if type(var) is list:
        var_list = var
    else:
        raise ValueError('Incorrect type {}'.format(type(var)))
    return var_list

def onehot(x, num_classes=2, mask=None):
    x = torch.nn.functional.one_hot(x, num_classes=num_classes)
    zero = torch.tensor(0, device=x.device, dtype=torch.long)
    if mask is not None:
        x = torch.where(mask, x, zero)
    return x



class PELICANModule(LightningModule):
    """Lightning wrapper around `PELICANClassifier` for the HLS4ML jet-tagging workflow.

    `PELICANClassifier` consumes a `data` dict (`Pmu`, `particle_mask`,
    `edge_mask`, `Nobj`, [`scalars`]); this wrapper converts the `(B, N, F)`
    feature tensors produced by `HLS4MLDataModule` into that format and runs
    the same train/val/test loop (with augmentation / symm_loss / equivariance
    support) shared by the other models in this repo.

    Note: the Hydra config calls the spurion-vs-invariant-input switch
    `spurion_method` (passed through to `PELICANClassifier` as `method`) to
    avoid a name clash with `method`, which here means the symmetry-handling
    method (`augmentation` / `symm_loss` / `equivariance`), matching the
    convention of the other modules and `data.method`.
    """

    SUPPORTED_METHODS = {"augmentation", "symm_loss", "equivariance"}

    def __init__(
        self,
        num_classes: int,
        nbr_feats: int,
        nconstituents: int,
        average_nobj: float,
        rank1_dim_multiplier: int,
        num_channels_scalar: int,
        num_channels_m: list,
        num_channels_2to2: list,
        num_channels_m_out: list,
        num_channels_out: list,
        stabilizer: str = "so13",
        spurion_method: str = "input",
        config: str = "M",
        config_out: str = "M",
        activation: str = "leakyrelu",
        batchnorm: Optional[str] = "b",
        masked: bool = True,
        activate_agg_in: bool = False,
        activate_lin_in: bool = True,
        activate_agg: bool = False,
        activate_lin: bool = True,
        activate_agg_out: bool = True,
        activate_lin_out: bool = False,
        factorize: bool = True,
        mlp_out: bool = True,
        irc_safe: bool = False,
        dropout: bool = False,
        drop_rate: float = 0.1,
        drop_rate_out: float = 0.1,
        method: str = "augmentation",
        group: str = "so2",
        symm_loss_weight: float = 0.1,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. Expected one of: {self.SUPPORTED_METHODS}."
            )
        self.method        = method
        self.group         = group
        validate_symmetry_group(method, group)
        self.nconstituents = nconstituents

        # Hydra hands these over as (nested) ListConfig objects, but
        # PELICANClassifier's `expand_var_list` requires `type(x) is list`
        # (and downstream layers concatenate the inner lists with `+`, which
        # ListConfig doesn't support cleanly) — convert to plain nested lists.
        def _plain(x):
            return OmegaConf.to_container(x, resolve=True) if isinstance(x, ListConfig) else x

        num_channels_m     = _plain(num_channels_m)
        num_channels_2to2  = _plain(num_channels_2to2)
        num_channels_m_out = _plain(num_channels_m_out)
        num_channels_out   = _plain(num_channels_out)

        self.net = PELICANClassifier(
            rank1_dim_multiplier=rank1_dim_multiplier,
            num_channels_scalar=num_channels_scalar,
            num_channels_m=num_channels_m,
            num_channels_2to2=num_channels_2to2,
            num_channels_out=num_channels_out,
            num_channels_m_out=num_channels_m_out,
            stabilizer=stabilizer,
            method=spurion_method,
            num_classes=num_classes,
            activate_agg_in=activate_agg_in,
            activate_lin_in=activate_lin_in,
            activate_agg=activate_agg,
            activate_lin=activate_lin,
            activation=activation,
            config=config,
            config_out=config_out,
            average_nobj=average_nobj,
            factorize=factorize,
            masked=masked,
            activate_agg_out=activate_agg_out,
            activate_lin_out=activate_lin_out,
            mlp_out=mlp_out,
            irc_safe=irc_safe,
            dropout=dropout,
            drop_rate=drop_rate,
            drop_rate_out=drop_rate_out,
            batchnorm=batchnorm,
            dataset='',
        )

        if compile:
            self.net = torch.compile(self.net)

        self.criterion = combine_with_symmetry_loss if method == "symm_loss" else nn.CrossEntropyLoss()

        self.train_acc    = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc      = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc     = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_loss   = MeanMetric()
        self.val_loss     = MeanMetric()
        self.test_loss    = MeanMetric()
        self.val_acc_best = MaxMetric()

        # feature_names / norm_scales / column indices populated in setup()
        # from the datamodule; defaults below match the HLS4ML feature order
        # ["px", "py", "pz", "etarel", "phirel"] so __init__-time tracing works.
        self.feature_names = None
        self.norm_scales   = None
        self._px_col = 0
        self._py_col = 1
        self._pz_col = 2

        self.example_input_array = torch.randn(2, nconstituents, nbr_feats)

        # Per-jet forward FLOPs (scales with the constituent count; see
        # src/utils/flops.py for the counting convention).
        record_flops(self, nconstituents, nbr_feats)

    # ------------------------------------------------------------------
    # Input conversion: (B, N, F) -> PELICAN `data` dict
    # ------------------------------------------------------------------
    def _features_to_pmu(self, x: torch.Tensor) -> torch.Tensor:
        """Build (B, N, 4) tensor [E, px, py, pz] from feature columns."""
        px = x[:, :, self._px_col]
        py = x[:, :, self._py_col]
        if self._pz_col is not None:
            pz = x[:, :, self._pz_col]
            E  = torch.sqrt(px**2 + py**2 + pz**2 + 1e-8)
        else:
            pz = torch.zeros_like(px)
            E  = torch.sqrt(px**2 + py**2 + 1e-8)
        return torch.stack([E, px, py, pz], dim=-1)  # (B, N, 4)

    def _build_data(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert (B, N, F) features into the dict PELICANClassifier.forward expects."""
        pmu = self._features_to_pmu(x)                                      # (B, N, 4)
        pt  = torch.sqrt(pmu[:, :, 1] ** 2 + pmu[:, :, 2] ** 2)
        particle_mask = pt > 1e-6                                           # (B, N) bool — real vs padded
        edge_mask     = particle_mask.unsqueeze(1) * particle_mask.unsqueeze(2)
        nobj          = particle_mask.sum(dim=-1)                           # (B,) — only used for spurions
        return {
            "Pmu":           pmu,
            "particle_mask": particle_mask,
            "edge_mask":     edge_mask,
            "Nobj":          nobj,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        data = self._build_data(x)

        # PELICANClassifier.device/.dtype are plain attributes fixed at
        # construction time (not registered buffers), so Lightning moving
        # this module to the accelerator does not update them — left alone,
        # `prepare_input` would move the (correctly-placed) input tensors back
        # to the construction-time device and crash on a device mismatch.
        self.net.device = x.device
        self.net.dtype  = x.dtype

        return self.net(data)["predict"]

    # ------------------------------------------------------------------
    # Shared step — branched cleanly by method
    # ------------------------------------------------------------------
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        if y.ndim > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)

        logits = self.forward(x)

        if self.method == "symm_loss":
            if self.feature_names is None:
                raise RuntimeError(
                    "feature_names is None — symm_loss requires feature_names. "
                    "Ensure setup() has been called so the datamodule populates it."
                )
            x_sym, _, keep_mask = build_symmetric_batch(
                x,
                group=self.group,
                feature_names=self.feature_names,
                norm_scales=self.norm_scales,
            )
            logits_sym = self.forward(x_sym)

            # keep_mask is set only for so3, which can drop jets rotated
            # outside detector acceptance — subset the original logits to
            # the same surviving jets so the symmetry term compares like
            # with like (classification loss still uses the full batch).
            logits_for_sym = logits if keep_mask is None else logits[keep_mask]

            cls_loss = cross_entropy(logits, y)
            loss, _  = self.criterion(
                cls_loss, logits_for_sym, logits_sym, weight=self.hparams.symm_loss_weight
            )
        else:
            # augmentation / equivariance — plain cross-entropy
            loss = self.criterion(logits, y)

        return loss, torch.argmax(logits, dim=1), y

    # ------------------------------------------------------------------
    # Setup — grab feature_names / column indices from the datamodule
    # ------------------------------------------------------------------
    def setup(self, stage: str) -> None:
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
            self.feature_names = list(dm.features)
            if self.method == "symm_loss" and not self.feature_names:
                raise RuntimeError(
                    "datamodule.features is empty after setup(). "
                    "Ensure HLS4MLDataModule is configured with a features list."
                )
            raw_params = getattr(dm, "norm_params", None)
            self.norm_scales = (
                {name: float(scale) for name, (_, scale) in raw_params.items()}
                if raw_params else None
            )
            self._resolve_col_indices(self.feature_names)

    def _resolve_col_indices(self, names: list) -> None:
        def _find(candidates):
            for n in candidates:
                if n in names:
                    return names.index(n)
            return None

        self._px_col = _find(["px", "j1_px"])
        self._py_col = _find(["py", "j1_py"])
        self._pz_col = _find(["pz", "j1_pz"])

        if self._px_col is None or self._py_col is None:
            raise RuntimeError(
                f"PELICANModule: px/py columns not found in features {names}. "
                "PELICAN requires px and py to reconstruct 4-momenta."
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc,  on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        train_loss = self.train_loss.compute()
        train_acc  = self.train_acc.compute()
        logger.info(
            f"[Epoch {self.current_epoch:>3d}] "
            f"Train | loss: {train_loss:.4f}  acc: {train_acc:.4f}"
        )
        garbage_collection_cuda()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc",  self.val_acc,  on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        val_loss = self.val_loss.compute()
        logger.info(
            f"[Epoch {self.current_epoch:>3d}]   Val | loss: {val_loss:.4f}  acc: {acc:.4f}"
            f"  best: {self.val_acc_best.compute():.4f}"
        )

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc",  self.test_acc,  on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor":   "val/acc",
                    "interval":  "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
