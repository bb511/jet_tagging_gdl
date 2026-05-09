from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchinfo import summary as torchinfo_summary
from torch.nn.functional import cross_entropy


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def unsorted_segment_mean(data, segment_ids, num_segments):
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def normsq4(p):
    """Minkowski squared norm: p0² - p1² - p2² - p3²"""
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def dotsq4(p, q):
    """Minkowski inner product: p0*q0 - p1*q1 - p2*q2 - p3*q3"""
    pq = p * q
    return 2 * pq[..., 0] - pq.sum(dim=-1)


def psi(p):
    """Signed log: sign(p) * log(|p| + 1)"""
    return torch.sign(p) * torch.log(torch.abs(p) + 1)


# ---------------------------------------------------------------------------
# LGEB — Lorentz Group Equivariant Block
# ---------------------------------------------------------------------------

class LGEB(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
        n_node_attr: int = 0,
        dropout_rate: float = 0.0,
        c_weight: float = 1.0,
        last_layer: bool = False,
    ):
        super().__init__()
        self.c_weight = c_weight
        self.last_layer = last_layer
        n_edge_attr = 2  # Minkowski norm + dot product

        self.phi_e = nn.Sequential(
            nn.Linear(n_input * 2 + n_edge_attr, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )

        self.phi_m = nn.Sequential(
            nn.Linear(n_hidden, 1),
            nn.Sigmoid(),
        )

        # phi_h: aggregated messages (n_hidden) + original node scalars (n_node_attr) → n_output
        self.phi_h = nn.Sequential(
            nn.Linear(n_hidden + n_node_attr, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

        if not last_layer:
            layer = nn.Linear(n_hidden, 1, bias=False)
            nn.init.xavier_uniform_(layer.weight)
            self.phi_x = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                layer,
            )

    def minkowski_feats(self, edges, x):
        i, j = edges
        x_diff = x[i] - x[j]
        norms = normsq4(x_diff).unsqueeze(1)
        dots = dotsq4(x[i], x[j]).unsqueeze(1)
        return psi(norms), psi(dots), x_diff

    def m_model(self, hi, hj, norms, dots):
        out = torch.cat([hi, hj, norms, dots], dim=1)
        out = self.phi_e(out)
        return out * self.phi_m(out)

    def x_model(self, x, edges, x_diff, m):
        i, _ = edges
        trans = x_diff * self.phi_x(m)
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, i, x.size(0))
        return x + agg * self.c_weight

    def h_model(self, h, edges, m, node_attr=None):
        i, _ = edges
        agg = unsorted_segment_mean(m, i, h.size(0))       # (n_nodes, n_hidden)
        inp = torch.cat([agg, node_attr], dim=1) if node_attr is not None else agg
        return h + self.phi_h(inp)

    def forward(self, h, x, edges, node_attr=None):
        norms, dots, x_diff = self.minkowski_feats(edges, x)
        i, j = edges
        m = self.m_model(h[i], h[j], norms, dots)
        if not self.last_layer:
            x = self.x_model(x, edges, x_diff, m)
        h = self.h_model(h, edges, m, node_attr)
        return h, x, m


# ---------------------------------------------------------------------------
# LorentzNet — core nn.Module (graph-format input)
# ---------------------------------------------------------------------------

class LorentzNet(nn.Module):
    """LorentzNet core: Lorentz-equivariant message-passing network.

    Input is in flat graph format (B*N nodes).  Use LorentzNetModule for the
    LightningModule wrapper that accepts (B, N, F) tensors.
    """

    def __init__(
        self,
        n_scalar: int,
        n_hidden: int,
        n_class: int = 5,
        n_layers: int = 6,
        c_weight: float = 1e-3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.embedding = nn.Linear(n_scalar, n_hidden)
        self.LGEBs = nn.ModuleList([
            LGEB(
                n_hidden, n_hidden, n_hidden,
                n_node_attr=n_scalar,
                dropout_rate=dropout,
                c_weight=c_weight,
                last_layer=(i == n_layers - 1),
            )
            for i in range(n_layers)
        ])
        self.graph_dec = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_class),
        )

    def forward(self, scalars, x, edges, node_mask, n_nodes):
        """
        Args:
            scalars:   (B*N, n_scalar) per-particle feature scalars
            x:         (B*N, 4)        Minkowski 4-momenta [E, px, py, pz]
            edges:     (2, E)          flat edge indices over the full batch
            node_mask: (B*N, 1)        1 for real particles, 0 for padding
            n_nodes:   int             N, number of particles per jet
        Returns:
            (B, n_class) logits
        """
        h = self.embedding(scalars)
        for lgeb in self.LGEBs:
            h, x, _ = lgeb(h, x, edges, node_attr=scalars)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.n_hidden)
        h = torch.mean(h, dim=1)
        return self.graph_dec(h)


# ---------------------------------------------------------------------------
# LorentzNetModule — LightningModule wrapper for the HLS4ML workflow
# ---------------------------------------------------------------------------

class LorentzNetModule(LightningModule):
    """Lightning wrapper for LorentzNet compatible with the HLS4ML datamodule.

    Accepts (B, N, F) tensors, converts to Lorentz 4-momenta + fully-connected
    graph, then runs LorentzNet.  Supports augmentation, symm_loss, and
    equivariance methods, matching the interface of the other modules.
    """

    SUPPORTED_METHODS = {"equivariance"}

    def __init__(
        self,
        n_scalar: int,
        n_hidden: int,
        output_dim: int = 5,
        n_layers: int = 6,
        c_weight: float = 1e-3,
        dropout: float = 0.0,
        nconstituents: int = 8,
        method: str = "equivariance",
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
        self.method = method
        self.group = group
        self.nconstituents = nconstituents

        self.net = LorentzNet(
            n_scalar=n_scalar,
            n_hidden=n_hidden,
            n_class=output_dim,
            n_layers=n_layers,
            c_weight=c_weight,
            dropout=dropout,
        )

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc    = Accuracy(task="multiclass", num_classes=output_dim)
        self.val_acc      = Accuracy(task="multiclass", num_classes=output_dim)
        self.test_acc     = Accuracy(task="multiclass", num_classes=output_dim)
        self.train_loss   = MeanMetric()
        self.val_loss     = MeanMetric()
        self.test_loss    = MeanMetric()
        self.val_acc_best = MaxMetric()

        self.feature_names = None
        self.norm_scales   = None
        self._px_col = 0
        self._py_col = 1
        self._pz_col = 2

        # Pre-build edge index template for a single jet (reused across batches)
        self._n_nodes = nconstituents
        self._register_edge_template(nconstituents)

        self.example_input_array = torch.randn(2, nconstituents, n_scalar)

        stats = torchinfo_summary(self, input_data=self.example_input_array, verbose=0)
        mult_adds = float(stats.total_mult_adds)
        total_flops = 2.0 * mult_adds
        self.hparams["flops_mult_adds"] = mult_adds
        self.hparams["flops_total"]     = total_flops
        self.hparams["flops"]           = total_flops

    # ------------------------------------------------------------------
    # Graph construction helpers
    # ------------------------------------------------------------------

    def _register_edge_template(self, n_nodes: int) -> None:
        """Register fully-connected edge template (no self-loops) for one jet."""
        idx = torch.arange(n_nodes)
        src, dst = torch.meshgrid(idx, idx, indexing="ij")
        keep = src != dst
        self.register_buffer("_edge_src_tmpl", src[keep])  # (N*(N-1),)
        self.register_buffer("_edge_dst_tmpl", dst[keep])

    def _build_edges(self, batch_size: int) -> torch.Tensor:
        """Expand edge template to cover all jets in the batch."""
        n = self._n_nodes
        device = self._edge_src_tmpl.device
        offsets = torch.arange(batch_size, device=device) * n  # (B,)
        src = (self._edge_src_tmpl.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        dst = (self._edge_dst_tmpl.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        return torch.stack([src, dst], dim=0)  # (2, B*N*(N-1))

    # ------------------------------------------------------------------
    # Input conversion: (B, N, F) → 4-momenta + masks
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        B, N, F = x.shape

        pmu = self._features_to_pmu(x)                           # (B, N, 4)
        pt  = torch.sqrt(pmu[:, :, 1]**2 + pmu[:, :, 2]**2)
        node_mask = (pt > 1e-6).float().view(B * N, 1)           # (B*N, 1)

        scalars = x.view(B * N, F)                               # (B*N, F)
        x_flat  = pmu.view(B * N, 4)                             # (B*N, 4)
        edges   = self._build_edges(B)                           # (2, B*N*(N-1))

        return self.net(scalars, x_flat, edges, node_mask, N)    # (B, output_dim)

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        if y.ndim > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)

        logits = self.forward(x)
        loss = self.criterion(logits, y)

        return loss, torch.argmax(logits, dim=1), y

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
            self.feature_names = list(dm.features)
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
                f"LorentzNetModule: px/py columns not found in features {names}. "
                "LorentzNet requires px and py to reconstruct 4-momenta."
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
        print(
            f"[Epoch {self.current_epoch:>3d}] "
            f"Train | loss: {train_loss:.4f}  acc: {train_acc:.4f}",
            flush=True,
        )

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
        print(
            f"[Epoch {self.current_epoch:>3d}]   Val | loss: {val_loss:.4f}  acc: {acc:.4f}"
            f"  best: {self.val_acc_best.compute():.4f}",
            flush=True,
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
