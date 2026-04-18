# Transformer, this implementation is based on the paper "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
# and the PyTorch implementation (https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer)
# This has been suggested by Maximilian Herde (but I am the one to blame for wrong usage)
from typing import Any, Dict, Tuple, Optional

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.utils.custom_loss import combine_with_symmetry_loss, build_symmetric_batch
from torch.nn.functional import cross_entropy
import torch.nn as nn

# This is a baseline implementation for the Transformer model. I need to modify it for the structure of this repo

class Transformer(LightningModule):
    def __init__(self,
                nbr_const: int,
                nbr_feat: int,
                lifting_dim: int,
                num_layers: int,
                d_models: int,
                nhead: int,
                output_size: int,
                norm: Optional[str] = None,
                use_mask: bool = True,
                fill_masked: bool = True,
                bias: bool = True,
                coef_norm: Optional[str] = None,
                fast_path: bool = False,
                optimizer: Optional[torch.optim.Optimizer] = None,
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                compile: bool = False,
                group: str = "so2"):
        
        super(Transformer, self).__init__()
        
        self.nbr_const = nbr_const
        self.nbr_feat = nbr_feat
        
        self.lifting_dim = lifting_dim
        self.num_layers = num_layers
        self.d_models = d_models
        self.nhead = nhead
        
        self.output_size = output_size
        self.norm = norm
        self.use_mask = use_mask
        self.fill_masked = fill_masked

        if self.method == "augmentation":   
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.method == "symm_loss":
            self.criterion = combine_with_symmetry_loss

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=output_size)
        self.val_acc = Accuracy(task="multiclass", num_classes=output_size)
        self.test_acc = Accuracy(task="multiclass", num_classes=output_size)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.group = group

        # Store normalization method
        if coef_norm is None:
            self.norm_method = 'none'
        else:
            raise NotImplementedError("Normalization inside the model is not supported anymore.")

        # Lifting layer for each constituent
        self.lifting_layer = nn.Linear(nbr_feat, d_models, bias=bias)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_models, nhead=self.nhead, batch_first=True, bias=bias)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers, enable_nested_tensor=fast_path)
        self.fc = nn.Linear(self.d_models * self.nbr_const, self.output_size)

        self.save_hyperparameters(logger=False)  # Save hyperparameters for checkpointing and logging
        
    def forward(self, x):
        x = x.to(torch.float32)
        # Create mask based on zero-padded constituents
        mask = (x.sum(dim=-1) == 0) if self.use_mask else None  # (batch, nbr_const)

        # Apply lifting layer to each constituent
        x = self.lifting_layer(x)  # (batch, nbr_const, d_models)

        # Pass through transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # (batch, nbr_const, d_models)

        # Zero out masked positions before flattening so FC doesn't see garbage
        if self.use_mask and self.fill_masked and mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        # Flatten and pass through final fully connected layer
        x = x.view(x.size(0), -1)  # (batch, nbr_const * d_models)
        x = self.fc(x)  # (batch, output_size)
        return x
    
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        # convert one-hot labels to class indices if needed
        if y.ndim > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)
        logits = self.forward(x)
        if self.method == "augmentation":
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            return loss, preds, y

        else:
            if self.group.lower() == "lorentz":
                feature_names = self.trainer.datamodule.feature_names if hasattr(self.trainer, "datamodule") else None
                x_sym, _ = build_symmetric_batch(x, group=self.group, feature_names=feature_names)
            else:
                x_sym, _ = build_symmetric_batch(x, group=self.group)
            cls_loss = cross_entropy(logits, y)
            logits_sym = self.forward(x_sym)
            loss = self.criterion(cls_loss, logits, logits_sym, weight=0.1)
            preds = torch.argmax(logits, dim=1)
            return loss, preds, y
    
    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        train_loss = self.train_loss.compute()
        train_acc = self.train_acc.compute()
        print(
            f"Epoch {self.current_epoch} ended. "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        val_loss = self.val_loss.compute()
        print(
            f"Epoch {self.current_epoch} ended. "
            f"Val Loss: {val_loss:.4f}, Val Acc: {acc:.4f}, "
            f"Val Acc Best: {self.val_acc_best.compute():.4f}"
        )

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        pass

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            num_training_steps = self.trainer.estimated_stepping_batches
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                num_training_steps=num_training_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
