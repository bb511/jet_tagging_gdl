import math
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch.optim as optim

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def plot_training_curves(
    csv_path: str,
    save_dir: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot training/validation loss, accuracy, and learning rate from a CSV logger file.

    :param csv_path: Path to the metrics.csv file produced by Lightning's CSVLogger.
    :param save_dir: Directory to save the plot. Defaults to the parent of csv_path.
    :param show: Whether to display the plot interactively.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    if "epoch" not in df.columns:
        log.warning("No 'epoch' column found in CSV metrics. Cannot plot training curves.")
        return

    df_epoch = df.groupby("epoch").first().reset_index()
    epochs = df_epoch["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Loss ---
    ax = axes[0]
    if "train/loss" in df_epoch.columns:
        ax.plot(epochs, df_epoch["train/loss"], label="Train Loss", linewidth=2)
    if "val/loss" in df_epoch.columns:
        ax.plot(epochs, df_epoch["val/loss"], label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Accuracy ---
    ax = axes[1]
    if "train/acc" in df_epoch.columns:
        ax.plot(epochs, df_epoch["train/acc"], label="Train Acc", linewidth=2)
    if "val/acc" in df_epoch.columns:
        ax.plot(epochs, df_epoch["val/acc"], label="Val Acc", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Learning Rate ---
    ax = axes[2]
    lr_cols = [c for c in df_epoch.columns if c.startswith("lr")]
    if lr_cols:
        lr_col = lr_cols[0]
        lr_vals = df_epoch[lr_col].dropna()
        ax.plot(df_epoch.loc[lr_vals.index, "epoch"], lr_vals.values, label=lr_col, linewidth=2, color="green")
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    output_dir = Path(save_dir) if save_dir is not None else Path(csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_curves.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    log.info(f"Saved training curves to: {output_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Optional[list] = None,
    save_dir: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot one-vs-rest ROC curves for each class with AUC values.

    :param y_true: Ground truth labels as class indices, shape [N].
    :param y_score: Predicted probabilities (softmax output), shape [N, num_classes].
    :param class_names: Optional list of class names.
    :param save_dir: Directory to save the plot.
    :param show: Whether to display the plot interactively.
    """
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    num_classes = y_score.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4), sharex=True, sharey=True)
    if num_classes == 1:
        axes = [axes]

    for k in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, k], y_score[:, k])
        roc_auc = auc(fpr, tpr)

        ax = axes[k]
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(class_names[k])
        ax.set_xlabel("False Positive Rate")
        if k == 0:
            ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Class ROC Curves (One-vs-Rest)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir is not None:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "roc_curves.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        log.info(f"Saved ROC curves to: {output_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def compute_kfold_accuracy(
    x_data: np.ndarray,
    y_data: np.ndarray,
    model_cfg: DictConfig,
    trainer_cfg: DictConfig,
    k: int = 5,
    batch_size: int = 256,
    seed: int = 42,
) -> Tuple[float, float, List[float]]:
    """Compute accuracy uncertainty via Stratified K-Fold cross-validation.

    For each fold a fresh model and trainer are instantiated from the Hydra
    configs, trained on the fold's training split, and evaluated on its
    validation split.  Returns mean accuracy, standard deviation, and the
    per-fold accuracy list.

    :param x_data: Feature array, shape [N, n_constituents, n_features].
    :param y_data: Label array, one-hot [N, n_classes] or indices [N].
    :param model_cfg: Hydra DictConfig for the model (with _target_, optimizer, etc.).
    :param trainer_cfg: Hydra DictConfig for the trainer.
    :param k: Number of folds.
    :param batch_size: Batch size for fold data loaders.
    :param seed: Random seed for the fold splitter.
    :return: (mean_accuracy, std_accuracy, per_fold_accuracies)
    """
    import hydra
    import lightning as L
    import torch
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader, TensorDataset

    y_labels = y_data.argmax(axis=1) if y_data.ndim > 1 else y_data.astype(int)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accuracies: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_data, y_labels)):
        print(f"\n--- Fold {fold + 1}/{k} ---")
        L.seed_everything(seed + fold, workers=True)

        x_train_fold = torch.from_numpy(x_data[train_idx]).float()
        y_train_fold = torch.from_numpy(y_data[train_idx]).float()
        x_val_fold = torch.from_numpy(x_data[val_idx]).float()
        y_val_fold = torch.from_numpy(y_data[val_idx]).float()

        train_loader = DataLoader(
            TensorDataset(x_train_fold, y_train_fold),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(x_val_fold, y_val_fold),
            batch_size=batch_size,
            shuffle=False,
        )

        model = hydra.utils.instantiate(model_cfg)
        trainer = hydra.utils.instantiate(
            trainer_cfg,
            callbacks=None,
            logger=False,
            enable_checkpointing=False,
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        val_acc = trainer.callback_metrics.get("val/acc")
        if val_acc is not None:
            acc = float(val_acc.item())
        else:
            val_acc_best = trainer.callback_metrics.get("val/acc_best")
            acc = float(val_acc_best.item()) if val_acc_best is not None else 0.0

        accuracies.append(acc)
        print(f"Fold {fold + 1} val accuracy: {acc:.4f}")

    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))
    print(f"\nK-Fold Results: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Per-fold: {accuracies}")

    return mean_acc, std_acc, accuracies


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """Custom learning rate scheduler for transformers"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_onecycle_schedule(optimizer, num_training_steps, max_lr, pct_start=0.2, last_epoch=-1):
    """OneCycleLR wrapper matching the (optimizer, num_training_steps) call signature used by configure_optimizers."""
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=num_training_steps,
        pct_start=pct_start,
        last_epoch=last_epoch,
    )
