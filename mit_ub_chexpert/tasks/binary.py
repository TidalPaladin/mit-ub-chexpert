from typing import Any, Dict, List, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from deep_helpers.structs import State
from deep_helpers.tasks import Task
from einops.layers.torch import Rearrange
from mit_ub.model import BACKBONES, ViT
from torch import Tensor
from torchvision.ops import sigmoid_focal_loss


class BinaryChexpert(Task):
    """

    Args:
        backbone: Name of the backbone to use for the task.
        optimizer_init: Initial configuration for the optimizer.
        lr_scheduler_init: Initial configuration for the learning rate scheduler.
        lr_interval: Frequency of learning rate update. Can be 'step' or 'epoch'.
        lr_monitor: Quantity to monitor for learning rate scheduler.
        named_datasets: If True, datasets are named, else they are indexed by integers.
        checkpoint: Path to the checkpoint file to initialize the model.
        strict_checkpoint: If True, the model must exactly match the checkpoint.
        log_train_metrics_interval: Interval (in steps) at which to log training metrics.
        log_train_metrics_on_epoch: If True, log training metrics at the end of each epoch.
        parameter_groups: List of parameter groups to use for the optimizer.
    """

    def __init__(
        self,
        backbone: str,
        focal_loss: bool = False,
        optimizer_init: Dict[str, Any] = {},
        lr_scheduler_init: Dict[str, Any] = {},
        lr_interval: str = "epoch",
        lr_monitor: str = "train/total_loss_epoch",
        named_datasets: bool = False,
        checkpoint: Optional[str] = None,
        strict_checkpoint: bool = True,
        log_train_metrics_interval: int = 1,
        log_train_metrics_on_epoch: bool = False,
        parameter_groups: List[Dict[str, Any]] = [],
    ):
        super().__init__(
            optimizer_init,
            lr_scheduler_init,
            lr_interval,
            lr_monitor,
            named_datasets,
            checkpoint,
            strict_checkpoint,
            log_train_metrics_interval,
            log_train_metrics_on_epoch,
            parameter_groups,
        )

        self.backbone = cast(ViT, self.prepare_backbone(backbone))
        dim = self.backbone.dim
        self.finding_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange("b c () () -> b c"),
            nn.LayerNorm(dim),
            nn.Dropout(0.1),
            nn.Linear(dim, 1),
        )
        self.criterion = nn.BCEWithLogitsLoss(reduction="none") if not focal_loss else sigmoid_focal_loss
        self.save_hyperparameters()

    def prepare_backbone(self, name: str) -> nn.Module:
        backbone = BACKBONES.get(name).instantiate_with_metadata().fn
        assert isinstance(backbone, nn.Module)
        return backbone

    def create_metrics(self, state: State, **kwargs) -> tm.MetricCollection:
        return tm.MetricCollection(
            {
                "auroc": tm.AUROC(task="binary"),
                "acc": tm.Accuracy(task="binary"),
            }
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.backbone(x)
        cls = self.finding_head(x)
        return {"finding": cls.view(-1, 1)}

    def step(
        self, batch: Any, batch_idx: int, state: State, metrics: Optional[tm.MetricCollection] = None
    ) -> Dict[str, Any]:
        x = batch["img"]
        y = batch["finding"]

        # forward pass
        result = self(x)

        # compute loss
        pred_logits = cast(Tensor, result["finding"].flatten())

        # Build ground truth and compute loss
        mask = y != -1
        if mask.any():
            loss = F.binary_cross_entropy_with_logits(pred_logits[mask], y[mask].float())
        else:
            loss = y.new_tensor(0.0, requires_grad=True)

        with torch.no_grad():
            pred = pred_logits.sigmoid()

        # log metrics
        with torch.no_grad():
            for metric in (metrics or {}).values():
                _pred = pred[mask]
                _label = y[mask].long()
                metric.update(_pred, _label)

        output = {
            "finding_score": pred.detach(),
            "log": {
                "loss_finding": loss,
            },
        }

        return output

    @torch.no_grad()
    def predict_step(self, batch: Any, *args, **kwargs) -> Dict[str, Any]:
        result = self(batch["img"])
        pred_logits = cast(Tensor, result["finding"].flatten())
        return {
            "finding_score": pred_logits.sigmoid(),
        }
