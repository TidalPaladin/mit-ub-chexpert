from copy import copy
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from deep_helpers.structs import State
from mit_ub.tasks.jepa import JEPAWithProbe


class JEPAChexpert(JEPAWithProbe):

    def create_probe_head(self) -> nn.Module:
        return nn.Sequential(nn.LayerNorm(self.backbone.dim), nn.Dropout(0.1), nn.Linear(self.backbone.dim, 1))

    def create_metrics(self, state: State) -> tm.MetricCollection:
        metrics = super().create_metrics(state)
        metrics.add_metrics({"acc": tm.Accuracy(task="binary")})
        metrics.add_metrics({"auroc": tm.AUROC(task="binary")})
        return metrics

    def step_linear_probe(
        self, batch: Dict[str, Any], output: Dict[str, Any], metrics: tm.MetricCollection | None
    ) -> Dict[str, Any]:
        # Forward pass of linear probe using target features
        features = self.get_probe_features_from_output(output)
        assert self.linear_probe is not None
        N = features.shape[0]
        linprobe_logits = self.linear_probe(features.mean(1).view(N, -1)).view(N)
        assert linprobe_logits.requires_grad or not self.training

        # Build ground truth and compute loss
        linprobe_gt = batch["finding"]
        mask = linprobe_gt != -1
        if mask.any():
            linprobe_loss = F.binary_cross_entropy_with_logits(linprobe_logits[mask], linprobe_gt[mask].float())
        else:
            linprobe_loss = linprobe_gt.new_tensor(0.0, requires_grad=True)

        # Logits -> probs
        with torch.no_grad():
            linprobe_probs = torch.sigmoid(linprobe_logits)

        # Compute metrics
        with torch.no_grad():
            if mask.any():
                for name, metric in (metrics or {}).items():
                    if name in {"acc", "auroc"}:
                        metric.update(linprobe_probs[mask], linprobe_gt[mask])

        output = copy(output)
        output["log"]["loss_linprobe"] = linprobe_loss
        return output
