"""Continuous geometry supervision with field filtering and global counts."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from fastfill.v2.boxes import bev_giou
from fastfill.v2.geometry import decode_yaw, encode_yaw
from fastfill.v2.matching import match_batch, validate_matching_options
from fastfill.v2.objective import local_objective_count, supervision_masks

ELEMENTWISE_TYPES = {"l1", "smooth_l1"}


@dataclass(frozen=True)
class LossConfig:
    position: float = 1.
    size: float = 1.
    yaw_cls: float = 1.
    yaw_reg: float = 1.
    box: float = 0.
    box_operator: str = "bev_oriented_giou_convex_hull"
    hungarian: bool = True
    alpha_position: float = 1.
    alpha_size: float = 1.
    collision: float = 0.
    boundary: float = 0.
    position_type: str = "l1"
    size_type: str = "l1"
    smooth_l1_beta: float = 1.

    def __post_init__(self):
        weights = [getattr(self, key) for key in ("position", "size", "yaw_cls", "yaw_reg", "box", "collision", "boundary")]
        if any(type(value) not in {int, float} or not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("loss weights must be native finite nonnegative numbers")
        if self.position_type not in ELEMENTWISE_TYPES or self.size_type not in ELEMENTWISE_TYPES:
            raise ValueError(f"position_type/size_type must be one of {sorted(ELEMENTWISE_TYPES)}")
        beta = self.smooth_l1_beta
        if type(beta) not in {int, float} or not math.isfinite(beta) or beta <= 0:
            raise ValueError("smooth_l1_beta must be a native finite positive number")
        validate_matching_options(self.hungarian, self.alpha_position, self.alpha_size)


def _elementwise(error, kind, beta):
    if kind == "l1":
        return error.abs()
    return F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none", beta=beta)


def _mean(total, local_count):
    # Every rank must reduce the same dtype, including ranks with no valid
    # float64 box terms. Counts also need no floating-point rounding.
    count = torch.tensor(local_count, dtype=torch.int64, device=total.device)
    world = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(count)
        world = torch.distributed.get_world_size()
    # DDP averages gradients, so compensate once for global count.
    return total * world / count.clamp_min(1), int(count.item())


def _gather(tensor, assignment):
    index = assignment.reshape(*assignment.shape, *([1] * (tensor.ndim - 2)))
    return tensor.gather(1, index.expand_as(tensor))


def _safe_zero(predictions, valid):
    return sum((value[valid] * 0).sum() for name, value in predictions.items()
               if name in {"position_normalized", "size", "yaw_logits", "yaw_residuals"})


class GeometryCriterion(nn.Module):
    def __init__(self, config=LossConfig()):
        super().__init__()
        if config.box and config.box_operator != "bev_oriented_giou_convex_hull":
            raise ValueError("only explicitly named BEV oriented convex-hull GIoU is implemented")
        self.config = config

    def forward(self, predictions, batch):
        mask = batch["slot_mask"]
        for key in ("position_normalized", "size", "yaw_logits", "yaw_residuals"):
            if not torch.isfinite(predictions[key][mask]).all():
                raise ValueError(f"nonfinite valid prediction: {key}")
        if (predictions["size"][mask] <= 0).any():
            raise ValueError("predicted valid sizes must be positive")
        cfg = self.config
        assignment = match_batch(predictions, batch, cfg.hungarian, cfg.alpha_position, cfg.alpha_size)
        targets = {k: _gather(v, assignment) for k, v in batch["targets"].items()}
        validity = {k: _gather(v, assignment) for k, v in batch["validity"].items() if k in {"position", "size", "yaw"}}
        masks = supervision_masks(batch, validity)
        active_local = local_objective_count(batch, cfg, validity)
        zero = _safe_zero(predictions, mask)
        # losses: per-term global means (gradient path). term_sums/term_counts: local
        # unweighted sums and valid counts, for window-level logging by the trainer.
        losses, counts, term_sums, term_counts = {}, {}, {}, {}
        for key, pred_key, target_key in (("position", "position_normalized", "position_normalized"), ("size", "size", "size")):
            learn = ~batch.get(f"fixed_{key}_mask", torch.zeros_like(validity[key]))
            valid = masks[key]
            pred, gt = predictions[pred_key][valid], targets[target_key][valid]
            if not torch.isfinite(gt).all() or (key == "size" and (gt <= 0).any()):
                raise ValueError(f"invalid {key} target marked valid")
            # Select learning coordinates before arithmetic. Fixed coords contribute zero /3.
            coordinate_mask = learn[valid]
            p, t = pred[coordinate_mask], gt[coordinate_mask]
            error = p.log() - t.log() if key == "size" else p - t
            total = _elementwise(error, getattr(cfg, f"{key}_type"), cfg.smooth_l1_beta).sum() / 3
            losses[key], counts[key] = _mean(total + zero, int(valid.sum()))
            term_sums[key], term_counts[key] = total.detach().float(), int(valid.sum())
        angle_valid = masks["yaw"]
        logits, residuals = predictions["yaw_logits"][angle_valid], predictions["yaw_residuals"][angle_valid]
        yaw = targets["yaw"][angle_valid]
        if not torch.isfinite(yaw).all():
            raise ValueError("nonfinite yaw target marked valid")
        orders = batch.get("yaw_symmetry_order", torch.ones_like(mask, dtype=torch.long))
        orders = _gather(orders, assignment)[angle_valid]
        cls_terms, reg_terms = [], []
        for a, r, y, order in zip(logits, residuals, yaw, orders):
            n = int(order)
            if n < 1 or n > 36:
                raise ValueError("yaw symmetry order must be 1..36 with source justification")
            candidates = y + torch.arange(n, device=y.device, dtype=y.dtype) * (2 * math.pi / n)
            bins, gt_residual = encode_yaw(candidates, a.shape[-1])
            cls = F.cross_entropy(a.expand(n, -1), bins, reduction="none")
            reg = F.smooth_l1_loss(r[bins], gt_residual, reduction="none", beta=1.)
            choice = (cfg.yaw_cls * cls + cfg.yaw_reg * reg).detach().argmin()
            cls_terms.append(cls[choice])
            reg_terms.append(reg[choice])
        losses["yaw_cls"], counts["yaw"] = _mean(sum(cls_terms, zero), int(angle_valid.sum()))
        losses["yaw_reg"], _ = _mean(sum(reg_terms, zero), int(angle_valid.sum()))
        for key, terms in (("yaw_cls", cls_terms), ("yaw_reg", reg_terms)):
            term_sums[key] = sum((t.detach().float() for t in terms), torch.zeros((), device=zero.device))
            term_counts[key] = int(angle_valid.sum())
        box_valid = masks["box"]
        box_terms = []
        yaw_pred = decode_yaw(predictions["yaw_logits"], predictions["yaw_residuals"])
        world_pred = predictions["position_normalized"] * batch["scale"][:, None] + batch["origin"][:, None]
        if cfg.box:
            for b, i in box_valid.nonzero().tolist():
                gt_pos = targets["position_normalized"][b, i] * batch["scale"][b] + batch["origin"][b]
                box_terms.append(1 - bev_giou(world_pred[b, i], predictions["size"][b, i], yaw_pred[b, i],
                                             gt_pos, targets["size"][b, i], targets["yaw"][b, i]))
        losses["box"], counts["box"] = _mean(sum(box_terms, zero), len(box_terms))
        term_sums["box"] = sum((t.detach().float() for t in box_terms), torch.zeros((), device=zero.device))
        term_counts["box"] = len(box_terms)
        if cfg.collision or cfg.boundary:
            from fastfill.v2.regularizers import scene_regularizers
            regularizers = scene_regularizers(world_pred, predictions["size"], yaw_pred, batch, cfg)
            term_sums.update(regularizers.pop("sums"))
            term_counts.update(regularizers.pop("counts"))
            losses.update(regularizers)
        else:
            losses.update(collision=zero, boundary=zero)
            term_sums.update(collision=zero.detach().float(), boundary=zero.detach().float())
            term_counts.update(collision=0, boundary=0)
        loss = sum(getattr(cfg, k) * v for k, v in losses.items())
        return {"loss": loss, **losses, "counts": counts, "assignment": assignment,
                "active_objective_count_local": active_local,
                "term_sums": term_sums, "term_counts": term_counts}
