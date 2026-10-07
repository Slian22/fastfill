"""Tensor geometry in meters, right-handed Z-up, full extents, bottom-center."""
from __future__ import annotations

import math

import torch


def wrap_yaw(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def encode_yaw(yaw: torch.Tensor, bins: int):
    if bins < 1 or not torch.isfinite(yaw).all():
        raise ValueError("yaw encoding needs finite radians and positive bin count")
    delta = 2 * math.pi / bins
    indices = torch.floor(torch.remainder(yaw + delta / 2, 2 * math.pi) / delta).long() % bins
    # Bin centres in the input dtype: long * python float would round to float32.
    residual = wrap_yaw(yaw - indices.to(yaw.dtype) * delta) / (delta / 2)
    tolerance = 32 * torch.finfo(yaw.dtype).eps * max(1, bins)
    if ((residual < -1 - tolerance) | (residual > 1 + tolerance)).any():
        raise ValueError("yaw residual outside encoding tolerance")
    return indices, residual.clamp(-1, 1)


def decode_yaw(logits: torch.Tensor, residuals: torch.Tensor):
    if logits.shape != residuals.shape or logits.shape[-1] < 1:
        raise ValueError("yaw logits and residual shape mismatch")
    k = logits.argmax(-1)
    selected = residuals.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    # No clipping: the trained tanh head constrains residuals, recorded in model config.
    # Bin centres need at least float32: a bf16 residual (autocast) would quantize them to ~0.03 rad.
    dtype = torch.promote_types(selected.dtype, torch.float32)
    return wrap_yaw(k.to(dtype) * (2 * math.pi / logits.shape[-1]) + selected.to(dtype) * (math.pi / logits.shape[-1]))


def bottom_to_center(position: torch.Tensor, size: torch.Tensor):
    return position + torch.stack((torch.zeros_like(size[..., 2]), torch.zeros_like(size[..., 2]), size[..., 2] / 2), -1)


def bev_corners(position, size, yaw):
    signs = size.new_tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    local = size[..., None, :2] * signs / 2
    c, s = yaw.cos()[..., None], yaw.sin()[..., None]
    x = local[..., 0] * c - local[..., 1] * s
    y = local[..., 0] * s + local[..., 1] * c
    return torch.stack((x, y), -1) + position[..., None, :2]


def box_corners(position, size, yaw):
    xy = bev_corners(position, size, yaw)
    low = position[..., None, 2:3].expand(*xy.shape[:-1], 1)
    high = low + size[..., None, 2:3]
    return torch.cat((torch.cat((xy, low), -1), torch.cat((xy, high), -1)), -2)
