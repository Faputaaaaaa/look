"""GT-aware ROI proposal sampling for multi-instance training.

This is a small optional extension to the project's own two-stage detector,
not a replacement for the RPN, backbone, or ROI head.
"""
from __future__ import annotations

import torch


def append_gt_proposals(proposals: torch.Tensor, gt_boxes: torch.Tensor):
    """Add detached GT proposals for training/validation *loss only*.

    Returns (proposals_with_gt, number_of_original_rpn_proposals).
    Inference must never use this function.
    """
    if proposals.ndim != 2 or proposals.shape[1] != 4:
        raise ValueError("proposals must be [N, 4]")
    if gt_boxes.ndim != 2 or gt_boxes.shape[1] != 4:
        raise ValueError("gt_boxes must be [M, 4]")
    if proposals.device != gt_boxes.device:
        raise ValueError("proposal and GT devices must match")
    if gt_boxes.numel() and (not bool(torch.isfinite(gt_boxes).all())):
        raise ValueError("GT boxes contain NaN/Inf")
    if gt_boxes.numel() and (not bool(torch.all(gt_boxes[:, 2:] > gt_boxes[:, :2]))):
        raise ValueError("GT boxes contain nonpositive area")
    n_rpn = proposals.size(0)
    return torch.cat([proposals, gt_boxes.detach().to(proposals.dtype)], dim=0), n_rpn


def sample_gt_aware_rois(
    roi_orig_indices: torch.Tensor,
    roi_pos_mask: torch.Tensor,
    roi_neg_mask: torch.Tensor,
    original_rpn_count: int,
    gt_count: int,
    max_positive: int = 32,
    negative_per_positive: int = 3,
):
    """Return ROI-feature indices (positive, negative), covering every GT.

    ``roi_orig_indices`` is the map from level-grouped ROI Align output to the
    original ``[RPN proposals, GT proposals]`` order. This is essential: the
    ROI Align implementation groups P2/P3/P4 and **reorders** its output.

    Extra positives are drawn randomly; GT-injected positives are mandatory.
    Negatives are strictly taken from the valid ROI negative mask, never from
    an arbitrary proposal with an unknown match quality.
    """
    if gt_count < 0 or original_rpn_count < 0:
        raise ValueError("counts must be nonnegative")
    if max_positive < gt_count:
        raise ValueError(f"{gt_count} GT instances exceed max_positive={max_positive}; increase the limit")
    if max_positive < 1 or negative_per_positive < 0:
        raise ValueError("invalid ROI sampling limits")
    if roi_orig_indices.ndim != 1:
        raise ValueError("roi_orig_indices must be a vector")
    if roi_orig_indices.shape != roi_pos_mask.shape or roi_orig_indices.shape != roi_neg_mask.shape:
        raise ValueError("ROI masks and index map must have equal lengths")
    if roi_pos_mask.dtype != torch.bool or roi_neg_mask.dtype != torch.bool:
        raise ValueError("ROI masks must be boolean")
    if bool((roi_pos_mask & roi_neg_mask).any()):
        raise ValueError("ROI cannot be both positive and negative")
    if roi_orig_indices.numel():
        if bool((roi_orig_indices < 0).any()) or bool((roi_orig_indices >= original_rpn_count + gt_count).any()):
            raise ValueError("ROI indices out of proposal bounds")

    device = roi_orig_indices.device
    mandatory = torch.where(roi_orig_indices >= original_rpn_count)[0]
    if mandatory.numel() != gt_count:
        raise ValueError("Every GT must have exactly one ROI Align feature")
    gt_ids = roi_orig_indices[mandatory] - original_rpn_count
    if not torch.equal(torch.sort(gt_ids).values, torch.arange(gt_count, device=device)):
        raise ValueError("GT ROI proposals must occur exactly once")
    if mandatory.numel() and not bool(roi_pos_mask[mandatory].all()):
        raise ValueError("Injected GT proposals must be classified as positives")

    pool = torch.where(roi_pos_mask & (roi_orig_indices < original_rpn_count))[0]
    n_extra = min(int(pool.numel()), max_positive - gt_count)
    if n_extra:
        pool = pool[torch.randperm(pool.numel(), device=device)[:n_extra]]
    else:
        pool = pool[:0]
    positives = torch.cat((mandatory, pool))

    neg_pool = torch.where(roi_neg_mask)[0]
    n_neg = min(int(neg_pool.numel()), int(positives.numel()) * negative_per_positive)
    if n_neg:
        negatives = neg_pool[torch.randperm(neg_pool.numel(), device=device)[:n_neg]]
    else:
        negatives = neg_pool[:0]
    return positives, negatives
