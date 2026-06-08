#!/usr/bin/env python3
"""
PQ (Panoptic Quality) Evaluation Script for EPS3D.

This script computes PQ/SQ/RQ metrics from the evaluation results.
It reads semantic and instance predictions, saves them as .npy files,
and computes panoptic quality metrics.

Usage:
    python evaluate_pq.py --results_dir results_panoptic/panoptic_scannet_8_views_sem2ins/8/
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import csv

from torchmetrics import Accuracy
from torchmetrics.segmentation import MeanIoU


# ============================================================================
# PQ Calculation Functions (from evaluate_filter_organize.py)
# ============================================================================

def _nested_tuple(nested_list: List) -> Tuple:
    return tuple(map(_nested_tuple, nested_list)) if isinstance(nested_list, list) else nested_list


def _totuple(t: torch.Tensor) -> Tuple:
    return _nested_tuple(t.tolist())


def _get_color_areas(img: torch.Tensor) -> Dict[Tuple, torch.Tensor]:
    unique_keys, unique_keys_area = torch.unique(img, dim=0, return_counts=True)
    return dict(zip(_totuple(unique_keys), unique_keys_area))


def _is_set_int(value: Any) -> bool:
    return isinstance(value, Set) and set(map(type, value)).issubset({int})


def _validate_categories(things: Set[int], stuff: Set[int]) -> None:
    if not _is_set_int(things):
        raise ValueError("Expected argument `things` to be of type `Set[int]`")
    if not _is_set_int(stuff):
        raise ValueError("Expected argument `stuff` to be of type `Set[int]`")
    if stuff & things:
        raise ValueError("Expected arguments `things` and `stuffs` to have distinct keys.")


def _validate_inputs(preds: torch.Tensor, target: torch.Tensor) -> None:
    if not isinstance(preds, torch.Tensor):
        raise ValueError("Expected argument `preds` to be of type `torch.Tensor`")
    if not isinstance(target, torch.Tensor):
        raise ValueError("Expected argument `target` to be of type `torch.Tensor`")
    if preds.shape != target.shape:
        raise ValueError("Expected argument `preds` and `target` to have the same shape")


def _get_void_color(things: Set[int], stuff: Set[int]) -> Tuple[int, int]:
    unused_category_id = 1 + max([0] + list(things) + list(stuff))
    return unused_category_id, 0


def _get_category_id_to_continous_id(things: Set[int], stuff: Set[int]) -> Dict[int, int]:
    thing_id_to_continuous_id = {thing_id: idx for idx, thing_id in enumerate(things)}
    stuff_id_to_continuous_id = {stuff_id: idx + len(things) for idx, stuff_id in enumerate(stuff)}
    cat_id_to_continuous_id = {}
    cat_id_to_continuous_id.update(thing_id_to_continuous_id)
    cat_id_to_continuous_id.update(stuff_id_to_continuous_id)
    return cat_id_to_continuous_id


def _isin(arr: torch.Tensor, values: List) -> torch.Tensor:
    return (arr[..., None] == arr.new(values)).any(-1)


def _prepocess_image(
    things: Set[int],
    stuff: Set[int],
    img: torch.Tensor,
    void_color: Tuple[int, int],
    allow_unknown_category: bool,
) -> torch.Tensor:
    img = torch.flatten(img, 0, -2)
    stuff_pixels = _isin(img[:, 0], list(stuff))
    things_pixels = _isin(img[:, 0], list(things))
    img[stuff_pixels, 1] = 0
    if not allow_unknown_category and not torch.all(things_pixels | stuff_pixels):
        raise ValueError("Unknown categories found in preds")
    img[~(things_pixels | stuff_pixels)] = img.new(void_color)
    return img


def _panoptic_quality_update(
    flatten_preds: torch.Tensor,
    flatten_target: torch.Tensor,
    cat_id_to_continuous_id: Dict[int, int],
    void_color: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = flatten_preds.device
    n_categories = len(cat_id_to_continuous_id)
    iou_sum = torch.zeros(n_categories, dtype=torch.double, device=device)
    true_positives = torch.zeros(n_categories, dtype=torch.int, device=device)
    false_positives = torch.zeros(n_categories, dtype=torch.int, device=device)
    false_negatives = torch.zeros(n_categories, dtype=torch.int, device=device)

    pred_areas = _get_color_areas(flatten_preds)
    target_areas = _get_color_areas(flatten_target)
    intersection_matrix = torch.transpose(torch.stack((flatten_preds, flatten_target), -1), -1, -2)
    intersection_areas = _get_color_areas(intersection_matrix)

    pred_segment_matched = set()
    target_segment_matched = set()
    for (pred_color, target_color), intersection in intersection_areas.items():
        if target_color == void_color:
            continue
        if pred_color[0] != target_color[0]:
            continue
        continuous_id = cat_id_to_continuous_id[pred_color[0]]
        pred_area = pred_areas[pred_color]
        target_area = target_areas[target_color]
        pred_void_area = intersection_areas.get((pred_color, void_color), 0)
        void_target_area = intersection_areas.get((void_color, target_color), 0)
        union = pred_area - pred_void_area + target_area - void_target_area - intersection
        iou = intersection / union
        if iou > 0.5:
            pred_segment_matched.add(pred_color)
            target_segment_matched.add(target_color)
            iou_sum[continuous_id] += iou
            true_positives[continuous_id] += 1

    false_negative_colors = set(target_areas.keys()).difference(target_segment_matched)
    false_negative_colors.discard(void_color)
    for target_color in false_negative_colors:
        void_target_area = intersection_areas.get((void_color, target_color), 0)
        if void_target_area / target_areas[target_color] > 0.5:
            continue
        continuous_id = cat_id_to_continuous_id[target_color[0]]
        false_negatives[continuous_id] += 1

    false_positive_colors = set(pred_areas.keys()).difference(pred_segment_matched)
    false_positive_colors.discard(void_color)
    for pred_color in false_positive_colors:
        pred_void_area = intersection_areas.get((pred_color, void_color), 0)
        if pred_void_area / pred_areas[pred_color] > 0.5:
            continue
        continuous_id = cat_id_to_continuous_id[pred_color[0]]
        false_positives[continuous_id] += 1

    return iou_sum, true_positives, false_positives, false_negatives


def _panoptic_quality_compute(
    things: Set[int],
    stuff: Set[int],
    iou_sum: torch.Tensor,
    true_positives: torch.Tensor,
    false_positives: torch.Tensor,
    false_negatives: torch.Tensor,
) -> Dict:
    denominator = (true_positives + 0.5 * false_positives + 0.5 * false_negatives).double()
    panoptic_quality = torch.where(denominator > 0.0, iou_sum / denominator, 0.0)
    segmentation_quality = torch.where(true_positives > 0.0, iou_sum / true_positives, 0.0)
    recognition_quality = torch.where(denominator > 0.0, true_positives / denominator, 0.0)

    metrics = dict(
        all=dict(
            pq=torch.mean(panoptic_quality),
            rq=torch.mean(recognition_quality),
            sq=torch.mean(segmentation_quality),
            n=len(things) + len(stuff),
        ),
        things=dict(
            pq=torch.mean(panoptic_quality[: len(things)]),
            rq=torch.mean(recognition_quality[: len(things)]),
            sq=torch.mean(segmentation_quality[: len(things)]),
            n=len(things),
        ),
        stuff=dict(
            pq=torch.mean(panoptic_quality[len(things):]),
            rq=torch.mean(recognition_quality[len(things):]),
            sq=torch.mean(segmentation_quality[len(things):]),
            n=len(stuff),
        ),
    )
    return metrics


def get_non_robust_classes_for_image(pred_sem, target_sem, robustness_thres=0.005):
    pred_unique, pred_counts = pred_sem.unique(return_counts=True)
    target_unique, target_counts = target_sem.unique(return_counts=True)
    pred_perc = pred_counts / pred_counts.sum()
    target_perc = target_counts / target_counts.sum()
    return set(
        pred_unique[pred_perc < robustness_thres].tolist()
        + target_unique[target_perc < robustness_thres].tolist()
    )


def panoptic_quality(
    preds: torch.Tensor,
    target: torch.Tensor,
    things: Set[int],
    stuff: Set[int],
    allow_unknown_preds_category: bool = False,
    robust: float = 0.005,
) -> Tuple[Any, Any, Any]:
    unused_classes = things.union(stuff) - set(preds[..., 0].unique().tolist() + target[..., 0].unique().tolist())
    non_robust_classes = get_non_robust_classes_for_image(preds[..., 0], target[..., 0], robust)
    things = things - unused_classes - non_robust_classes
    stuff = stuff - unused_classes - non_robust_classes

    if len(things) == 0 and len(stuff) == 0:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    _validate_categories(things, stuff)
    _validate_inputs(preds, target)
    void_color = _get_void_color(things, stuff)
    cat_id_to_continuous_id = _get_category_id_to_continous_id(things, stuff)
    flatten_preds = _prepocess_image(things, stuff, preds, void_color, allow_unknown_preds_category)
    flatten_target = _prepocess_image(things, stuff, target, void_color, True)
    iou_sum, true_positives, false_positives, false_negatives = _panoptic_quality_update(
        flatten_preds, flatten_target, cat_id_to_continuous_id, void_color
    )
    results = _panoptic_quality_compute(things, stuff, iou_sum, true_positives, false_positives, false_negatives)
    return results["all"]["pq"], results["all"]["sq"], results["all"]["rq"]


# ============================================================================
# Configuration
# ============================================================================

# ScanNet 8-class configuration
# Classes: 0=background, 1=wall, 2=floor, 3=ceiling, 4=chair, 5=table, 6=sofa, 7=bed, 8=other
# Stuff: background(0), wall(1), floor(2), ceiling(3), other(8)
# Things: chair(4), table(5), sofa(6), bed(7)
THINGS = {4, 5, 6, 7}  # chair, table, sofa, bed
STUFF = {0, 1, 2, 3, 8}  # background, wall, floor, ceiling, other

RESIZE = (256, 256)
CROP_MARGIN = 25
def my_crop(image):
    """Crop border margins from image."""
    return image[..., CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN]


# ============================================================================
# Main Evaluation
# ============================================================================

def load_npy_files(results_dir, scene_id, batch_id):
    """Load semantic and instance predictions from .npy files."""
    batch_dir = os.path.join(results_dir, f"scene_{scene_id}", f"batch_{batch_id}")
    seg_dir = os.path.join(batch_dir, "segmentation_id")
    mask_dir = os.path.join(batch_dir, "mask_npy")

    if not os.path.exists(seg_dir) or not os.path.exists(mask_dir):
        return None, None, None, None, 0

    # Count frames
    frame_idx = 0
    gt_semantics, pred_semantics = [], []
    gt_instances, pred_instances = [], []

    while True:
        gt_sem_file = os.path.join(seg_dir, f"gt_seg{frame_idx:02d}.npy")
        pred_sem_file = os.path.join(seg_dir, f"pred_seg{frame_idx:02d}.npy")
        gt_inst_file = os.path.join(mask_dir, f"instance_mask_GT_{frame_idx}.npy")
        pred_inst_file = os.path.join(mask_dir, f"instance_mask_prediction_{frame_idx}.npy")

        if not all(os.path.exists(f) for f in [gt_sem_file, pred_sem_file, gt_inst_file, pred_inst_file]):
            break

        gt_semantics.append(np.load(gt_sem_file))
        pred_semantics.append(np.load(pred_sem_file))
        gt_instances.append(np.load(gt_inst_file))
        pred_instances.append(np.load(pred_inst_file))
        frame_idx += 1

    if frame_idx == 0:
        return None, None, None, None, 0

    return gt_semantics, pred_semantics, gt_instances, pred_instances, frame_idx


def evaluate_pq_for_batch(gt_semantics, pred_semantics, gt_instances, pred_instances,
                          num_frames, device='cuda'):
    """Compute PQ/SQ/RQ for a single batch."""
    things = THINGS.copy()
    stuff = STUFF.copy()
    pred_list, target_list = [], []

    # Process each frame
    for frame_idx in range(num_frames):
        # Load and resize
        gt_sem = torch.from_numpy(gt_semantics[frame_idx]).to(device)
        pred_sem = torch.from_numpy(pred_semantics[frame_idx]).to(device)
        gt_inst = torch.from_numpy(gt_instances[frame_idx]).to(device)
        pred_inst = torch.from_numpy(pred_instances[frame_idx]).to(device)

        # Resize to standard size
        gt_sem = F.interpolate(gt_sem.float().unsqueeze(0).unsqueeze(0), size=RESIZE, mode='nearest').squeeze().long()
        pred_sem = F.interpolate(pred_sem.float().unsqueeze(0).unsqueeze(0), size=RESIZE, mode='nearest').squeeze().long()
        gt_inst = F.interpolate(gt_inst.float().unsqueeze(0).unsqueeze(0), size=RESIZE, mode='nearest').squeeze().long()
        pred_inst = F.interpolate(pred_inst.float().unsqueeze(0).unsqueeze(0), size=RESIZE, mode='nearest').squeeze().long()

        # Crop
        gt_sem = my_crop(gt_sem)
        pred_sem = my_crop(pred_sem)
        gt_inst = my_crop(gt_inst)
        pred_inst = my_crop(pred_inst)

        # Build valid mask: exclude background only.
        valid_mask = gt_sem > 0

        if valid_mask.sum() == 0:
            continue

        pred_sem_valid = pred_sem[valid_mask].unsqueeze(-1)
        pred_inst_valid = pred_inst[valid_mask].unsqueeze(-1)
        target_sem_valid = gt_sem[valid_mask].unsqueeze(-1)
        target_inst_valid = gt_inst[valid_mask].unsqueeze(-1)

        pred_ = torch.cat([pred_sem_valid, pred_inst_valid], dim=1).reshape(-1, 2)
        target_ = torch.cat([target_sem_valid, target_inst_valid], dim=1).reshape(-1, 2)
        pred_list.append(pred_)
        target_list.append(target_)

    if len(pred_list) == 0:
        return 0.0, 0.0, 0.0

    pq, sq, rq = panoptic_quality(
        torch.cat(pred_list, dim=0).to(device),
        torch.cat(target_list, dim=0).to(device),
        things, stuff,
        allow_unknown_preds_category=True,
    )

    return pq.item(), sq.item(), rq.item()


def main():
    parser = argparse.ArgumentParser('PQ Evaluation for EPS3D')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to results directory (e.g., results_panoptic/xxx/8/)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    # All test batches (each batch = one ScanNet scene) are saved under scene_0/
    scene_dir = os.path.join(args.results_dir, "scene_0")

    print("=" * 70)
    print("PQ Evaluation for EPS3D")
    print("=" * 70)
    print(f"Results dir: {args.results_dir}")
    print(f"Things classes: {THINGS}")
    print(f"Stuff classes: {STUFF}")

    if not os.path.exists(scene_dir):
        print(f"scene_0 not found in {args.results_dir}")
        return

    batch_dirs = sorted([d for d in os.listdir(scene_dir) if d.startswith("batch_")])
    print(f"Found {len(batch_dirs)} batches (scenes)")

    pq_list, sq_list, rq_list = [], [], []

    for batch_name in batch_dirs:
        batch_id = int(batch_name.split("_")[1])

        gt_sem, pred_sem, gt_inst, pred_inst, num_frames = load_npy_files(
            args.results_dir, 0, batch_id
        )

        if num_frames == 0:
            print(f"  [Batch {batch_id}] No .npy files found, skipping")
            continue

        pq, sq, rq = evaluate_pq_for_batch(
            gt_sem, pred_sem, gt_inst, pred_inst, num_frames, args.device
        )

        print(f"  [Batch {batch_id}] PQ={pq:.4f}, SQ={sq:.4f}, RQ={rq:.4f}")
        pq_list.append(pq)
        sq_list.append(sq)
        rq_list.append(rq)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    if len(pq_list) > 0:
        final_pq = np.mean(pq_list)
        final_sq = np.mean(sq_list)
        final_rq = np.mean(rq_list)
        print(f"PQ: {final_pq:.4f}")
        print(f"SQ: {final_sq:.4f}")
        print(f"RQ: {final_rq:.4f}")

        csv_file = os.path.join(args.results_dir, "pq_metrics.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['PQ', f'{final_pq:.4f}'])
            writer.writerow(['SQ', f'{final_sq:.4f}'])
            writer.writerow(['RQ', f'{final_rq:.4f}'])
        print(f"\nSaved to: {csv_file}")
    else:
        print("No valid results found!")
    print("=" * 70)


if __name__ == "__main__":
    main()
