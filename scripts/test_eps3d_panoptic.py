#!/usr/bin/env python3
"""
EPS3D Panoptic Testing on ScanNet

Evaluates 8-class semantic segmentation and instance segmentation.
"""
from data_utils.path_manager import init_all_submodules
init_all_submodules()

import os
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import json

import torch
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '../..'))
sys.path.append(PROJECT_ROOT)

# EPS3D model imports
from src.model.model.eps3d_panoptic import EPS3DPanoptic
from src.model.model.eps3d import EncoderEPS3DCfg
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.misc.image_io import save_image

# Data loading
from data_utils.testdata_panoptic import PanopticTestDataset as TestDataset
import dust3r.datasets
dust3r.datasets.TestDataset = TestDataset

# Metrics
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics import Accuracy
from torchmetrics.segmentation import MeanIoU
import lpips

from dust3r.datasets import get_data_loader
from dust3r.losses import *  # noqa: F401

import croco.utils.misc as misc
from lseg import LSegFeatureExtractor
from safetensors.torch import load_file as load_safetensors

import csv
import time
import colorsys
import hdbscan
from scipy.optimize import linear_sum_assignment
import PIL.Image as PILImage


# ==============================================================================
# Utility functions
# ==============================================================================

def id2rgb(id, max_num_obj=256):
    """Convert instance ID to RGB color."""
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)
    s = 0.5 + (id % 2) * 0.5
    l = 0.5
    rgb = np.zeros((3,), dtype=np.uint8)
    if id == 0:
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r * 255), int(g * 255), int(b * 255)
    return rgb


def get_semantic_color_map():
    """Get 8-class + background color map."""
    colors = torch.tensor([
        [59, 118, 175],   # background
        [238, 133, 54],   # wall
        [82, 158, 63],    # floor
        [197, 58, 50],    # ceiling
        [133, 89, 78],    # chair
        [213, 125, 191],  # table
        [127, 127, 127],  # sofa
        [188, 189, 69],   # bed
        [89, 187, 204],   # other
    ], dtype=torch.float32) / 255.0
    return colors


def segmentation_to_color_semantic(seg_map, colors=None):
    """Convert semantic segmentation to colored image."""
    if colors is None:
        colors = get_semantic_color_map()
    h, w = seg_map.shape
    colored = torch.zeros(3, h, w)
    for class_id in range(colors.shape[0]):
        mask = (seg_map == class_id)
        if mask.any():
            colored[:, mask] = colors[class_id].unsqueeze(1)
    return colored


def segmentation_to_color_instance(seg_map, max_instances=256):
    """Convert instance segmentation to colored image."""
    h, w = seg_map.shape
    colored = torch.zeros(3, h, w)
    for inst_id in torch.unique(seg_map):
        inst_id_val = int(inst_id.item())
        if inst_id_val == 0:
            continue
        mask = (seg_map == inst_id_val)
        if mask.any():
            rgb = id2rgb(inst_id_val % max_instances)
            color = torch.tensor(rgb, dtype=torch.float32) / 255.0
            colored[:, mask] = color.unsqueeze(1)
    return colored


def rotation_6d_to_matrix(d6):
    """Convert 6D rotation to matrix."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def my_crop(image):
    """Crop borders for metric computation."""
    return image[..., 25:-25, 25:-25]


def save_metrics_to_csv(results_dir, dataset_name, metrics):
    """Save metrics to CSV file."""
    csv_file = os.path.join(results_dir, 'test_metrics.csv')
    file_exists = os.path.exists(csv_file)

    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [
                'dataset', 'mIoU_semantic', 'Acc', 'PSNR', 'SSIM', 'LPIPS',
                'instance_mIoU', 'instance_Precision@0.5', 'instance_Recall@0.5',
                'instance_F-score', 'avg_inference_time'
            ]
            writer.writerow(header)

        row = [
            dataset_name,
            f"{metrics['mIoU_mean']:.4f}",
            f"{metrics['mAcc_mean']:.4f}",
            f"{metrics['PSNR_mean']:.4f}",
            f"{metrics['SSIM_mean']:.4f}",
            f"{metrics['LPIPS_mean']:.4f}",
            f"{metrics['instance_mIoU_mean']:.4f}",
            f"{metrics['instance_precision_mean']:.4f}",
            f"{metrics['instance_recall_mean']:.4f}",
            f"{metrics['instance_fscore_mean']:.4f}",
            f"{metrics['avg_inference_time']:.4f}",
        ]
        writer.writerow(row)

    print(f"Metrics saved to: {csv_file}")


# ==============================================================================
# LSeg Feature Extractor (lazy initialization)
# ==============================================================================

lseg_feature_extractor = None


def init_lseg_feature_extractor(model_path):
    """Initialize LSeg from model_path/demo_e200.ckpt"""
    global lseg_feature_extractor
    lseg_ckpt_path = os.path.join(model_path, "demo_e200.ckpt")
    print(f">> Loading LSeg from: {lseg_ckpt_path}")
    lseg_feature_extractor = LSegFeatureExtractor.from_pretrained(
        pretrained_model_name_or_path=lseg_ckpt_path,
        half_res=True
    )
    lseg_feature_extractor.eval()
    for param in lseg_feature_extractor.parameters():
        param.requires_grad = False
    return lseg_feature_extractor


# ==============================================================================
# Test Loss / Evaluation Module
# ==============================================================================

class EPS3DPanopticTestLoss(torch.nn.Module):
    """Test evaluation module for EPS3D Panoptic on ScanNet."""

    def __init__(self, num_classes=9, labels=None, save_images=True,
                 label_flag=1, root=None, save_dir='./test_results/',
                 save_instance_features=True):
        super().__init__()

        if labels is None:
            labels = ['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other']

        self.labels = labels
        self.num_classes = len(labels)
        self.label_flag = label_flag
        self.root = root

        # Image quality metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
        self.lpips_vgg = lpips.LPIPS(net='vgg').cuda()
        self.lpips_scores = []

        # Semantic segmentation metrics
        self.miou = MeanIoU(
            num_classes=self.num_classes + 1,
            include_background=False,
            per_class=True,
            input_format="index"
        )
        self.accuracy = Accuracy(
            num_classes=self.num_classes + 1,
            task='multiclass',
            ignore_index=0
        )

        # Instance segmentation metrics
        self.instance_ious = []
        self.instance_precisions = []
        self.instance_recalls = []
        self.instance_fscores = []

        # Visualization
        self.save_images = save_images
        self.save_instance_features = save_instance_features
        if self.save_images:
            self.save_dir = save_dir
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        self.scene_counter = 0
        self.batch_counter = 0
        self.current_batch_dir = None
        self.semantic_colors = get_semantic_color_map()

    def set_scene_batch(self, scene_idx=None, batch_idx=None):
        if scene_idx is not None:
            self.scene_counter = scene_idx
        if batch_idx is not None:
            self.batch_counter = batch_idx

        if self.save_images:
            self.current_batch_dir = Path(self.save_dir) / f"scene_{self.scene_counter}" / f"batch_{self.batch_counter}"
            self.current_batch_dir.mkdir(parents=True, exist_ok=True)
            (self.current_batch_dir / "semantic").mkdir(exist_ok=True)
            (self.current_batch_dir / "instance").mkdir(exist_ok=True)

    def update_lpips(self, pred, gt):
        score = self.lpips_vgg(pred.unsqueeze(0), gt.unsqueeze(0))
        self.lpips_scores.append(score.item())

    def compute_lpips_mean(self):
        if len(self.lpips_scores) == 0:
            return 0.0
        return sum(self.lpips_scores) / len(self.lpips_scores)

    def forward(self, context_views, pred, target_view=None, model=None,
                pose_deltas=None, evaluate=True):
        gaussians = pred['gaussians']
        extrinsics = pred['poses']
        intrinsics = pred['intrinsics']

        rendered_images = []
        rendered_feats = []
        rendered_instance_feats = []
        gt_images = []
        gt_segmentations = []
        gt_instance_maps = []

        identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).cuda()

        for i, target in enumerate(target_view):
            pose_deltas_ = pose_deltas[i].unsqueeze(0)
            dx, drot = pose_deltas_[..., :3], pose_deltas_[..., 3:]
            rot = rotation_6d_to_matrix(drot + identity.expand(pose_deltas_.size(0), -1))
            transform = torch.eye(4, device=pose_deltas_.device).repeat((pose_deltas_.size(0), 1, 1))
            transform[..., :3, :3] = rot
            transform[..., :3, 3] = dx
            target_extrinsics_ = extrinsics[0, i, ...] @ transform.squeeze(0)
            target_extrinsics_ = target_extrinsics_.unsqueeze(0).unsqueeze(0)

            image_shape = (target['true_shape'][0][0], target['true_shape'][0][1])
            near = torch.ones(1, 1, device=gaussians.means.device) * 0.01
            far = torch.ones(1, 1, device=gaussians.means.device) * 100.0
            target_intrinsics = intrinsics[:, i:i+1, ...]

            rendered_output = model.decoder.rendering_fn(
                gaussians, target_extrinsics_, target_intrinsics,
                near, far, image_shape, depth_mode="depth"
            )

            rendered_img = rendered_output.color[0, 0]
            rendered_images.append(rendered_img)

            if evaluate:
                if model.encoder.semantic_fea:
                    semantic_feat = model.decoder.semantic_forward(
                        gaussians, target_extrinsics_, target_intrinsics,
                        near, far, image_shape, depth_mode="depth"
                    )
                    rendered_feats.append(semantic_feat[0])

                if model.encoder.instance_fea:
                    instance_feat = model.decoder.instance_forward(
                        gaussians, target_extrinsics_, target_intrinsics,
                        near, far, image_shape, depth_mode="depth"
                    )
                    rendered_instance_feats.append(instance_feat[0])

            gt_img = target['img'][0] * 0.5 + 0.5
            gt_images.append(gt_img)
            gt_seg = target["labelmap"].long()
            gt_segmentations.append(gt_seg)

            if "instance_labelmap" in target:
                gt_inst = target["instance_labelmap"]
                if isinstance(gt_inst, np.ndarray):
                    gt_inst = torch.from_numpy(gt_inst).long().to(gt_seg.device)
                else:
                    gt_inst = gt_inst.long()
            else:
                gt_inst = torch.zeros_like(gt_seg)
            gt_instance_maps.append(gt_inst)

        rendered_images = torch.stack(rendered_images, dim=0)
        gt_images = torch.stack(gt_images, dim=0)
        gt_segmentations = torch.stack(gt_segmentations, dim=0)
        gt_instance_maps = torch.stack(gt_instance_maps, dim=0)

        image_loss = torch.abs(rendered_images - gt_images).mean()

        if not evaluate:
            return image_loss, {'image_loss': float(image_loss)}

        # Semantic evaluation
        new_height, new_width = 256, 256

        if len(rendered_feats) > 0:
            rendered_feats_cat = torch.cat(rendered_feats, dim=0)
            logits = lseg_feature_extractor.decode_feature(rendered_feats_cat, self.labels)

        # Instance clustering
        all_instance_labels = None
        if len(rendered_instance_feats) > 0:
            all_inst_feats = []
            feat_h, feat_w = None, None
            for inst_feat in rendered_instance_feats:
                if inst_feat.dim() == 4:
                    feat = inst_feat[0]
                else:
                    feat = inst_feat
                feat_h, feat_w = feat.shape[1], feat.shape[2]
                all_inst_feats.append(feat)

            all_inst_stacked = torch.stack(all_inst_feats, dim=0)
            all_inst_stacked = all_inst_stacked / (torch.norm(all_inst_stacked, dim=1, keepdim=True) + 1e-6)

            if self.save_images and self.save_instance_features and self.current_batch_dir is not None:
                self._save_instance_features(all_inst_stacked, gt_instance_maps)

            all_inst_flat = all_inst_stacked.permute(0, 2, 3, 1).reshape(-1, all_inst_stacked.shape[1])

            all_inst_np = all_inst_flat.cpu().numpy()
            n_total = all_inst_np.shape[0]
            n_sample = min(50000, n_total)
            sample_idx = np.random.choice(n_total, n_sample, replace=False)
            sampled_feats = all_inst_np[sample_idx]

            print(f"  [Instance] Global clustering: {n_total} pixels, sampling {n_sample}")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=400, min_samples=1,
                prediction_data=True, allow_single_cluster=True
            )
            clusterer.fit(sampled_feats)

            labels_sampled = clusterer.labels_
            unique_cluster_ids = np.unique(labels_sampled)
            unique_cluster_ids = unique_cluster_ids[unique_cluster_ids >= 0]
            print(f"  [Instance] Found {len(unique_cluster_ids)} clusters")

            if len(unique_cluster_ids) > 0:
                global_centroids = np.stack([
                    clusterer.weighted_cluster_centroid(cluster_id=cid)
                    for cid in unique_cluster_ids
                ])

                centroids_t = torch.FloatTensor(global_centroids).to(all_inst_stacked.device)
                chunksize = 10**6
                all_labels_list = []
                for chunk_start in range(0, n_total, chunksize):
                    chunk = torch.FloatTensor(all_inst_np[chunk_start:chunk_start+chunksize]).to(all_inst_stacked.device)
                    distances = torch.cdist(chunk, centroids_t)
                    chunk_labels = torch.argmin(distances, dim=-1, keepdim=True).cpu().numpy()
                    all_labels_list.append(chunk_labels)

                all_labels_np = np.concatenate(all_labels_list, axis=0)
                all_instance_labels = torch.tensor(all_labels_np, dtype=torch.long).reshape(
                    len(rendered_instance_feats), feat_h, feat_w
                )

        # Per-view metrics
        for i in range(len(rendered_images)):
            ri = len(rendered_images) - 1 - i

            resized_rendered = F.interpolate(
                rendered_images[ri].unsqueeze(0),
                size=(new_height, new_width),
                mode='bilinear', align_corners=False, antialias=True
            ).squeeze(0)

            resized_gt = F.interpolate(
                gt_images[ri].unsqueeze(0),
                size=(new_height, new_width),
                mode='bilinear', align_corners=False, antialias=True
            ).squeeze(0)

            self.psnr.update(my_crop(resized_rendered), my_crop(resized_gt))
            self.ssim.update(my_crop(resized_rendered).unsqueeze(0), my_crop(resized_gt).unsqueeze(0))
            self.update_lpips(resized_rendered, resized_gt)

            if len(rendered_feats) > 0:
                resize_logits = F.interpolate(
                    logits[ri].unsqueeze(0),
                    size=(new_height, new_width),
                    mode='bilinear', align_corners=False, antialias=True
                )

                resized_gt_seg = F.interpolate(
                    gt_segmentations[ri].unsqueeze(0).float(),
                    size=(new_height, new_width),
                    mode='nearest'
                ).long()

                pred_seg = resize_logits.argmax(dim=1, keepdim=True)
                pred_seg = pred_seg.clamp(max=self.num_classes - 1) + 1

                resized_original_gt_segmentation = resized_gt_seg
                resize_pred_segmentations = torch.where(
                    resized_original_gt_segmentation != 0, pred_seg, 0
                )

                self.miou.update(my_crop(resize_pred_segmentations), my_crop(resized_original_gt_segmentation))
                self.accuracy.update(my_crop(resize_pred_segmentations), my_crop(resized_original_gt_segmentation))

            if self.save_images and self.current_batch_dir is not None:
                self._save_semantic_visualization(
                    resized_rendered, resized_gt,
                    resize_pred_segmentations[0, 0] if len(rendered_feats) > 0 else None,
                    resized_original_gt_segmentation[0, 0] if len(rendered_feats) > 0 else None,
                    i
                )

        # Instance evaluation
        if all_instance_labels is not None:
            gt_inst_list = []
            for gi in gt_instance_maps:
                if isinstance(gi, np.ndarray):
                    gi = torch.from_numpy(gi).long()
                else:
                    gi = gi.long()
                if gi.shape[-2:] != (feat_h, feat_w):
                    gi_resized = F.interpolate(
                        gi.float().unsqueeze(0).unsqueeze(0),
                        size=(feat_h, feat_w), mode='nearest'
                    ).long().squeeze()
                else:
                    gi_resized = gi.squeeze()
                gt_inst_list.append(gi_resized)
            gt_instance_stacked = torch.stack(gt_inst_list, dim=0)

            pred_instance_masked = all_instance_labels.clone()
            pred_instance_masked[gt_instance_stacked == 0] = 10000

            gt_label_idx = torch.unique(gt_instance_stacked)
            gt_label_idx = gt_label_idx[gt_label_idx > 0]
            num_gt_mask = len(gt_label_idx)

            pred_label_idx = torch.unique(all_instance_labels)
            num_pred_mask = len(pred_label_idx)

            print(f"  [Instance] Evaluating: {num_gt_mask} GT, {num_pred_mask} pred masks")

            if num_gt_mask > 0 and num_pred_mask > 0:
                iou_matrix = torch.zeros((num_gt_mask, max(num_gt_mask, num_pred_mask)))
                for ii in range(num_gt_mask):
                    for jj in range(num_pred_mask):
                        all_view_ious = []
                        for vi in range(len(gt_inst_list)):
                            gt_binary = (gt_instance_stacked[vi] == gt_label_idx[ii])
                            pred_binary = (pred_instance_masked[vi] == pred_label_idx[jj])
                            if gt_binary.sum() == 0:
                                continue
                            intersection = (gt_binary & pred_binary).sum()
                            union = (gt_binary | pred_binary).sum()
                            if union > 0:
                                all_view_ious.append((intersection.float() / union.float()).item())
                        if len(all_view_ious) > 0:
                            iou_matrix[ii, jj] = sum(all_view_ious) / len(all_view_ious)

                row_ind, col_ind = linear_sum_assignment(iou_matrix.numpy(), maximize=True)
                paired_iou = iou_matrix[row_ind, col_ind]
                mean_iou = paired_iou.mean().item()

                num_hit_05 = (paired_iou > 0.5).sum().item()
                precision_05 = num_hit_05 / num_pred_mask if num_pred_mask > 0 else 0.0
                recall_05 = num_hit_05 / num_gt_mask if num_gt_mask > 0 else 0.0
                f_score = 2 * precision_05 * recall_05 / (precision_05 + recall_05 + 1e-6)

                print(f"  [Instance] mIoU: {mean_iou:.4f}, P@0.5: {precision_05:.4f}, R@0.5: {recall_05:.4f}")

                self.instance_ious.append(mean_iou)
                self.instance_precisions.append(precision_05)
                self.instance_recalls.append(recall_05)
                self.instance_fscores.append(f_score)

            if self.save_images and self.current_batch_dir is not None:
                for vi in range(all_instance_labels.shape[0]):
                    self._save_instance_visualization(
                        all_instance_labels[vi], gt_instance_stacked[vi], vi
                    )

        return image_loss, {'image_loss': float(image_loss)}

    def _save_semantic_visualization(self, rendered_img, gt_img, pred_seg, gt_seg, view_idx):
        if self.current_batch_dir is None:
            return

        sem_dir = self.current_batch_dir / "semantic"
        rendered_img_clamped = torch.clamp(rendered_img, 0, 1)
        gt_img_clamped = torch.clamp(gt_img, 0, 1)

        save_image(rendered_img_clamped, sem_dir / f"rgb_pred_{view_idx:02d}.png")
        save_image(gt_img_clamped, sem_dir / f"rgb_gt_{view_idx:02d}.png")

        if pred_seg is not None and gt_seg is not None:
            pred_color = segmentation_to_color_semantic(pred_seg.cpu(), self.semantic_colors)
            gt_color = segmentation_to_color_semantic(gt_seg.cpu(), self.semantic_colors)
            save_image(pred_color, sem_dir / f"seg_pred_{view_idx:02d}.png")
            save_image(gt_color, sem_dir / f"seg_gt_{view_idx:02d}.png")

            seg_id_dir = self.current_batch_dir / "segmentation_id"
            seg_id_dir.mkdir(exist_ok=True)
            np.save(seg_id_dir / f"pred_seg{view_idx:02d}.npy", pred_seg.cpu().numpy())
            np.save(seg_id_dir / f"gt_seg{view_idx:02d}.npy", gt_seg.cpu().numpy())

    def _save_instance_visualization(self, pred_instance, gt_instance, view_idx):
        if self.current_batch_dir is None:
            return

        inst_dir = self.current_batch_dir / "instance"
        pred_color = segmentation_to_color_instance(pred_instance.cpu())
        gt_color = segmentation_to_color_instance(gt_instance.cpu())
        save_image(pred_color, inst_dir / f"instance_pred_{view_idx:02d}.png")
        save_image(gt_color, inst_dir / f"instance_gt_{view_idx:02d}.png")

        mask_dir = self.current_batch_dir / "mask_npy"
        mask_dir.mkdir(exist_ok=True)
        np.save(mask_dir / f"instance_mask_prediction_{view_idx}.npy", pred_instance.cpu().numpy())
        np.save(mask_dir / f"instance_mask_GT_{view_idx}.npy", gt_instance.cpu().numpy())

    def _save_instance_features(self, inst_feats_normed, gt_instance_maps):
        """Save normalized instance features and GT masks for offline clustering experiments."""
        if self.current_batch_dir is None:
            return
        feat_dir = self.current_batch_dir / "instance_feats"
        feat_dir.mkdir(exist_ok=True)
        np.save(feat_dir / "inst_feats.npy", inst_feats_normed.cpu().numpy())
        for vi, gi in enumerate(gt_instance_maps):
            gi_np = gi.cpu().numpy() if torch.is_tensor(gi) else gi
            np.save(feat_dir / f"gt_instance_{vi}.npy", gi_np)


# ==============================================================================
# Main inference loop
# ==============================================================================

def eps3d_panoptic_loss_of_one_batch(batch_id, batch, model, criterion, device,
                                     symmetrize_batch=False, use_amp=False,
                                     ret=None, total_time=None, save_dir=None,
                                     eval_context=False):
    """Process one batch for panoptic evaluation."""
    context_views = []
    target_views = []

    assert len(batch) != 0

    if len(batch) < 5:
        for i in range(len(batch)):
            if i < 2:
                context_views.append(batch[i])
            else:
                target_views.append(batch[i])
    else:
        split_len = (len(batch) + 1) // 2
        for i in range(len(batch)):
            if i < split_len:
                context_views.append(batch[i])
            else:
                target_views.append(batch[i])

    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape',
                       'rng', 'scene_id', 'view_idx', 'instance_labelmap'])
    for view in batch:
        for name in view.keys():
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    actual_model = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        images = []
        for view_idx, view in enumerate(context_views):
            img = view['img'][0]
            images.append(img)
            if save_dir is not None:
                save_path = Path(f"{save_dir}/scene_0/batch_{batch_id}/context_views/")
                save_path.mkdir(parents=True, exist_ok=True)
                save_image(torch.clamp((img + 1) * 0.5, 0, 1), save_path / f"context_{view_idx:02d}.jpg")

        ctx_images = torch.stack(images, dim=0).unsqueeze(0).to(device)
        ctx_images = (ctx_images + 1) * 0.5

        start_time = time.time()
        encoder_output = actual_model.encoder(ctx_images, global_step=0, visualization_dump={})
        encoder_time = time.time() - start_time

        gaussians, pred_context_pose = encoder_output.gaussians, encoder_output.pred_context_pose

        if eval_context:
            pred = {
                'gaussians': gaussians,
                'poses': pred_context_pose['extrinsic'],
                'intrinsics': pred_context_pose['intrinsic'].float(),
            }
            render_views = context_views
        else:
            tgt_images = []
            for view in target_views:
                img = view['img'][0]
                tgt_images.append(img)

            tgt_images = torch.stack(tgt_images, dim=0).unsqueeze(0).to(device)
            tgt_images = (tgt_images + 1) * 0.5

            b, v, _, h, w = tgt_images.shape
            num_context_view = ctx_images.shape[1]

            vggt_input_image = torch.cat((ctx_images, tgt_images), dim=1).to(torch.bfloat16)
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=False, dtype=torch.bfloat16):
                aggregated_tokens_list, patch_start_idx = model.encoder.aggregator(
                    vggt_input_image,
                    intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx
                )

            with torch.amp.autocast('cuda', enabled=False):
                fp32_tokens = [token.float() for token in aggregated_tokens_list]
                pred_all_pose_enc = model.encoder.camera_head(fp32_tokens)[-1]
                pred_all_extrinsic, pred_all_intrinsic = pose_encoding_to_extri_intri(
                    pred_all_pose_enc, vggt_input_image.shape[-2:]
                )

            extrinsic_padding = torch.tensor(
                [0, 0, 0, 1], device=pred_all_extrinsic.device, dtype=pred_all_extrinsic.dtype
            ).view(1, 1, 1, 4).repeat(b, vggt_input_image.shape[1], 1, 1)
            pred_all_extrinsic = torch.cat([pred_all_extrinsic, extrinsic_padding], dim=2).inverse()

            pred_all_intrinsic[:, :, 0] = pred_all_intrinsic[:, :, 0] / w
            pred_all_intrinsic[:, :, 1] = pred_all_intrinsic[:, :, 1] / h

            pred_all_context_extrinsic = pred_all_extrinsic[:, :num_context_view]
            pred_all_target_extrinsic = pred_all_extrinsic[:, num_context_view:]
            pred_all_target_intrinsic = pred_all_intrinsic[:, num_context_view:]

            scale_factor = (pred_context_pose['extrinsic'][:, :, :3, 3].mean() /
                           pred_all_context_extrinsic[:, :, :3, 3].mean())
            pred_all_target_extrinsic[..., :3, 3] *= scale_factor

            pred = {
                'gaussians': gaussians,
                'poses': pred_all_target_extrinsic,
                'intrinsics': pred_all_target_intrinsic.float(),
            }
            render_views = target_views

    with torch.amp.autocast('cuda', enabled=False):
        pose_embeds = torch.zeros([len(render_views), 9], requires_grad=False, device="cuda")

        with torch.no_grad():
            loss_value, loss_details = criterion(
                context_views, pred, target_view=render_views,
                model=actual_model, pose_deltas=pose_embeds, evaluate=True
            )

    if total_time is not None:
        total_time.append(encoder_time)

    result = dict(
        view=context_views,
        target_view=target_views,
        pred=pred,
        loss=(loss_value, loss_details)
    )
    return result[ret] if ret else result


# ==============================================================================
# Argument parser
# ==============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser('EPS3D Panoptic Testing on ScanNet', add_help=False)
    parser.add_argument('--test_dataset', default='[None]', type=str, help="testing set")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--amp', type=int, default=0, choices=[0, 1])
    parser.add_argument('--print_freq', default=1, type=int)
    parser.add_argument('--num_views_value', default=8, type=int)
    parser.add_argument('--test_results_dir', default='./test_results/', type=str)
    parser.add_argument('--model_path', required=True, type=str,
                       help="path to model directory (config.json, model.safetensors, demo_e200.ckpt)")
    parser.add_argument('--label_flag', default=1, type=int)
    parser.add_argument('--root', default='./', type=str)
    parser.add_argument('--max_batches', default=-1, type=int)
    parser.add_argument('--use_sem2ins', action='store_true', default=True)
    parser.add_argument('--skip_instance_features', action='store_true',
                       help='Skip saving dense instance feature dumps under instance_feats/')
    parser.add_argument('--eval_context', action='store_true',
                       help='Evaluate on context views instead of target views')
    return parser


# ==============================================================================
# Model building
# ==============================================================================

def build_panoptic_model(args, device):
    """Build EPS3DPanoptic model."""
    model_path = args.model_path

    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    config['encoder_cfg']['instance_fea'] = True
    config['encoder_cfg']['gs_instance_fea'] = 32
    if args.use_sem2ins:
        config['encoder_cfg']['use_sem2ins'] = True
        print(">> Sem2Ins fusion module ENABLED")

    encoder_cfg = EncoderEPS3DCfg(**{
        k: v for k, v in config['encoder_cfg'].items()
        if k in EncoderEPS3DCfg.__dataclass_fields__
    })
    decoder_cfg = DecoderSplattingCUDACfg(**{
        k: v for k, v in config['decoder_cfg'].items()
        if k in DecoderSplattingCUDACfg.__dataclass_fields__
    })

    model = EPS3DPanoptic(
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        output_mode='p'
    )

    print(f">> Loading weights from: {model_path}/model.safetensors")
    safetensors_path = os.path.join(model_path, "model.safetensors")
    all_weights = load_safetensors(safetensors_path)
    missing_keys, unexpected_keys = model.load_state_dict(all_weights, strict=False)
    print(f"  Loaded {len(all_weights)} keys, Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")

    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    model.eval()

    return model


# ==============================================================================
# Test epoch
# ==============================================================================

def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader: {dataset}')
    loader = get_data_loader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        pin_mem=True, shuffle=not test, drop_last=not test
    )
    print(f"{split} dataset length: {len(loader)}")
    return loader


def test_one_epoch(model, criterion, data_loader, device, epoch, num_views_value, args,
                   log_writer=None, prefix='test'):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = f'Test Epoch: [{epoch}]'

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(1)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(1)

    total_time = []

    for batch_id, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        if args.max_batches > 0 and batch_id >= args.max_batches:
            print(f"\n--- Reached max_batches={args.max_batches}, stopping ---")
            break

        if hasattr(criterion, 'set_scene_batch'):
            criterion.set_scene_batch(scene_idx=0, batch_idx=batch_id)

        print(f"\n--- Processing batch {batch_id} ---")

        res = eps3d_panoptic_loss_of_one_batch(
            batch_id, batch, model, criterion, device,
            symmetrize_batch=False, use_amp=bool(args.amp),
            total_time=total_time, save_dir=args.test_results_dir,
            eval_context=args.eval_context
        )

        loss_tuple = res['loss']
        loss_value, loss_details = loss_tuple
        metric_logger.update(loss=float(loss_value), **loss_details)

        del res, batch
        torch.cuda.empty_cache()

    # Print final metrics
    print("\n" + "=" * 70)
    print("EPS3D PANOPTIC EVALUATION RESULTS")
    print("=" * 70)

    psnr_value = criterion.psnr.compute()
    ssim_value = criterion.ssim.compute()
    lpips_value = criterion.compute_lpips_mean()
    miou_value = criterion.miou.compute()
    miou_value = torch.clamp(miou_value, 0, 1)
    miou_mean = miou_value.mean()
    acc_value = criterion.accuracy.compute()
    acc_mean = acc_value.mean() if acc_value.dim() > 0 else acc_value

    print(f"\n[Semantic Segmentation - 8 Classes]")
    print(f"  mIoU:     {miou_mean:.4f}")
    print(f"  Accuracy: {acc_mean:.4f}")
    print(f"  Per-class: {miou_value.tolist()}")

    print(f"\n[Image Quality]")
    print(f"  PSNR:  {psnr_value:.4f}")
    print(f"  SSIM:  {ssim_value:.4f}")
    print(f"  LPIPS: {lpips_value:.4f}")

    if len(criterion.instance_ious) > 0:
        inst_miou_mean = np.mean(criterion.instance_ious)
        inst_prec_mean = np.mean(criterion.instance_precisions)
        inst_rec_mean = np.mean(criterion.instance_recalls)
        inst_fscore_mean = np.mean(criterion.instance_fscores)
    else:
        inst_miou_mean = inst_prec_mean = inst_rec_mean = inst_fscore_mean = 0.0

    print(f"\n[Instance Segmentation]")
    print(f"  Mean IoU:      {inst_miou_mean:.4f}")
    print(f"  Precision@0.5: {inst_prec_mean:.4f}")
    print(f"  Recall@0.5:    {inst_rec_mean:.4f}")
    print(f"  F-score:       {inst_fscore_mean:.4f}")

    avg_time = sum(total_time) / len(total_time) if total_time else 0.0
    print(f"\n[Timing]")
    print(f"  Avg inference time: {avg_time:.4f}s per batch")
    print("=" * 70)

    save_metrics_to_csv(args.test_results_dir, f"EPS3D_Panoptic_{num_views_value}views", {
        'mIoU_mean': float(miou_mean),
        'mAcc_mean': float(acc_mean),
        'PSNR_mean': float(psnr_value),
        'SSIM_mean': float(ssim_value),
        'LPIPS_mean': float(lpips_value),
        'instance_mIoU_mean': inst_miou_mean,
        'instance_precision_mean': inst_prec_mean,
        'instance_recall_mean': inst_rec_mean,
        'instance_fscore_mean': inst_fscore_mean,
        'avg_inference_time': avg_time,
    })

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr)
               for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    return results


# ==============================================================================
# Main
# ==============================================================================

def main(args):
    misc.init_distributed_mode(args)

    num_views_value = args.num_views_value
    print(f"Test results dir: {args.test_results_dir}")
    if args.test_results_dir:
        Path(args.test_results_dir).mkdir(parents=True, exist_ok=True)

    print(f'Job dir: {os.path.dirname(os.path.realpath(__file__))}')
    print(f"{args}".replace(', ', ',\n'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize LSeg
    init_lseg_feature_extractor(args.model_path)

    # Build test dataset
    print(f'Building test dataset: {args.test_dataset}')
    data_loader_test = {
        dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
        for dataset in args.test_dataset.split('+')
    }

    # Build model
    print('>> Building EPS3D Panoptic model')
    model = build_panoptic_model(args, device)
    print(f">> Model output_mode: {model.output_mode}")
    print(f">> Encoder semantic_fea: {model.encoder.semantic_fea}")
    print(f">> Encoder instance_fea: {model.encoder.instance_fea}")

    # Create test criterion
    print('>> Creating panoptic test criterion')
    test_criterion = EPS3DPanopticTestLoss(
        labels=['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other'],
        save_images=True,
        save_dir=args.test_results_dir,
        label_flag=args.label_flag,
        root=args.root,
        save_instance_features=not args.skip_instance_features
    ).to(device)

    # Run evaluation
    test_stats = {}
    for test_name, testset in data_loader_test.items():
        stats = test_one_epoch(
            model, test_criterion, testset,
            device, 1, num_views_value, args=args, prefix=test_name
        )
        test_stats[test_name] = stats

    print("\nAll EPS3D panoptic tests completed!")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
