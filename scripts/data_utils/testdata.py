"""
Test dataset for EPS3D panoptic evaluation on ScanNet.
"""
import os
import os.path as osp
import json
from collections import deque
import numpy as np
import cv2
import torch
import pandas as pd
from PIL import Image

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2

from .scannet_utils import camera_normalization


def process_image(img_path):
    """Process image: resize and center crop to 448x448."""
    img = Image.open(img_path).convert("RGB")
    width, height = img.size

    if width > height:
        new_height = 448
        new_width = int(width * (new_height / height))
    else:
        new_width = 448
        new_height = int(height * (new_width / width))
    img = img.resize((new_width, new_height))

    left = (new_width - 448) // 2
    top = (new_height - 448) // 2
    img = img.crop((left, top, left + 448, top + 448))
    return img


def process_depth_mask_map(depthmap, labelmap, maskmap=None):
    """Process depth and label maps with same resize/crop as images."""
    if maskmap is None:
        maskmap = np.ones_like(depthmap) * 255

    height, width = depthmap.shape[:2]

    if width > height:
        new_height = 448
        new_width = int(width * (new_height / height))
    else:
        new_width = 448
        new_height = int(height * (new_width / width))

    depthmap_resized = cv2.resize(depthmap, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    labelmap_resized = cv2.resize(labelmap, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    maskmap_resized = cv2.resize(maskmap, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    left = (new_width - 448) // 2
    top = (new_height - 448) // 2

    depthmap_cropped = depthmap_resized[top:top+448, left:left+448]
    labelmap_cropped = labelmap_resized[top:top+448, left:left+448]
    maskmap_cropped = maskmap_resized[top:top+448, left:left+448]

    return np.stack([depthmap_cropped, maskmap_cropped, labelmap_cropped], axis=-1)


def create_label_mapping(label_flag, label_path, labels=None):
    """Create label mapping function for semantic segmentation."""
    if labels is None:
        labels = ['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other']

    if label_flag == 1:
        labels = [label.lower() for label in labels]
        df = pd.read_csv(label_path, sep='\t')
        id_to_nyu40class = pd.Series(df['nyu40class'].str.lower().values, index=df['id']).to_dict()
        nyu40class_to_newid = {
            cls: labels.index(cls) + 1 if cls in labels else labels.index('other') + 1
            for cls in set(id_to_nyu40class.values())
        }
        id_to_newid = {id_: nyu40class_to_newid[cls] for id_, cls in id_to_nyu40class.items()}
        print(f"label_flag={label_flag}, classes={labels}")
        return np.vectorize(lambda x: id_to_newid.get(x, labels.index('other') + 1) if x != 0 else 0)
    else:
        df = pd.read_csv(label_path, sep='\t')
        all_labels = sorted(list(set(df['nyu40class'].str.lower().values)))
        id_to_nyu40class = pd.Series(df['nyu40class'].str.lower().values, index=df['id']).to_dict()
        nyu40class_to_newid = {cls: all_labels.index(cls) + 1 for cls in set(id_to_nyu40class.values())}
        id_to_newid = {id_: nyu40class_to_newid[cls] for id_, cls in id_to_nyu40class.items()}
        print(f"label_flag={label_flag}, nyu40class={all_labels}")
        return np.vectorize(lambda x: id_to_newid.get(x, 0) if x != 0 else 0)


class TestDataset(BaseStereoViewDataset):
    """Test dataset for ScanNet scenes."""

    def __init__(self, mask_bg=True, llff_hold=8, test_ids=[1, 4], is_training=False,
                 instance_flag=0, num_views=2, normalize_camera=True,
                 *args, ROOT, label_flag=1, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)

        self.mask_bg = mask_bg
        self.num_views = num_views
        self.label_flag = label_flag
        self.instance_flag = instance_flag

        self.map_func = create_label_mapping(
            self.label_flag,
            os.path.join(ROOT, 'scannetv2-labels.combined.tsv')
        )

        self._load_scenes()

        self.scene_list = list(self.scenes.keys())
        self.invalidate = {scene: {} for scene in self.scene_list}
        self.llff_hold = llff_hold
        self.test_ids = test_ids
        self.is_training = is_training
        self.all_views = self._get_all_views()
        self.views_per_scene = len(self.all_views) // max(len(self.scene_list), 1)

    def _load_scenes(self):
        """Load scene information based on num_views."""
        ignored_scenes = ['scene0696_02', 'scene0692_00', 'scene0693_00']

        if self.num_views == 2:
            with open(osp.join(self.ROOT, f'selected_seqs_{self.split}.json'), 'r') as f:
                self.scenes = json.load(f)
                self.scenes = {k: sorted(v) for k, v in self.scenes.items() if len(v) > 0}
        elif self.num_views in (8, 16, 32):
            self.scenes = {}
            scene_names = sorted([f for f in os.listdir(self.ROOT)
                                if os.path.isdir(os.path.join(self.ROOT, f))])

            for scene_name in scene_names:
                images_dir = osp.join(self.ROOT, scene_name, 'images')
                if not os.path.isdir(images_dir):
                    continue
                images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')])
                indices = np.arange(0, self.num_views * 2, 2).astype(int)
                context_views = [images[i][:-4] for i in indices if i < len(images)]
                filter_images = images[:self.num_views * 2 - 1]
                target_views = [f[:-4] for f in filter_images if f[:-4] not in context_views]
                self.scenes[scene_name] = context_views + target_views
        else:
            raise ValueError(f"Unsupported num_views: {self.num_views}")

        for scene in ignored_scenes:
            self.scenes.pop(scene, None)

    def _get_all_views(self):
        """Get all view combinations for evaluation."""
        views = []
        for scene_id in self.scene_list:
            if not self.is_training:
                if self.num_views == 2:
                    selected = [i for i in range(len(self.scenes[scene_id]))
                               if i % self.llff_hold in self.test_ids]
                    for target_view in selected:
                        src1 = max(target_view - 1, 0)
                        src2 = min(target_view + 1, len(self.scenes[scene_id]) - 1)
                        views.append((scene_id, (target_view, src2, src1)))
                else:
                    selected = list(range(self.num_views * 2 - 1))[::-1]
                    views.append((scene_id, selected))
        return views

    def __len__(self):
        return len(self.all_views)

    def _get_views(self, idx, resolution, rng):
        scene_id, imgs_idxs = self.all_views[idx]
        image_pool = self.scenes[scene_id]

        if resolution not in self.invalidate[scene_id]:
            self.invalidate[scene_id][resolution] = [False] * len(image_pool)

        views = []
        imgs_idxs = deque(imgs_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.pop()

            if self.invalidate[scene_id][resolution][im_idx]:
                direction = 2 * rng.choice(2) - 1
                for offset in range(1, len(image_pool)):
                    tentative = (im_idx + direction * offset) % len(image_pool)
                    if not self.invalidate[scene_id][resolution][tentative]:
                        im_idx = tentative
                        break

            view_idx = image_pool[im_idx]
            impath = osp.join(self.ROOT, scene_id, 'images', f'{view_idx}.jpg')
            meta_path = impath.replace('jpg', 'npz')
            depth_path = impath.replace('images', 'depths').replace('.jpg', '.png')

            if self.instance_flag == 1:
                label_path = impath.replace('images', 'instance').replace('.jpg', '.png')
            else:
                label_path = impath.replace('images', 'labels').replace('.jpg', '.png')

            sam_path = impath.replace('images', 'segment').replace('.jpg', '.png')

            metadata = np.load(meta_path)
            camera_pose = metadata['camera_pose'].astype(np.float32)

            if np.any(np.isinf(camera_pose)):
                self.invalidate[scene_id][resolution][im_idx] = True
                imgs_idxs.append(im_idx)
                continue

            intrinsics = metadata['camera_intrinsics'].astype(np.float32)
            rgb_image = process_image(impath)
            depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
            labelmap = imread_cv2(label_path, cv2.IMREAD_UNCHANGED)
            if os.path.exists(sam_path):
                sam_map = imread_cv2(sam_path, cv2.IMREAD_UNCHANGED)
            else:
                sam_map = np.zeros_like(depthmap, dtype=np.uint8)

            height, width = depthmap.shape[:2]
            labelmap = cv2.resize(labelmap, (width, height), interpolation=cv2.INTER_NEAREST)

            depth_mask_map = np.stack([depthmap, sam_map, labelmap], axis=-1)
            _, original_depth_mask_map, _ = self._crop_resize_if_necessary(
                rgb_image, depth_mask_map, intrinsics, resolution, rng=rng, info=impath
            )

            depth_mask_map = process_depth_mask_map(depthmap, labelmap, sam_map)
            depthmap = depth_mask_map[:, :, 0]
            sam_map = depth_mask_map[:, :, 1].astype(np.uint8)
            labelmap = depth_mask_map[:, :, 2]

            if self.instance_flag == 1:
                labelmap = labelmap.astype(np.uint8)
                original_label = original_depth_mask_map[:, :, 2].astype(np.uint8)
            else:
                labelmap = self.map_func(labelmap)
                original_label = self.map_func(original_depth_mask_map[:, :, 2])

            depthmap = depthmap.astype(np.float32) / 1000.0

            if (depthmap > 0.0).sum() == 0:
                self.invalidate[scene_id][resolution][im_idx] = True
                imgs_idxs.append(im_idx)
                continue

            view = dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                extrinsics=np.copy(camera_pose),
                scale=np.float32(1.0),
                dataset='ScanNet',
                label=scene_id,
                instance=osp.split(impath)[1],
                labelmap=labelmap,
                original_label=original_label,
                view_idx=view_idx,
                scene_id=scene_id,
                sam_segmentmap=sam_map
            )
            views.append(view)

        if self.num_views == 2 and len(views) >= 2:
            scale = np.linalg.norm(views[0]['extrinsics'][:3, 3] - views[1]['extrinsics'][:3, 3])
        else:
            scale = 1.0

        extrinsics0 = None
        for i, view in enumerate(views):
            view['extrinsics'][:3, 3] /= max(scale, 1e-6)
            if i == 0:
                extrinsics0 = view['extrinsics']
            view['extrinsics'] = camera_normalization(extrinsics0, view['extrinsics'])
            view['scale'] = np.float32(scale)

        return views
