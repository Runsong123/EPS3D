"""
Panoptic test dataset: loads both semantic and instance labels.
"""
import os.path as osp
import numpy as np
import cv2

from dust3r.utils.image import imread_cv2
from .testdata import TestDataset, process_depth_mask_map


def process_instance_map(instance_map, depthmap):
    """Process instance map with same resize/crop as depth map."""
    height, width = depthmap.shape[:2]

    if width > height:
        new_height = 448
        new_width = int(width * (new_height / height))
    else:
        new_width = 448
        new_height = int(height * (new_width / width))

    instance_resized = cv2.resize(instance_map, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    left = (new_width - 448) // 2
    top = (new_height - 448) // 2

    return instance_resized[top:top+448, left:left+448]


class PanopticTestDataset(TestDataset):
    """Test dataset that loads both semantic and instance ground truth."""

    def _get_views(self, idx, resolution, rng):
        views = super()._get_views(idx, resolution, rng)

        for view in views:
            view_idx = view['view_idx']
            scene_id = view['scene_id']
            impath = osp.join(self.ROOT, scene_id, 'images', f'{view_idx}.jpg')
            instance_path = impath.replace('images', 'instance').replace('.jpg', '.png')
            depth_path = impath.replace('images', 'depths').replace('.jpg', '.png')

            if osp.exists(instance_path):
                instance_map = imread_cv2(instance_path, cv2.IMREAD_UNCHANGED)
                depthmap_raw = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
                instance_cropped = process_instance_map(instance_map, depthmap_raw)
                view['instance_labelmap'] = instance_cropped.astype(np.int64)
            else:
                h, w = view['depthmap'].shape[:2]
                view['instance_labelmap'] = np.zeros((h, w), dtype=np.int64)

        return views
