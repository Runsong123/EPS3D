"""ScanNet utilities for camera normalization."""
import numpy as np


def camera_normalization(pivotal_pose: np.ndarray, poses: np.ndarray):
    """Normalize camera poses relative to a pivotal pose."""
    canonical_camera_extrinsics = np.eye(4, dtype=np.float32)
    pivotal_pose_inv = np.linalg.inv(pivotal_pose)
    camera_norm_matrix = canonical_camera_extrinsics @ pivotal_pose_inv
    poses = camera_norm_matrix @ poses
    return poses
