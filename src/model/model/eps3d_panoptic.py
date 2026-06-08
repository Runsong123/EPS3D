"""
EPS3D Panoptic: Panoptic segmentation extension for EPS3D.

Supports three output modes:
    's' - semantic only
    'i' - instance only
    'p' - panoptic (both semantic and instance)
"""
from typing import Optional, Literal

import torch

from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
from src.model.model.eps3d import EPS3D, EncoderEPS3DCfg


class EPS3DPanoptic(EPS3D):
    """
    EPS3D with panoptic (semantic + instance) branch support.

    output_mode:
        's' - semantic only (default)
        'i' - instance only
        'p' - panoptic: both semantic and instance
    """

    def __init__(
        self,
        encoder_cfg: EncoderEPS3DCfg,
        decoder_cfg: DecoderSplattingCUDACfg,
        output_mode: Literal['p', 's', 'i'] = 's',
    ):
        super(EPS3DPanoptic, self).__init__(encoder_cfg, decoder_cfg)
        self.output_mode = output_mode

    def set_output_mode(self, mode: Literal['p', 's', 'i']):
        assert mode in ('p', 's', 'i'), f"output_mode must be 'p', 's', or 'i', got '{mode}'"
        self.output_mode = mode

    def forward(self,
        context_image: torch.Tensor,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        near: float = 0.01,
        far: float = 100.0,
    ):
        b, v, c, h, w = context_image.shape
        device = context_image.device
        encoder_output = self.encoder(context_image, global_step, visualization_dump=visualization_dump)
        gaussians, pred_context_pose = encoder_output.gaussians, encoder_output.pred_context_pose

        output = self.decoder.forward(
            gaussians,
            pred_context_pose['extrinsic'],
            pred_context_pose["intrinsic"],
            torch.ones(1, v, device=device) * near,
            torch.ones(1, v, device=device) * far,
            (h, w),
            "depth",
        )

        if self.output_mode in ('s', 'p'):
            if self.encoder.semantic_fea:
                output.feature = self.decoder.semantic_forward(
                    gaussians,
                    pred_context_pose['extrinsic'],
                    pred_context_pose["intrinsic"],
                    torch.ones(1, v, device=device) * near,
                    torch.ones(1, v, device=device) * far,
                    (h, w),
                    "depth",
                )

        if self.output_mode in ('i', 'p'):
            if self.encoder.instance_fea:
                output.instance_feature = self.decoder.instance_forward(
                    gaussians,
                    pred_context_pose['extrinsic'],
                    pred_context_pose["intrinsic"],
                    torch.ones(1, v, device=device) * near,
                    torch.ones(1, v, device=device) * far,
                    (h, w),
                    "depth",
                )

        return encoder_output, output


__all__ = [
    'EPS3DPanoptic',
    'EncoderEPS3DCfg',
    'DecoderSplattingCUDACfg',
]
