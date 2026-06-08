"""
EPS3D: Efficient Panoptic Semantic 3D Gaussian Splatting
"""
import os
from copy import deepcopy
import time
from typing import Optional
from einops import rearrange
import huggingface_hub
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass

from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
from src.model.encoder.eps3d_encoder import EncoderEPS3D, EncoderEPS3DCfg, OpacityMappingCfg


class EPS3D(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """EPS3D: Efficient Panoptic Semantic 3D Gaussian Splatting model."""

    def __init__(
        self,
        encoder_cfg: EncoderEPS3DCfg,
        decoder_cfg: DecoderSplattingCUDACfg,
    ):
        super(EPS3D, self).__init__()
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.build_encoder(encoder_cfg)
        self.build_decoder(decoder_cfg)

    def convert_nested_config(self, cfg_dict: dict, target_class: type):
        if isinstance(cfg_dict, dict):
            return target_class(**cfg_dict)
        elif isinstance(cfg_dict, target_class):
            return cfg_dict
        elif cfg_dict is None:
            return None
        else:
            raise ValueError(f"Cannot convert {type(cfg_dict)} to {target_class}")

    def convert_config_recursively(self, cfg_obj, conversion_map: dict):
        if not hasattr(cfg_obj, '__dict__'):
            return cfg_obj

        cfg_dict = cfg_obj.__dict__.copy()

        for field_name, target_class in conversion_map.items():
            if field_name in cfg_dict:
                cfg_dict[field_name] = self.convert_nested_config(
                    cfg_dict[field_name],
                    target_class
                )

        return type(cfg_obj)(**cfg_dict)

    def convert_encoder_config(self, encoder_cfg: EncoderEPS3DCfg) -> EncoderEPS3DCfg:
        conversion_map = {
            'gaussian_adapter': GaussianAdapterCfg,
            'opacity_mapping': OpacityMappingCfg,
        }
        return self.convert_config_recursively(encoder_cfg, conversion_map)

    def build_encoder(self, encoder_cfg: EncoderEPS3DCfg):
        encoder_cfg = self.convert_encoder_config(encoder_cfg)
        self.encoder = EncoderEPS3D(encoder_cfg)

    def build_decoder(self, decoder_cfg: DecoderSplattingCUDACfg):
        self.decoder = DecoderSplattingCUDA(decoder_cfg)

    @torch.no_grad()
    def inference(self, context_image: torch.Tensor):
        self.encoder.distill = False
        encoder_output = self.encoder(context_image, global_step=0, visualization_dump=None)
        gaussians, pred_context_pose = encoder_output.gaussians, encoder_output.pred_context_pose
        return gaussians, pred_context_pose

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

        return encoder_output, output


__all__ = [
    'EPS3D',
    'EncoderEPS3D',
    'EncoderEPS3DCfg',
    'OpacityMappingCfg',
    'DecoderSplattingCUDA',
    'DecoderSplattingCUDACfg',
    'GaussianAdapterCfg',
]
