from typing import Optional, Union

from ..encoder import Encoder
from ..encoder.visualization.encoder_visualizer import EncoderVisualizer
from ..decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
from torch import nn

from .eps3d import EPS3D, EncoderEPS3D, EncoderEPS3DCfg

MODELS = {
    "eps3d": EPS3D,
}

EncoderCfg = Union[EncoderEPS3DCfg]
DecoderCfg = DecoderSplattingCUDACfg


def get_model(encoder_cfg: EncoderCfg, decoder_cfg: DecoderCfg) -> nn.Module:
    model = MODELS['eps3d'](encoder_cfg, decoder_cfg)
    return model
