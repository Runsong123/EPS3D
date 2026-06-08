from typing import Optional, Union

from .encoder import Encoder
from .visualization.encoder_visualizer import EncoderVisualizer
from .eps3d_encoder import EncoderEPS3D, EncoderEPS3DCfg

ENCODERS = {
    "eps3d": (EncoderEPS3D, None),
}

EncoderCfg = Union[EncoderEPS3DCfg]


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
