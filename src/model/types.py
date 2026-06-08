from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    scales: Float[Tensor, "batch gaussian 3"]
    rotations: Float[Tensor, "batch gaussian 4"]
    # levels: Float[Tensor, "batch gaussian"]


@dataclass
class SemanticGaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    scales: Float[Tensor, "batch gaussian 3"]
    rotations: Float[Tensor, "batch gaussian 4"]
    semantics: Float[Tensor, "batch gaussian 64"]


@dataclass
class PanopticGaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    scales: Float[Tensor, "batch gaussian 3"]
    rotations: Float[Tensor, "batch gaussian 4"]
    semantics: Float[Tensor, "batch gaussian 64"]
    instances: Float[Tensor, "batch gaussian 64"]
