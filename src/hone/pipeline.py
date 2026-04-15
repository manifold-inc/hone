"""ResBM: Residual Bottleneck Models for low-bandwidth pipeline parallelism.

Implements activation compression for pipeline-parallel training over
low-bandwidth links. Based on the ResBM paper (Aboudib et al., 2025).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BottleneckEncoder",
    "BottleneckDecoder",
    "IdentityProjection",
    "PipelineStageBoundary",
    "PipelineStage",
    "create_pipeline_stages",
]


class BottleneckEncoder(nn.Module):
    """Compresses activations at pipeline stage output boundaries.

    Maps hidden_dim -> bottleneck_dim through a two-layer network with SiLU
    activation. The bottleneck_dim is typically hidden_dim / compression_ratio
    (e.g., 2048/128 = 16).
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        mid_dim = max(bottleneck_dim * 4, 64)
        self.fc1 = nn.Linear(hidden_dim, mid_dim, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(mid_dim, bottleneck_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class BottleneckDecoder(nn.Module):
    """Expands activations at pipeline stage input boundaries.

    Maps bottleneck_dim -> hidden_dim through a two-layer network with SiLU.
    """

    def __init__(self, bottleneck_dim: int, hidden_dim: int):
        super().__init__()
        mid_dim = max(bottleneck_dim * 4, 64)
        self.fc1 = nn.Linear(bottleneck_dim, mid_dim, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(mid_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class IdentityProjection(nn.Module):
    """Rectangular identity matrix for residual path dimension changes.

    When going from dim_in to dim_out:
    - If dim_out > dim_in: zero-pads the extra dimensions
    - If dim_out < dim_in: truncates to first dim_out dimensions
    - If dim_out == dim_in: identity (no-op)

    This preserves the identity property of the residual connection
    as described in the ResBM paper (Eq. 6).
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim_in == self.dim_out:
            return x
        if self.dim_out < self.dim_in:
            return x[..., : self.dim_out]
        pad_size = self.dim_out - self.dim_in
        return F.pad(x, (0, pad_size))


class PipelineStageBoundary(nn.Module):
    """Handles activation compression/decompression at a pipeline stage boundary.

    At the sending side: encodes hidden_dim -> bottleneck_dim
    At the receiving side: decodes bottleneck_dim -> hidden_dim
    The residual path uses IdentityProjection for dimension matching.

    In practice, encoder lives on the sending stage's device and
    decoder lives on the receiving stage's device.
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.compression_ratio = hidden_dim // bottleneck_dim

        self.encoder = BottleneckEncoder(hidden_dim, bottleneck_dim)
        self.decoder = BottleneckDecoder(bottleneck_dim, hidden_dim)

        self.id_down = IdentityProjection(hidden_dim, bottleneck_dim)
        self.id_up = IdentityProjection(bottleneck_dim, hidden_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compress activation for transmission. Returns bottleneck-dim tensor."""
        return self.encoder(x) + self.id_down(x)

    def decode(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decompress received activation. Returns hidden-dim tensor."""
        return self.decoder(compressed) + self.id_up(compressed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode -> decode pass (for single-device testing/training)."""
        return self.decode(self.encode(x))

    def init_weights(self):
        """Initialize bottleneck weights for near-identity behavior at init."""
        for module in [self.encoder, self.decoder]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01)


class PipelineStage(nn.Module):
    """A pipeline stage containing a subset of transformer layers.

    Each stage processes its layers sequentially, with optional bottleneck
    boundaries for inter-stage communication.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        stage_idx: int,
        num_stages: int,
        hidden_dim: int,
        bottleneck_dim: int = 16,
    ):
        super().__init__()
        self.layers = layers
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.is_first = stage_idx == 0
        self.is_last = stage_idx == num_stages - 1

        self.input_boundary: PipelineStageBoundary | None = None
        if not self.is_first:
            self.input_boundary = PipelineStageBoundary(hidden_dim, bottleneck_dim)

        self.output_boundary: PipelineStageBoundary | None = None
        if not self.is_last:
            self.output_boundary = PipelineStageBoundary(hidden_dim, bottleneck_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        *,
        compressed_input: bool = False,
    ) -> torch.Tensor:
        """Process layers. If compressed_input, decode first. Returns potentially compressed output."""
        h = hidden_states

        if compressed_input and self.input_boundary is not None:
            h = self.input_boundary.decode(h)

        for layer in self.layers:
            result = layer(h, position_embeddings, attention_mask)
            if isinstance(result, tuple):
                h = result[0]
            else:
                h = result

        if not self.is_last and self.output_boundary is not None:
            h = self.output_boundary.encode(h)

        return h

    def init_weights(self):
        """Initialize weights for all layers and boundaries."""
        for layer in self.layers:
            layer.init_weights()
        if self.input_boundary is not None:
            self.input_boundary.init_weights()
        if self.output_boundary is not None:
            self.output_boundary.init_weights()


def create_pipeline_stages(
    model_layers: nn.ModuleList,
    num_stages: int,
    hidden_dim: int,
    bottleneck_dim: int = 16,
) -> nn.ModuleList:
    """Partition transformer layers into pipeline stages with ResBM boundaries.

    Args:
        model_layers: The nn.ModuleList of DecoderLayer instances
        num_stages: Number of pipeline stages to create
        hidden_dim: Model hidden dimension
        bottleneck_dim: Bottleneck dimension for activation compression
            (hidden_dim / bottleneck_dim = compression ratio)

    Returns:
        nn.ModuleList of PipelineStage instances
    """
    n_layers = len(model_layers)
    layers_per_stage = n_layers // num_stages
    remainder = n_layers % num_stages

    stages = []
    start = 0
    for i in range(num_stages):
        end = start + layers_per_stage + (1 if i < remainder else 0)
        stage_layers = nn.ModuleList(list(model_layers[start:end]))

        stage = PipelineStage(
            layers=stage_layers,
            stage_idx=i,
            num_stages=num_stages,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )
        stages.append(stage)
        start = end

    return nn.ModuleList(stages)
