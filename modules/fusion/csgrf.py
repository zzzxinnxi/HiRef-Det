"""Cross-scale semantic-guided residual fusion for YOLO feature pyramids."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv


class CSGRF(nn.Module):
    """Calibrate two adjacent-scale features before preserving Concat layout.

    The module starts as an exact identity Concat because both residual scales
    are initialized to zero. This keeps downstream C3k2 input channels and the
    pretrained feature distribution unchanged at initialization.
    """

    def __init__(
        self,
        channels: Sequence[int],
        reduction: int = 4,
        kernel_size: int = 3,
        calibration_mode: str = "independent",
        relation_mode: str = "add_mul",
        alpha_init: float = 0.0,
    ):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"CSGRF expects two input features, got channels={channels}")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        c_deep, c_lateral = (int(c) for c in channels)
        c_embed = max(min(c_deep, c_lateral) // reduction, 8)
        if calibration_mode not in {"independent", "shared"}:
            raise ValueError(f"Unsupported calibration_mode={calibration_mode}")
        if relation_mode not in {"add_mul", "add_only", "mul_only"}:
            raise ValueError(f"Unsupported relation_mode={relation_mode}")
        if calibration_mode == "shared" and c_deep != c_lateral:
            raise ValueError("Shared calibration requires equal branch channels")

        self.channels = (c_deep, c_lateral)
        self.calibration_mode = calibration_mode
        self.relation_mode = relation_mode
        self.deep_proj = Conv(c_deep, c_embed, 1, 1)
        self.lateral_proj = Conv(c_lateral, c_embed, 1, 1)
        self.relation = nn.Sequential(
            DWConv(c_embed, c_embed, kernel_size, 1),
            Conv(c_embed, c_embed, 1, 1),
        )
        if calibration_mode == "shared":
            self.shared_calibration = nn.Conv2d(c_embed, c_deep, 1, bias=True)
        else:
            self.deep_calibration = nn.Conv2d(c_embed, c_deep, 1, bias=True)
            self.lateral_calibration = nn.Conv2d(c_embed, c_lateral, 1, bias=True)

        # ReZero-style initialization: forward(x) == torch.cat(x, 1) at start.
        self.deep_scale = nn.Parameter(torch.full((1,), float(alpha_init)))
        self.lateral_scale = nn.Parameter(torch.full((1,), float(alpha_init)))

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(x) != 2:
            raise ValueError(f"CSGRF expects two tensors, got {len(x)}")
        # Older released checkpoints predate these explicit mode attributes.
        calibration_mode = getattr(self, "calibration_mode", "independent")
        relation_mode = getattr(self, "relation_mode", "add_mul")
        deep, lateral = x
        if deep.shape[-2:] != lateral.shape[-2:]:
            raise ValueError(
                "CSGRF inputs must already share spatial resolution, "
                f"got {deep.shape[-2:]} and {lateral.shape[-2:]}"
            )

        deep_embed = self.deep_proj(deep)
        lateral_embed = self.lateral_proj(lateral)
        if relation_mode == "mul_only":
            relation = deep_embed * lateral_embed
        else:
            relation = self.relation(deep_embed + lateral_embed)
        if relation_mode == "add_mul":
            relation = relation + deep_embed * lateral_embed

        if calibration_mode == "shared":
            deep_weight = lateral_weight = torch.tanh(self.shared_calibration(relation))
        else:
            deep_weight = torch.tanh(self.deep_calibration(relation))
            lateral_weight = torch.tanh(self.lateral_calibration(relation))
        deep = deep * (1.0 + self.deep_scale * deep_weight)
        lateral = lateral * (1.0 + self.lateral_scale * lateral_weight)
        return torch.cat((deep, lateral), dim=1)


class ScalarWeightedConcat(nn.Module):
    """BiFPN-style scalar branch reweighting with retained concatenation."""

    def __init__(self, channels: Sequence[int], eps: float = 1e-4):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"ScalarWeightedConcat expects two inputs, got {channels}")
        self.channels = tuple(int(c) for c in channels)
        self.eps = float(eps)
        self.deep_weight = nn.Parameter(torch.ones(1))
        self.lateral_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(x) != 2:
            raise ValueError(f"ScalarWeightedConcat expects two tensors, got {len(x)}")
        deep, lateral = x
        positive = torch.relu(torch.cat((self.deep_weight, self.lateral_weight)))
        weights = 2.0 * positive / (positive.sum() + self.eps)
        return torch.cat((weights[0] * deep, weights[1] * lateral), dim=1)


class AdaptiveCompetitiveConcat(nn.Module):
    """Content-adaptive competitive weighting with retained concatenation."""

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"AdaptiveCompetitiveConcat expects two inputs, got {channels}")
        c_deep, c_lateral = (int(c) for c in channels)
        if c_deep != c_lateral:
            raise ValueError("AdaptiveCompetitiveConcat requires equal branch channels")
        self.channels = (c_deep, c_lateral)
        self.logits = nn.Conv2d(c_deep + c_lateral, 2 * c_deep, 1, bias=True)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(x) != 2:
            raise ValueError(f"AdaptiveCompetitiveConcat expects two tensors, got {len(x)}")
        deep, lateral = x
        batch, channels, height, width = deep.shape
        logits = self.logits(torch.cat((deep, lateral), dim=1))
        weights = 2.0 * logits.reshape(batch, 2, channels, height, width).softmax(dim=1)
        return torch.cat((weights[:, 0] * deep, weights[:, 1] * lateral), dim=1)



if __name__ == "__main__":
    module = CSGRF((64, 64)).eval()
    inputs = (torch.randn(2, 64, 40, 40), torch.randn(2, 64, 40, 40))
    with torch.no_grad():
        output = module(inputs)
        reference = torch.cat(inputs, dim=1)
    assert output.shape == (2, 128, 40, 40)
    assert torch.equal(output, reference)
