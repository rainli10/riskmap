from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
import importlib


class SimpleRiskCNN(nn.Module):
    """Simple RGBD CNN baseline for per-pixel risk-class logits."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: tuple[int, ...] = (32, 64, 64),
        out_channels: int = 5,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        current_channels = in_channels
        final_out_channels = int(out_channels)

        for hidden_out_channels in hidden_channels:
            layers.append(nn.Conv2d(current_channels, hidden_out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(hidden_out_channels))
            layers.append(nn.ReLU(inplace=True))
            current_channels = hidden_out_channels

        layers.append(nn.Conv2d(current_channels, final_out_channels, kernel_size=1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SimpleRiskLinear(nn.Module):
    """
    Per-pixel linear baseline over RGBD channels.

    This is a 1x1 convolution that acts like an independent linear classifier
    on each pixel's input feature vector.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 5) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class SegFormerRisk(nn.Module):
    """
    SegFormer-based dense risk predictor for transfer learning.

    Returns logits [B, num_labels, H, W]. Apply sigmoid/softmax at inference time.
    """

    def __init__(
        self,
        in_channels: int = 4,
        pretrained_model_name: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        freeze_backbone: bool = False,
        adapter_hidden: int | None = None,
        num_labels: int = 1,
    ) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        try:
            transformers_module = importlib.import_module("transformers")
            SegformerForSemanticSegmentation = getattr(
                transformers_module, "SegformerForSemanticSegmentation"
            )
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "transformers is required for SegFormerRisk. "
                "Install with: pip install transformers"
            ) from exc

        self.backbone = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

        if in_channels == 3:
            self.input_adapter = nn.Identity()
        elif adapter_hidden is None:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
            self._initialize_rgb_preserving_adapter(self.input_adapter, in_channels)
        else:
            self.input_adapter = nn.Sequential(
                nn.Conv2d(in_channels, adapter_hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(adapter_hidden, 3, kernel_size=1),
            )

        if freeze_backbone:
            for param in self.backbone.segformer.parameters():
                param.requires_grad = False

    @staticmethod
    def _initialize_rgb_preserving_adapter(adapter: nn.Conv2d, in_channels: int) -> None:
        """Start from an RGB pass-through so frozen pretrained features remain useful."""
        with torch.no_grad():
            adapter.weight.zero_()
            if adapter.bias is not None:
                adapter.bias.zero_()
            for rgb_channel in range(min(3, in_channels)):
                adapter.weight[rgb_channel, rgb_channel, 0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_h, input_w = x.shape[-2], x.shape[-1]
        x_rgb_like = self.input_adapter(x)
        outputs = self.backbone(pixel_values=x_rgb_like)
        logits = outputs.logits
        return F.interpolate(logits, size=(input_h, input_w), mode="bilinear", align_corners=False)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        if self.num_labels == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)