from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
import importlib


class SimpleRiskCNN(nn.Module):
    """A minimal fully convolutional baseline for dense risk prediction logits."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: tuple[int, ...] = (32, 64, 128, 256, 64, 32),
        out_channels: int = 1,
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



import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutput


class DepthEncoder(nn.Module):
    """
    Build a 4-level depth pyramid matching SegFormer stage resolutions/channels.
    """
    def __init__(self, hidden_sizes):
        super().__init__()
        c1, c2, c3, c4 = hidden_sizes

        self.stage1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

    def forward(self, depth):
        d1 = self.stage1(depth)   # 1/4
        d2 = self.stage2(d1)      # 1/8
        d3 = self.stage3(d2)      # 1/16
        d4 = self.stage4(d3)      # 1/32
        return [d1, d2, d3, d4]


class TokenFusion(nn.Module):
    """
    Fuse depth into token sequence BEFORE a transformer block.
    rgb_tokens:   [B, N, C]
    depth_feat:   [B, C, H, W]
    """
    def __init__(self, channels):
        super().__init__()
        self.depth_proj = nn.Linear(channels, channels)
        self.gate = nn.Linear(channels * 2, channels)
        self.out_norm = nn.LayerNorm(channels)

    def forward(self, rgb_tokens, depth_feat):
        depth_tokens = depth_feat.flatten(2).transpose(1, 2)   # [B, N, C]
        depth_tokens = self.depth_proj(depth_tokens)

        gate = torch.sigmoid(
            self.gate(torch.cat([rgb_tokens, depth_tokens], dim=-1))
        )
        fused = rgb_tokens + gate * depth_tokens
        return self.out_norm(fused)


class SegformerEncoderWithDepth(nn.Module):
    """
    Reuse original SegFormer patch embeddings, transformer blocks, and norms,
    but inject depth BEFORE each transformer block.
    """
    def __init__(self, base_encoder, config):
        super().__init__()
        self.config = config

        # reuse pretrained SegFormer pieces
        self.patch_embeddings = base_encoder.patch_embeddings
        self.block = base_encoder.block
        self.layer_norm = base_encoder.layer_norm

        # new depth path
        self.depth_encoder = DepthEncoder(config.hidden_sizes)

        # one fusion module per transformer block
        self.token_fusions = nn.ModuleList([
            nn.ModuleList([
                TokenFusion(config.hidden_sizes[stage_idx])
                for _ in range(config.depths[stage_idx])
            ])
            for stage_idx in range(config.num_encoder_blocks)
        ])

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        depth_values: torch.FloatTensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = pixel_values.shape[0]

        hidden_states = pixel_values           # RGB image path
        depth_feats = self.depth_encoder(depth_values)   # 4 depth feature maps

        for stage_idx, (embedding_layer, block_layer, norm_layer) in enumerate(
            zip(self.patch_embeddings, self.block, self.layer_norm)
        ):
            # original SegFormer patch embedding
            hidden_states, height, width = embedding_layer(hidden_states)   # [B, N, C]

            depth_feat = depth_feats[stage_idx]
            if depth_feat.shape[-2:] != (height, width):
                depth_feat = F.interpolate(
                    depth_feat,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )

            # inject depth BEFORE every transformer block in this stage
            for blk_idx, blk in enumerate(block_layer):
                hidden_states = self.token_fusions[stage_idx][blk_idx](
                    hidden_states, depth_feat
                )
                layer_outputs = blk(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # original SegFormer norm
            hidden_states = norm_layer(hidden_states)

            # original reshape logic
            if stage_idx != len(self.patch_embeddings) - 1 or (
                stage_idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage
            ):
                hidden_states = hidden_states.reshape(
                    batch_size, height, width, -1
                ).permute(0, 3, 1, 2).contiguous()

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class SegFormerRiskWithDepth(nn.Module):
    """
    Depth is fused into the SegFormer transformer, not only the decoder.
    """
    def __init__(
        self,
        in_channels: int = 4,
        pretrained_model_name: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        freeze_backbone: bool = False,
        num_labels: int = 5,
    ):
        super().__init__()
        self.num_labels = num_labels

        if in_channels != 4:
            raise ValueError("Expected RGB-D input with 4 channels.")

        transformers_module = importlib.import_module("transformers")
        SegformerForSemanticSegmentation = getattr(
            transformers_module, "SegformerForSemanticSegmentation"
        )

        self.backbone = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

        # replace encoder with depth-aware encoder
        old_encoder = self.backbone.segformer.encoder
        self.backbone.segformer.encoder = SegformerEncoderWithDepth(
            old_encoder, self.backbone.config
        )

        if freeze_backbone:
            for p in self.backbone.segformer.patch_embeddings.parameters():
                p.requires_grad = False
            for p in self.backbone.segformer.encoder.block.parameters():
                p.requires_grad = False
            for p in self.backbone.decode_head.parameters():
                p.requires_grad = False

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"Expected [B, 4, H, W], got {tuple(x.shape)}")

        rgb = x[:, :3, :, :]
        depth = x[:, 3:4, :, :]

        # IMPORTANT:
        # bypass self.backbone.segformer(...) because HF SegformerModel.forward
        # only passes pixel_values to the encoder.
        encoder_outputs = self.backbone.segformer.encoder(
            pixel_values=rgb,
            depth_values=depth,
            output_hidden_states=True,
            return_dict=True,
        )

        encoder_hidden_states = encoder_outputs.hidden_states
        logits = self.backbone.decode_head(encoder_hidden_states)

        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits