from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, random_split

from dataloader import (
    RISK_BIN_EDGES,
    RISK_CLASS_LABELS,
    RISK_CLASS_VALUES,
    RiskMapDataset,
    build_blocked_risk_map,
    build_dense_risk_map,
    quantize_risk_map,
    risk_classes_to_values,
)
from model import SegFormerRisk, SimpleRiskCNN
from train import (
    COMPONENT_CONNECTIVITY as TRAIN_COMPONENT_CONNECTIVITY,
    DEPTH_MAX as TRAIN_DEPTH_MAX,
    DEPTH_MIN as TRAIN_DEPTH_MIN,
    SEGFORMER_ADAPTER_HIDDEN as TRAIN_SEGFORMER_ADAPTER_HIDDEN,
    SEGFORMER_FREEZE_BACKBONE as TRAIN_SEGFORMER_FREEZE_BACKBONE,
    SEGFORMER_PRETRAINED_MODEL as TRAIN_SEGFORMER_PRETRAINED_MODEL,
    SEMANTIC_WEIGHTS as TRAIN_SEMANTIC_WEIGHTS,
    TARGET_MODE as TRAIN_TARGET_MODE,
)

TRAIN_ARCHITECTURE = "segformer"
CHECKPOINT_PATH = Path("ckpts_classification/ours_segformer_transfer_ce_bins5_ori_weights/segformer_transfer_epoch_029.pt")
VALIDATION_DATASET_ROOT = Path("data/cityscape_prepro/train")
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_RANDOM_SEED = 42
DEFAULT_USE_TRAIN_SPLIT_FOR_VALIDATION = True
OUTPUT_ROOT = Path("output/validation")
BATCH_SIZE = 1
NUM_WORKERS = 0

RISK_CLASS_INFO = {
    0: ("[0.0, 0.2)", (49, 54, 149)),
    1: ("[0.2, 0.4)", (69, 117, 180)),
    2: ("[0.4, 0.6)", (255, 255, 191)),
    3: ("[0.6, 0.8)", (253, 174, 97)),
    4: ("[0.8, 1.0]", (215, 25, 28)),
}

RISK_CLASS_BOUNDS = (0.0,) + tuple(float(edge) for edge in RISK_BIN_EDGES) + (1.0,)
RISK_LEVELS = [
    (
        str(RISK_CLASS_INFO[class_id][0]),
        float(RISK_CLASS_BOUNDS[class_id]),
        float(RISK_CLASS_BOUNDS[class_id + 1]),
        tuple(int(channel) for channel in RISK_CLASS_INFO[class_id][1]),
    )
    for class_id in sorted(RISK_CLASS_INFO.keys())
]
RISK_COLOR_ANCHORS = [
    (float(RISK_CLASS_BOUNDS[class_id]), tuple(int(channel) for channel in RISK_CLASS_INFO[class_id][1]))
    for class_id in sorted(RISK_CLASS_INFO.keys())
] + [(1.0, tuple(int(channel) for channel in RISK_CLASS_INFO[max(RISK_CLASS_INFO.keys())][1]))]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_runtime_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "architecture": checkpoint.get("architecture", TRAIN_ARCHITECTURE),
        "task_mode": str(checkpoint.get("task_mode", "risk_regression")),
        "semantic_weights": checkpoint.get("semantic_weights", TRAIN_SEMANTIC_WEIGHTS),
        "depth_min": float(checkpoint.get("depth_min", TRAIN_DEPTH_MIN)),
        "depth_max": float(checkpoint.get("depth_max", TRAIN_DEPTH_MAX)),
        "target_mode": str(checkpoint.get("target_mode", TRAIN_TARGET_MODE)),
        "component_connectivity": int(
            checkpoint.get("component_connectivity", TRAIN_COMPONENT_CONNECTIVITY)
        ),
        "dataset_root": str(checkpoint.get("dataset_root", VALIDATION_DATASET_ROOT)),
        "val_split": float(checkpoint.get("val_split", DEFAULT_VAL_SPLIT)),
        "random_seed": int(checkpoint.get("random_seed", DEFAULT_RANDOM_SEED)),
        "use_train_split_for_validation": bool(
            checkpoint.get(
                "use_train_split_for_validation",
                DEFAULT_USE_TRAIN_SPLIT_FOR_VALIDATION,
            )
        ),
        "num_risk_classes": int(checkpoint.get("num_risk_classes", 1)),
        "risk_scale": float(checkpoint.get("risk_scale", 1.0)),
        "risk_bin_edges": tuple(checkpoint.get("risk_bin_edges", RISK_BIN_EDGES)),
        "risk_class_values": tuple(checkpoint.get("risk_class_values", RISK_CLASS_VALUES)),
        "risk_class_labels": tuple(checkpoint.get("risk_class_labels", RISK_CLASS_LABELS)),
        "segformer_pretrained_model": str(
            checkpoint.get("segformer_pretrained_model", TRAIN_SEGFORMER_PRETRAINED_MODEL)
        ),
        "segformer_freeze_backbone": bool(
            checkpoint.get("segformer_freeze_backbone", TRAIN_SEGFORMER_FREEZE_BACKBONE)
        ),
        "segformer_adapter_hidden": checkpoint.get(
            "segformer_adapter_hidden", TRAIN_SEGFORMER_ADAPTER_HIDDEN
        ),
    }


def build_model(runtime_config: dict[str, Any]) -> torch.nn.Module:
    architecture = runtime_config["architecture"]
    if architecture == "simplest_cnn":
        return SimpleRiskCNN(out_channels=runtime_config["num_risk_classes"])
    if architecture == "segformer":
        return SegFormerRisk(
            in_channels=4,
            pretrained_model_name=runtime_config["segformer_pretrained_model"],
            freeze_backbone=runtime_config["segformer_freeze_backbone"],
            adapter_hidden=runtime_config["segformer_adapter_hidden"],
            num_labels=runtime_config["num_risk_classes"],
        )

    raise ValueError(f"Unsupported architecture '{architecture}'.")


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    runtime_config = build_runtime_config(checkpoint)

    model = build_model(runtime_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, runtime_config


def build_target_risk_map(
    label_map: np.ndarray,
    depth_map: np.ndarray,
    runtime_config: dict[str, Any],
) -> np.ndarray:
    if runtime_config["target_mode"] == "dense":
        return build_dense_risk_map(
            label_map=label_map,
            depth_map=depth_map,
            semantic_weights=runtime_config["semantic_weights"],
            depth_min=runtime_config["depth_min"],
            depth_max=runtime_config["depth_max"],
        )

    return build_blocked_risk_map(
        label_map=label_map,
        depth_map=depth_map,
        semantic_weights=runtime_config["semantic_weights"],
        depth_min=runtime_config["depth_min"],
        depth_max=runtime_config["depth_max"],
        connectivity=runtime_config["component_connectivity"],
    )


def risk_targets_to_class_indices(
    targets: torch.Tensor,
    runtime_config: dict[str, Any],
) -> torch.Tensor:
    # Match utils/run_gt_classes.py exactly: NumPy quantization path.
    risk_scale = float(runtime_config.get("risk_scale", 1.0))
    target_risk = targets.squeeze(1).detach().cpu().numpy().astype(np.float32, copy=False)
    scaled_targets = np.clip(target_risk * risk_scale, 0.0, 1.0).astype(np.float32)
    class_map = quantize_risk_map(scaled_targets, bin_edges=runtime_config["risk_bin_edges"])
    return torch.from_numpy(class_map).to(device=targets.device, dtype=torch.long)


def risk_class_indices_to_values_tensor(
    class_map: torch.Tensor,
    runtime_config: dict[str, Any],
) -> torch.Tensor:
    values = torch.tensor(
        runtime_config["risk_class_values"],
        device=class_map.device,
        dtype=torch.float32,
    )
    return values[class_map]


def convert_risk_map_for_task(
    risk_map: np.ndarray,
    runtime_config: dict[str, Any],
) -> np.ndarray:
    risk = np.clip(np.asarray(risk_map, dtype=np.float32), 0.0, 1.0)
    if runtime_config["task_mode"] == "risk_classification":
        risk_scale = float(runtime_config.get("risk_scale", 1.0))
        scaled_risk = np.clip(risk * risk_scale, 0.0, 1.0).astype(np.float32)
        class_map = quantize_risk_map(scaled_risk, bin_edges=runtime_config["risk_bin_edges"])
        return risk_classes_to_values(class_map, class_values=runtime_config["risk_class_values"])
    return risk.astype(np.float32)


def predictions_to_risk_tensor(
    raw_predictions: torch.Tensor,
    runtime_config: dict[str, Any],
) -> torch.Tensor:
    if runtime_config["task_mode"] == "risk_classification":
        predicted_classes = torch.argmax(raw_predictions, dim=1)
        predicted_risk = risk_class_indices_to_values_tensor(predicted_classes, runtime_config)
        return predicted_risk.unsqueeze(1)
    return torch.sigmoid(raw_predictions)


def targets_to_risk_tensor(
    targets: torch.Tensor,
    runtime_config: dict[str, Any],
) -> torch.Tensor:
    if runtime_config["task_mode"] == "risk_classification":
        target_classes = risk_targets_to_class_indices(targets, runtime_config)
        target_risk = risk_class_indices_to_values_tensor(target_classes, runtime_config)
        return target_risk.unsqueeze(1)
    return targets


def create_validation_loader(runtime_config: dict[str, Any]) -> DataLoader:
    full_dataset = RiskMapDataset(
        dataset_root=Path(runtime_config.get("dataset_root", VALIDATION_DATASET_ROOT)),
        semantic_weights=runtime_config["semantic_weights"],
        depth_min=runtime_config["depth_min"],
        depth_max=runtime_config["depth_max"],
        target_mode=runtime_config["target_mode"],
        component_connectivity=runtime_config["component_connectivity"],
    )

    if runtime_config.get("use_train_split_for_validation", False):
        if len(full_dataset) < 2:
            raise RuntimeError("Need at least 2 samples to reconstruct validation split.")
        val_split = float(runtime_config.get("val_split", DEFAULT_VAL_SPLIT))
        val_size = max(1, int(len(full_dataset) * val_split))
        train_size = len(full_dataset) - val_size
        if train_size == 0:
            train_size = len(full_dataset) - 1
            val_size = 1
        generator = torch.Generator().manual_seed(
            int(runtime_config.get("random_seed", DEFAULT_RANDOM_SEED))
        )
        _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        dataset = val_dataset
    else:
        dataset = full_dataset

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )


def tensor_to_rgb_image(input_tensor: torch.Tensor) -> np.ndarray:
    rgb_tensor = input_tensor[:3].detach().cpu().clamp(0.0, 1.0)
    rgb_image = np.transpose(rgb_tensor.numpy(), (1, 2, 0))
    return (rgb_image * 255.0).round().astype(np.uint8)


def colorize_risk_map(risk_map: np.ndarray) -> np.ndarray:
    risk = np.clip(np.asarray(risk_map, dtype=np.float32), 0.0, 1.0)
    color_image = np.zeros(risk.shape + (3,), dtype=np.uint8)

    for _, lower, upper, color in RISK_LEVELS:
        mask = (risk >= lower) & (risk < upper)
        if upper >= 1.0:
            mask = risk >= lower
        color_image[mask] = color

    return color_image


def colorize_risk_map_continuous(
    risk_map: np.ndarray,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> np.ndarray:
    risk = np.asarray(risk_map, dtype=np.float32)
    if value_max > value_min:
        risk = np.clip((risk - value_min) / (value_max - value_min), 0.0, 1.0)
    else:
        risk = np.zeros_like(risk)
    flat_risk = risk.reshape(-1)
    anchor_positions = np.asarray([anchor[0] for anchor in RISK_COLOR_ANCHORS], dtype=np.float32)
    anchor_colors = np.asarray([anchor[1] for anchor in RISK_COLOR_ANCHORS], dtype=np.float32)

    color_channels = [
        np.interp(flat_risk, anchor_positions, anchor_colors[:, channel_idx])
        for channel_idx in range(3)
    ]
    stacked = np.stack(color_channels, axis=1)
    return stacked.reshape(risk.shape + (3,)).astype(np.uint8)


def overlay_risk_on_rgb(
    rgb_image: np.ndarray,
    risk_map: np.ndarray,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> np.ndarray:
    rgb = np.asarray(rgb_image, dtype=np.float32)
    risk = np.asarray(risk_map, dtype=np.float32)
    if value_max > value_min:
        normalized = np.clip((risk - value_min) / (value_max - value_min), 0.0, 1.0)
    else:
        normalized = np.zeros_like(risk)
    colorized = colorize_risk_map_continuous(risk_map, value_min, value_max).astype(np.float32)
    alpha = np.where(normalized > 0.0, 0.15 + 0.55 * normalized, 0.0)[..., None]
    blended = rgb * (1.0 - alpha) + colorized * alpha
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def build_legend_image(
    height: int,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> np.ndarray:
    legend_width = 170
    legend = Image.new("RGB", (legend_width, height), color=(24, 24, 24))
    draw = ImageDraw.Draw(legend)
    font = ImageFont.load_default()

    if height <= 0:
        raise ValueError(f"Legend height must be positive, got {height}.")

    draw.text((12, 8), "Risk Scale", fill=(255, 255, 255), font=font)

    bar_x0 = 16
    bar_x1 = 44
    top_offset = 26
    bottom_padding = 8
    bar_height = max(1, height - top_offset - bottom_padding)

    gradient_values = np.linspace(1.0, 0.0, bar_height, dtype=np.float32)[:, None]
    gradient_map = np.repeat(gradient_values, bar_x1 - bar_x0, axis=1)
    gradient_image = Image.fromarray(colorize_risk_map_continuous(gradient_map))
    legend.paste(gradient_image, (bar_x0, top_offset))
    draw.rectangle((bar_x0, top_offset, bar_x1 - 1, top_offset + bar_height - 1), outline=(255, 255, 255))

    tick_values = [1.0, 0.75, 0.50, 0.25, 0.0] if height >= 140 else [1.0, 0.5, 0.0]
    value_range = value_max - value_min
    for norm_val in tick_values:
        value = value_min + norm_val * value_range
        y = int(round(top_offset + (1.0 - norm_val) * max(bar_height - 1, 0)))
        draw.line((bar_x1 + 4, y, bar_x1 + 12, y), fill=(255, 255, 255), width=1)
        draw.text((bar_x1 + 16, max(0, y - 5)), f"{value:.2f}", fill=(235, 235, 235), font=font)

    draw.text((bar_x1 + 16, top_offset), "High", fill=(255, 255, 255), font=font)
    draw.text(
        (bar_x1 + 16, max(top_offset + bar_height - 14, 0)),
        "Low",
        fill=(255, 255, 255),
        font=font,
    )

    return np.asarray(legend)


def render_continuous_risk_map(
    risk_map: np.ndarray,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> np.ndarray:
    return np.concatenate(
        (
            colorize_risk_map_continuous(risk_map, value_min, value_max),
            build_legend_image(risk_map.shape[0], value_min, value_max),
        ),
        axis=1,
    )


def add_panel_title(title: str, image_array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(image_array)
    title_height = 28
    canvas = Image.new("RGB", (image.width, image.height + title_height), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(255, 255, 255), font=font)
    canvas.paste(image, (0, title_height))
    return np.asarray(canvas)


def build_class_legend(height: int) -> np.ndarray:
    legend_width = 210
    legend = Image.new("RGB", (legend_width, height), color=(24, 24, 24))
    draw = ImageDraw.Draw(legend)
    font = ImageFont.load_default()

    draw.text((12, 8), "Risk Classes", fill=(255, 255, 255), font=font)

    y = 28
    row_height = 18
    for class_id, (label, color) in RISK_CLASS_INFO.items():
        draw.rectangle((12, y, 28, y + 11), fill=color)
        draw.text((36, y - 1), f"{class_id}: {label}", fill=(235, 235, 235), font=font)
        y += row_height

    return np.asarray(legend)


def render_binned_risk_map(class_map: np.ndarray) -> np.ndarray:
    classes = np.asarray(class_map, dtype=np.int64)
    color_map = np.zeros(classes.shape + (3,), dtype=np.uint8)

    for class_id, (_, color) in RISK_CLASS_INFO.items():
        color_map[classes == int(class_id)] = color

    legend = build_class_legend(classes.shape[0])
    return np.concatenate((color_map, legend), axis=1)


def make_comparison_panel(
    sample_id: str,
    rgb_image: np.ndarray,
    predicted_risk: np.ndarray,
    predicted_class_map: np.ndarray | None = None,
    target_risk: np.ndarray | None = None,
    target_class_map: np.ndarray | None = None,
    pred_value_max: float = 1.0,
    target_value_max: float = 1.0,
) -> Image.Image:
    panels = [
        add_panel_title("RGB Image", rgb_image),
        add_panel_title(
            "Prediction Overlay",
            overlay_risk_on_rgb(rgb_image, predicted_risk, value_max=pred_value_max),
        ),
    ]
    if predicted_class_map is not None:
        panels.append(
            add_panel_title(
                "Predicted Risk Classes",
                render_binned_risk_map(predicted_class_map),
            )
        )
    else:
        panels.append(
            add_panel_title(
                "Predicted Risk",
                colorize_risk_map_continuous(predicted_risk, value_max=pred_value_max),
            )
        )

    if target_class_map is not None:
        panels.append(
            add_panel_title(
                "Target Risk Classes",
                render_binned_risk_map(target_class_map),
            )
        )
    elif target_risk is not None:
        panels.append(
            add_panel_title(
                "Target Risk",
                colorize_risk_map_continuous(target_risk, value_max=target_value_max),
            )
        )

    panel_height = max(panel.shape[0] for panel in panels)
    normalized_panels: list[np.ndarray] = []
    for panel in panels:
        if panel.shape[0] < panel_height:
            padded = np.zeros((panel_height, panel.shape[1], 3), dtype=np.uint8)
            padded[: panel.shape[0], :, :] = panel
            normalized_panels.append(padded)
        else:
            normalized_panels.append(panel)

    combined = np.concatenate(normalized_panels, axis=1)
    legend = build_legend_image(combined.shape[0], value_max=max(pred_value_max, target_value_max))
    full_panel = np.concatenate((combined, legend), axis=1)
    full_image = Image.fromarray(full_panel)

    header_height = 28
    canvas = Image.new("RGB", (full_image.width, full_image.height + header_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), f"Sample: {sample_id}", fill=(255, 255, 255), font=font)
    canvas.paste(full_image, (0, header_height))
    return canvas


def create_output_directories(output_root: Path) -> dict[str, Path]:
    directories = {
        "pred_npy": output_root / "pred_npy",
        "target_npy": output_root / "target_npy",
        "pred_png": output_root / "pred_png",
        "target_png": output_root / "target_png",
        "overlay_png": output_root / "overlay_png",
        "comparison_png": output_root / "comparison_png",
    }

    output_root.mkdir(parents=True, exist_ok=True)
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    return directories


def save_sample_outputs(
    sample_id: str,
    rgb_image: np.ndarray,
    predicted_risk: np.ndarray,
    predicted_class_map: np.ndarray | None,
    target_risk: np.ndarray | None,
    target_class_map: np.ndarray | None,
    output_dirs: dict[str, Path],
    pred_value_max: float = 1.0,
    target_value_max: float = 1.0,
) -> None:
    if predicted_class_map is not None:
        np.save(output_dirs["pred_npy"] / f"{sample_id}.npy", predicted_class_map.astype(np.int64))
        pred_image = Image.fromarray(render_binned_risk_map(predicted_class_map))
    else:
        np.save(output_dirs["pred_npy"] / f"{sample_id}.npy", predicted_risk.astype(np.float32))
        pred_image = Image.fromarray(
            render_continuous_risk_map(predicted_risk, value_max=pred_value_max)
        )
    pred_image.save(output_dirs["pred_png"] / f"{sample_id}.png")

    overlay_image = Image.fromarray(
        np.concatenate(
            (
                overlay_risk_on_rgb(rgb_image, predicted_risk, value_max=pred_value_max),
                build_legend_image(predicted_risk.shape[0], value_max=pred_value_max),
            ),
            axis=1,
        )
    )
    overlay_image.save(output_dirs["overlay_png"] / f"{sample_id}.png")

    if target_class_map is not None:
        np.save(output_dirs["target_npy"] / f"{sample_id}.npy", target_class_map.astype(np.int64))
        target_image = Image.fromarray(render_binned_risk_map(target_class_map))
        target_image.save(output_dirs["target_png"] / f"{sample_id}.png")
    elif target_risk is not None:
        np.save(output_dirs["target_npy"] / f"{sample_id}.npy", target_risk.astype(np.float32))
        target_image = Image.fromarray(
            render_continuous_risk_map(target_risk, value_max=target_value_max)
        )
        target_image.save(output_dirs["target_png"] / f"{sample_id}.png")

    comparison_panel = make_comparison_panel(
        sample_id=sample_id,
        rgb_image=rgb_image,
        predicted_risk=predicted_risk,
        predicted_class_map=predicted_class_map,
        target_risk=target_risk,
        target_class_map=target_class_map,
        pred_value_max=pred_value_max,
        target_value_max=target_value_max,
    )
    comparison_panel.save(output_dirs["comparison_png"] / f"{sample_id}.png")


def write_validation_reports(
    output_root: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    runtime_config: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    metrics_csv_path = output_root / "metrics.csv"
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_id", "mse", "l1"])
        writer.writeheader()
        writer.writerows(rows)

    mean_mse = float(np.mean([row["mse"] for row in rows])) if rows else 0.0
    mean_l1 = float(np.mean([row["l1"] for row in rows])) if rows else 0.0

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_val_loss": float(checkpoint.get("val_loss", 0.0)),
        "num_samples": len(rows),
        "mean_mse": mean_mse,
        "mean_l1": mean_l1,
        "runtime_config": {
            "architecture": runtime_config["architecture"],
            "task_mode": runtime_config["task_mode"],
            "depth_min": runtime_config["depth_min"],
            "depth_max": runtime_config["depth_max"],
            "target_mode": runtime_config["target_mode"],
            "component_connectivity": runtime_config["component_connectivity"],
            "dataset_root": runtime_config["dataset_root"],
            "val_split": runtime_config["val_split"],
            "random_seed": runtime_config["random_seed"],
            "use_train_split_for_validation": runtime_config["use_train_split_for_validation"],
            "num_risk_classes": runtime_config["num_risk_classes"],
            "risk_scale": runtime_config["risk_scale"],
            "risk_bin_edges": list(runtime_config["risk_bin_edges"]),
            "risk_class_values": list(runtime_config["risk_class_values"]),
            "risk_class_labels": list(runtime_config["risk_class_labels"]),
            "segformer_pretrained_model": runtime_config["segformer_pretrained_model"],
            "segformer_freeze_backbone": runtime_config["segformer_freeze_backbone"],
            "segformer_adapter_hidden": runtime_config["segformer_adapter_hidden"],
        },
    }

    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)


def main() -> None:
    device = get_device()
    model, checkpoint, runtime_config = load_model_from_checkpoint(CHECKPOINT_PATH, device=device)
    loader = create_validation_loader(runtime_config)
    output_dirs = create_output_directories(OUTPUT_ROOT)

    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            raw_predictions = model(inputs)
            predictions = predictions_to_risk_tensor(raw_predictions, runtime_config)
            task_targets = targets_to_risk_tensor(targets, runtime_config)

            sample_id = str(batch["image_id"][0])
            rgb_image = tensor_to_rgb_image(batch["input"][0])
            predicted_risk = predictions[0, 0].detach().cpu().numpy()
            target_risk = task_targets[0, 0].detach().cpu().numpy()
            predicted_class_map: np.ndarray | None = None
            target_class_map: np.ndarray | None = None
            if runtime_config["task_mode"] == "risk_classification":
                predicted_class_map = torch.argmax(raw_predictions, dim=1)[0].detach().cpu().numpy()
                target_class_map = risk_targets_to_class_indices(targets, runtime_config)[0].detach().cpu().numpy()

            mse_value = float(F.mse_loss(predictions, task_targets).item())
            l1_value = float(F.l1_loss(predictions, task_targets).item())

            save_sample_outputs(
                sample_id=sample_id,
                rgb_image=rgb_image,
                predicted_risk=predicted_risk,
                predicted_class_map=predicted_class_map,
                target_risk=target_risk,
                target_class_map=target_class_map,
                output_dirs=output_dirs,
                pred_value_max=1.0,
                target_value_max=1.0,
            )

            rows.append({"image_id": sample_id, "mse": mse_value, "l1": l1_value})
            print(f"[{index}/{len(loader)}] Saved validation outputs for {sample_id}")

    write_validation_reports(
        output_root=OUTPUT_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint=checkpoint,
        runtime_config=runtime_config,
        rows=rows,
    )

    print(f"Validation results saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
