from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from dataloader import RiskMapDataset, build_blocked_risk_map, build_dense_risk_map
from model import SimpleRiskCNN
from train import (
    ARCHITECTURE as TRAIN_ARCHITECTURE,
    COMPONENT_CONNECTIVITY as TRAIN_COMPONENT_CONNECTIVITY,
    DEPTH_MAX as TRAIN_DEPTH_MAX,
    DEPTH_MIN as TRAIN_DEPTH_MIN,
    SEMANTIC_WEIGHTS as TRAIN_SEMANTIC_WEIGHTS,
    TARGET_MODE as TRAIN_TARGET_MODE,
)


CHECKPOINT_PATH = Path("checkpoints/baseline_cnn_best.pt")
VALIDATION_DATASET_ROOT = Path("data/cityscape_prepro/val")
OUTPUT_ROOT = Path("output/validation")
BATCH_SIZE = 1
NUM_WORKERS = 0

RISK_LEVELS = [
    ("Very Low", 0.00, 0.20, (49, 54, 149)),
    ("Low", 0.20, 0.40, (69, 117, 180)),
    ("Medium", 0.40, 0.60, (255, 255, 191)),
    ("High", 0.60, 0.80, (253, 174, 97)),
    ("Critical", 0.80, 1.01, (215, 25, 28)),
]

RISK_COLOR_ANCHORS = [
    (0.00, (48, 18, 59)),
    (0.20, (50, 96, 182)),
    (0.40, (31, 180, 173)),
    (0.60, (154, 217, 60)),
    (0.80, (253, 174, 97)),
    (1.00, (215, 25, 28)),
]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_runtime_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "architecture": checkpoint.get("architecture", TRAIN_ARCHITECTURE),
        "semantic_weights": checkpoint.get("semantic_weights", TRAIN_SEMANTIC_WEIGHTS),
        "depth_min": float(checkpoint.get("depth_min", TRAIN_DEPTH_MIN)),
        "depth_max": float(checkpoint.get("depth_max", TRAIN_DEPTH_MAX)),
        "target_mode": str(checkpoint.get("target_mode", TRAIN_TARGET_MODE)),
        "component_connectivity": int(
            checkpoint.get("component_connectivity", TRAIN_COMPONENT_CONNECTIVITY)
        ),
    }


def build_model(architecture: str) -> SimpleRiskCNN:
    if architecture == "simplest_cnn":
        return SimpleRiskCNN()

    raise ValueError(f"Unsupported architecture '{architecture}'.")


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[SimpleRiskCNN, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    runtime_config = build_runtime_config(checkpoint)

    model = build_model(runtime_config["architecture"]).to(device)
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


def create_validation_loader(runtime_config: dict[str, Any]) -> DataLoader:
    dataset = RiskMapDataset(
        dataset_root=VALIDATION_DATASET_ROOT,
        semantic_weights=runtime_config["semantic_weights"],
        depth_min=runtime_config["depth_min"],
        depth_max=runtime_config["depth_max"],
        target_mode=runtime_config["target_mode"],
        component_connectivity=runtime_config["component_connectivity"],
    )

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


def colorize_risk_map_continuous(risk_map: np.ndarray) -> np.ndarray:
    risk = np.clip(np.asarray(risk_map, dtype=np.float32), 0.0, 1.0)
    flat_risk = risk.reshape(-1)
    anchor_positions = np.asarray([anchor[0] for anchor in RISK_COLOR_ANCHORS], dtype=np.float32)
    anchor_colors = np.asarray([anchor[1] for anchor in RISK_COLOR_ANCHORS], dtype=np.float32)

    color_channels = [
        np.interp(flat_risk, anchor_positions, anchor_colors[:, channel_idx])
        for channel_idx in range(3)
    ]
    stacked = np.stack(color_channels, axis=1)
    return stacked.reshape(risk.shape + (3,)).astype(np.uint8)


def overlay_risk_on_rgb(rgb_image: np.ndarray, risk_map: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_image, dtype=np.float32)
    colorized = colorize_risk_map(risk_map).astype(np.float32)
    alpha = np.clip(np.asarray(risk_map, dtype=np.float32), 0.0, 1.0)[..., None]
    alpha = np.where(alpha > 0.0, 0.15 + 0.55 * alpha, 0.0)
    blended = rgb * (1.0 - alpha) + colorized * alpha
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def build_legend_image(height: int) -> np.ndarray:
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
    for value in tick_values:
        y = int(round(top_offset + (1.0 - value) * max(bar_height - 1, 0)))
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


def render_continuous_risk_map(risk_map: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (colorize_risk_map_continuous(risk_map), build_legend_image(risk_map.shape[0])),
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


def make_comparison_panel(
    sample_id: str,
    rgb_image: np.ndarray,
    predicted_risk: np.ndarray,
    target_risk: np.ndarray | None = None,
) -> Image.Image:
    panels = [
        add_panel_title("RGB Image", rgb_image),
        add_panel_title("Prediction Overlay", overlay_risk_on_rgb(rgb_image, predicted_risk)),
        add_panel_title("Predicted Risk", colorize_risk_map_continuous(predicted_risk)),
    ]

    if target_risk is not None:
        panels.append(add_panel_title("Target Risk", colorize_risk_map_continuous(target_risk)))

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
    legend = build_legend_image(combined.shape[0])
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
    target_risk: np.ndarray | None,
    output_dirs: dict[str, Path],
) -> None:
    np.save(output_dirs["pred_npy"] / f"{sample_id}.npy", predicted_risk.astype(np.float32))
    pred_image = Image.fromarray(render_continuous_risk_map(predicted_risk))
    pred_image.save(output_dirs["pred_png"] / f"{sample_id}.png")

    overlay_image = Image.fromarray(
        np.concatenate(
            (overlay_risk_on_rgb(rgb_image, predicted_risk), build_legend_image(predicted_risk.shape[0])),
            axis=1,
        )
    )
    overlay_image.save(output_dirs["overlay_png"] / f"{sample_id}.png")

    if target_risk is not None:
        np.save(output_dirs["target_npy"] / f"{sample_id}.npy", target_risk.astype(np.float32))
        target_image = Image.fromarray(render_continuous_risk_map(target_risk))
        target_image.save(output_dirs["target_png"] / f"{sample_id}.png")

    comparison_panel = make_comparison_panel(
        sample_id=sample_id,
        rgb_image=rgb_image,
        predicted_risk=predicted_risk,
        target_risk=target_risk,
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
            "depth_min": runtime_config["depth_min"],
            "depth_max": runtime_config["depth_max"],
            "target_mode": runtime_config["target_mode"],
            "component_connectivity": runtime_config["component_connectivity"],
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
            predictions = model(inputs)

            sample_id = str(batch["image_id"][0])
            rgb_image = tensor_to_rgb_image(batch["input"][0])
            predicted_risk = predictions[0, 0].detach().cpu().numpy()
            target_risk = targets[0, 0].detach().cpu().numpy()

            mse_value = float(F.mse_loss(predictions, targets).item())
            l1_value = float(F.l1_loss(predictions, targets).item())

            save_sample_outputs(
                sample_id=sample_id,
                rgb_image=rgb_image,
                predicted_risk=predicted_risk,
                target_risk=target_risk,
                output_dirs=output_dirs,
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
