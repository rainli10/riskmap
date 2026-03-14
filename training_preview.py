from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import RISK_CLASS_LABELS


RISK_CLASS_COLORS = {
    0: (49, 54, 149),
    1: (69, 117, 180),
    2: (255, 255, 191),
    3: (253, 174, 97),
    4: (215, 25, 28),
}


def tensor_to_rgb_image(input_tensor: torch.Tensor) -> np.ndarray:
    rgb_tensor = input_tensor[:3].detach().cpu().clamp(0.0, 1.0)
    rgb_image = np.transpose(rgb_tensor.numpy(), (1, 2, 0))
    return (rgb_image * 255.0).round().astype(np.uint8)


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
    legend_width = 220
    legend = Image.new("RGB", (legend_width, height), color=(24, 24, 24))
    draw = ImageDraw.Draw(legend)
    font = ImageFont.load_default()
    draw.text((12, 8), "Risk Classes", fill=(255, 255, 255), font=font)

    y = 30
    row_height = 18
    for class_id, label in enumerate(RISK_CLASS_LABELS):
        color = RISK_CLASS_COLORS[class_id]
        draw.rectangle((12, y, 28, y + 11), fill=color)
        draw.text((36, y - 1), f"{class_id}: {label}", fill=(235, 235, 235), font=font)
        y += row_height

    draw.text((12, y + 6), "Green = correct", fill=(150, 255, 150), font=font)
    draw.text((12, y + 22), "Red = wrong", fill=(255, 150, 150), font=font)
    return np.asarray(legend)


def render_class_map(class_map: np.ndarray) -> np.ndarray:
    classes = np.asarray(class_map, dtype=np.int64)
    color_map = np.zeros(classes.shape + (3,), dtype=np.uint8)
    for class_id, color in RISK_CLASS_COLORS.items():
        color_map[classes == int(class_id)] = color
    legend = build_class_legend(classes.shape[0])
    return np.concatenate((color_map, legend), axis=1)


def render_correctness_map(predicted_classes: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted_classes, dtype=np.int64)
    target = np.asarray(target_classes, dtype=np.int64)
    correct = predicted == target
    correctness = np.zeros(predicted.shape + (3,), dtype=np.uint8)
    correctness[correct] = (40, 180, 99)
    correctness[~correct] = (220, 60, 60)
    legend = Image.new("RGB", (220, predicted.shape[0]), color=(24, 24, 24))
    draw = ImageDraw.Draw(legend)
    font = ImageFont.load_default()
    draw.text((12, 8), "Prediction vs GT", fill=(255, 255, 255), font=font)
    draw.rectangle((12, 30, 28, 41), fill=(40, 180, 99))
    draw.text((36, 28), "Correct class", fill=(235, 235, 235), font=font)
    draw.rectangle((12, 52, 28, 63), fill=(220, 60, 60))
    draw.text((36, 50), "Wrong class", fill=(235, 235, 235), font=font)
    return np.concatenate((correctness, np.asarray(legend)), axis=1)


def build_preview_image(
    sample_id: str,
    rgb_image: np.ndarray,
    predicted_classes: np.ndarray,
    target_classes: np.ndarray,
) -> np.ndarray:
    panels = [
        add_panel_title("RGB Image", rgb_image),
        add_panel_title("Predicted Classes", render_class_map(predicted_classes)),
        add_panel_title("GT Classes", render_class_map(target_classes)),
        add_panel_title("Correctness", render_correctness_map(predicted_classes, target_classes)),
    ]

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
    combined_image = Image.fromarray(combined)
    header_height = 28
    canvas = Image.new("RGB", (combined_image.width, combined_image.height + header_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), f"Sample: {sample_id}", fill=(255, 255, 255), font=font)
    canvas.paste(combined_image, (0, header_height))
    return np.asarray(canvas)


def save_training_previews(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    output_root: Path,
    target_to_classes_fn: Callable[[torch.Tensor], torch.Tensor],
    num_samples: int = 3,
    writer: SummaryWriter | None = None,
    writer_prefix: str = "preview",
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    epoch_dir = output_root / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    model_was_training = model.training
    model.eval()

    saved_samples = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            logits = model(inputs)
            predicted_classes_batch = torch.argmax(logits, dim=1).detach().cpu()
            target_classes_batch = target_to_classes_fn(targets).detach().cpu()

            image_ids = batch["image_id"]
            batch_size = inputs.shape[0]
            for item_idx in range(batch_size):
                if saved_samples >= num_samples:
                    break

                sample_id = str(image_ids[item_idx])
                rgb_image = tensor_to_rgb_image(batch["input"][item_idx])
                predicted_classes = predicted_classes_batch[item_idx].numpy()
                target_classes = target_classes_batch[item_idx].numpy()
                preview_image = build_preview_image(
                    sample_id=sample_id,
                    rgb_image=rgb_image,
                    predicted_classes=predicted_classes,
                    target_classes=target_classes,
                )

                Image.fromarray(preview_image).save(epoch_dir / f"{saved_samples + 1:02d}_{sample_id}.png")

                if writer is not None:
                    preview_tensor = torch.from_numpy(preview_image).permute(2, 0, 1).float() / 255.0
                    writer.add_image(
                        f"{writer_prefix}/{saved_samples + 1:02d}_{sample_id}",
                        preview_tensor,
                        global_step=epoch,
                    )

                saved_samples += 1

            if saved_samples >= num_samples:
                break

    model.train(model_was_training)
