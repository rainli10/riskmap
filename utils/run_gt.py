from __future__ import annotations

import json
from math import ceil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.append("/home/rain/Desktop/workspace/APS360/riskmap")
from dataloader import build_blocked_risk_map, compute_depth_weight
from train import COMPONENT_CONNECTIVITY, DEPTH_MAX, DEPTH_MIN, SEMANTIC_WEIGHTS
from validation import (
    add_panel_title,
    colorize_risk_map_continuous,
    make_comparison_panel,
    overlay_risk_on_rgb,
    render_continuous_risk_map,
)


# Edit these paths before running if needed.
INPUT_ROOT = Path("data/cityscape_prepro/train")
OUTPUT_ROOT = Path("output/gt_risk_train")

IMAGE_FOLDER = "image_png"
LABEL_FOLDER = "label"
DEPTH_FOLDER = "depth"

CLASS_INFO = {
    -1: ("ignore", (0, 0, 0)),
    0: ("road", (128, 64, 128)),
    1: ("sidewalk", (244, 35, 232)),
    2: ("building", (70, 70, 70)),
    3: ("wall", (102, 102, 156)),
    4: ("fence", (190, 153, 153)),
    5: ("pole", (153, 153, 153)),
    6: ("traffic light", (250, 170, 30)),
    7: ("traffic sign", (220, 220, 0)),
    8: ("vegetation", (107, 142, 35)),
    9: ("terrain", (152, 251, 152)),
    10: ("sky", (70, 130, 180)),
    11: ("person", (220, 20, 60)),
    12: ("rider", (255, 0, 0)),
    13: ("car", (0, 0, 142)),
    14: ("truck", (0, 0, 70)),
    15: ("bus", (0, 60, 100)),
    16: ("train", (0, 80, 100)),
    17: ("motorcycle", (0, 0, 230)),
    18: ("bicycle", (119, 11, 32)),
}


def collect_samples(input_root: Path) -> list[dict[str, Path | str]]:
    image_dir = input_root / IMAGE_FOLDER
    label_dir = input_root / LABEL_FOLDER
    depth_dir = input_root / DEPTH_FOLDER

    for directory in (image_dir, label_dir, depth_dir):
        if not directory.exists():
            raise FileNotFoundError(f"Required dataset directory does not exist: {directory}")

    samples: list[dict[str, Path | str]] = []
    for image_path in sorted(image_dir.glob("*.png")):
        sample_id = image_path.stem
        label_path = label_dir / f"{sample_id}.npy"
        depth_path = depth_dir / f"{sample_id}.npy"

        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file for {sample_id}: {label_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth file for {sample_id}: {depth_path}")

        samples.append(
            {
                "id": sample_id,
                "image": image_path,
                "label": label_path,
                "depth": depth_path,
            }
        )

    if not samples:
        raise RuntimeError(f"No PNG images found in {image_dir}")

    return samples


def create_output_directories(output_root: Path) -> dict[str, Path]:
    directories = {
        "risk_npy": output_root / "risk_npy",
        "risk_png": output_root / "risk_png",
        "overlay_png": output_root / "overlay_png",
        "comparison_png": output_root / "comparison_png",
        "depth_com_risk": output_root / "depth_com_risk",
    }

    output_root.mkdir(parents=True, exist_ok=True)
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    return directories


def save_risk_visualization(risk_map: np.ndarray, output_path: Path) -> None:
    Image.fromarray(render_continuous_risk_map(risk_map)).save(output_path)


def build_scalar_legend_image(
    height: int,
    title: str,
    value_min: float,
    value_max: float,
) -> np.ndarray:
    legend_width = 170
    legend = Image.new("RGB", (legend_width, height), color=(24, 24, 24))
    draw = ImageDraw.Draw(legend)
    font = ImageFont.load_default()

    if height <= 0:
        raise ValueError(f"Legend height must be positive, got {height}.")

    draw.text((12, 8), title, fill=(255, 255, 255), font=font)

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
    for tick in tick_values:
        y = int(round(top_offset + (1.0 - tick) * max(bar_height - 1, 0)))
        value = value_min + tick * value_range
        draw.line((bar_x1 + 4, y, bar_x1 + 12, y), fill=(255, 255, 255), width=1)
        draw.text((bar_x1 + 16, max(0, y - 5)), f"{value:.3f}", fill=(235, 235, 235), font=font)

    draw.text((bar_x1 + 16, top_offset), "High", fill=(255, 255, 255), font=font)
    draw.text(
        (bar_x1 + 16, max(top_offset + bar_height - 14, 0)),
        "Low",
        fill=(255, 255, 255),
        font=font,
    )

    return np.asarray(legend)


def render_continuous_depth_map(depth_map: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    valid_mask = depth > 0

    if np.any(valid_mask):
        value_min = float(depth[valid_mask].min())
        value_max = float(depth[valid_mask].max())
    else:
        value_min = 0.0
        value_max = 1.0

    if value_max <= value_min:
        normalized_depth = np.zeros_like(depth, dtype=np.float32)
    else:
        normalized_depth = (1/depth - 1/value_min) / (1/value_max - 1/value_min)
        normalized_depth = np.clip(normalized_depth, 0.0, 1.0)

    depth_color = colorize_risk_map_continuous(normalized_depth)
    depth_color[~valid_mask] = 0
    legend = build_scalar_legend_image(
        height=depth.shape[0],
        title="Depth",
        value_min=value_min,
        value_max=value_max,
    )
    return np.concatenate((depth_color, legend), axis=1)


def render_continuous_scalar_map(
    scalar_map: np.ndarray,
    title: str,
    value_min: float,
    value_max: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    scalar = np.asarray(scalar_map, dtype=np.float32)

    if value_max <= value_min:
        normalized = np.zeros_like(scalar, dtype=np.float32)
    else:
        normalized = np.clip((scalar - value_min) / (value_max - value_min), 0.0, 1.0)

    scalar_color = colorize_risk_map_continuous(normalized)
    if valid_mask is not None:
        scalar_color[~valid_mask] = 0

    legend = build_scalar_legend_image(
        height=scalar.shape[0],
        title=title,
        value_min=value_min,
        value_max=value_max,
    )
    return np.concatenate((scalar_color, legend), axis=1)


def build_label_legend_block(width: int) -> np.ndarray:
    font = ImageFont.load_default()
    side_padding = 8
    top_offset = 22
    bottom_padding = 8
    row_height = 12
    entries = list(CLASS_INFO.items())
    column_count = 2 if width < 360 else 3
    column_count = min(column_count, max(1, len(entries)))
    usable_width = max(width - 2 * side_padding, column_count * 60)
    column_width = max(60, usable_width // column_count)
    row_count = ceil(len(entries) / column_count)
    height = top_offset + row_count * row_height + bottom_padding

    legend = Image.new("RGB", (width, height), color=(24, 24, 24))
    draw = ImageDraw.Draw(legend)
    draw.text((side_padding, 6), "Classes", fill=(255, 255, 255), font=font)

    for index, (class_id, (class_name, class_color)) in enumerate(entries):
        col = index % column_count
        row = index // column_count
        x0 = side_padding + col * column_width
        y0 = top_offset + row * row_height
        y1 = min(y0 + 8, height - bottom_padding)

        draw.rectangle((x0, y0, x0 + 12, y1), fill=class_color)
        draw.text(
            (x0 + 18, max(0, y0 - 1)),
            f"{class_id}: {class_name}",
            fill=(235, 235, 235),
            font=font,
        )

    return np.asarray(legend)


def render_label_mask(label_map: np.ndarray) -> np.ndarray:
    label_ids = np.rint(np.asarray(label_map)).astype(np.int64)
    label_color = np.zeros(label_ids.shape + (3,), dtype=np.uint8)

    for class_id in np.unique(label_ids):
        class_color = CLASS_INFO.get(int(class_id), (f"class_{int(class_id)}", (255, 255, 255)))[1]
        label_color[label_ids == int(class_id)] = class_color

    legend = build_label_legend_block(label_ids.shape[1])
    return np.concatenate((label_color, legend), axis=0)


def make_panel_grid(
    sample_id: str,
    panel_rows: list[list[np.ndarray]],
    gap: int = 12,
    outer_padding: int = 12,
) -> Image.Image:
    row_heights = [max(panel.shape[0] for panel in row) for row in panel_rows]
    column_count = max(len(row) for row in panel_rows)
    column_widths = []
    for column_idx in range(column_count):
        column_width = max(
            row[column_idx].shape[1] if column_idx < len(row) else 0 for row in panel_rows
        )
        column_widths.append(column_width)

    header_height = 32
    canvas_height = (
        header_height
        + outer_padding * 2
        + sum(row_heights)
        + gap * max(len(panel_rows) - 1, 0)
    )
    canvas_width = (
        outer_padding * 2
        + sum(column_widths)
        + gap * max(column_count - 1, 0)
    )

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((outer_padding, 8), f"Sample: {sample_id}", fill=(255, 255, 255), font=font)

    y_offset = header_height + outer_padding
    for row_idx, row in enumerate(panel_rows):
        x_offset = outer_padding
        for column_idx, panel in enumerate(row):
            target_width = column_widths[column_idx]
            target_height = row_heights[row_idx]

            cell = np.full((target_height, target_width, 3), 18, dtype=np.uint8)
            y_pad = max((target_height - panel.shape[0]) // 2, 0)
            x_pad = max((target_width - panel.shape[1]) // 2, 0)
            cell[y_pad : y_pad + panel.shape[0], x_pad : x_pad + panel.shape[1], :] = panel

            canvas.paste(Image.fromarray(cell), (x_offset, y_offset))
            x_offset += target_width + gap

        y_offset += row_heights[row_idx] + gap

    return canvas


def save_depth_comparison_visualization(
    sample_id: str,
    rgb_image: np.ndarray,
    label_map: np.ndarray,
    depth_map: np.ndarray,
    risk_map: np.ndarray,
    output_path: Path,
) -> None:
    valid_depth_mask = np.asarray(depth_map, dtype=np.float32) > 0
    depth_weight_map = compute_depth_weight(
        depth_map=depth_map,
        depth_min=DEPTH_MIN,
        depth_max=DEPTH_MAX,
    )
    # depth_weight_map = 2*depth_weight_map

    original_panel = add_panel_title("Original Photo", rgb_image)
    label_panel = add_panel_title("Label Mask", render_label_mask(label_map))
    depth_weight_panel = add_panel_title(
        "Depth Weight",
        render_continuous_scalar_map(
            scalar_map=depth_weight_map,
            title="Depth Weight",
            value_min=0.0,
            value_max=1.0,
            valid_mask=valid_depth_mask,
        ),
    )
    risk_panel = add_panel_title("Risk", render_continuous_risk_map(risk_map))
    grid = make_panel_grid(
        sample_id=sample_id,
        panel_rows=[
            [original_panel, label_panel],
            [depth_weight_panel, risk_panel],
        ],
    )
    grid.save(output_path)


def process_sample(
    sample: dict[str, Path | str],
    output_dirs: dict[str, Path],
) -> dict[str, float | int | str]:
    sample_id = str(sample["id"])

    image = Image.open(sample["image"]).convert("RGB")
    rgb_image = np.asarray(image, dtype=np.uint8)
    label_map = np.load(sample["label"])
    depth_map = np.load(sample["depth"]).astype(np.float32)

    label_map = np.squeeze(label_map)
    depth_map = np.squeeze(depth_map)

    if label_map.shape != depth_map.shape:
        raise ValueError(
            f"Label shape {label_map.shape} does not match depth shape {depth_map.shape} "
            f"for sample {sample_id}"
        )
    if label_map.shape != rgb_image.shape[:2]:
        raise ValueError(
            f"Image size {rgb_image.shape[:2]} does not match label/depth size {label_map.shape} "
            f"for sample {sample_id}"
        )

    risk_map = build_blocked_risk_map(
        label_map=label_map,
        depth_map=depth_map,
        semantic_weights=SEMANTIC_WEIGHTS,
        depth_min=DEPTH_MIN,
        depth_max=DEPTH_MAX,
        connectivity=COMPONENT_CONNECTIVITY,
    )

    np.save(output_dirs["risk_npy"] / f"{sample_id}.npy", risk_map.astype(np.float32))
    save_risk_visualization(risk_map, output_dirs["risk_png"] / f"{sample_id}.png")
    save_depth_comparison_visualization(
        sample_id=sample_id,
        rgb_image=rgb_image,
        label_map=label_map,
        depth_map=depth_map,
        risk_map=risk_map,
        output_path=output_dirs["depth_com_risk"] / f"{sample_id}.png",
    )

    overlay = overlay_risk_on_rgb(rgb_image, risk_map)
    Image.fromarray(overlay).save(output_dirs["overlay_png"] / f"{sample_id}.png")

    comparison = make_comparison_panel(
        sample_id=sample_id,
        rgb_image=rgb_image,
        predicted_risk=risk_map,
        target_risk=None,
    )
    comparison.save(output_dirs["comparison_png"] / f"{sample_id}.png")

    return {
        "image_id": sample_id,
        "min_risk": float(risk_map.min()),
        "max_risk": float(risk_map.max()),
        "mean_risk": float(risk_map.mean()),
    }


def write_summary(output_root: Path, input_root: Path, rows: list[dict[str, float | int | str]]) -> None:
    summary = {
        "input_root": str(input_root),
        "num_samples": len(rows),
        "depth_min": DEPTH_MIN,
        "depth_max": DEPTH_MAX,
        "component_connectivity": COMPONENT_CONNECTIVITY,
        "semantic_weights": SEMANTIC_WEIGHTS,
        "samples": rows,
    }

    with (output_root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def main() -> None:
    samples = collect_samples(INPUT_ROOT)
    output_dirs = create_output_directories(OUTPUT_ROOT)

    rows: list[dict[str, float | int | str]] = []
    for index, sample in enumerate(samples, start=1):
        row = process_sample(sample, output_dirs)
        rows.append(row)
        print(f"[{index}/{len(samples)}] Saved GT blocked risk visualization for {row['image_id']}")

    write_summary(OUTPUT_ROOT, INPUT_ROOT, rows)
    print(f"GT blocked risk outputs saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
