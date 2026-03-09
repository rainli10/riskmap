from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Edit these paths before running.
DEPTH_NPY_PATH = Path("data/cityscape_prepro/train/depth/0.npy")
OUTPUT_PATH = Path("output/depth_debug/0_depth.png")

IGNORE_NONPOSITIVE_AS_INVALID = True

COLOR_ANCHORS = [
    (0.00, (48, 18, 59)),
    (0.20, (50, 96, 182)),
    (0.40, (31, 180, 173)),
    (0.60, (154, 217, 60)),
    (0.80, (253, 174, 97)),
    (1.00, (215, 25, 28)),
]


def load_depth_map(depth_path: Path) -> np.ndarray:
    depth = np.load(depth_path)
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.squeeze(depth)

    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map after squeeze, got shape {depth.shape}.")

    return depth


def colorize_continuous_map(normalized_map: np.ndarray) -> np.ndarray:
    normalized = np.clip(np.asarray(normalized_map, dtype=np.float32), 0.0, 1.0)
    flat_values = normalized.reshape(-1)
    anchor_positions = np.asarray([anchor[0] for anchor in COLOR_ANCHORS], dtype=np.float32)
    anchor_colors = np.asarray([anchor[1] for anchor in COLOR_ANCHORS], dtype=np.float32)

    color_channels = [
        np.interp(flat_values, anchor_positions, anchor_colors[:, channel_idx])
        for channel_idx in range(3)
    ]
    stacked = np.stack(color_channels, axis=1)
    return stacked.reshape(normalized.shape + (3,)).astype(np.uint8)


def build_color_bar(height: int, title: str, value_min: float, value_max: float) -> np.ndarray:
    bar_width = 170
    image = Image.new("RGB", (bar_width, height), color=(24, 24, 24))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((12, 8), title, fill=(255, 255, 255), font=font)

    x0 = 16
    x1 = 44
    top_offset = 26
    bottom_padding = 8
    gradient_height = max(1, height - top_offset - bottom_padding)

    gradient_values = np.linspace(1.0, 0.0, gradient_height, dtype=np.float32)[:, None]
    gradient_map = np.repeat(gradient_values, x1 - x0, axis=1)
    gradient_image = Image.fromarray(colorize_continuous_map(gradient_map))
    image.paste(gradient_image, (x0, top_offset))
    draw.rectangle((x0, top_offset, x1 - 1, top_offset + gradient_height - 1), outline=(255, 255, 255))

    tick_values = [1.0, 0.75, 0.50, 0.25, 0.0] if height >= 140 else [1.0, 0.5, 0.0]
    value_range = value_max - value_min

    for tick in tick_values:
        y = int(round(top_offset + (1.0 - tick) * max(gradient_height - 1, 0)))
        value = value_min + tick * value_range
        draw.line((x1 + 4, y, x1 + 12, y), fill=(255, 255, 255), width=1)
        draw.text((x1 + 16, max(0, y - 5)), f"{value:.4f}", fill=(235, 235, 235), font=font)

    draw.text((x1 + 16, top_offset), "High", fill=(255, 255, 255), font=font)
    draw.text(
        (x1 + 16, max(top_offset + gradient_height - 14, 0)),
        "Low",
        fill=(255, 255, 255),
        font=font,
    )

    return np.asarray(image)


def render_raw_depth_map(depth_map: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    finite_mask = np.isfinite(depth)

    if IGNORE_NONPOSITIVE_AS_INVALID:
        valid_mask = finite_mask & (depth > 0)
    else:
        valid_mask = finite_mask

    if np.any(valid_mask):
        value_min = float(depth[valid_mask].min())
        value_max = float(depth[valid_mask].max())
    else:
        value_min = 0.0
        value_max = 1.0

    if value_max <= value_min:
        normalized = np.zeros_like(depth, dtype=np.float32)
    else:
        normalized = (depth - value_min) / (value_max - value_min)
        normalized = np.clip(normalized, 0.0, 1.0)

    depth_color = colorize_continuous_map(normalized)
    depth_color[~valid_mask] = 0

    color_bar = build_color_bar(
        height=depth.shape[0],
        title="Raw Depth",
        value_min=value_min,
        value_max=value_max,
    )
    return np.concatenate((depth_color, color_bar), axis=1)


def print_depth_stats(depth_map: np.ndarray) -> None:
    depth = np.asarray(depth_map, dtype=np.float32)
    finite_mask = np.isfinite(depth)

    if IGNORE_NONPOSITIVE_AS_INVALID:
        valid_mask = finite_mask & (depth > 0)
    else:
        valid_mask = finite_mask

    print(f"Shape: {depth.shape}")
    print(f"Raw min: {float(np.min(depth)):.6f}")
    print(f"Raw max: {float(np.max(depth)):.6f}")
    print(f"Raw mean: {float(np.mean(depth)):.6f}")

    if np.any(valid_mask):
        valid_values = depth[valid_mask]
        quantiles = np.quantile(valid_values, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        print(f"Valid min: {float(valid_values.min()):.6f}")
        print(f"Valid max: {float(valid_values.max()):.6f}")
        print(f"Valid mean: {float(valid_values.mean()):.6f}")
        print("Valid quantiles:", [round(float(value), 6) for value in quantiles])
    else:
        print("No valid depth values found.")


def main() -> None:
    depth_map = load_depth_map(DEPTH_NPY_PATH)
    visualization = render_raw_depth_map(depth_map)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visualization).save(OUTPUT_PATH)

    print(f"Saved depth visualization to: {OUTPUT_PATH}")
    print_depth_stats(depth_map)


if __name__ == "__main__":
    main()
