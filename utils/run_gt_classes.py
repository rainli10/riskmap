from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path("/home/rain/Desktop/workspace/APS360/riskmap")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dataloader import (  # noqa: E402
    RISK_BIN_EDGES,
    RISK_CLASS_LABELS,
    build_blocked_risk_map,
    quantize_risk_map,
)
from train_seg_head_simple import (  # noqa: E402
    COMPONENT_CONNECTIVITY,
    DEPTH_MAX,
    DEPTH_MIN,
    RISK_SCALE,
    SEMANTIC_WEIGHTS,
)
from utils.run_gt import collect_samples, make_panel_grid, render_label_mask  # noqa: E402
from validation import add_panel_title, render_continuous_risk_map  # noqa: E402


SPLIT_NAME = "train"  # Change to "val" if you want validation GTs instead.
INPUT_ROOT = Path("data/cityscape_prepro") / SPLIT_NAME
OUTPUT_ROOT = Path(f"output/gt_risk_classes_{SPLIT_NAME}")

RISK_CLASS_INFO = {
    0: ("[0.0, 0.2)", (49, 54, 149)),
    1: ("[0.2, 0.4)", (69, 117, 180)),
    2: ("[0.4, 0.6)", (255, 255, 191)),
    3: ("[0.6, 0.8)", (253, 174, 97)),
    4: ("[0.8, 1.0]", (215, 25, 28)),
}


def create_output_directories(output_root: Path) -> dict[str, Path]:
    directories = {
        "class_npy": output_root / "class_npy",
        "class_png": output_root / "class_png",
        "comparison_png": output_root / "comparison_png",
    }

    output_root.mkdir(parents=True, exist_ok=True)
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    return directories


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


def save_class_visualization(class_map: np.ndarray, output_path: Path) -> None:
    Image.fromarray(render_binned_risk_map(class_map)).save(output_path)


def save_comparison_visualization(
    sample_id: str,
    rgb_image: np.ndarray,
    label_map: np.ndarray,
    continuous_risk: np.ndarray,
    class_map: np.ndarray,
    output_path: Path,
) -> None:
    original_panel = add_panel_title("Original Photo", rgb_image)
    label_panel = add_panel_title("Label Mask", render_label_mask(label_map))
    class_panel = add_panel_title("Risk Classes", render_binned_risk_map(class_map))
    continuous_panel = add_panel_title(
        "Continuous Risk",
        render_continuous_risk_map(continuous_risk, value_max=1.0),
    )

    grid = make_panel_grid(
        sample_id=sample_id,
        panel_rows=[
            [original_panel, label_panel],
            [class_panel, continuous_panel],
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
    label_map = np.squeeze(np.load(sample["label"]))
    depth_map = np.squeeze(np.load(sample["depth"]).astype(np.float32))

    continuous_risk_raw = build_blocked_risk_map(
        label_map=label_map,
        depth_map=depth_map,
        semantic_weights=SEMANTIC_WEIGHTS,
        depth_min=DEPTH_MIN,
        depth_max=DEPTH_MAX,
        connectivity=COMPONENT_CONNECTIVITY,
    )
    continuous_risk = np.clip(continuous_risk_raw * float(RISK_SCALE), 0.0, 1.0).astype(np.float32)
    class_map = quantize_risk_map(continuous_risk, bin_edges=RISK_BIN_EDGES)

    np.save(output_dirs["class_npy"] / f"{sample_id}.npy", class_map.astype(np.int64))
    save_class_visualization(class_map, output_dirs["class_png"] / f"{sample_id}.png")
    save_comparison_visualization(
        sample_id=sample_id,
        rgb_image=rgb_image,
        label_map=label_map,
        continuous_risk=continuous_risk,
        class_map=class_map,
        output_path=output_dirs["comparison_png"] / f"{sample_id}.png",
    )

    unique_classes, counts = np.unique(class_map, return_counts=True)
    class_histogram = {int(class_id): int(count) for class_id, count in zip(unique_classes, counts)}

    return {
        "image_id": sample_id,
        "min_risk_raw": float(continuous_risk_raw.min()),
        "max_risk_raw": float(continuous_risk_raw.max()),
        "mean_risk_raw": float(continuous_risk_raw.mean()),
        "min_risk_scaled": float(continuous_risk.min()),
        "max_risk_scaled": float(continuous_risk.max()),
        "mean_risk_scaled": float(continuous_risk.mean()),
        "class_histogram": class_histogram,
    }


def write_summary(output_root: Path, input_root: Path, rows: list[dict[str, float | int | str]]) -> None:
    summary = {
        "input_root": str(input_root),
        "num_samples": len(rows),
        "depth_min": DEPTH_MIN,
        "depth_max": DEPTH_MAX,
        "component_connectivity": COMPONENT_CONNECTIVITY,
        "semantic_weights": SEMANTIC_WEIGHTS,
        "risk_scale": float(RISK_SCALE),
        "risk_bin_edges": list(RISK_BIN_EDGES),
        "risk_class_labels": list(RISK_CLASS_LABELS),
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
        print(f"[{index}/{len(samples)}] Saved class GT outputs for {sample['id']}")

    write_summary(output_root=OUTPUT_ROOT, input_root=INPUT_ROOT, rows=rows)
    print(f"Binned GT outputs saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
