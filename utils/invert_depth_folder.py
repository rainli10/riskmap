from __future__ import annotations

from pathlib import Path

import numpy as np


# Edit these paths before running.
INPUT_FOLDER = Path("data/cityscape_prepro/train/depth_inv")
OUTPUT_FOLDER = Path("data/cityscape_prepro/train/depth")
MIN_SOURCE_DEPTH = 0.005


def invert_depth_array(
    depth_array: np.ndarray,
    min_source_depth: float = MIN_SOURCE_DEPTH,
    eps: float = 1e-6,
) -> np.ndarray:
    """Invert depth values while zeroing invalid or too-small inputs."""
    depth = np.asarray(depth_array, dtype=np.float32)
    inverted = np.zeros_like(depth, dtype=np.float32)

    valid_mask = np.isfinite(depth) & (depth >= max(min_source_depth, eps))
    inverted[valid_mask] = 1.0 / depth[valid_mask]

    return inverted


def transform_depth_folder(input_folder: Path, output_folder: Path) -> None:
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(input_folder.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in: {input_folder}")
        return

    for npy_path in npy_files:
        depth_array = np.load(npy_path)
        inverted_depth = invert_depth_array(depth_array)

        output_path = output_folder / npy_path.name
        np.save(output_path, inverted_depth.astype(np.float32))

        valid_mask = np.isfinite(inverted_depth) & (inverted_depth > 0)
        if np.any(valid_mask):
            min_val = float(inverted_depth[valid_mask].min())
            max_val = float(inverted_depth[valid_mask].max())
            print(
                f"Saved: {output_path} | "
                f"source threshold >= {MIN_SOURCE_DEPTH:.4f} | "
                f"valid range: {min_val:.6f} to {max_val:.6f}"
            )
        else:
            print(f"Saved: {output_path} | no valid positive depth values")


if __name__ == "__main__":
    transform_depth_folder(INPUT_FOLDER, OUTPUT_FOLDER)
