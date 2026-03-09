from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from validation import (
    CHECKPOINT_PATH as DEFAULT_CHECKPOINT_PATH,
    build_target_risk_map,
    create_output_directories,
    get_device,
    load_model_from_checkpoint,
    save_sample_outputs,
)


CHECKPOINT_PATH = DEFAULT_CHECKPOINT_PATH
TEST_IMAGE_PATH = Path("data/cityscape_prepro/val/image_png/0.png")
TEST_DEPTH_PATH = Path("data/cityscape_prepro/val/depth/0.npy")
# Set this to None if no label file is available for the test image.
TEST_LABEL_PATH: Path | None = Path("data/cityscape_prepro/val/label/0.npy")
MODEL_INPUT_SIZE = (256, 128)  # (width, height)
OUTPUT_ROOT = Path("output/test")


def resize_map(
    array: np.ndarray,
    size: tuple[int, int],
    mode: str,
) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None, ...]

    if mode == "nearest":
        resized = F.interpolate(tensor, size=(size[1], size[0]), mode=mode)
    else:
        resized = F.interpolate(tensor, size=(size[1], size[0]), mode=mode, align_corners=False)

    return resized[0, 0].cpu().numpy()


def align_map_to_image(image_size: tuple[int, int], array: np.ndarray, mode: str) -> np.ndarray:
    squeezed = np.squeeze(np.asarray(array))
    if squeezed.ndim != 2:
        raise ValueError(f"Expected 2D array after squeeze, got shape {squeezed.shape}")

    array_size = (squeezed.shape[1], squeezed.shape[0])
    if array_size == image_size:
        return squeezed.astype(np.float32)

    return resize_map(squeezed.astype(np.float32), size=image_size, mode=mode)


def load_test_inputs(
    image_path: Path,
    depth_path: Path,
    expected_size: tuple[int, int],
    depth_max: float,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, tuple[int, int], bool]:
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    original_rgb = np.asarray(image, dtype=np.uint8)

    depth_array = np.load(depth_path).astype(np.float32)
    depth_array = align_map_to_image(original_size, depth_array, mode="bilinear")

    resized = original_size != expected_size
    model_image = image.resize(expected_size, Image.BILINEAR) if resized else image
    model_depth = resize_map(depth_array, size=expected_size, mode="bilinear") if resized else depth_array

    image_array = np.asarray(model_image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1))).float()
    depth_tensor = torch.from_numpy(
        np.clip(model_depth / max(depth_max, 1e-6), 0.0, 1.0)[None, ...]
    ).float()

    input_tensor = torch.cat((image_tensor, depth_tensor), dim=0)
    return input_tensor, original_rgb, depth_array, original_size, resized


def load_optional_label(label_path: Path | None, image_size: tuple[int, int]) -> np.ndarray | None:
    if label_path is None:
        return None
    if not label_path.exists():
        return None

    label_array = np.load(label_path)
    return align_map_to_image(image_size, label_array, mode="nearest")


def resize_prediction_back(
    prediction: np.ndarray,
    original_size: tuple[int, int],
) -> np.ndarray:
    prediction_size = (prediction.shape[1], prediction.shape[0])
    if prediction_size == original_size:
        return prediction.astype(np.float32)

    return resize_map(prediction.astype(np.float32), size=original_size, mode="bilinear")


def write_test_summary(
    output_root: Path,
    sample_id: str,
    checkpoint_path: Path,
    runtime_config: dict[str, Any],
    original_size: tuple[int, int],
    resized: bool,
    mse_value: float | None,
    l1_value: float | None,
) -> None:
    summary = {
        "sample_id": sample_id,
        "checkpoint_path": str(checkpoint_path),
        "original_size": {"width": original_size[0], "height": original_size[1]},
        "model_input_size": {"width": MODEL_INPUT_SIZE[0], "height": MODEL_INPUT_SIZE[1]},
        "resized_for_model": resized,
        "target_mode": runtime_config["target_mode"],
        "component_connectivity": runtime_config["component_connectivity"],
        "mse": mse_value,
        "l1": l1_value,
    }

    with (output_root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def main() -> None:
    device = get_device()
    model, _, runtime_config = load_model_from_checkpoint(CHECKPOINT_PATH, device=device)
    output_dirs = create_output_directories(OUTPUT_ROOT)

    input_tensor, original_rgb, original_depth, original_size, resized = load_test_inputs(
        image_path=TEST_IMAGE_PATH,
        depth_path=TEST_DEPTH_PATH,
        expected_size=MODEL_INPUT_SIZE,
        depth_max=runtime_config["depth_max"],
    )
    label_array = load_optional_label(TEST_LABEL_PATH, image_size=original_size)

    with torch.no_grad():
        prediction = model(input_tensor.unsqueeze(0).to(device))

    predicted_risk = prediction[0, 0].detach().cpu().numpy()
    predicted_risk = resize_prediction_back(predicted_risk, original_size=original_size)

    target_risk = None
    mse_value = None
    l1_value = None

    if label_array is not None:
        target_risk = build_target_risk_map(
            label_map=label_array,
            depth_map=original_depth,
            runtime_config=runtime_config,
        )
        mse_value = float(np.mean((predicted_risk - target_risk) ** 2))
        l1_value = float(np.mean(np.abs(predicted_risk - target_risk)))

    sample_id = TEST_IMAGE_PATH.stem
    save_sample_outputs(
        sample_id=sample_id,
        rgb_image=original_rgb,
        predicted_risk=predicted_risk,
        target_risk=target_risk,
        output_dirs=output_dirs,
    )

    write_test_summary(
        output_root=OUTPUT_ROOT,
        sample_id=sample_id,
        checkpoint_path=CHECKPOINT_PATH,
        runtime_config=runtime_config,
        original_size=original_size,
        resized=resized,
        mse_value=mse_value,
        l1_value=l1_value,
    )

    print(f"Test outputs saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
