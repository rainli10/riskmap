"""
Prepare one RGB + depth + semantic label sample for inference (train_ours / validation_ours style).

What the pipeline expects
-------------------------
``RiskMapDataset`` (and ``test.py``) assume this layout under a single root directory::

    <root>/
      image_png/<sample_id>.png   # RGB, uint8
      depth/<sample_id>.npy       # 2D float32, same H×W as the image
      label/<sample_id>.npy       # 2D int64 train-style class ids (see below)

Depth is later divided by ``depth_max`` from the checkpoint (often 100) and clipped to [0, 1],
so store **raw depth in the same units as training** (e.g. meters in [DEPTH_MIN, DEPTH_MAX]).

Labels must use **integer class ids that exist as keys in** ``SEMANTIC_WEIGHTS`` (see
``train.py`` / checkpoint): typically Cityscapes **trainId** 0–18, with **-1** (or 255 in an
8-bit image) for void/ignore. The dataloader rounds to int64; every id that appears in the map
needs a weight in ``semantic_weights`` or ``build_*_risk_map`` will raise.

Different spatial sizes
-----------------------
If RGB, depth, and label do not share width×height, this script **resamples** depth to the RGB
size with **bilinear** interpolation and the label with **nearest** neighbor so all three align.

Different size from the SegFormer checkpoint
--------------------------------------------
Checkpoints are trained at a fixed input resolution (e.g. 256×128 in ``test.py``). You can:

1. **Prepare at full resolution** (``model_input_size=None``): folder is valid for
   ``RiskMapDataset`` at native resolution; resize inside your inference script before the
   forward pass (as ``test.py`` does), then resize predictions back to the original size.
2. **Prepare already resized** (set ``model_input_size=(W, H)``): writes PNG/NPY at that
   resolution so you can point the loader or a minimal script at this folder without extra
   resize logic (trades away full-res GT overlays unless you keep a separate full-res copy).

This module only writes files; run ``test.py`` (or your own loop) for the actual forward pass.

For the ``test/images`` + ``test/depth`` + ``test/label`` tree (including WebM sidecars), use
``prepare_test_folder.py`` instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# --- Edit these for a one-off run -------------------------------------------------
RGB_PATH = Path("path/to/your/image.png")
DEPTH_PATH = Path("path/to/your/depth.npy")
LABEL_PATH = Path("path/to/your/label.png")  # or .npy
OUTPUT_ROOT = Path("output/prepared_single")
SAMPLE_ID = "custom_001"
# Set to (width, height) to match SegFormer training, or None to keep RGB resolution.
MODEL_INPUT_SIZE: tuple[int, int] | None = (256, 128)
# How to interpret the label file when it is an image:
#   "auto" — .npy always load as array; .png/.jpg use rules below
#   "npy" — force numpy load (path must be .npy)
#   "grayscale_trainid" — single channel: values 0–18, 255 -> -1 (void)
#   "cityscapes_color" — RGB PNG with official Cityscapes label colors -> trainId (-1 if unknown)
LABEL_MODE: Literal["auto", "npy", "grayscale_trainid", "cityscapes_color"] = "auto"


# Cityscapes label colors (R, G, B) -> trainId. Unknown RGB -> -1 (ignore).
_CITYSCAPES_COLOR_TO_TRAIN_ID: dict[tuple[int, int, int], int] = {
    (128, 64, 128): 0,
    (244, 35, 232): 1,
    (70, 70, 70): 2,
    (102, 102, 156): 3,
    (190, 153, 153): 4,
    (153, 153, 153): 5,
    (250, 170, 30): 6,
    (220, 220, 0): 7,
    (107, 142, 35): 8,
    (152, 251, 152): 9,
    (70, 130, 180): 10,
    (220, 20, 60): 11,
    (255, 0, 0): 12,
    (0, 0, 142): 13,
    (0, 0, 70): 14,
    (0, 60, 100): 15,
    (0, 80, 100): 16,
    (0, 0, 230): 17,
    (119, 11, 32): 18,
}


def resize_map(array: np.ndarray, size: tuple[int, int], mode: str) -> np.ndarray:
    """Resize 2D map to ``size`` (width, height). ``mode`` is ``nearest`` or ``bilinear``."""
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None, ...]
    if mode == "nearest":
        resized = F.interpolate(tensor, size=(size[1], size[0]), mode=mode)
    else:
        resized = F.interpolate(tensor, size=(size[1], size[0]), mode=mode, align_corners=False)
    return resized[0, 0].cpu().numpy()


def align_map_to_image(image_size: tuple[int, int], array: np.ndarray, mode: str) -> np.ndarray:
    """Resize ``array`` to ``image_size`` (width, height) if needed."""
    squeezed = np.squeeze(np.asarray(array))
    if squeezed.ndim != 2:
        raise ValueError(f"Expected 2D map after squeeze, got shape {squeezed.shape}")
    array_size = (squeezed.shape[1], squeezed.shape[0])
    if array_size == image_size:
        return squeezed.astype(np.float32) if mode != "nearest" else squeezed.astype(np.float32)
    return resize_map(squeezed.astype(np.float32), size=image_size, mode=mode)


def rgb_label_to_train_ids(rgb: np.ndarray) -> np.ndarray:
    """Map H×W×3 uint8 Cityscapes colors to trainIds (-1 unknown)."""
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    colors, inv = np.unique(flat, axis=0, return_inverse=True)
    tids = np.array(
        [
            int(_CITYSCAPES_COLOR_TO_TRAIN_ID.get((int(c[0]), int(c[1]), int(c[2])), -1))
            for c in colors
        ],
        dtype=np.int64,
    )
    return tids[inv].reshape(h, w)


def load_semantic_label(path: Path, mode: Literal["auto", "npy", "grayscale_trainid", "cityscapes_color"]) -> np.ndarray:
    """Load a 2D integer label map (trainIds / semantic_weights keys)."""
    suffix = path.suffix.lower()
    effective = mode
    if mode == "auto":
        if suffix == ".npy":
            effective = "npy"
        elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
            effective = "grayscale_trainid"
        else:
            raise ValueError(f"Cannot infer label mode for suffix {suffix}; set LABEL_MODE explicitly.")

    if effective == "npy":
        arr = np.squeeze(np.load(path))
        if arr.ndim != 2:
            raise ValueError(f"Label .npy must be 2D after squeeze, got {arr.shape}")
        return np.rint(arr).astype(np.int64)

    img = Image.open(path)
    if effective == "cityscapes_color":
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        return rgb_label_to_train_ids(rgb)

    # grayscale_trainid
    gray = np.asarray(img.convert("L"), dtype=np.int64)
    gray = np.where(gray == 255, -1, gray)
    return gray.astype(np.int64)


def finalize_riskmap_sample(
    image: Image.Image,
    depth: np.ndarray,
    label_ids: np.ndarray,
    output_root: Path,
    sample_id: str,
    model_input_size: tuple[int, int] | None,
) -> dict[str, Path]:
    """
    Align ``depth`` and ``label_ids`` to the RGB ``image`` size, optionally resize the full
    bundle, and write ``image_png`` / ``depth`` / ``label`` under ``output_root``.
    """
    output_root = Path(output_root)
    image = image.convert("RGB")
    native_size = image.size  # (W, H)

    depth = np.squeeze(np.asarray(depth, dtype=np.float32))
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D after squeeze, got shape {depth.shape}")
    depth = align_map_to_image(native_size, depth, mode="bilinear")

    label_f = np.asarray(label_ids, dtype=np.float32)
    if label_f.ndim != 2:
        raise ValueError(f"Label map must be 2D, got shape {label_f.shape}")
    label_out = np.rint(align_map_to_image(native_size, label_f, mode="nearest")).astype(np.int64)

    target_size = model_input_size if model_input_size is not None else native_size
    if target_size != native_size:
        image = image.resize(target_size, Image.BILINEAR)
        depth = resize_map(depth, size=target_size, mode="bilinear").astype(np.float32)
        label_f = label_out.astype(np.float32)
        label_out = np.rint(resize_map(label_f, size=target_size, mode="nearest")).astype(np.int64)

    image_dir = output_root / "image_png"
    depth_dir = output_root / "depth"
    label_dir = output_root / "label"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    out_image = image_dir / f"{sample_id}.png"
    out_depth = depth_dir / f"{sample_id}.npy"
    out_label = label_dir / f"{sample_id}.npy"

    image.save(out_image)
    np.save(out_depth, depth.astype(np.float32))
    np.save(out_label, label_out)

    return {"image": out_image, "depth": out_depth, "label": out_label}


def prepare_inference_bundle(
    rgb_path: Path,
    depth_path: Path,
    label_path: Path,
    output_root: Path,
    sample_id: str,
    model_input_size: tuple[int, int] | None,
    label_mode: Literal["auto", "npy", "grayscale_trainid", "cityscapes_color"] = "auto",
) -> dict[str, Path]:
    """
    Align modalities, optionally resize to model resolution, and write ``image_png`` / ``depth`` / ``label``.

    Returns paths to the written image, depth, and label files.
    """
    rgb_path = Path(rgb_path)
    depth_path = Path(depth_path)
    label_path = Path(label_path)

    image = Image.open(rgb_path).convert("RGB")
    depth = np.squeeze(np.load(depth_path).astype(np.float32))
    label_ids = load_semantic_label(label_path, mode=label_mode)
    return finalize_riskmap_sample(
        image=image,
        depth=depth,
        label_ids=label_ids,
        output_root=output_root,
        sample_id=sample_id,
        model_input_size=model_input_size,
    )


def main() -> None:
    paths = prepare_inference_bundle(
        rgb_path=RGB_PATH,
        depth_path=DEPTH_PATH,
        label_path=LABEL_PATH,
        output_root=OUTPUT_ROOT,
        sample_id=SAMPLE_ID,
        model_input_size=MODEL_INPUT_SIZE,
        label_mode=LABEL_MODE,
    )
    print("Wrote prepared sample:")
    for key, p in paths.items():
        print(f"  {key}: {p.resolve()}")
    print(
        "\nPoint test.py at these paths (or set dataset_root to OUTPUT_ROOT and use one sample), "
        "or load the tensors like test.py does. If you prepared at native resolution, keep "
        "MODEL_INPUT_SIZE in test.py and let it resize for the model."
    )


if __name__ == "__main__":
    main()
