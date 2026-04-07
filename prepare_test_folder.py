"""
Convert the ``test/`` layout (RGB + depth + label, possibly different sizes / WebM sidecars)
into ``RiskMapDataset`` format: ``image_png/``, ``depth/*.npy``, ``label/*.npy``.

Expected source layout (configurable via constants)::

    <SOURCE_ROOT>/
      images/<id>.png|jpg|...
      depth/<id>.webm|mp4|png|npy|...
      label/<id>.webm|mp4|png|npy|...  (optional; omitted → pseudo label, all class 1)

For each ``<id>`` taken from files under ``images/``, this script picks the first existing
file in ``depth/`` and ``label/`` whose name matches ``<id>`` and one of the supported
extensions (WebM tried before PNG so you can swap in video without renaming).

If there is **no** ``label/`` directory or no matching label file for an id, a **pseudo**
label map is used: every pixel is set to train class ``PSEUDO_LABEL_CLASS_ID`` (default **1**,
Cityscapes sidewalk). The ``label/`` folder is optional.

Depth from **video or RGB image** is only a heuristic: by default luminance is mapped
linearly to ``[0, DEPTH_RANGE_MAX]`` (same units as training ``DEPTH_MAX`` when set to 100).
Use **``.npy``** for real metric depth maps.

Labels from **video**: first frame is decoded like still images — ``cityscapes_color`` for RGB
frames, or luminance/``L`` semantics with 255 → -1 (void) for grayscale-style encodings.

All outputs are resampled to **256×128 (width × height)** for RGB, depth, and label together
(bilinear for image and depth, nearest for label), matching typical SegFormer training input.

After conversion, point ``test.py`` at ``OUTPUT_ROOT`` (and set ``SAMPLE_ID_STEM`` / ``None``).
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from prepare_inference_ours import (
    finalize_riskmap_sample,
    load_semantic_label,
    rgb_label_to_train_ids,
)

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "prepare_test_folder.py needs OpenCV to read WebM/video. "
        "Install with: pip install opencv-python-headless"
    ) from exc


# --- Source tree (your ``riskmap/test`` folder) -----------------------------------
SOURCE_ROOT = Path("real_data/2")
IMAGES_SUBDIR = "images"
DEPTH_SUBDIR = "depth"
LABEL_SUBDIR = "label"

OUTPUT_ROOT = Path("output/prepared_from_real/2")
# Fixed output resolution (width, height) for image, depth, and label maps.
OUTPUT_WIDTH = 256
OUTPUT_HEIGHT = 128
OUTPUT_SIZE: tuple[int, int] = (OUTPUT_WIDTH, OUTPUT_HEIGHT)

# First frame index for WebM/MP4 depth and label.
VIDEO_FRAME_INDEX = 0

# Depth when loaded from video or RGB image (not from .npy).
DEPTH_DECODE = "luminance_u8_to_range"
DEPTH_RANGE_MAX = 100.0

# Label decoding for image files (see ``prepare_inference_ours.load_semantic_label``).
# Use ``cityscapes_color`` for RGB label PNGs; ``auto`` picks grayscale for PNG/JPEG.
LABEL_MODE: Literal["auto", "npy", "grayscale_trainid", "cityscapes_color"] = "auto"

# When no label file exists, fill H×W with this train id (default 1 = sidewalk).
PSEUDO_LABEL_CLASS_ID = 1

RGB_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
# Order matters: first match wins when multiple files exist for the same stem.
SIDE_EXTENSIONS = (".webm", ".mp4", ".mkv", ".mov", ".avi", ".png", ".jpg", ".jpeg", ".webp", ".npy")


def read_video_frame(path: Path, frame_index: int = 0) -> np.ndarray:
    """Return ``uint8`` array: ``(H, W)`` grayscale or ``(H, W, 3)`` RGB."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, bgr = cap.read()
    cap.release()
    if not ret or bgr is None:
        raise RuntimeError(f"No frame at index {frame_index} in {path}")
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_depth_frame(frame: np.ndarray, decode: str, range_max: float) -> np.ndarray:
    """Turn one video/image frame into a 2D float depth map."""
    if decode == "luminance_u8_to_range":
        if frame.ndim == 2:
            u = frame.astype(np.float32)
        else:
            u = (
                0.299 * frame[..., 0]
                + 0.587 * frame[..., 1]
                + 0.114 * frame[..., 2]
            ).astype(np.float32)
        return (u / 255.0) * float(range_max)
    if decode == "red_channel_u8_to_range":
        if frame.ndim == 2:
            u = frame.astype(np.float32)
        else:
            u = frame[..., 0].astype(np.float32)
        return (u / 255.0) * float(range_max)
    raise ValueError(f"Unknown DEPTH_DECODE: {decode!r}")


def label_frame_to_ids(
    frame: np.ndarray,
    label_mode: Literal["auto", "npy", "grayscale_trainid", "cityscapes_color"],
) -> np.ndarray:
    """Decode a single label frame to int64 train-style ids (void = -1)."""
    if label_mode == "cityscapes_color":
        if frame.ndim != 3:
            raise ValueError("cityscapes_color requires an RGB (H, W, 3) frame")
        return rgb_label_to_train_ids(frame.astype(np.uint8))

    if frame.ndim == 3:
        gray = (
            0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
        ).astype(np.int64)
    else:
        gray = frame.astype(np.int64)
    return np.where(gray == 255, -1, gray).astype(np.int64)


def find_sidecar(directory: Path | None, stem: str) -> Path | None:
    if directory is None or not directory.is_dir():
        return None
    for ext in SIDE_EXTENSIONS:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def pseudo_label_map(width: int, height: int, class_id: int) -> np.ndarray:
    """H×W label map filled with a single semantic class id (e.g. trainId 1)."""
    return np.full((height, width), int(class_id), dtype=np.int64)


def load_depth_array(path: Path, frame_index: int) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        depth = np.squeeze(np.load(path).astype(np.float32))
        if depth.ndim != 2:
            raise ValueError(f"Depth .npy must be 2D after squeeze, got {depth.shape} ({path})")
        return depth

    if ext in {".webm", ".mp4", ".mkv", ".mov", ".avi"}:
        frame = read_video_frame(path, frame_index)
        return decode_depth_frame(frame, DEPTH_DECODE, DEPTH_RANGE_MAX)

    image = Image.open(path).convert("RGB")
    frame = np.asarray(image, dtype=np.uint8)
    return decode_depth_frame(frame, DEPTH_DECODE, DEPTH_RANGE_MAX)


def load_label_array(
    path: Path,
    frame_index: int,
    label_mode: Literal["auto", "npy", "grayscale_trainid", "cityscapes_color"],
) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        return load_semantic_label(path, mode="npy")

    if ext in {".webm", ".mp4", ".mkv", ".mov", ".avi"}:
        frame = read_video_frame(path, frame_index)
        effective = label_mode
        if effective == "auto":
            effective = "cityscapes_color" if frame.ndim == 3 and frame.shape[2] >= 3 else "grayscale_trainid"
        return label_frame_to_ids(frame, effective)

    return load_semantic_label(path, mode=label_mode)


def iter_image_stems(images_dir: Path) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for path in sorted(
        chain.from_iterable(images_dir.glob(f"*{ext}") for ext in RGB_EXTENSIONS)
    ):
        if not path.is_file():
            continue
        stem = path.stem
        if stem in seen:
            continue
        seen.add(stem)
        pairs.append((stem, path))
    return pairs


def convert_one(
    stem: str,
    rgb_path: Path,
    depth_dir: Path,
    label_dir: Path | None,
) -> dict[str, Path]:
    depth_path = find_sidecar(depth_dir, stem)
    label_path = find_sidecar(label_dir, stem)
    if depth_path is None:
        raise FileNotFoundError(f"No depth file for stem {stem!r} under {depth_dir}")

    image = Image.open(rgb_path).convert("RGB")
    width, height = image.size
    depth = load_depth_array(depth_path, VIDEO_FRAME_INDEX)
    if label_path is None:
        label_ids = pseudo_label_map(width, height, PSEUDO_LABEL_CLASS_ID)
    else:
        label_ids = load_label_array(label_path, VIDEO_FRAME_INDEX, LABEL_MODE)

    return finalize_riskmap_sample(
        image=image,
        depth=depth,
        label_ids=label_ids,
        output_root=OUTPUT_ROOT,
        sample_id=stem,
        model_input_size=OUTPUT_SIZE,
    )


def main() -> None:
    images_dir = SOURCE_ROOT / IMAGES_SUBDIR
    depth_dir = SOURCE_ROOT / DEPTH_SUBDIR
    label_dir_candidate = SOURCE_ROOT / LABEL_SUBDIR
    label_dir: Path | None = label_dir_candidate if label_dir_candidate.is_dir() else None

    for d in (images_dir, depth_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Missing directory: {d}")
    if label_dir is None:
        print(
            f"No {LABEL_SUBDIR!r} directory under {SOURCE_ROOT}; "
            f"using pseudo labels (class {PSEUDO_LABEL_CLASS_ID} everywhere)."
        )

    pairs = iter_image_stems(images_dir)
    if not pairs:
        raise RuntimeError(f"No RGB images found under {images_dir}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for stem, rgb_path in pairs:
        try:
            lp = find_sidecar(label_dir, stem)
            paths = convert_one(stem, rgb_path, depth_dir, label_dir)
            ok += 1
            tag = "pseudo-label" if lp is None else "ok"
            print(f"OK {stem} ({tag}) -> {paths['image']}")
        except Exception as err:  # noqa: BLE001 — batch helper; show all failures
            print(f"SKIP {stem}: {err}")
    print(
        f"Converted {ok}/{len(pairs)} samples into {OUTPUT_ROOT.resolve()} "
        f"at {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} (W×H)."
    )
    print("Set test.py SAMPLE_DATASET_ROOT to this folder and SAMPLE_ID_STEM or None.")


if __name__ == "__main__":
    main()
