from pathlib import Path

import numpy as np
from PIL import Image

# Edit these two paths before running.
INPUT_FOLDER = Path("data/cityscape_prepro/val/image")
OUTPUT_FOLDER = Path("data/cityscape_prepro/val/image_png")


def to_uint8_image(array: np.ndarray) -> np.ndarray:
    """Convert a numpy array into uint8 image data."""
    arr = np.asarray(array)

    # Remove singleton dimensions like (1, H, W) or (H, W, 1).
    arr = np.squeeze(arr)

    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported array shape {arr.shape}. Expected 2D or 3D array.")

    # Normalize numeric data to [0, 255] unless already uint8.
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val > min_val:
            arr = (arr - min_val) / (max_val - min_val)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    # If shape is channel-first (C, H, W), convert to (H, W, C).
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    return arr


def convert_folder_npy_to_png(input_folder: Path, output_folder: Path) -> None:
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(input_folder.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in: {input_folder}")
        return

    for npy_path in npy_files:
        try:
            array = np.load(npy_path)
            # import pdb; pdb.set_trace()
            image_data = to_uint8_image(array.squeeze())
            img = Image.fromarray(image_data)

            out_path = output_folder / f"{npy_path.stem}.png"
            img.save(out_path)
            print(f"Saved: {out_path}")
        except Exception as exc:
            print(f"Failed to convert {npy_path.name}: {exc}")


if __name__ == "__main__":
    convert_folder_npy_to_png(INPUT_FOLDER, OUTPUT_FOLDER)
