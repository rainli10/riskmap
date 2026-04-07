"""
Single-sample (or single-id) inference using the same path as ``validation_ours.py``:

``RiskMapDataset`` + ``DataLoader`` → model forward → ``predictions_to_risk_tensor`` /
``targets_to_risk_tensor`` → ``save_sample_outputs``.

Point ``SAMPLE_DATASET_ROOT`` at a directory with ``image_png/``, ``depth/``, and ``label/``
(see ``prepare_inference_ours.py``). Resolution is whatever the PNG/NPY files use—no extra
resize step; this matches full-dataset validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataloader import RiskMapDataset
from validation_ours import (
    CHECKPOINT_PATH as OURS_CHECKPOINT_PATH,
    create_output_directories,
    get_device,
    load_model_from_checkpoint,
    predictions_to_risk_tensor,
    risk_targets_to_class_indices,
    save_sample_outputs,
    targets_to_risk_tensor,
    tensor_to_rgb_image,
)

# Checkpoint bundled with ``validation_ours`` defaults; change OURS_CHECKPOINT_PATH import if needed.
CHECKPOINT_PATH = OURS_CHECKPOINT_PATH
# Root with image_png/*.png, depth/<id>.npy, label/<id>.npy — same layout as training/validation.
SAMPLE_DATASET_ROOT = Path("/home/rain/Desktop/workspace/APS360/riskmap/output/prepared_from_real/2")
# PNG stem to run; must exist under SAMPLE_DATASET_ROOT. Use None to run every sample in the folder.
SAMPLE_ID_STEM: str | None = "dvp2"
OUTPUT_ROOT = Path("output/test_ours")
BATCH_SIZE = 1
NUM_WORKERS = 0


def build_loader(
    runtime_config: dict[str, object],
    dataset_root: Path,
    sample_id_stem: str | None,
) -> DataLoader:
    """Same ``RiskMapDataset`` construction as ``validation_ours.create_validation_loader`` (no val split)."""
    dataset: RiskMapDataset | Subset = RiskMapDataset(
        dataset_root=dataset_root,
        semantic_weights=runtime_config["semantic_weights"],
        depth_min=runtime_config["depth_min"],
        depth_max=runtime_config["depth_max"],
        target_mode=runtime_config["target_mode"],
        component_connectivity=runtime_config["component_connectivity"],
    )
    if sample_id_stem is not None:
        indices = [i for i, s in enumerate(dataset.samples) if str(s["id"]) == sample_id_stem]
        if len(indices) != 1:
            available = [str(s["id"]) for s in dataset.samples]
            raise ValueError(
                f"Expected exactly one sample with id {sample_id_stem!r} under {dataset_root}, "
                f"found {len(indices)}. Available ids (first 30): {available[:30]}"
            )
        dataset = Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )


def write_test_summary(
    output_root: Path,
    checkpoint_path: Path,
    runtime_config: dict[str, object],
    rows: list[dict[str, float | str]],
) -> None:
    mean_mse = float(sum(r["mse"] for r in rows) / len(rows)) if rows else 0.0
    mean_l1 = float(sum(r["l1"] for r in rows) / len(rows)) if rows else 0.0
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "sample_dataset_root": str(SAMPLE_DATASET_ROOT),
        "sample_id_filter": SAMPLE_ID_STEM,
        "num_samples": len(rows),
        "mean_mse": mean_mse,
        "mean_l1": mean_l1,
        "task_mode": runtime_config["task_mode"],
        "target_mode": runtime_config["target_mode"],
        "depth_min": runtime_config["depth_min"],
        "depth_max": runtime_config["depth_max"],
        "per_image": rows,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def main() -> None:
    device = get_device()
    model, _checkpoint, runtime_config = load_model_from_checkpoint(CHECKPOINT_PATH, device=device)

    runtime_config["dataset_root"] = str(SAMPLE_DATASET_ROOT.resolve())
    runtime_config["use_train_split_for_validation"] = False

    loader = build_loader(runtime_config, SAMPLE_DATASET_ROOT, SAMPLE_ID_STEM)
    output_dirs = create_output_directories(OUTPUT_ROOT)

    rows: list[dict[str, float | str]] = []

    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            raw_predictions = model(inputs)
            predictions = predictions_to_risk_tensor(raw_predictions, runtime_config)
            task_targets = targets_to_risk_tensor(targets, runtime_config)

            sample_id = str(batch["image_id"][0])
            rgb_image = tensor_to_rgb_image(batch["input"][0])
            predicted_risk = predictions[0, 0].detach().cpu().numpy()
            target_risk = task_targets[0, 0].detach().cpu().numpy()

            predicted_class_map: np.ndarray | None = None
            target_class_map: np.ndarray | None = None
            if runtime_config["task_mode"] == "risk_classification":
                predicted_class_map = torch.argmax(raw_predictions, dim=1)[0].detach().cpu().numpy()
                target_class_map = risk_targets_to_class_indices(targets, runtime_config)[
                    0
                ].detach().cpu().numpy()

            mse_value = float(F.mse_loss(predictions, task_targets).item())
            l1_value = float(F.l1_loss(predictions, task_targets).item())

            save_sample_outputs(
                sample_id=sample_id,
                rgb_image=rgb_image,
                predicted_risk=predicted_risk,
                predicted_class_map=predicted_class_map,
                target_risk=target_risk,
                target_class_map=target_class_map,
                output_dirs=output_dirs,
                pred_value_max=1.0,
                target_value_max=1.0,
            )

            rows.append({"image_id": sample_id, "mse": mse_value, "l1": l1_value})
            print(f"[{index}/{len(loader)}] Saved test outputs for {sample_id}")

    write_test_summary(
        output_root=OUTPUT_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
        runtime_config=runtime_config,
        rows=rows,
    )
    print(f"Test results saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
