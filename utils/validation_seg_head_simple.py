from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from validation import (
    create_output_directories,
    create_validation_loader,
    get_device,
    load_model_from_checkpoint,
    predictions_to_risk_tensor,
    save_sample_outputs,
    targets_to_risk_tensor,
    tensor_to_rgb_image,
    write_validation_reports,
)


CHECKPOINT_PATH = Path("ckpts_classification/simple_segformer_head_only/segformer_head_only_best.pt")
OUTPUT_ROOT = Path("output/validation_seg_head_simple")


def main() -> None:
    device = get_device()
    model, checkpoint, runtime_config = load_model_from_checkpoint(CHECKPOINT_PATH, device=device)
    loader = create_validation_loader(runtime_config)
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

            mse_value = float(F.mse_loss(predictions, task_targets).item())
            l1_value = float(F.l1_loss(predictions, task_targets).item())

            save_sample_outputs(
                sample_id=sample_id,
                rgb_image=rgb_image,
                predicted_risk=predicted_risk,
                target_risk=target_risk,
                output_dirs=output_dirs,
                pred_value_max=1.0,
                target_value_max=1.0,
            )

            rows.append({"image_id": sample_id, "mse": mse_value, "l1": l1_value})
            print(f"[{index}/{len(loader)}] Saved validation outputs for {sample_id}")

    write_validation_reports(
        output_root=OUTPUT_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint=checkpoint,
        runtime_config=runtime_config,
        rows=rows,
    )
    print(f"Validation results saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
