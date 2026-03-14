from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataloader_baseline import RISK_BIN_EDGES, RISK_CLASS_LABELS, RISK_CLASS_VALUES, RiskMapDataset
from model_baseline import SimpleRiskCNN, SimpleRiskLinear
from training_preview import save_training_previews


DATASET_ROOT = Path("data/cityscape_prepro/train")
VAL_SPLIT = 0.2
NICKNAME = "baseline_rgbd_simple"
CHECKPOINT_DIR = Path("ckpts_baseline") / NICKNAME
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "baseline_best.pt"
TENSORBOARD_LOG_DIR = Path("tensorboard_baseline") / NICKNAME
PREVIEW_OUTPUT_DIR = Path("output/training_previews") / NICKNAME

BATCH_SIZE = 8
NUM_EPOCHS = 80
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 4
RANDOM_SEED = 42
LOG_EVERY_N_BATCHES = 10
LABEL_SMOOTHING = 0.02
CLASS_WEIGHT_POWER = 0.5
CLASS_WEIGHT_MIN = 0.25
CLASS_WEIGHT_MAX = 4.0
CLASS_COUNT_MAX_BATCHES = 24
RISK_SCALE = 10.0
PREVIEW_EVERY_N_EPOCHS = 1
NUM_PREVIEW_SAMPLES = 3
NUM_RISK_CLASSES = len(RISK_CLASS_VALUES)
USE_LABEL_AS_INPUT = True
INPUT_CHANNELS = 5 if USE_LABEL_AS_INPUT else 4  # RGBD(+label)

# Choose: "cnn" or "linear".
BASELINE_MODEL = "cnn"

TARGET_MODE = "blocked"
COMPONENT_CONNECTIVITY = 4
DEPTH_MIN = 1.0
DEPTH_MAX = 100.0

SEMANTIC_WEIGHTS = {
    -1: 0.0,
    0: 0.0,
    1: 0.0,
    2: 0.1,
    3: 0.2,
    4: 0.2,
    5: 0.3,
    6: 0.3,
    7: 0.3,
    8: 0.1,
    9: 0.1,
    10: 0.0,
    11: 3.0,
    12: 1.0,
    13: 0.6,
    14: 0.7,
    15: 0.8,
    16: 0.8,
    17: 0.6,
    18: 0.6,
}


def risk_targets_to_classes(targets: torch.Tensor) -> torch.Tensor:
    edges = targets.new_tensor(RISK_BIN_EDGES, dtype=torch.float32)
    scaled_targets = torch.clamp(targets.squeeze(1) * RISK_SCALE, 0.0, 1.0)
    return torch.bucketize(scaled_targets, boundaries=edges, right=False).long()


def risk_classes_to_values_tensor(class_map: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(RISK_CLASS_VALUES, device=class_map.device, dtype=torch.float32)
    return values[class_map]


def build_dataset(dataset_root: Path) -> RiskMapDataset:
    return RiskMapDataset(
        dataset_root=dataset_root,
        semantic_weights=SEMANTIC_WEIGHTS,
        depth_min=DEPTH_MIN,
        depth_max=DEPTH_MAX,
        target_mode=TARGET_MODE,
        component_connectivity=COMPONENT_CONNECTIVITY,
        include_label_in_input=USE_LABEL_AS_INPUT,
    )


def create_data_loaders(device: torch.device) -> tuple[DataLoader, DataLoader]:
    full_dataset = build_dataset(DATASET_ROOT)
    if len(full_dataset) < 2:
        raise RuntimeError("Need at least 2 samples to create train/validation splits.")

    val_size = max(1, int(len(full_dataset) * VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    if train_size == 0:
        train_size = len(full_dataset) - 1
        val_size = 1

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def compute_class_counts(loader: DataLoader, max_batches: int | None = None) -> torch.Tensor:
    counts = torch.zeros(NUM_RISK_CLASSES, dtype=torch.long)
    for batch_idx, batch in enumerate(loader, start=1):
        target_classes = risk_targets_to_classes(batch["target"])
        counts += torch.bincount(target_classes.reshape(-1), minlength=NUM_RISK_CLASSES)
        if max_batches is not None and batch_idx >= max_batches:
            break
    return counts


def build_class_weights(class_counts: torch.Tensor) -> torch.Tensor:
    counts = class_counts.to(torch.float32)
    present_mask = counts > 0
    weights = torch.ones_like(counts, dtype=torch.float32)
    if not torch.any(present_mask):
        return weights

    frequencies = counts[present_mask] / counts[present_mask].sum().clamp(min=1.0)
    median_frequency = frequencies.median()
    balanced_weights = torch.pow(median_frequency / frequencies.clamp(min=1e-12), CLASS_WEIGHT_POWER)
    balanced_weights = balanced_weights / balanced_weights.mean().clamp(min=1e-12)
    balanced_weights = balanced_weights.clamp(min=CLASS_WEIGHT_MIN, max=CLASS_WEIGHT_MAX)
    weights[present_mask] = balanced_weights
    weights[~present_mask] = 0.0
    return weights


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    target_classes: torch.Tensor,
    predicted_classes: torch.Tensor,
) -> None:
    linear_indices = target_classes.reshape(-1) * NUM_RISK_CLASSES + predicted_classes.reshape(-1)
    confusion_matrix += torch.bincount(
        linear_indices,
        minlength=NUM_RISK_CLASSES * NUM_RISK_CLASSES,
    ).reshape(NUM_RISK_CLASSES, NUM_RISK_CLASSES)


def summarize_confusion_matrix(confusion_matrix: torch.Tensor) -> dict[str, float]:
    confusion = confusion_matrix.to(torch.float32)
    total_pixels = float(confusion.sum().item())
    true_positives = confusion.diag()
    target_totals = confusion.sum(dim=1)
    predicted_totals = confusion.sum(dim=0)
    present_mask = target_totals > 0

    class_accuracy = torch.zeros(NUM_RISK_CLASSES, dtype=torch.float32)
    class_iou = torch.zeros(NUM_RISK_CLASSES, dtype=torch.float32)
    if torch.any(present_mask):
        class_accuracy[present_mask] = true_positives[present_mask] / target_totals[present_mask].clamp(min=1.0)
        unions = target_totals + predicted_totals - true_positives
        valid_iou_mask = present_mask & (unions > 0)
        class_iou[valid_iou_mask] = true_positives[valid_iou_mask] / unions[valid_iou_mask].clamp(min=1.0)
        macro_accuracy = float(class_accuracy[present_mask].mean().item())
        mean_iou = float(class_iou[present_mask].mean().item())
    else:
        macro_accuracy = 0.0
        mean_iou = 0.0

    pixel_accuracy = float(true_positives.sum().item() / total_pixels) if total_pixels > 0 else 0.0
    return {
        "pixel_acc": pixel_accuracy,
        "macro_acc": macro_accuracy,
        "miou": mean_iou,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    writer: SummaryWriter | None = None,
    global_step: int = 0,
) -> tuple[dict[str, float], int]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    confusion_matrix = torch.zeros((NUM_RISK_CLASSES, NUM_RISK_CLASSES), dtype=torch.long)

    for batch_idx, batch in enumerate(loader, start=1):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        target_classes = risk_targets_to_classes(targets)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(inputs)
            loss = criterion(logits, target_classes)
            predicted_classes = torch.argmax(logits, dim=1)
            predicted_risk = risk_classes_to_values_tensor(predicted_classes)
            target_risk = risk_classes_to_values_tensor(target_classes)
            mae = torch.abs(predicted_risk - target_risk).mean()
            batch_pixel_acc = (predicted_classes == target_classes).float().mean()

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_mae += mae.item() * batch_size
        total_samples += batch_size
        update_confusion_matrix(
            confusion_matrix,
            target_classes=target_classes.detach().cpu(),
            predicted_classes=predicted_classes.detach().cpu(),
        )

        if is_training:
            global_step += 1
            if writer is not None and LOG_EVERY_N_BATCHES > 0 and batch_idx % LOG_EVERY_N_BATCHES == 0:
                writer.add_scalar("train/batch_loss", loss.item(), global_step)
                writer.add_scalar("train/batch_mae", mae.item(), global_step)
                writer.add_scalar("train/batch_pixel_acc", batch_pixel_acc.item(), global_step)
                writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)

    metrics = summarize_confusion_matrix(confusion_matrix)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["mae"] = total_mae / max(total_samples, 1)
    return metrics, global_step


def build_model() -> nn.Module:
    model_key = BASELINE_MODEL.strip().lower()
    if model_key == "cnn":
        return SimpleRiskCNN(in_channels=INPUT_CHANNELS, out_channels=NUM_RISK_CLASSES)
    if model_key == "linear":
        return SimpleRiskLinear(in_channels=INPUT_CHANNELS, out_channels=NUM_RISK_CLASSES)
    raise ValueError(f"Unsupported BASELINE_MODEL '{BASELINE_MODEL}'. Use 'cnn' or 'linear'.")


def save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    metrics: dict[str, float],
    class_weights: torch.Tensor,
    train_class_counts: torch.Tensor,
    val_class_counts: torch.Tensor,
    path: Path,
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_loss": metrics["loss"],
            "val_mae": metrics["mae"],
            "val_accuracy": metrics["pixel_acc"],
            "val_macro_accuracy": metrics["macro_acc"],
            "val_miou": metrics["miou"],
            "selection_metric": "val_miou",
            "architecture": "simplest_cnn",
            "baseline_model": BASELINE_MODEL,
            "task_mode": "risk_classification",
            "num_risk_classes": NUM_RISK_CLASSES,
            "risk_scale": RISK_SCALE,
            "risk_bin_edges": list(RISK_BIN_EDGES),
            "risk_class_values": list(RISK_CLASS_VALUES),
            "risk_class_labels": list(RISK_CLASS_LABELS),
            "class_weights": class_weights.cpu(),
            "train_class_counts": train_class_counts.cpu(),
            "val_class_counts": val_class_counts.cpu(),
            "dataset_root": str(DATASET_ROOT),
            "val_split": float(VAL_SPLIT),
            "random_seed": int(RANDOM_SEED),
            "use_train_split_for_validation": True,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "semantic_weights": SEMANTIC_WEIGHTS,
            "depth_min": DEPTH_MIN,
            "depth_max": DEPTH_MAX,
            "target_mode": TARGET_MODE,
            "component_connectivity": COMPONENT_CONNECTIVITY,
            "label_smoothing": LABEL_SMOOTHING,
            "learning_rate": LEARNING_RATE,
            "min_learning_rate": MIN_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        path,
    )


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_data_loaders(device=device)
    train_class_counts = compute_class_counts(train_loader, max_batches=CLASS_COUNT_MAX_BATCHES)
    val_class_counts = compute_class_counts(val_loader, max_batches=CLASS_COUNT_MAX_BATCHES)
    class_weights = build_class_weights(train_class_counts)

    print(f"Device: {device}")
    print(f"Baseline model: {BASELINE_MODEL}")
    print(f"Target mode: {TARGET_MODE} (connectivity={COMPONENT_CONNECTIVITY})")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Validation split ratio: {VAL_SPLIT}")
    print(f"Risk scale before binning: {RISK_SCALE}")
    print(f"Input channels: {INPUT_CHANNELS} (RGBD{' + label' if USE_LABEL_AS_INPUT else ''})")
    print(f"Train class counts: {train_class_counts.tolist()}")
    print(f"Val class counts: {val_class_counts.tolist()}")
    if CLASS_COUNT_MAX_BATCHES is not None:
        print(f"Class counts estimated from first {CLASS_COUNT_MAX_BATCHES} batches.")
    print(f"Class weights: {[round(float(value), 4) for value in class_weights.tolist()]}")
    print(f"Preview output dir: {PREVIEW_OUTPUT_DIR}")

    model = build_model().to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Trainable parameters: {trainable_parameters:,}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=LABEL_SMOOTHING,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=MIN_LEARNING_RATE)

    TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(TENSORBOARD_LOG_DIR))
    writer.add_text(
        "config/summary",
        "\n".join(
            [
                f"dataset_root: {DATASET_ROOT}",
                f"val_split: {VAL_SPLIT}",
                f"random_seed: {RANDOM_SEED}",
                f"baseline_model: {BASELINE_MODEL}",
                f"batch_size: {BATCH_SIZE}",
                f"num_epochs: {NUM_EPOCHS}",
                f"learning_rate: {LEARNING_RATE}",
                f"min_learning_rate: {MIN_LEARNING_RATE}",
                f"weight_decay: {WEIGHT_DECAY}",
                f"label_smoothing: {LABEL_SMOOTHING}",
                f"risk_scale: {RISK_SCALE}",
                f"preview_output_dir: {PREVIEW_OUTPUT_DIR}",
                f"preview_every_n_epochs: {PREVIEW_EVERY_N_EPOCHS}",
                f"num_preview_samples: {NUM_PREVIEW_SAMPLES}",
                f"target_mode: {TARGET_MODE}",
                f"component_connectivity: {COMPONENT_CONNECTIVITY}",
                f"use_label_as_input: {USE_LABEL_AS_INPUT}",
                f"class_weights: {[round(float(value), 4) for value in class_weights.tolist()]}",
                f"train_class_counts: {train_class_counts.tolist()}",
                f"val_class_counts: {val_class_counts.tolist()}",
                f"trainable_parameters: {trainable_parameters}",
            ]
        ),
    )

    best_score = (float("-inf"), float("-inf"), float("-inf"))
    global_step = 0

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            train_metrics, global_step = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                writer=writer,
                global_step=global_step,
            )
            val_metrics, global_step = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                writer=None,
                global_step=global_step,
            )

            writer.add_scalar("train/epoch_loss", train_metrics["loss"], epoch)
            writer.add_scalar("train/epoch_mae", train_metrics["mae"], epoch)
            writer.add_scalar("train/epoch_pixel_acc", train_metrics["pixel_acc"], epoch)
            writer.add_scalar("train/epoch_macro_acc", train_metrics["macro_acc"], epoch)
            writer.add_scalar("train/epoch_miou", train_metrics["miou"], epoch)
            writer.add_scalar("val/epoch_loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/epoch_mae", val_metrics["mae"], epoch)
            writer.add_scalar("val/epoch_pixel_acc", val_metrics["pixel_acc"], epoch)
            writer.add_scalar("val/epoch_macro_acc", val_metrics["macro_acc"], epoch)
            writer.add_scalar("val/epoch_miou", val_metrics["miou"], epoch)
            writer.add_scalar("train/lr_epoch", float(optimizer.param_groups[0]["lr"]), epoch)

            print(
                f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
                f"train_loss={train_metrics['loss']:.6f} | train_mae={train_metrics['mae']:.6f} | "
                f"train_pixel_acc={train_metrics['pixel_acc']:.6f} | train_macro_acc={train_metrics['macro_acc']:.6f} | "
                f"train_miou={train_metrics['miou']:.6f} | val_loss={val_metrics['loss']:.6f} | "
                f"val_mae={val_metrics['mae']:.6f} | val_pixel_acc={val_metrics['pixel_acc']:.6f} | "
                f"val_macro_acc={val_metrics['macro_acc']:.6f} | val_miou={val_metrics['miou']:.6f}"
            )

            if PREVIEW_EVERY_N_EPOCHS > 0 and epoch % PREVIEW_EVERY_N_EPOCHS == 0:
                save_training_previews(
                    model=model,
                    loader=val_loader,
                    device=device,
                    epoch=epoch,
                    output_root=PREVIEW_OUTPUT_DIR,
                    target_to_classes_fn=risk_targets_to_classes,
                    num_samples=NUM_PREVIEW_SAMPLES,
                    writer=writer,
                    writer_prefix="preview/val",
                )
                print(f"Saved preview images to {PREVIEW_OUTPUT_DIR / f'epoch_{epoch:03d}'}")

            epoch_ckpt = CHECKPOINT_DIR / f"baseline_epoch_{epoch:03d}.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                class_weights=class_weights,
                train_class_counts=train_class_counts,
                val_class_counts=val_class_counts,
                path=epoch_ckpt,
            )

            current_score = (
                val_metrics["miou"],
                val_metrics["macro_acc"],
                -val_metrics["loss"],
            )
            if current_score > best_score:
                best_score = current_score
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    class_weights=class_weights,
                    train_class_counts=train_class_counts,
                    val_class_counts=val_class_counts,
                    path=BEST_CHECKPOINT_PATH,
                )
                print(f"Saved best checkpoint to {BEST_CHECKPOINT_PATH}")

            scheduler.step()
    finally:
        writer.close()


if __name__ == "__main__":
    main()

