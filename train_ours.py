from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from dataloader import RISK_BIN_EDGES, RISK_CLASS_LABELS, RISK_CLASS_VALUES, RiskMapDataset
from model import SegFormerRisk
from training_preview import save_training_previews


DATASET_ROOT = Path("data/cityscape_prepro/train")
VAL_SPLIT = 0.2
NICKNAME = "ours_segformer_transfer_ce_bins5_ori_weights"
CHECKPOINT_DIR = Path("ckpts_classification") / NICKNAME
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "segformer_transfer_best.pt"
TENSORBOARD_LOG_DIR = Path("tensorboard_classification") / NICKNAME
PREVIEW_OUTPUT_DIR = Path("output/training_previews") / NICKNAME
ARCHITECTURE = "segformer"
BATCH_SIZE = 8
NUM_EPOCHS = 80
FREEZE_EPOCHS = 8
STAGE2_EPOCHS = max(NUM_EPOCHS - FREEZE_EPOCHS, 1)
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
INPUT_CHANNELS = 4  # RGB + depth

TARGET_MODE = "blocked"
COMPONENT_CONNECTIVITY = 4
DEPTH_MIN = 1.0
DEPTH_MAX = 100.0

SEGFORMER_PRETRAINED_MODEL = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
SEGFORMER_ADAPTER_HIDDEN: int | None = None

WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0
# STAGE1_HEAD_LR = 1e-5
# STAGE1_MIN_LR = 1e-6
# STAGE2_BACKBONE_LR = 2e-6
# STAGE2_HEAD_LR = 1e-5
# STAGE2_BACKBONE_MIN_LR = 2e-7
# STAGE2_HEAD_MIN_LR = 1e-6

STAGE1_HEAD_LR = 5e-5
STAGE1_MIN_LR = 5e-6
STAGE2_BACKBONE_LR = 1e-5
STAGE2_HEAD_LR = 5e-5
STAGE2_BACKBONE_MIN_LR = 1e-6
STAGE2_HEAD_MIN_LR = 5e-6



AUG_FLIP_PROB = 0.5
# AUG_BRIGHTNESS = 0.05
# AUG_CONTRAST = 0.05

AUG_BRIGHTNESS = 0.1
AUG_CONTRAST = 0.1

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


class TrainAugmentDataset(Dataset):
    """Lightweight augmentation that preserves image/target geometry."""

    def __init__(
        self,
        base_dataset: Dataset,
        flip_prob: float = 0.5,
        brightness: float = 0.1,
        contrast: float = 0.1,
    ) -> None:
        self.base_dataset = base_dataset
        self.flip_prob = float(flip_prob)
        self.brightness = float(brightness)
        self.contrast = float(contrast)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base_dataset[index]
        input_tensor = sample["input"].clone()
        target_tensor = sample["target"].clone()

        if torch.rand(1).item() < self.flip_prob:
            input_tensor = torch.flip(input_tensor, dims=[2])
            target_tensor = torch.flip(target_tensor, dims=[2])

        rgb = input_tensor[:3]
        if self.brightness > 0:
            brightness_factor = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.brightness
            rgb = torch.clamp(rgb * brightness_factor, 0.0, 1.0)
        if self.contrast > 0:
            contrast_factor = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.contrast
            mean = rgb.mean(dim=(1, 2), keepdim=True)
            rgb = torch.clamp((rgb - mean) * contrast_factor + mean, 0.0, 1.0)

        input_tensor = torch.cat((rgb, input_tensor[3:]), dim=0)
        return {
            "input": input_tensor,
            "target": target_tensor,
            "image_id": sample["image_id"],
        }


def risk_targets_to_classes(targets: torch.Tensor) -> torch.Tensor:
    edges = targets.new_tensor(RISK_BIN_EDGES, dtype=torch.float32)
    scaled_targets = torch.clamp(targets.squeeze(1) * RISK_SCALE, 0.0, 1.0)
    return torch.bucketize(scaled_targets, boundaries=edges, right=False).long()


def risk_classes_to_values_tensor(class_map: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(RISK_CLASS_VALUES, device=class_map.device, dtype=torch.float32)
    return values[class_map]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_backbone_trainable(model: SegFormerRisk, trainable: bool) -> None:
    for parameter in model.backbone.segformer.parameters():
        parameter.requires_grad = trainable


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_dataset(dataset_root: Path) -> RiskMapDataset:
    return RiskMapDataset(
        dataset_root=dataset_root,
        semantic_weights=SEMANTIC_WEIGHTS,
        depth_min=DEPTH_MIN,
        depth_max=DEPTH_MAX,
        target_mode=TARGET_MODE,
        component_connectivity=COMPONENT_CONNECTIVITY,
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
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_dataset = TrainAugmentDataset(
        base_dataset=train_subset,
        flip_prob=AUG_FLIP_PROB,
        brightness=AUG_BRIGHTNESS,
        contrast=AUG_CONTRAST,
    )
    val_dataset = val_subset

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


def build_optimizer(model: SegFormerRisk, backbone_lr: float | None, head_lr: float) -> AdamW:
    backbone_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone.segformer."):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    parameter_groups: list[dict[str, Any]] = []
    if backbone_params and backbone_lr is not None and backbone_lr > 0.0:
        parameter_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        parameter_groups.append({"params": head_params, "lr": head_lr})
    if not parameter_groups:
        raise RuntimeError("No trainable parameters were found for the optimizer.")

    return AdamW(parameter_groups, weight_decay=WEIGHT_DECAY)


def build_scheduler(
    optimizer: AdamW,
    stage_epochs: int,
    min_lrs: list[float],
) -> CosineAnnealingLR:
    eta_min = min(min_lrs) if min_lrs else 0.0
    return CosineAnnealingLR(optimizer, T_max=max(stage_epochs, 1), eta_min=eta_min)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    scaler: GradScaler | None = None,
    writer: SummaryWriter | None = None,
    global_step: int = 0,
) -> tuple[dict[str, float], int]:
    is_training = optimizer is not None
    model.train(is_training)
    use_amp = scaler is not None and scaler.is_enabled()

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
            with autocast(enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, target_classes)
                predicted_classes = torch.argmax(logits, dim=1)
                predicted_risk = risk_classes_to_values_tensor(predicted_classes)
                target_risk = risk_classes_to_values_tensor(target_classes)
                mae = torch.abs(predicted_risk - target_risk).mean()
                batch_pixel_acc = (predicted_classes == target_classes).float().mean()

            if is_training:
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
                for lr_idx, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"train/lr_group_{lr_idx}", float(param_group["lr"]), global_step)

    metrics = summarize_confusion_matrix(confusion_matrix)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["mae"] = total_mae / max(total_samples, 1)
    return metrics, global_step


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
    stage_name: str,
    backbone_frozen: bool,
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
            "architecture": "segformer",
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
            "segformer_pretrained_model": SEGFORMER_PRETRAINED_MODEL,
            "segformer_freeze_backbone": backbone_frozen,
            "segformer_adapter_hidden": SEGFORMER_ADAPTER_HIDDEN,
            "label_smoothing": LABEL_SMOOTHING,
            "weight_decay": WEIGHT_DECAY,
            "freeze_epochs": FREEZE_EPOCHS,
            "stage1_head_lr": STAGE1_HEAD_LR,
            "stage2_backbone_lr": STAGE2_BACKBONE_LR,
            "stage2_head_lr": STAGE2_HEAD_LR,
            "stage_name": stage_name,
        },
        path,
    )


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.benchmark = True

    device = get_device()
    train_loader, val_loader = create_data_loaders(device=device)
    train_class_counts = compute_class_counts(train_loader, max_batches=CLASS_COUNT_MAX_BATCHES)
    val_class_counts = compute_class_counts(val_loader, max_batches=CLASS_COUNT_MAX_BATCHES)
    class_weights = build_class_weights(train_class_counts)

    print(f"Device: {device}")
    print(f"Target mode: {TARGET_MODE} (connectivity={COMPONENT_CONNECTIVITY})")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Validation split ratio: {VAL_SPLIT}")
    print(f"Risk scale before binning: {RISK_SCALE}")
    print(f"Input channels: {INPUT_CHANNELS} (RGBD)")
    print(f"Train class counts: {train_class_counts.tolist()}")
    print(f"Val class counts: {val_class_counts.tolist()}")
    if CLASS_COUNT_MAX_BATCHES is not None:
        print(f"Class counts estimated from first {CLASS_COUNT_MAX_BATCHES} batches.")
    print(f"Class weights: {[round(float(value), 4) for value in class_weights.tolist()]}")
    print(f"Preview output dir: {PREVIEW_OUTPUT_DIR}")

    model = SegFormerRisk(
        in_channels=INPUT_CHANNELS,
        pretrained_model_name=SEGFORMER_PRETRAINED_MODEL,
        freeze_backbone=True,
        adapter_hidden=SEGFORMER_ADAPTER_HIDDEN,
        num_labels=NUM_RISK_CLASSES,
    ).to(device)

    backbone_frozen = FREEZE_EPOCHS > 0
    if backbone_frozen:
        set_backbone_trainable(model, trainable=False)
        optimizer = build_optimizer(model=model, backbone_lr=None, head_lr=STAGE1_HEAD_LR)
        scheduler = build_scheduler(
            optimizer=optimizer,
            stage_epochs=FREEZE_EPOCHS,
            min_lrs=[STAGE1_MIN_LR],
        )
        stage_name = "freeze"
    else:
        optimizer = build_optimizer(
            model=model,
            backbone_lr=STAGE2_BACKBONE_LR,
            head_lr=STAGE2_HEAD_LR,
        )
        scheduler = build_scheduler(
            optimizer=optimizer,
            stage_epochs=NUM_EPOCHS,
            min_lrs=[STAGE2_BACKBONE_MIN_LR, STAGE2_HEAD_MIN_LR],
        )
        stage_name = "finetune"

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=LABEL_SMOOTHING,
    ).to(device)
    scaler = GradScaler(enabled=device.type == "cuda")

    print(f"Trainable parameters ({stage_name}): {count_trainable_parameters(model):,}")

    TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(TENSORBOARD_LOG_DIR))
    writer.add_text(
        "config/summary",
        "\n".join(
            [
                f"dataset_root: {DATASET_ROOT}",
                f"val_split: {VAL_SPLIT}",
                f"random_seed: {RANDOM_SEED}",
                f"batch_size: {BATCH_SIZE}",
                f"num_epochs: {NUM_EPOCHS}",
                f"freeze_epochs: {FREEZE_EPOCHS}",
                f"stage1_head_lr: {STAGE1_HEAD_LR}",
                f"stage2_backbone_lr: {STAGE2_BACKBONE_LR}",
                f"stage2_head_lr: {STAGE2_HEAD_LR}",
                f"weight_decay: {WEIGHT_DECAY}",
                f"label_smoothing: {LABEL_SMOOTHING}",
                f"risk_scale: {RISK_SCALE}",
                f"preview_output_dir: {PREVIEW_OUTPUT_DIR}",
                f"preview_every_n_epochs: {PREVIEW_EVERY_N_EPOCHS}",
                f"num_preview_samples: {NUM_PREVIEW_SAMPLES}",
                f"target_mode: {TARGET_MODE}",
                f"component_connectivity: {COMPONENT_CONNECTIVITY}",
                f"class_weights: {[round(float(value), 4) for value in class_weights.tolist()]}",
                f"train_class_counts: {train_class_counts.tolist()}",
                f"val_class_counts: {val_class_counts.tolist()}",
                f"segformer_pretrained_model: {SEGFORMER_PRETRAINED_MODEL}",
                f"segformer_adapter_hidden: {SEGFORMER_ADAPTER_HIDDEN}",
                f"trainable_parameters: {count_trainable_parameters(model)}",
            ]
        ),
    )

    best_score = (float("-inf"), float("-inf"), float("-inf"))
    global_step = 0

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            if backbone_frozen and epoch == FREEZE_EPOCHS + 1:
                set_backbone_trainable(model, trainable=True)
                backbone_frozen = False
                optimizer = build_optimizer(
                    model=model,
                    backbone_lr=STAGE2_BACKBONE_LR,
                    head_lr=STAGE2_HEAD_LR,
                )
                scheduler = build_scheduler(
                    optimizer=optimizer,
                    stage_epochs=STAGE2_EPOCHS,
                    min_lrs=[STAGE2_BACKBONE_MIN_LR, STAGE2_HEAD_MIN_LR],
                )
                stage_name = "finetune"
                print(f"Stage switch at epoch {epoch}: unfroze backbone | trainable params={count_trainable_parameters(model):,}")
                writer.add_text(
                    "events/stage_switch",
                    f"Epoch {epoch}: unfroze backbone and rebuilt optimizer/scheduler.",
                    epoch,
                )

            train_metrics, global_step = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                writer=writer,
                global_step=global_step,
            )
            val_metrics, global_step = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=None,
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
            for lr_idx, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(f"train/epoch_lr_group_{lr_idx}", float(param_group["lr"]), epoch)

            print(
                f"Epoch {epoch:03d}/{NUM_EPOCHS} | stage={stage_name} | "
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

            epoch_ckpt = CHECKPOINT_DIR / f"segformer_transfer_epoch_{epoch:03d}.pt"
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
                stage_name=stage_name,
                backbone_frozen=backbone_frozen,
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
                    stage_name=stage_name,
                    backbone_frozen=backbone_frozen,
                )
                print(f"Saved best checkpoint to {BEST_CHECKPOINT_PATH}")

            scheduler.step()
    finally:
        writer.close()


if __name__ == "__main__":
    main()
