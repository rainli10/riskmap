from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from dataloader import RiskMapDataset, compute_depth_weight_value
from model import SegFormerRisk, SimpleRiskCNN


DATASET_ROOT = Path("data/cityscape_prepro/train")
NICKNAME = "baseline_cnn_or_segformer_head_only"
CHECKPOINT_DIR = Path("ckpts") / NICKNAME

BATCH_SIZE = 64# 64
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
SEGFORMER_LEARNING_RATE = 1e-4
SEGFORMER_WEIGHT_DECAY = 0.01
VAL_SPLIT = 0.2
NUM_WORKERS = 0
RANDOM_SEED = 42
TARGET_MODE = "blocked"
COMPONENT_CONNECTIVITY = 4

# Options: "simplest_cnn", "segformer"
ARCHITECTURE = "simplest_cnn"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / f"{ARCHITECTURE}_best.pt"
LOG_EVERY_N_BATCHES = 3
TENSORBOARD_LOG_DIR = Path("tensorboard") / NICKNAME / ARCHITECTURE
SEGFORMER_PRETRAINED_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
SEGFORMER_FREEZE_BACKBONE = True
SEGFORMER_ADAPTER_HIDDEN: int | None = None
POSITIVE_RISK_THRESHOLD = 1e-4
POSITIVE_PIXEL_WEIGHT = 10
LOSS_NAME = "weighted_bce"  # Options: "weighted_bce", "weighted_mse", "weighted_l1", "boundary_aware"
DICE_LOSS_WEIGHT = 0.1
EDGE_LOSS_WEIGHT = 0.05

# Keep these weights in [0, 1] so they match bounded risk targets.
SEMANTIC_WEIGHTS = {
    -1: 0.0,   # unlabeled / ignore
    0: 0.0,    # road
    1: 0.00,   # sidewalk
    2: 0.1,   # building
    3: 0.2,   # wall
    4: 0.2,   # fence
    5: 0.3,   # pole
    6: 0.3,   # traffic light
    7: 0.3,   # traffic sign
    8: 0.1,   # vegetation
    9: 0.1,   # terrain
    10: 0.0,   # sky
    11: 3.0,   # person
    12: 1.0,  # rider
    13: 0.6,  # car
    14: 0.7,   # truck
    15: 0.8,  # bus
    16: 0.8,  # train
    17: 0.6,  # motorcycle
    18: 0.6,  # bicycle
}

# Edit these values if your depth representation uses a different scale.
DEPTH_MIN = 1.0
DEPTH_MAX = 100.0


class WeightedMSELoss(nn.Module):
    """Pixel-weighted MSE to emphasize sparse non-zero risk targets."""

    def __init__(self, positive_weight: float = 10.0, threshold: float = 1e-4) -> None:
        super().__init__()
        self.positive_weight = float(positive_weight)
        self.threshold = float(threshold)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight_map = torch.ones_like(targets)
        weight_map = torch.where(
            targets > self.threshold,
            torch.full_like(targets, self.positive_weight),
            weight_map,
        )
        loss_map = (predictions - targets) ** 2
        return (weight_map * loss_map).mean()


class WeightedBCEWithLogitsLoss(nn.Module):
    """Pixel-weighted BCEWithLogits for segformer output logits."""

    def __init__(self, positive_weight: float = 10.0, threshold: float = 1e-4) -> None:
        super().__init__()
        self.positive_weight = float(positive_weight)
        self.threshold = float(threshold)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight_map = torch.ones_like(targets)
        weight_map = torch.where(
            targets > self.threshold,
            torch.full_like(targets, self.positive_weight),
            weight_map,
        )
        loss_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (weight_map * loss_map).mean()


class WeightedL1Loss(nn.Module):
    """Pixel-weighted L1 for sparse non-zero risk targets."""

    def __init__(
        self,
        positive_weight: float = 10.0,
        threshold: float = 1e-4,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        self.positive_weight = float(positive_weight)
        self.threshold = float(threshold)
        self.from_logits = bool(from_logits)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(predictions) if self.from_logits else predictions
        weight_map = torch.ones_like(targets)
        weight_map = torch.where(
            targets > self.threshold,
            torch.full_like(targets, self.positive_weight),
            weight_map,
        )
        loss_map = torch.abs(probs - targets)
        return (weight_map * loss_map).mean()


class BoundaryAwareLoss(nn.Module):
    """Weighted BCE + Dice + Sobel-edge L1 for sharper boundaries."""

    def __init__(
        self,
        positive_weight: float = 10.0,
        threshold: float = 1e-4,
        dice_weight: float = 0.5,
        edge_weight: float = 0.2,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.positive_weight = float(positive_weight)
        self.threshold = float(threshold)
        self.dice_weight = float(dice_weight)
        self.edge_weight = float(edge_weight)
        self.smooth = float(smooth)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _dice_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()

    def _edge_map(self, values: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(values, self.sobel_x, padding=1)
        grad_y = F.conv2d(values, self.sobel_y, padding=1)
        return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight_map = torch.ones_like(targets)
        weight_map = torch.where(
            targets > self.threshold,
            torch.full_like(targets, self.positive_weight),
            weight_map,
        )

        bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce_loss = (weight_map * bce_map).mean()

        probs = torch.sigmoid(logits)
        dice_loss = self._dice_loss(probs, targets)

        pred_edge = self._edge_map(probs)
        target_edge = self._edge_map(targets)
        edge_loss = F.l1_loss(pred_edge, target_edge)

        return bce_loss + self.dice_weight * dice_loss + self.edge_weight * edge_loss


def build_loss() -> nn.Module:
    if LOSS_NAME == "weighted_bce":
        return WeightedBCEWithLogitsLoss(
            positive_weight=POSITIVE_PIXEL_WEIGHT,
            threshold=POSITIVE_RISK_THRESHOLD,
        )
    if LOSS_NAME == "weighted_mse":
        return WeightedMSELoss(
            positive_weight=POSITIVE_PIXEL_WEIGHT,
            threshold=POSITIVE_RISK_THRESHOLD,
        )
    if LOSS_NAME == "weighted_l1":
        return WeightedL1Loss(
            positive_weight=POSITIVE_PIXEL_WEIGHT,
            threshold=POSITIVE_RISK_THRESHOLD,
            from_logits=True,
        )
    if LOSS_NAME == "boundary_aware":
        return BoundaryAwareLoss(
            positive_weight=POSITIVE_PIXEL_WEIGHT,
            threshold=POSITIVE_RISK_THRESHOLD,
            dice_weight=DICE_LOSS_WEIGHT,
            edge_weight=EDGE_LOSS_WEIGHT,
        )
    raise ValueError(
        "Unknown LOSS_NAME "
        f"'{LOSS_NAME}'. Use 'weighted_bce', 'weighted_mse', 'weighted_l1', or 'boundary_aware'."
    )


def set_backbone_trainable(model: SegFormerRisk, trainable: bool) -> None:
    for parameter in model.backbone.segformer.parameters():
        parameter.requires_grad = trainable


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def debug_depth(depth_values: list[float] | None = None) -> list[dict[str, float | None]]:
    """Print how the current inverse-depth weighting changes with depth."""
    if depth_values is None:
        depth_values = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]

    results: list[dict[str, float | None]] = []
    previous_weight: float | None = None

    print(
        f"Debugging depth weights with DEPTH_MIN={DEPTH_MIN:.4f} "
        f"and DEPTH_MAX={DEPTH_MAX:.4f}"
    )
    print("depth\tweight\tdelta_from_previous")

    for depth in depth_values:
        weight = compute_depth_weight_value(
            depth_value=float(depth),
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
        )
        delta = None if previous_weight is None else weight - previous_weight

        results.append(
            {
                "depth": float(depth),
                "weight": float(weight),
                "delta_from_previous": None if delta is None else float(delta),
            }
        )

        delta_text = "N/A" if delta is None else f"{delta:.6f}"
        print(f"{float(depth):.4f}\t{weight:.6f}\t{delta_text}")
        previous_weight = weight

    return results


def create_data_loaders() -> tuple[DataLoader, DataLoader]:
    dataset = RiskMapDataset(
        dataset_root=DATASET_ROOT,
        semantic_weights=SEMANTIC_WEIGHTS,
        depth_min=DEPTH_MIN,
        depth_max=DEPTH_MAX,
        target_mode=TARGET_MODE,
        component_connectivity=COMPONENT_CONNECTIVITY,
    )

    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 samples to create train/validation splits.")

    val_size = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size
    if train_size == 0:
        train_size = len(dataset) - 1
        val_size = 1

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
    writer: SummaryWriter | None = None,
    epoch: int | None = None,
    global_step: int = 0,
) -> tuple[float, int]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(loader, start=1):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            predictions = model(inputs)
            loss = criterion(predictions, targets)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if is_training:
            global_step += 1
            if (
                writer is not None
                and LOG_EVERY_N_BATCHES > 0
                and batch_idx % LOG_EVERY_N_BATCHES == 0
            ):
                writer.add_scalar("train/batch_loss", loss.item(), global_step)
                if epoch is not None:
                    writer.add_scalar("train/epoch_progress", float(epoch), global_step)

    return total_loss / max(total_samples, 1), global_step


def save_checkpoint(
    model: nn.Module,
    optimizer: Adam | AdamW,
    epoch: int,
    val_loss: float,
    checkpoint_path: Path,
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "architecture": ARCHITECTURE,
            "semantic_weights": SEMANTIC_WEIGHTS,
            "depth_min": DEPTH_MIN,
            "depth_max": DEPTH_MAX,
            "target_mode": TARGET_MODE,
            "component_connectivity": COMPONENT_CONNECTIVITY,
            "segformer_pretrained_model": SEGFORMER_PRETRAINED_MODEL,
            "segformer_freeze_backbone": SEGFORMER_FREEZE_BACKBONE,
            "segformer_adapter_hidden": SEGFORMER_ADAPTER_HIDDEN,
            "positive_risk_threshold": POSITIVE_RISK_THRESHOLD,
            "positive_pixel_weight": POSITIVE_PIXEL_WEIGHT,
            "loss_name": LOSS_NAME,
            "dice_loss_weight": DICE_LOSS_WEIGHT,
            "edge_loss_weight": EDGE_LOSS_WEIGHT,
        },
        checkpoint_path,
    )


def main() -> None:
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_data_loaders()
    print(f"Target mode: {TARGET_MODE} (connectivity={COMPONENT_CONNECTIVITY})")

    first_batch = next(iter(train_loader))
    print(f"Train input shape: {tuple(first_batch['input'].shape)}")
    print(f"Train target shape: {tuple(first_batch['target'].shape)}")
    print(
        "Train target range: "
        f"{first_batch['target'].min().item():.4f} to {first_batch['target'].max().item():.4f}"
    )

    if ARCHITECTURE == "simplest_cnn":
        model = SimpleRiskCNN().to(device)
        criterion = build_loss().to(device)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    elif ARCHITECTURE == "segformer":
        model = SegFormerRisk(
            in_channels=4,
            pretrained_model_name=SEGFORMER_PRETRAINED_MODEL,
            freeze_backbone=SEGFORMER_FREEZE_BACKBONE,
            adapter_hidden=SEGFORMER_ADAPTER_HIDDEN,
        ).to(device)
        # Head-only transfer learning: keep pretrained encoder fixed.
        if SEGFORMER_FREEZE_BACKBONE:
            set_backbone_trainable(model, trainable=False)
        criterion = build_loss().to(device)
        optimizer = AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=SEGFORMER_LEARNING_RATE,
            weight_decay=SEGFORMER_WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unknown architecture: {ARCHITECTURE}")

    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(TENSORBOARD_LOG_DIR))
    writer.add_text(
        "config/summary",
        "\n".join(
            [
                f"architecture: {ARCHITECTURE}",
                f"batch_size: {BATCH_SIZE}",
                f"learning_rate: {LEARNING_RATE}",
                f"segformer_learning_rate: {SEGFORMER_LEARNING_RATE}",
                f"segformer_weight_decay: {SEGFORMER_WEIGHT_DECAY}",
                f"num_epochs: {NUM_EPOCHS}",
                f"log_every_n_batches: {LOG_EVERY_N_BATCHES}",
                f"target_mode: {TARGET_MODE}",
                f"component_connectivity: {COMPONENT_CONNECTIVITY}",
                f"depth_min: {DEPTH_MIN}",
                f"depth_max: {DEPTH_MAX}",
                f"segformer_pretrained_model: {SEGFORMER_PRETRAINED_MODEL}",
                f"segformer_freeze_backbone: {SEGFORMER_FREEZE_BACKBONE}",
                f"segformer_adapter_hidden: {SEGFORMER_ADAPTER_HIDDEN}",
                f"positive_risk_threshold: {POSITIVE_RISK_THRESHOLD}",
                f"positive_pixel_weight: {POSITIVE_PIXEL_WEIGHT}",
                f"loss_name: {LOSS_NAME}",
                f"dice_loss_weight: {DICE_LOSS_WEIGHT}",
                f"edge_loss_weight: {EDGE_LOSS_WEIGHT}",
                f"trainable_parameters: {count_trainable_parameters(model)}",
            ]
        ),
    )

    best_val_loss = float("inf")
    global_step = 0

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, global_step = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                writer=writer,
                epoch=epoch,
                global_step=global_step,
            )
            val_loss, global_step = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                writer=None,
                epoch=epoch,
                global_step=global_step,
            )

            writer.add_scalar("train/epoch_loss", train_loss, epoch)
            writer.add_scalar("val/epoch_loss", val_loss, epoch)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)

            print(
                f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )

            epoch_checkpoint_path = CHECKPOINT_DIR / f"{ARCHITECTURE}_epoch_{epoch:03d}.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                checkpoint_path=epoch_checkpoint_path,
            )
            print(f"Saved epoch checkpoint to {epoch_checkpoint_path}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_loss=val_loss,
                    checkpoint_path=BEST_CHECKPOINT_PATH,
                )
                print(f"Saved best checkpoint to {BEST_CHECKPOINT_PATH}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
