# SegFormer-Based Transfer Learning for Dense Risk Classification (Ours)

## 1. Task and Objective

The model is trained for **per-pixel risk-class prediction** from RGB-D imagery, using the same 5-class risk taxonomy and data pipeline as the simple CNN baseline. The difference is the architecture: a **pretrained SegFormer** encoder is used as backbone, with a **staged optimization** strategy—first training only the decoding head (and input adapter) with the backbone frozen, then unfreezing the backbone and fine-tuning with a lower learning rate for the backbone and a separate learning rate for the head. The goal is to leverage strong semantic features from Cityscapes while adapting to the risk-class task with minimal catastrophic forgetting.

---

## 2. Architecture (Detailed)

**Model:** `SegFormerRisk`.

### 2.1 Input and adapter

- **Input:** 4-channel tensor `[B, 4, H, W]`: RGB (3 ch) + normalized depth (1 ch), same normalization as in the CNN baseline (§3.2 of TRAIN_CNN_SIMPLE).
- **Input adapter:** The SegFormer backbone expects 3-channel RGB. A **learned** mapping \(4 \to 3\) is used:
  - **Default (adapter_hidden = None):** Single \(1\times 1\) convolution: \(\mathbb{R}^4 \to \mathbb{R}^3\). Initialization is **RGB-preserving**: weight and bias are zeroed, then for \(i \in \{0,1,2\}\), \(W_{i,i,0,0} \leftarrow 1\). So at init, the first three output channels copy the RGB input and the depth channel does not feed through; pretrained backbone features remain valid.
  - **Alternative:** If `adapter_hidden` is set, the adapter is a small MLP in 1×1 space: Conv2d(4, adapter_hidden, 1) → ReLU → Conv2d(adapter_hidden, 3, 1).
- **Output of adapter:** `[B, 3, H, W]`, passed to the backbone as `pixel_values`.

### 2.2 Backbone

- **Model:** Hugging Face `SegformerForSemanticSegmentation` loaded from `nvidia/segformer-b5-finetuned-cityscapes-1024-1024` (SegFormer-B5 pretrained and fine-tuned on Cityscapes). The **decoder head** is replaced at load time to output `num_labels=5` (risk classes) instead of the original Cityscapes classes; encoder weights are kept.
- **Freezing:** When `freeze_backbone=True`, every parameter under `model.backbone.segformer` has `requires_grad=False`. Only the adapter and the new decode head are trained in Stage 1.

### 2.3 Decoder and output

- The backbone returns logits of shape `[B, 5, H', W']` (typically downsampled). These are **bilinearly upsampled** to the **input** spatial size \((H, W)\) with `align_corners=False`.
- **Output:** Logits `[B, 5, H, W]`; no softmax in the forward pass. Cross-entropy is applied on these logits in the training loop.

---

## 3. Data and Dataloader (Detailed)

### 3.1 Dataset and target construction

**Dataset:** Same `RiskMapDataset` from `dataloader.py` as the CNN baseline. Input composition is identical: RGB + depth, same normalization and depth range (\(D_{\min}=1\), \(D_{\max}=100\)). Target is the same **continuous** blocked risk map built with `build_blocked_risk_map` (object-level risk \(R = w_c \cdot g(D_{\min}^{\mathrm{obj}})\), 4-connectivity, same `SEMANTIC_WEIGHTS`). See TRAIN_CNN_SIMPLE §3.3 for the full risk formula and semantic weight table.

**Risk scale and binning:** Identical to the CNN: in the training loop, continuous targets are scaled by `RISK_SCALE = 10`, clamped to \([0,1]\), and binned with `RISK_BIN_EDGES = (0.2, 0.4, 0.6, 0.8)` (left-open) to obtain class indices in \(\{0,\ldots,4\}\). Representative values \((0.1, 0.3, 0.5, 0.7, 0.9)\) are used for MAE and logging.

### 3.2 Train/validation split

Deterministic 80/20 `random_split` from a single `DATASET_ROOT` with seed 42. Same as CNN baseline.

### 3.3 Data augmentation (train only)

A wrapper **`TrainAugmentDataset`** is applied to the train subset only; the validation subset is used as-is.

- **Horizontal flip:** With probability `AUG_FLIP_PROB = 0.5`, the input tensor and target tensor are flipped along the width dimension (dim=2). Applied consistently so geometry is preserved.
- **Brightness jitter (RGB only):** Sample \(\alpha \sim U[1 - 0.1, 1 + 0.1]\); set \(\texttt{rgb} \leftarrow \mathrm{clip}(\texttt{rgb} \cdot \alpha,\; 0,\; 1)\). Depth channel is unchanged.
- **Contrast jitter (RGB only):** Sample \(\beta \sim U[1 - 0.1, 1 + 0.1]\); set \(\texttt{rgb} \leftarrow \mathrm{clip}((\texttt{rgb} - \bar{\texttt{rgb}}) \cdot \beta + \bar{\texttt{rgb}},\; 0,\; 1)\) (per-channel mean \(\bar{\texttt{rgb}}\)). Depth channel is unchanged.

So the depth channel is **never** augmented; only RGB is. This preserves the physical meaning of depth for the risk formula.

### 3.4 DataLoader

Batch size 8; train loader shuffled, val loader not; `num_workers=4`, `pin_memory=True` on CUDA, `persistent_workers=True`.

---

## 4. Loss, Class Balancing, and Metrics

- **Loss:** Same as CNN baseline: `CrossEntropyLoss` with **class weights** (median-frequency balancing over the first 24 batches, power 0.5, bounds [0.25, 4.0]) and **label smoothing** 0.02. See TRAIN_CNN_SIMPLE §4.2 for the exact weight formula.
- **Metrics:** Same definitions: pixel accuracy, macro accuracy (mean of per-class accuracies over present classes), mean IoU over present risk classes, loss, and MAE in risk-value space. Confusion matrix is accumulated over the epoch; IoU per class is \(\mathrm{IoU}_k = C_{kk} / (C_{k\cdot} + C_{\cdot k} - C_{kk})\) for present classes.
- **Best-model selection:** Same tuple \((\texttt{val\_miou},\; \texttt{val\_macro\_acc},\; -\texttt{val\_loss})\) (lexicographic).

---

## 5. Optimization Strategy (Detailed)

Training is divided into **two stages**. The optimizer and scheduler are **rebuilt** at the transition (epoch 9); the previous optimizer is discarded.

### 5.1 Stage 1 — Frozen backbone (epochs 1 to FREEZE_EPOCHS = 8)

- **Trainable parameters:** Only those **not** under `backbone.segformer` (i.e. input adapter + decode head).
- **Optimizer:** AdamW with a **single** parameter group:
  - Learning rate: \(\texttt{STAGE1\_HEAD\_LR} = 5\times 10^{-5}\).
  - Weight decay: \(\texttt{WEIGHT\_DECAY} = 10^{-2}\).
- **Scheduler:** `CosineAnnealingLR` with \(T_{\max} = 8\) and \(\eta_{\min} = \texttt{STAGE1\_MIN\_LR} = 5\times 10^{-6}\). So over 8 epochs, the head LR decays from \(5\times 10^{-5}\) to \(5\times 10^{-6}\) following a cosine curve.
- **Gradient clipping:** After backward, global gradient norm is clipped to 1.0 (before optimizer step; when using AMP, after `scaler.unscale_`).
- **Mixed precision:** On CUDA, `GradScaler` and `autocast` are enabled. Forward and loss computation run in FP16 where safe; optimizer step uses unscaled gradients. This reduces memory and can speed up training.

### 5.2 Stage 2 — Full fine-tuning (epochs 9 to NUM_EPOCHS = 80)

- **Trainable parameters:** All parameters (backbone + adapter + head). Backbone is unfrozen at the start of epoch 9.
- **Optimizer:** AdamW with **two** parameter groups:
  - **Group 0 (backbone):** Parameters whose name starts with `backbone.segformer.` use \(\texttt{STAGE2\_BACKBONE\_LR} = 1\times 10^{-5}\).
  - **Group 1 (head + adapter):** All other trainable parameters use \(\texttt{STAGE2\_HEAD\_LR} = 5\times 10^{-5}\).
  - Both groups use the same weight decay \(10^{-2}\).
- **Scheduler:** A **new** `CosineAnnealingLR` with \(T_{\max} = \texttt{STAGE2\_EPOCHS} = 72\) (remaining epochs) and \(\eta_{\min} = \min(\texttt{STAGE2\_BACKBONE\_MIN\_LR},\; \texttt{STAGE2\_HEAD\_MIN\_LR}) = 1\times 10^{-6}\). So both groups follow one cosine cycle from their respective initial LRs down to \(1\times 10^{-6}\) over 72 epochs. Note: PyTorch’s CosineAnnealingLR uses a single `eta_min` for all groups; the **initial** LRs differ (backbone 1e-5, head 5e-5), and both decay toward 1e-6.
- **Gradient clipping and AMP:** Same as Stage 1 (clip 1.0, AMP on CUDA).

### 5.3 Hyperparameter summary table

| Hyperparameter | Stage 1 | Stage 2 (backbone) | Stage 2 (head) |
|----------------|---------|---------------------|----------------|
| Learning rate (initial) | \(5\times 10^{-5}\) | \(1\times 10^{-5}\) | \(5\times 10^{-5}\) |
| \(\eta_{\min}\) | \(5\times 10^{-6}\) | \(1\times 10^{-6}\) | \(1\times 10^{-6}\) |
| Schedule length | 8 epochs | 72 epochs | 72 epochs |
| Weight decay | \(10^{-2}\) | \(10^{-2}\) | \(10^{-2}\) |
| Gradient clip (max norm) | 1.0 | 1.0 | 1.0 |

**Rationale:** A lower LR for the backbone in Stage 2 reduces catastrophic forgetting of the pretrained features while still allowing adaptation to the risk task.

### 5.4 Training step (per batch)

1. `optimizer.zero_grad(set_to_none=True)`.
2. Forward under `autocast()` (CUDA): logits, loss, predicted classes, MAE, batch pixel acc.
3. Backward: `scaler.scale(loss).backward()` (or `loss.backward()` without AMP).
4. Unscale gradients if AMP; then `clip_grad_norm_(model.parameters(), 1.0)`.
5. `scaler.step(optimizer)` (or `optimizer.step()`); `scaler.update()` if AMP.
6. Accumulate confusion matrix and scalars for logging.

At the **end of each epoch**, `scheduler.step()` is called once.

---

## 6. Checkpointing and Logging

- **Per-epoch checkpoint:** Saved every epoch; includes `stage_name` ("freeze" or "finetune") and backbone frozen flag.
- **Best checkpoint:** Same selection rule as CNN; metadata includes stage and all LR/scale parameters for reproducibility.
- **TensorBoard:** Epoch scalars (train/val loss, MAE, pixel_acc, macro_acc, mIoU); per-batch scalars every 10 batches (batch loss, MAE, pixel acc, and **per-group** learning rates `train/lr_group_0`, `train/lr_group_1` in Stage 2). Preview images (RGB, predicted classes, GT classes, correctness) every epoch from the validation set. A text summary of the config (dataset, LRs, freeze_epochs, etc.) is written at startup.

---

## 7. Input/Output Summary Table

| Item | Shape / Type | Description |
|------|----------------|-------------|
| **Input** | `[B, 4, H, W]`, float32 | RGB (3 ch) + depth (1 ch) in \([0,1]\); adapter maps to 3 ch for backbone. |
| **Target (dataset)** | `[B, 1, H, W]`, float32 | Continuous blocked risk in \([0,1]\). |
| **Target (loss)** | Class indices in \(\{0,\ldots,4\}\) | From RISK_SCALE and RISK_BIN_EDGES. |
| **Output** | `[B, 5, H, W]`, float32 | Logits; argmax gives predicted class per pixel. |

---

## 8. Summary of Differences vs. Simple CNN

| Aspect | Simple CNN | Ours (SegFormer) |
|--------|------------|-------------------|
| **Architecture** | Small FCN (4→32→64→128→256→64→32→5) | SegFormer-B5 backbone + 4→3 adapter + 5-class head |
| **Pretraining** | None | Backbone: Cityscapes SegFormer-B5 |
| **Input** | RGB + depth (4 ch) | Same; adapter converts to 3 ch for backbone |
| **Optimization** | Single phase, one LR (5e-5 → 5e-6 cosine over 80 epochs) | Two phases: 8 epochs head-only (5e-5 → 5e-6), then 72 epochs full (backbone 1e-5→1e-6, head 5e-5→1e-6) |
| **Augmentation** | None in the described setup | Train only: H-flip 0.5, RGB brightness/contrast ±0.1 |
| **Precision** | FP32 | AMP (FP16) on CUDA |
| **Batch size** | 16 | 8 |
| **Data & loss** | Same RiskMapDataset, same risk formula, same class weights and label smoothing | Same |

The **data pipeline, risk definition, risk scale, binning, and class weighting** are shared so that comparisons are aligned on the same task and labels.
