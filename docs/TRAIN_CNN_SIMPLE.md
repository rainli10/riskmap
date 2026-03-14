# Simple CNN Baseline for Dense Risk Classification

## 1. Task and Objective

The model is trained for **per-pixel risk-class prediction** from RGB-D imagery. Each pixel is assigned one of five discrete risk classes derived from an object-level risk formulation. The task is framed as semantic segmentation with \(K = 5\) classes; the model outputs logits over these classes and is trained with class-weighted cross-entropy loss and label smoothing. There is no pretrained backbone; the network is trained from scratch.

---

## 2. Architecture (Detailed)

**Model:** `SimpleRiskCNN` — a fully convolutional encoder with no downsampling (constant spatial resolution).

### 2.1 Layer specification

- **Input:** Tensor of shape `[B, 4, H, W]`: 3 RGB channels + 1 normalized depth channel, float32 in \([0, 1]\).
- **Convolutional body:** A single `nn.Sequential` of blocks. Each block is:
  - `Conv2d(in_ch, out_ch, kernel_size=3, padding=1)` (same padding),
  - `BatchNorm2d(out_ch)`,
  - `ReLU(inplace=True)`.
- **Default channel progression:**  
  \(4 \to 32 \to 64 \to 128 \to 256 \to 64 \to 32\).  
  So six hidden blocks with channels (32, 64, 128, 256, 64, 32), then the head.
- **Classification head:** One `Conv2d(32, 5, kernel_size=1)` producing logits for the 5 risk classes.
- **Output:** Logits of shape `[B, 5, H, W]`. No softmax in the forward pass; `CrossEntropyLoss` is applied on flattened spatial dimensions with class indices.

All convolutions use kernel size 3 and padding 1 (or 1×1 for the head), so **spatial dimensions H, W are unchanged** throughout the network.

---

## 3. Data and Dataloader (Detailed)

### 3.1 Dataset and folder layout

**Dataset class:** `RiskMapDataset` (from `dataloader.py`).

**Folder layout under `dataset_root`:**

- `image_png/`: RGB images as PNG; filenames (stems) identify the sample.
- `depth/`: Depth maps as NPY; same stem as the corresponding PNG.
- `label/`: Semantic label maps as NPY (integer class IDs per pixel); same stem.

All three must be aligned (same height and width). The dataset raises if any of the three is missing for a given stem.

### 3.2 Input composition (per sample)

1. **RGB:** Loaded with PIL, converted to RGB, cast to float32 and normalized to \([0,1]\), then transposed to shape `[3, H, W]`.
2. **Depth:** Loaded from NPY, squeezed to 2D, float32. Normalization:
   \[
   \texttt{depth\_norm} = \mathrm{clip}\left(\frac{\texttt{depth}}{\max(D_{\max}, 10^{-6})},\; 0,\; 1\right),
   \]
   with \(D_{\max} = 100\) (meters). Added as one channel: `[1, H, W]`.
3. **Final input:** `input = concat(RGB, depth_norm)` along the channel dimension → shape `[4, H, W]`.

No semantic label is used as input; only RGB and depth.

### 3.3 Target composition: object-level blocked risk map

The dataset returns a **continuous** risk map of shape `[1, H, W]` in \([0,1]\). Class indices are computed only inside the training loop (see §4.1).

**Step 1 — Object-level risk (blocked map):**

- **Mode:** `target_mode="blocked"` with **connectivity 4** (up/down/left/right) for connected components.
- For each **semantic class** \(c\) present in the label map:
  - Build a binary mask of pixels with that class.
  - Run **connected components** on that mask (4-connectivity); each connected region is one “object”.
  - For each object:
    - Collect all depth values in that object with depth \(> \varepsilon\) (\(\varepsilon = 10^{-6}\)).
    - If there is at least one valid depth, set \(D_{\min}^{\mathrm{obj}} = \min\) of those depths; otherwise set object risk to 0.
    - **Depth weight** (inverse-depth normalization):
      \[
      g(D) = \mathrm{clip}\left(\frac{1/D - 1/D_{\max}}{1/D_{\min} - 1/D_{\max}},\; 0,\; 1\right),
      \]
      with \(D_{\min} = 1\), \(D_{\max} = 100\) (same units as depth).
    - **Object risk:** \(R_{\mathrm{obj}} = w_c \cdot g(D_{\min}^{\mathrm{obj}})\), where \(w_c\) is the semantic weight for class \(c\).
  - Write \(R_{\mathrm{obj}}\) to every pixel of that object in the risk map.
- Background (unlabeled or zero weight) stays 0. Final continuous map is clipped to \([0,1]\).

**Step 2 — Semantic weights (fixed, not learned):**

\(w_c\) is given by a fixed dictionary `SEMANTIC_WEIGHTS` keyed by **train ID** (Cityscapes-style). Used values in this trainer:

| Train ID | Semantic weight | Train ID | Semantic weight |
|----------|-----------------|----------|-----------------|
| -1, 0, 1 | 0.0 | 10 | 0.0 |
| 2 | 0.1 | 11 (person) | **3.0** |
| 3 | 0.2 | 12 | 1.0 |
| 4 | 0.2 | 13 | 0.6 |
| 5 | 0.3 | 14 | 0.7 |
| 6 | 0.3 | 15 | 0.8 |
| 7 | 0.3 | 16 | 0.8 |
| 8 | 0.1 | 17 | 0.6 |
| 9 | 0.1 | 18 | 0.6 |

So “person” (11) has the highest base weight (3.0); road/sky/void have 0.

**Step 3 — Risk scale and binning (in training loop):**

The dataset does **not** quantize; it returns the continuous map. In the training script:

1. **Scale:** \(r_{\mathrm{scaled}} = \mathrm{clip}(r \cdot \texttt{RISK\_SCALE},\; 0,\; 1)\) with \(\texttt{RISK\_SCALE} = 10\).
2. **Binning:** Class index is obtained by bucketizing \(r_{\mathrm{scaled}}\) with boundaries \(\texttt{RISK\_BIN\_EDGES} = (0.2,\, 0.4,\, 0.6,\, 0.8)\) and `right=False` (left-open intervals):
   - Class 0: \(r_{\mathrm{scaled}} \in [0, 0.2)\)
   - Class 1: \([0.2, 0.4)\), Class 2: \([0.4, 0.6)\), Class 3: \([0.6, 0.8)\), Class 4: \([0.8, 1]\).
3. **Representative values (for MAE / visualization):** Class indices are mapped to \(\texttt{RISK\_CLASS\_VALUES} = (0.1,\, 0.3,\, 0.5,\, 0.7,\, 0.9)\).

### 3.4 Train/validation split and DataLoader

- **Split:** One root; 80% train / 20% validation via `torch.utils.data.random_split` with a fixed generator seed (42). No separate validation folder.
- **DataLoader:** Batch size 16; train loader shuffled, val loader not; `num_workers=4`, `pin_memory=True` on CUDA, `persistent_workers=True` when workers > 0.

---

## 4. Loss, Class Balancing, and Metrics

### 4.1 Loss

- **Criterion:** `nn.CrossEntropyLoss` with:
  - **Class weights** \(\mathbf{w} \in \mathbb{R}^5\): applied to the loss per class; see §4.2.
  - **Label smoothing:** 0.02 (smoothing toward a uniform distribution over the 5 classes).
- The loss is computed over the flattened spatial dimensions: each pixel is one sample, target is the class index in \(\{0,\ldots,4\}\).

### 4.2 Class weight computation (median-frequency balancing)

Class weights are computed **once** at the start of training from the **training** set:

1. **Count pixels per risk class:** Over the training loader, convert each batch’s continuous target to class indices (same scale + binning as above). Accumulate counts \(n_k\) for \(k = 0,\ldots,4\). Counting is capped at the first \(\texttt{CLASS\_COUNT\_MAX\_BATCHES} = 24\) batches to keep startup fast.
2. **Frequencies:** \(f_k = n_k / \sum_j n_j\) for classes with \(n_k > 0\).
3. **Median frequency:** \(f_{\mathrm{med}} = \mathrm{median}\{f_k : n_k > 0\}\).
4. **Balanced weights (only for present classes):**
   \[
   \tilde{w}_k = \left(\frac{f_{\mathrm{med}}}{f_k}\right)^{\texttt{CLASS\_WEIGHT\_POWER}},
   \]
   with \(\texttt{CLASS\_WEIGHT\_POWER} = 0.5\).
5. **Normalize:** \(\tilde{w}_k \leftarrow \tilde{w}_k / \mathrm{mean}(\tilde{w}_k)\) over present classes.
6. **Clip:** \(w_k = \mathrm{clip}(\tilde{w}_k,\; \texttt{CLASS\_WEIGHT\_MIN},\; \texttt{CLASS\_WEIGHT\_MAX})\) with (0.25, 4.0). Classes with \(n_k = 0\) get weight 0.

These \(w_k\) are passed to `CrossEntropyLoss(weight=...)` so that rare risk classes (e.g. high risk) receive a higher loss weight.

### 4.3 Evaluation metrics (per epoch)

All metrics are computed from a **confusion matrix** \(C \in \mathbb{N}^{5\times 5}\) (accumulated over the epoch):

- **Pixel accuracy:** \(\frac{\sum_k C_{kk}}{\sum_{ij} C_{ij}}\).
- **Per-class accuracy (for present classes):** For class \(k\) with \(\sum_j C_{kj} > 0\), \(\mathrm{acc}_k = C_{kk} / \sum_j C_{kj}\). **Macro accuracy** is the mean of \(\mathrm{acc}_k\) over all \(k\) with \(\sum_j C_{kj} > 0\).
- **Per-class IoU (for present classes):** \(\mathrm{IoU}_k = \frac{C_{kk}}{\sum_j C_{kj} + \sum_i C_{ik} - C_{kk}}\). **Mean IoU (mIoU)** is the mean of \(\mathrm{IoU}_k\) over all \(k\) with target count \(> 0\) (and union \(> 0\) where defined).
- **MAE (risk space):** After converting predicted and target class indices to the representative values (0.1, 0.3, 0.5, 0.7, 0.9), mean absolute error is computed over all pixels and averaged over the epoch.

Best-model selection uses the tuple \((\texttt{val\_miou},\; \texttt{val\_macro\_acc},\; -\texttt{val\_loss})\) in lexicographic order.

---

## 5. Optimization Strategy (Detailed)

| Hyperparameter | Value | Notes |
|----------------|--------|------|
| Optimizer | AdamW | One parameter group; all parameters trained. |
| Learning rate | \(5\times 10^{-5}\) | Applied to all parameters. |
| Weight decay | \(10^{-2}\) | L2 regularization. |
| Schedule | CosineAnnealingLR | \(T_{\max} = 80\) (NUM_EPOCHS), \(\eta_{\min} = 5\times 10^{-6}\). So LR decays from \(5\times 10^{-5}\) to \(5\times 10^{-6}\) following \(\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\pi t/T_{\max}))\). |
| Gradient clipping | Global norm | \(\|\nabla\| \mapsto \min(\|\nabla\|, 1)\) (GRAD_CLIP_NORM=1.0). |
| Batch size | 16 | No gradient accumulation. |
| Precision | FP32 | No automatic mixed precision in this script. |
| Epochs | 80 | Single phase; no staged unfreezing. |

**Training step (per batch):** `optimizer.zero_grad(set_to_none=True)` → forward → loss backward → `clip_grad_norm_(model.parameters(), 1.0)` → `optimizer.step()` → `scheduler.step()` at end of each epoch.

---

## 6. Checkpointing and Logging

- **Per-epoch checkpoint:** Saved every epoch (e.g. `simple_cnn_epoch_001.pt`, …).
- **Best checkpoint:** Saved whenever the validation tuple \((\texttt{val\_miou},\; \texttt{val\_macro\_acc},\; -\texttt{val\_loss})\) strictly improves (lexicographic order).
- **TensorBoard:** Scalars logged every epoch (train/val loss, MAE, pixel_acc, macro_acc, mIoU) and every LOG_EVERY_N_BATCHES (default 10) for batch loss, batch MAE, batch pixel accuracy, and learning rate. Preview images (RGB, predicted classes, GT classes, correctness) are logged every PREVIEW_EVERY_N_EPOCHS (e.g. 1) from the validation set.

---

## 7. Input/Output Summary Table

| Item | Shape / Type | Description |
|------|----------------|-------------|
| **Input** | `[B, 4, H, W]`, float32 | RGB (3 ch) + depth (1 ch), each in \([0,1]\). |
| **Target (dataset)** | `[B, 1, H, W]`, float32 | Continuous blocked risk in \([0,1]\). |
| **Target (loss)** | Class indices in \(\{0,\ldots,4\}\) | From scaled risk + bin edges (0.2, 0.4, 0.6, 0.8). |
| **Output** | `[B, 5, H, W]`, float32 | Logits; argmax along channel gives predicted class per pixel. |

---

## 8. Reproducibility

- **Random seed:** 42 for PyTorch (and CUDA if available); used for dataset split and any randomness in training.
- **Determinism:** Same `SEMANTIC_WEIGHTS`, `DEPTH_MIN`, `DEPTH_MAX`, `TARGET_MODE`, `COMPONENT_CONNECTIVITY` (4), `RISK_SCALE`, and `RISK_BIN_EDGES` ensure identical ground-truth risk and class labels across runs. DataLoader workers and hardware may still cause non-determinism unless fixed (e.g. `worker_init_fn`, deterministic cuDNN).
