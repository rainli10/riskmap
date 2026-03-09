from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#     Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#     Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#     Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#     Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#     Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#     Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#     Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#     Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#     Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#     Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#     Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#     Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#     Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
#     Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#     Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#     Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#     Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#     Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#     Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#     Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#     Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#     Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#     Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#     Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#     Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#     Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#     Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#     Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#     Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
# ]
custom_weight = 10

def _validate_depth_range(depth_min: float, depth_max: float) -> None:
    if depth_min <= 0:
        raise ValueError("depth_min must be greater than 0.")
    if depth_max <= depth_min:
        raise ValueError("depth_max must be greater than depth_min.")


def _prepare_label_map(label_map: np.ndarray) -> np.ndarray:
    label = np.asarray(label_map)
    label = np.squeeze(label)

    if label.ndim != 2:
        raise ValueError(f"Label map must be 2D after squeeze, got shape {label.shape}.")

    return np.rint(label).astype(np.int64)


def _prepare_depth_map(depth_map: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    depth = np.squeeze(depth)

    if depth.ndim != 2:
        raise ValueError(f"Depth map must be 2D after squeeze, got shape {depth.shape}.")

    return depth


def compute_depth_weight(
    depth_map: np.ndarray,
    depth_min: float,
    depth_max: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute the clipped inverse-depth weight g(D)."""
    _validate_depth_range(depth_min, depth_max)

    depth = _prepare_depth_map(depth_map)
    valid_mask = depth > eps
    depth_weight = np.zeros_like(depth, dtype=np.float32)

    denom = (1.0 / depth_min) - (1.0 / depth_max)
    inverse_depth = np.zeros_like(depth, dtype=np.float32)
    inverse_depth[valid_mask] = 1.0 / depth[valid_mask]
    depth_weight[valid_mask] = np.clip(
        (inverse_depth[valid_mask] - (1.0 / depth_max)) / denom,
        0.0,
        1.0,
    )

    return depth_weight


def compute_depth_weight_value(
    depth_value: float,
    depth_min: float,
    depth_max: float,
    eps: float = 1e-6,
) -> float:
    """Compute g(D) for a single depth value."""
    _validate_depth_range(depth_min, depth_max)

    if depth_value <= eps:
        return 0.0

    denom = (1.0 / depth_min) - (1.0 / depth_max)
    weight = ((1.0 / depth_value) - (1.0 / depth_max)) / denom
    return float(np.clip(custom_weight*weight, 0.0, 1.0))


def connected_components(binary_mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Return a component-id map for a 2D binary mask."""
    mask = np.asarray(binary_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Binary mask must be 2D, got shape {mask.shape}.")

    if connectivity == 4:
        neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    elif connectivity == 8:
        neighbors = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )
    else:
        raise ValueError(f"Unsupported connectivity {connectivity}. Use 4 or 8.")

    height, width = mask.shape
    component_map = np.full((height, width), -1, dtype=np.int32)
    current_component_id = 0

    for start_y, start_x in np.argwhere(mask):
        start_y = int(start_y)
        start_x = int(start_x)

        if component_map[start_y, start_x] != -1:
            continue

        queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
        component_map[start_y, start_x] = current_component_id

        while queue:
            y, x = queue.popleft()

            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx

                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if not mask[ny, nx] or component_map[ny, nx] != -1:
                    continue

                component_map[ny, nx] = current_component_id
                queue.append((ny, nx))

        current_component_id += 1

    return component_map, current_component_id


def build_object_component_map(label_map: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Approximate object ids by running connected components within each class mask."""
    label_ids = _prepare_label_map(label_map)
    object_map = np.full(label_ids.shape, -1, dtype=np.int32)
    current_object_id = 0

    for class_id in np.unique(label_ids):
        class_mask = label_ids == int(class_id)
        component_map, num_components = connected_components(class_mask, connectivity=connectivity)

        for component_id in range(num_components):
            object_map[component_map == component_id] = current_object_id
            current_object_id += 1

    return object_map


def build_dense_risk_map(
    label_map: np.ndarray,
    depth_map: np.ndarray,
    semantic_weights: Dict[int, float],
    depth_min: float,
    depth_max: float,
) -> np.ndarray:
    """Construct the original dense target risk map R(x, y) = w_c * g(D)."""
    label_ids = _prepare_label_map(label_map)
    depth_weight = compute_depth_weight(depth_map, depth_min=depth_min, depth_max=depth_max)

    unique_labels = {int(class_id) for class_id in np.unique(label_ids)}
    missing_weights = sorted(unique_labels.difference(semantic_weights))
    if missing_weights:
        raise KeyError(
            "Missing semantic weights for class ids: "
            f"{missing_weights}. Update your SEMANTIC_WEIGHTS config."
        )

    semantic_weight_map = np.zeros_like(depth_weight, dtype=np.float32)
    for class_id in unique_labels:
        semantic_weight_map[label_ids == class_id] = float(semantic_weights[class_id])

    risk_map = semantic_weight_map * depth_weight

    return np.clip(risk_map, 0.0, 1.0).astype(np.float32)


def build_blocked_risk_map(
    label_map: np.ndarray,
    depth_map: np.ndarray,
    semantic_weights: Dict[int, float],
    depth_min: float,
    depth_max: float,
    connectivity: int = 4,
    eps: float = 1e-6,
) -> np.ndarray:
    """Construct an object-blocked risk map using minimum valid depth per component."""
    label_ids = _prepare_label_map(label_map)
    depth = _prepare_depth_map(depth_map)

    unique_labels = {int(class_id) for class_id in np.unique(label_ids)}
    missing_weights = sorted(unique_labels.difference(semantic_weights))
    if missing_weights:
        raise KeyError(
            "Missing semantic weights for class ids: "
            f"{missing_weights}. Update your SEMANTIC_WEIGHTS config."
        )

    object_map = build_object_component_map(label_ids, connectivity=connectivity)
    blocked_risk_map = np.zeros_like(depth, dtype=np.float32)

    for object_id in np.unique(object_map):
        if object_id < 0:
            continue

        object_mask = object_map == int(object_id)
        class_id = int(label_ids[object_mask][0])
        valid_depths = depth[object_mask & (depth > eps)]

        if valid_depths.size == 0:
            object_risk = 0.0
        else:
            min_depth = float(valid_depths.min())
            object_risk = float(semantic_weights[class_id]) * compute_depth_weight_value(
                depth_value=min_depth,
                depth_min=depth_min,
                depth_max=depth_max,
                eps=eps,
            )

        blocked_risk_map[object_mask] = object_risk

    return np.clip(custom_weight*blocked_risk_map, 0.0, 1.0).astype(np.float32)


def build_risk_map(
    label_map: np.ndarray,
    depth_map: np.ndarray,
    semantic_weights: Dict[int, float],
    depth_min: float,
    depth_max: float,
) -> np.ndarray:
    """Backward-compatible alias for the original dense risk map."""
    return build_dense_risk_map(
        label_map=label_map,
        depth_map=depth_map,
        semantic_weights=semantic_weights,
        depth_min=depth_min,
        depth_max=depth_max,
    )


class RiskMapDataset(Dataset):
    """Loads aligned RGB images, depth maps, labels, and on-the-fly risk targets."""

    def __init__(
        self,
        dataset_root: str | Path,
        semantic_weights: Dict[int, float],
        depth_min: float,
        depth_max: float,
        target_mode: str = "blocked",
        component_connectivity: int = 8,
        image_folder: str = "image_png",
        label_folder: str = "label",
        depth_folder: str = "depth",
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.image_dir = self.dataset_root / image_folder
        self.label_dir = self.dataset_root / label_folder
        self.depth_dir = self.dataset_root / depth_folder
        self.semantic_weights = {int(key): float(value) for key, value in semantic_weights.items()}
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.target_mode = target_mode
        self.component_connectivity = int(component_connectivity)

        if self.target_mode not in {"dense", "blocked"}:
            raise ValueError(f"Unsupported target_mode '{self.target_mode}'. Use 'dense' or 'blocked'.")

        for directory in (self.image_dir, self.label_dir, self.depth_dir):
            if not directory.exists():
                raise FileNotFoundError(f"Required dataset directory does not exist: {directory}")

        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(f"No PNG images found in {self.image_dir}")

    def _build_index(self) -> list[dict[str, Path | str]]:
        samples: list[dict[str, Path | str]] = []

        for image_path in sorted(self.image_dir.glob("*.png")):
            sample_id = image_path.stem
            label_path = self.label_dir / f"{sample_id}.npy"
            depth_path = self.depth_dir / f"{sample_id}.npy"

            if not label_path.exists():
                raise FileNotFoundError(f"Missing label file for {sample_id}: {label_path}")
            if not depth_path.exists():
                raise FileNotFoundError(f"Missing depth file for {sample_id}: {depth_path}")

            samples.append(
                {
                    "id": sample_id,
                    "image": image_path,
                    "label": label_path,
                    "depth": depth_path,
                }
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]

        image = Image.open(sample["image"]).convert("RGB")
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1))).float()

        label_array = np.load(sample["label"])
        depth_array = np.load(sample["depth"]).astype(np.float32)

        label_array = np.squeeze(label_array)
        depth_array = np.squeeze(depth_array)

        if label_array.ndim != 2:
            raise ValueError(f"Expected 2D label map, got shape {label_array.shape}")
        if depth_array.ndim != 2:
            raise ValueError(f"Expected 2D depth map, got shape {depth_array.shape}")

        image_height, image_width = image_array.shape[:2]
        if label_array.shape != (image_height, image_width):
            raise ValueError(
                f"Label shape {label_array.shape} does not match image size "
                f"{(image_height, image_width)} for sample {sample['id']}"
            )
        if depth_array.shape != (image_height, image_width):
            raise ValueError(
                f"Depth shape {depth_array.shape} does not match image size "
                f"{(image_height, image_width)} for sample {sample['id']}"
            )

        depth_tensor = torch.from_numpy(
            np.clip(depth_array / max(self.depth_max, 1e-6), 0.0, 1.0)[None, ...]
        ).float()

        if self.target_mode == "dense":
            risk_map = build_dense_risk_map(
                label_map=label_array,
                depth_map=depth_array,
                semantic_weights=self.semantic_weights,
                depth_min=self.depth_min,
                depth_max=self.depth_max,
            )
        else:
            risk_map = build_blocked_risk_map(
                label_map=label_array,
                depth_map=depth_array,
                semantic_weights=self.semantic_weights,
                depth_min=self.depth_min,
                depth_max=self.depth_max,
                connectivity=self.component_connectivity,
            )
        risk_tensor = torch.from_numpy(risk_map[None, ...]).float()

        input_tensor = torch.cat((image_tensor, depth_tensor), dim=0)

        return {
            "input": input_tensor,
            "target": risk_tensor,
            "image_id": str(sample["id"]),
        }
