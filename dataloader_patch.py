"""
Patchified dataset loader for optical ONN training.

Creates 4x4 patch grids (16 channels) from resized inputs. For each patch, it
can either:
- build a deterministic 2x2 rotation tile (0/90/180/270 clockwise), then resize
  to the full canvas; or
- directly interpolate each patch to the full canvas without rotation tiling.

Also supports a direct single-channel mode (`upsampler`) that keeps the native
input resolution (no resize in dataloader).
"""

from __future__ import annotations

import csv
import math
import re
import string
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.io import ImageReadMode, read_image

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency for image I/O path
    Image = None

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency for facial dataset
    pd = None

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - optional dependency for VLM
    AutoTokenizer = None

_FACIAL_DEFAULT_ZIP = Path(
    "/home/adminlwe/Documents/lwe-opu/experiment/model_training/onn_online_training/Data_Facial_keypoints/training.zip"
)
_FACIAL_SHAPES_LOGGED = False
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_CAPTION_FILTER_ARTICLES = {"a", "an", "the"}
_DIGIT_LABEL_NAMES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]
_FASHION_LABEL_NAMES = [
    "t-shirt/top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]
_NOTMNIST_LABEL_NAMES = [chr(ord("A") + i) for i in range(10)]
_SYNTHETIC_SLOT_NAMES = ("top left", "top right", "bottom center")
_SYNTHETIC_FAMILY_NAMES = ("mnist", "fashion", "notmnist")
_NOTMNIST_DOWNLOAD_URLS = (
    "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/notMNIST_small.tar.gz",
    "http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz",
)


class PatchifiedDataset(Dataset):
    """Wraps a base dataset and returns either patchified or upsampled inputs."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        num_channels: int = 16,
        hw: Tuple[int, int] = (112, 112),
        use_rot4_tiling: bool = True,
        method: str = "patchify",
        overlap_percent: float = 0.0,
    ) -> None:
        self.dataset = dataset
        self.num_channels = int(num_channels)
        self.hw = (int(hw[0]), int(hw[1]))
        self.use_rot4_tiling = bool(use_rot4_tiling)
        self.method = str(method).strip().lower()
        self.overlap_percent = float(overlap_percent)
        if self.method not in {"patchify", "upsampler"}:
            raise ValueError("method must be one of {'patchify', 'upsampler'}.")
        if self.overlap_percent < 0.0:
            raise ValueError("overlap_percent must be non-negative.")
        if self.overlap_percent >= 100.0:
            raise ValueError("overlap_percent must be < 100.")

        self.grid = 1
        self.patch_hw = self.hw
        if self.method == "patchify":
            grid = int(round(math.sqrt(self.num_channels)))
            if grid * grid != self.num_channels:
                raise ValueError("num_channels must be a perfect square for 2D patch grids.")
            if self.hw[0] % grid != 0 or self.hw[1] % grid != 0:
                raise ValueError("hw must be divisible by sqrt(num_channels).")
            self.grid = grid
            self.patch_hw = (self.hw[0] // grid, self.hw[1] // grid)

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _coerce_label(y):
        if isinstance(y, torch.Tensor):
            if y.dim() == 0:
                return int(y.item())
            return y.to(dtype=torch.float32)
        if isinstance(y, np.ndarray):
            if y.ndim == 0:
                return int(y.item())
            return torch.from_numpy(y.astype(np.float32, copy=False))
        if isinstance(y, (list, tuple)):
            if len(y) == 1:
                return int(y[0])
            return torch.tensor(y, dtype=torch.float32)
        return int(y)

    @staticmethod
    def _interp_patch_to_canvas(
        patches: torch.Tensor,
        output_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Resize each patch directly to the output canvas."""
        return F.interpolate(patches, size=output_hw, mode="bilinear", align_corners=False)

    @staticmethod
    def _rot4_tile_and_resize(
        patches: torch.Tensor,
        output_hw: Tuple[int, int],
        patch_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Build 2x2 deterministic rotation tiles per patch, then resize to canvas."""
        patch_h, patch_w = patch_hw
        rot0 = patches
        rot90 = torch.rot90(patches, k=-1, dims=(-2, -1))   # 90 deg clockwise
        rot180 = torch.rot90(patches, k=2, dims=(-2, -1))   # 180 deg
        rot270 = torch.rot90(patches, k=1, dims=(-2, -1))   # 270 deg clockwise

        # Keep a stable tile shape even when patches are not square.
        if rot90.shape[-2:] != (patch_h, patch_w):
            rot90 = F.interpolate(rot90, size=(patch_h, patch_w), mode="bilinear", align_corners=False)
            rot270 = F.interpolate(rot270, size=(patch_h, patch_w), mode="bilinear", align_corners=False)

        top = torch.cat((rot0, rot90), dim=-1)
        bottom = torch.cat((rot180, rot270), dim=-1)
        tiled = torch.cat((top, bottom), dim=-2)
        return F.interpolate(tiled, size=output_hw, mode="bilinear", align_corners=False)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        if not isinstance(x, torch.Tensor):
            x = transforms.functional.to_tensor(x)
        x = x.to(dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"Expected input tensor (C,H,W), got {tuple(x.shape)}")

        # Upsampler mode: no patch splitting and no resize in dataloader.
        if self.method == "upsampler":
            y_out = self._coerce_label(y)
            return x.clamp(0.0, 1.0), y_out

        if x.size(0) != 1:
            # Patchify mode expects a single channel; fold multi-channel inputs to grayscale.
            x = x.mean(dim=0, keepdim=True)

        # 1) Resize to target canvas.
        x_resized = F.interpolate(
            x.unsqueeze(0),
            size=self.hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (1, H, W)

        # 2) Split into 4x4 patches (16 channels) with optional overlap.
        _, H, W = x_resized.shape
        base_stride_h = H // self.grid
        base_stride_w = W // self.grid
        overlap_ratio = self.overlap_percent / 100.0
        patch_h = int(round(base_stride_h / (1.0 - overlap_ratio)))
        patch_w = int(round(base_stride_w / (1.0 - overlap_ratio)))
        patch_h = min(max(1, patch_h), H)
        patch_w = min(max(1, patch_w), W)

        center_shift_h = (patch_h - base_stride_h) // 2
        center_shift_w = (patch_w - base_stride_w) // 2
        y_positions = [i * base_stride_h - center_shift_h for i in range(self.grid)]
        x_positions = [i * base_stride_w - center_shift_w for i in range(self.grid)]

        patch_list = []
        for y_start in y_positions:
            y_start = int(min(max(y_start, 0), max(H - 1, 0)))
            y_end = int(min(y_start + patch_h, H))
            for x_start in x_positions:
                x_start = int(min(max(x_start, 0), max(W - 1, 0)))
                x_end = int(min(x_start + patch_w, W))
                patch = x_resized[:, y_start:y_end, x_start:x_end]
                # Keep a fixed tensor size for stacking when boundary clipping occurs.
                if patch.shape[-2:] != (patch_h, patch_w):
                    patch = F.interpolate(
                        patch.unsqueeze(0),
                        size=(patch_h, patch_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                patch_list.append(patch)
        patches = torch.stack(patch_list, dim=0)
        if patches.size(0) != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} patches, got {patches.size(0)}."
            )

        # 3) Convert each patch to a full-canvas channel.
        if self.use_rot4_tiling:
            # Deterministic 4-tile rotation pattern (0/90/180/270 clockwise).
            patches_up = self._rot4_tile_and_resize(patches, self.hw, (patch_h, patch_w))
        else:
            # No rotation tiling; direct interpolation only.
            patches_up = self._interp_patch_to_canvas(patches, self.hw)
        x_out = patches_up.squeeze(1).clamp(0.0, 1.0)
        y_out = self._coerce_label(y)
        return x_out, y_out


@dataclass(frozen=True)
class VLMSample:
    """Caption sample tied to one image and one caption string."""

    sample_index: int
    image_index: int
    caption_index: int
    image_filename: str
    image_path: Path
    caption: str


@dataclass(frozen=True)
class SyntheticTriSlotSample:
    """Synthetic caption sample with 3 grayscale slots on a triangular layout."""

    sample_index: int
    image_filename: str
    caption: str
    family_ids: Tuple[int, int, int]
    image_indices: Tuple[int, int, int]
    class_labels: Tuple[int, int, int]


def _synthetic_class_text(family_id: int, class_label: int) -> str:
    if int(family_id) == 0:
        if class_label < 0 or class_label >= len(_DIGIT_LABEL_NAMES):
            raise ValueError(f"MNIST label must be in [0, 9], got {class_label}.")
        return f"mnist digit {_DIGIT_LABEL_NAMES[int(class_label)]}"
    if int(family_id) == 1:
        if class_label < 0 or class_label >= len(_FASHION_LABEL_NAMES):
            raise ValueError(f"Fashion-MNIST label must be in [0, 9], got {class_label}.")
        return f"fashion {_FASHION_LABEL_NAMES[int(class_label)]}"
    if int(family_id) == 2:
        if class_label < 0 or class_label >= len(_NOTMNIST_LABEL_NAMES):
            raise ValueError(f"notMNIST label must be in [0, 9], got {class_label}.")
        return f"notmnist letter {_NOTMNIST_LABEL_NAMES[int(class_label)]}"
    raise ValueError(f"Unsupported family id: {family_id}")


def _caption_from_tri_labels(*, family_ids: Tuple[int, int, int], class_labels: Tuple[int, int, int]) -> str:
    if len(family_ids) != 3 or len(class_labels) != 3:
        raise ValueError("Tri-slot captions require exactly 3 family ids and 3 class labels.")
    slot_descriptions: List[str] = []
    for family_id, class_label in zip(family_ids, class_labels):
        slot_descriptions.append(_synthetic_class_text(int(family_id), int(class_label)))
    return (
        f"the {_SYNTHETIC_SLOT_NAMES[0]} image is {slot_descriptions[0]}, "
        f"the {_SYNTHETIC_SLOT_NAMES[1]} image is {slot_descriptions[1]}, "
        f"and the {_SYNTHETIC_SLOT_NAMES[2]} image is {slot_descriptions[2]}"
    )


def _build_tri_family_samples(
    *,
    mnist_labels: torch.Tensor,
    fashion_labels: torch.Tensor,
    notmnist_labels: torch.Tensor,
    num_samples: int,
    seed: int,
    split_name: str,
) -> List[SyntheticTriSlotSample]:
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be positive.")
    counts = (
        int(mnist_labels.numel()),
        int(fashion_labels.numel()),
        int(notmnist_labels.numel()),
    )
    if any(count <= 0 for count in counts):
        raise ValueError("MNIST/Fashion-MNIST/notMNIST labels must be non-empty.")

    rng = np.random.default_rng(int(seed))

    labels_by_family = (
        mnist_labels.to(dtype=torch.long),
        fashion_labels.to(dtype=torch.long),
        notmnist_labels.to(dtype=torch.long),
    )

    samples: List[SyntheticTriSlotSample] = []
    for sample_idx in range(int(num_samples)):
        # Force one sample to contain all three families; only slot positions are shuffled.
        family_ids_row = tuple(int(x) for x in rng.permutation(np.arange(3, dtype=np.int64)).tolist())
        if set(family_ids_row) != {0, 1, 2}:
            raise RuntimeError(f"Invalid synthetic family assignment for sample {sample_idx}: {family_ids_row}")
        image_idx_row = [0, 0, 0]
        class_label_row = [0, 0, 0]
        for slot_pos, family_id in enumerate(family_ids_row):
            family_count = counts[family_id]
            img_idx = int(rng.integers(0, family_count))
            family_labels = labels_by_family[family_id]
            image_idx_row[slot_pos] = img_idx
            class_label_row[slot_pos] = int(family_labels[img_idx].item())
        class_label_tuple = tuple(int(x) for x in class_label_row)
        caption = _caption_from_tri_labels(
            family_ids=family_ids_row,
            class_labels=class_label_tuple,
        )
        samples.append(
            SyntheticTriSlotSample(
                sample_index=sample_idx,
                image_filename=f"{split_name}_tri_{sample_idx:07d}.png",
                caption=caption,
                family_ids=family_ids_row,
                image_indices=tuple(int(x) for x in image_idx_row),
                class_labels=class_label_tuple,
            )
        )
    return samples


class _SyntheticTriSlotImageDataset(Dataset):
    """Returns 3-image triangular-layout grayscale canvases with deterministic indexing."""

    def __init__(
        self,
        *,
        samples: List[SyntheticTriSlotSample],
        mnist_images: torch.Tensor,
        fashion_images: torch.Tensor,
        notmnist_images: torch.Tensor,
        combined_hw: Tuple[int, int] = (112, 112),
        slot_gap: int = 2,
    ) -> None:
        if int(slot_gap) < 0:
            raise ValueError("slot_gap must be non-negative.")
        self.samples = samples
        self.combined_hw = (int(combined_hw[0]), int(combined_hw[1]))
        self.slot_gap = int(slot_gap)
        if self.combined_hw[0] <= 0 or self.combined_hw[1] <= 0:
            raise ValueError("combined_hw must contain positive dimensions.")

        if not isinstance(mnist_images, torch.Tensor):
            mnist_images = torch.as_tensor(mnist_images)
        if not isinstance(fashion_images, torch.Tensor):
            fashion_images = torch.as_tensor(fashion_images)
        if not isinstance(notmnist_images, torch.Tensor):
            notmnist_images = torch.as_tensor(notmnist_images)
        if mnist_images.dim() != 3 or fashion_images.dim() != 3 or notmnist_images.dim() != 3:
            raise ValueError("MNIST/Fashion/notMNIST image tensors must have shape (N,H,W).")
        self.images_by_family = (
            mnist_images.to(dtype=torch.float32).unsqueeze(1) / 255.0,
            fashion_images.to(dtype=torch.float32).unsqueeze(1) / 255.0,
            notmnist_images.to(dtype=torch.float32).unsqueeze(1) / 255.0,
        )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _ensure_slot_hw(image: torch.Tensor, slot_hw: Tuple[int, int]) -> torch.Tensor:
        if image.shape[-2:] == slot_hw:
            return image
        return F.interpolate(
            image.unsqueeze(0),
            size=slot_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _compose_tri(self, sample: SyntheticTriSlotSample) -> torch.Tensor:
        slot_hw = (28, 28)
        top_y = 0
        left_x = 0
        right_x = 28 + self.slot_gap
        bottom_y = 28 + self.slot_gap
        raw_side = 56 + self.slot_gap
        bottom_x = (raw_side - 28) // 2
        canvas = torch.zeros((1, raw_side, raw_side), dtype=torch.float32)

        placements = (
            (top_y, left_x),
            (top_y, right_x),
            (bottom_y, bottom_x),
        )
        for slot_idx in range(3):
            family_id = int(sample.family_ids[slot_idx])
            image_idx = int(sample.image_indices[slot_idx])
            image_bank = self.images_by_family[family_id]
            slot_img = self._ensure_slot_hw(image_bank[image_idx], slot_hw)
            y0, x0 = placements[slot_idx]
            canvas[:, y0 : y0 + slot_hw[0], x0 : x0 + slot_hw[1]] = slot_img
        if canvas.shape[-2:] != self.combined_hw:
            canvas = F.interpolate(
                canvas.unsqueeze(0),
                size=self.combined_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return canvas.clamp(0.0, 1.0)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = self._compose_tri(sample)
        return image, idx


class SyntheticTriFamilyCaptionDataset(Dataset):
    """
    Synthetic VLM caption dataset:
    triangular 3-slot MNIST/Fashion-MNIST/notMNIST composition with deterministic captions.
    """

    def __init__(
        self,
        *,
        samples: List[SyntheticTriSlotSample],
        tokenizer: Any,
        max_caption_length: int,
        mnist_images: torch.Tensor,
        fashion_images: torch.Tensor,
        notmnist_images: torch.Tensor,
        image_size: Tuple[int, int] = (112, 112),
        num_channels: int = 16,
        overlap_percent: float = 0.0,
        slot_gap: int = 2,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_caption_length = int(max_caption_length)
        if self.max_caption_length <= 0:
            raise ValueError("max_caption_length must be positive.")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("image_size must contain positive values.")

        tri_images = _SyntheticTriSlotImageDataset(
            samples=samples,
            mnist_images=mnist_images,
            fashion_images=fashion_images,
            notmnist_images=notmnist_images,
            combined_hw=self.image_size,
            slot_gap=int(slot_gap),
        )
        # Always patchify here (4x4 patches) to get 16 optical channels.
        self.image_pipeline = PatchifiedDataset(
            tri_images,
            num_channels=int(num_channels),
            hw=self.image_size,
            use_rot4_tiling=False,
            method="patchify",
            overlap_percent=float(overlap_percent),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_tensor, sample_idx = self.image_pipeline[idx]
        sample = self.samples[int(sample_idx)]
        tokenized = self.tokenizer(
            sample.caption,
            max_length=self.max_caption_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        caption_ids = tokenized["input_ids"].squeeze(0).to(dtype=torch.long)
        attention_mask = tokenized["attention_mask"].squeeze(0).to(dtype=torch.long)
        return {
            "image": image_tensor,
            "caption": sample.caption,
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "image_filename": sample.image_filename,
            "image_index": sample.sample_index,
            "sample_index": sample.sample_index,
            "caption_index": 0,
        }


def _parse_hw(cfg_value: object, *, default: Tuple[int, int]) -> Tuple[int, int]:
    if cfg_value is None:
        return default
    if isinstance(cfg_value, (list, tuple)) and len(cfg_value) == 2:
        h = int(cfg_value[0])
        w = int(cfg_value[1])
        if h <= 0 or w <= 0:
            raise ValueError("image_size/hw values must be positive.")
        return (h, w)
    raise ValueError("image_size/hw must be a 2-element list or tuple.")


def _contains_image_files(folder: Path) -> bool:
    try:
        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in _IMAGE_EXTENSIONS:
                return True
    except OSError:
        return False
    return False


def _has_notmnist_layout(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    for letter in _NOTMNIST_LABEL_NAMES:
        class_dir = folder / letter
        if not class_dir.is_dir():
            return False
    return True


def _resolve_notmnist_dir(dataset_root: Path) -> Optional[Path]:
    candidates = (
        dataset_root / "notMNIST_small",
        dataset_root / "notMNIST",
        dataset_root / "notmnist_small",
        dataset_root / "notmnist",
    )
    for candidate in candidates:
        if _has_notmnist_layout(candidate):
            return candidate.resolve()
    return None


def _ensure_notmnist_available(dataset_root: Path) -> Path:
    existing = _resolve_notmnist_dir(dataset_root)
    if existing is not None:
        return existing
    last_error: Optional[Exception] = None
    for url in _NOTMNIST_DOWNLOAD_URLS:
        try:
            archive_name = url.split("/")[-1].split("?", 1)[0]
            download_and_extract_archive(
                url=url,
                download_root=str(dataset_root),
                extract_root=str(dataset_root),
                filename=archive_name,
                remove_finished=False,
            )
        except Exception as exc:
            last_error = exc
            continue
        extracted = _resolve_notmnist_dir(dataset_root)
        if extracted is not None:
            return extracted
    err_msg = "notMNIST A-J dataset was not found and automatic download failed."
    if last_error is not None:
        err_msg += f" Last error: {last_error}"
    raise FileNotFoundError(err_msg)


def _load_notmnist_aj(dataset_root: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    notmnist_dir = _ensure_notmnist_available(dataset_root)
    image_tensors: List[torch.Tensor] = []
    label_tensors: List[int] = []
    for class_idx, letter in enumerate(_NOTMNIST_LABEL_NAMES):
        class_dir = notmnist_dir / letter
        file_paths = sorted(path for path in class_dir.rglob("*") if path.is_file())
        for img_path in file_paths:
            if img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            try:
                img = read_image(str(img_path), mode=ImageReadMode.GRAY)  # (1,H,W), uint8
            except Exception:
                continue
            img_f = img.to(dtype=torch.float32)
            if img_f.shape[-2:] != (28, 28):
                img_f = F.interpolate(
                    img_f.unsqueeze(0),
                    size=(28, 28),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            img_u8 = img_f.clamp(0.0, 255.0).round().to(dtype=torch.uint8)
            image_tensors.append(img_u8.squeeze(0))
            label_tensors.append(int(class_idx))
    if not image_tensors:
        raise ValueError(f"No readable notMNIST images found under {notmnist_dir}.")
    images = torch.stack(image_tensors, dim=0)  # (N,28,28), uint8
    labels = torch.tensor(label_tensors, dtype=torch.long)
    return images, labels


def _resolve_vlm_images_dir(dataset_root: Path, images_dir: Optional[str]) -> Path:
    requested = ("" if images_dir is None else str(images_dir).strip())
    if requested and requested.lower() not in {"auto", "infer", "none", "null"}:
        images_path = Path(requested).expanduser()
        if not images_path.is_absolute():
            images_path = dataset_root / images_path
        images_path = images_path.resolve()
        if not images_path.is_dir():
            raise FileNotFoundError(f"images_dir does not exist: {images_path}")
        return images_path

    candidates: List[Path] = []
    if _contains_image_files(dataset_root):
        candidates.append(dataset_root)
    for child in sorted(dataset_root.iterdir()):
        if child.is_dir() and _contains_image_files(child):
            candidates.append(child.resolve())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No image folder found under dataset_root={dataset_root}. "
            "Set data.images_dir explicitly."
        )
    pretty = ", ".join(str(path) for path in candidates)
    raise ValueError(
        "Multiple candidate image folders found. Set data.images_dir explicitly. "
        f"Candidates: {pretty}"
    )


def _resolve_vlm_captions_file(dataset_root: Path, images_dir: Path, captions_file: Optional[str]) -> Path:
    candidates: List[Path] = []
    requested = ("" if captions_file is None else str(captions_file).strip())
    if requested:
        cap_path = Path(requested).expanduser()
        if cap_path.is_absolute():
            candidates.append(cap_path)
        else:
            candidates.append((dataset_root / cap_path).resolve())
            candidates.append((images_dir / cap_path).resolve())

    # Common local defaults used by Flickr-style exports.
    for name in ("caption.txt", "captions.txt"):
        candidates.append((dataset_root / name).resolve())
        candidates.append((images_dir / name).resolve())

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    pretty = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find captions file. Checked: "
        f"{pretty}"
    )


def _read_flickr_style_captions(captions_path: Path) -> Dict[str, List[str]]:
    """
    Read `image,caption` rows and group all captions per image filename.

    Supports repeated image names (multiple captions per image).
    """
    grouped: Dict[str, List[str]] = {}
    with captions_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            if row_idx == 0 and len(row) >= 2:
                c0 = row[0].strip().lower()
                c1 = row[1].strip().lower()
                if c0 == "image" and c1 == "caption":
                    continue
            image_name = row[0].strip()
            if not image_name:
                continue
            # Handle both "image.jpg" and "image.jpg#0" variants.
            if "#" in image_name:
                image_name = image_name.split("#", 1)[0].strip()
            image_name = Path(image_name).name
            caption = ",".join(row[1:]).strip() if len(row) > 1 else ""
            if not caption:
                continue
            grouped.setdefault(image_name, []).append(caption)
    if not grouped:
        raise ValueError(f"No valid caption rows found in {captions_path}.")
    return grouped


def _preprocess_caption_for_filtering(caption: str) -> List[str]:
    """Lowercase, remove punctuation/articles, and split by whitespace."""
    content = "" if caption is None else str(caption)
    content = unicodedata.normalize("NFKC", content)
    content = content.lower()
    punctuation_table = str.maketrans({ch: " " for ch in string.punctuation})
    content = content.translate(punctuation_table)
    tokens = [tok for tok in content.split() if tok and tok not in _CAPTION_FILTER_ARTICLES]
    return tokens


def _select_shortest_filtered_caption(
    captions: List[str],
    *,
    max_word: int,
) -> Optional[Tuple[int, str]]:
    """
    Keep shortest preprocessed caption with length <= max_word.

    Ties are resolved by preserving the first caption order.
    """
    best: Optional[Tuple[int, str, int]] = None
    for caption_idx, caption in enumerate(captions):
        tokens = _preprocess_caption_for_filtering(caption)
        token_len = len(tokens)
        if token_len == 0 or token_len > int(max_word):
            continue
        filtered_caption = " ".join(tokens)
        if best is None or token_len < best[2]:
            best = (caption_idx, filtered_caption, token_len)
    if best is None:
        return None
    return best[0], best[1]


def _build_shortest_filtered_caption_map(
    image_filenames: List[str],
    *,
    captions_by_image: Dict[str, List[str]],
    max_word: int,
) -> Dict[str, Tuple[int, str]]:
    selected_by_image: Dict[str, Tuple[int, str]] = {}
    for image_filename in image_filenames:
        chosen = _select_shortest_filtered_caption(
            captions_by_image.get(image_filename, []),
            max_word=int(max_word),
        )
        if chosen is not None:
            selected_by_image[image_filename] = chosen
    return selected_by_image


def _split_images_by_ratio(
    image_filenames: List[str],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    if val_ratio < 0.0 or val_ratio >= 1.0:
        raise ValueError("val_ratio must be in [0, 1).")
    ratio_sum = float(train_ratio) + float(val_ratio)
    if ratio_sum <= 0.0:
        raise ValueError("train_ratio + val_ratio must be positive.")
    if ratio_sum > 1.0 + 1e-12:
        raise ValueError("train_ratio + val_ratio must be <= 1.0 for train/val-only split.")
    if len(image_filenames) < 2:
        raise ValueError("Need at least 2 unique images for train/val split.")

    shuffled = list(image_filenames)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(shuffled)
    total = len(shuffled)

    # VLM captioning uses train/val only: no test split.
    # If train+val < 1, renormalize proportions to consume all images.
    train_share = float(train_ratio) / ratio_sum
    val_share = float(val_ratio) / ratio_sum

    train_count = max(1, int(round(total * train_share)))
    val_count = int(round(total * val_share))
    if val_share > 0.0:
        val_count = max(1, val_count)

    max_val = max(0, total - train_count)
    val_count = min(val_count, max_val)

    train_images = shuffled[:train_count]
    val_images = shuffled[train_count:]
    if not val_images and total >= 2:
        # Keep at least one validation sample when possible.
        val_images = train_images[-1:]
        train_images = train_images[:-1]
    test_images: List[str] = []
    return train_images, val_images, test_images


def _split_indices_by_ratio(
    total: int,
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if int(total) < 2:
        raise ValueError("Need at least 2 samples for train/val split.")
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    if val_ratio < 0.0 or val_ratio >= 1.0:
        raise ValueError("val_ratio must be in [0, 1).")
    ratio_sum = float(train_ratio) + float(val_ratio)
    if ratio_sum <= 0.0:
        raise ValueError("train_ratio + val_ratio must be positive.")
    if ratio_sum > 1.0 + 1e-12:
        raise ValueError("train_ratio + val_ratio must be <= 1.0 for train/val-only split.")

    train_share = float(train_ratio) / ratio_sum
    train_count = max(1, int(round(int(total) * train_share)))
    if train_count >= int(total):
        train_count = int(total) - 1

    indices = np.arange(int(total), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    train_idx = indices[:train_count]
    val_idx = indices[train_count:]
    if val_idx.size == 0:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    return train_idx, val_idx


def _expand_vlm_samples(
    image_filenames: List[str],
    *,
    selected_caption_by_image: Dict[str, Tuple[int, str]],
    images_dir: Path,
    image_index_map: Dict[str, int],
) -> List[VLMSample]:
    samples: List[VLMSample] = []
    for image_filename in image_filenames:
        chosen = selected_caption_by_image.get(image_filename)
        if chosen is None:
            continue
        image_path = (images_dir / image_filename).resolve()
        caption_idx, caption = chosen
        samples.append(
            VLMSample(
                sample_index=len(samples),
                image_index=image_index_map[image_filename],
                caption_index=int(caption_idx),
                image_filename=image_filename,
                image_path=image_path,
                caption=caption,
            )
        )
    return samples


def _build_vlm_image_transform(image_size: Tuple[int, int], *, color_mode: str = "rgb"):
    mode = str(color_mode).strip().lower()
    if mode not in {"rgb", "grayscale", "gray"}:
        raise ValueError("color_mode must be one of {'rgb', 'grayscale'}.")
    tfms: List[object] = [transforms.Resize(image_size)]
    if mode in {"grayscale", "gray"}:
        tfms.insert(0, transforms.Grayscale(num_output_channels=1))
    tfms.append(transforms.ToTensor())
    return transforms.Compose(tfms)


class _VLMImageIndexDataset(Dataset):
    """Loads image tensors and returns (image, sample_index)."""

    def __init__(
        self,
        samples: List[VLMSample],
        image_size: Tuple[int, int],
        *,
        color_mode: str = "rgb",
    ) -> None:
        self.samples = samples
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.color_mode = str(color_mode).strip().lower()
        self.transform = _build_vlm_image_transform(image_size, color_mode=self.color_mode)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        if Image is not None:
            with Image.open(sample.image_path) as img:
                img = img.convert("RGB")
                image_tensor = self.transform(img)
        else:
            # PIL-free path: read tensor directly and apply equivalent preprocessing.
            image_tensor = read_image(str(sample.image_path), mode=ImageReadMode.RGB).to(dtype=torch.float32) / 255.0
            if self.color_mode in {"grayscale", "gray"} and image_tensor.size(0) != 1:
                image_tensor = image_tensor.mean(dim=0, keepdim=True)
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return image_tensor, idx


class VLMCaptionDataset(Dataset):
    """
    VLM dataset with optical image preprocessing plus caption tokenization.

    The image branch reuses PatchifiedDataset so `method='upsampler'` follows the
    exact same path used by the existing ONN pipeline.
    """

    def __init__(
        self,
        *,
        samples: List[VLMSample],
        tokenizer: Any,
        max_caption_length: int,
        image_size: Tuple[int, int],
        color_mode: str = "rgb",
        method: str = "upsampler",
        num_channels: int = 16,
        use_rot4_tiling: bool = False,
        overlap_percent: float = 0.0,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_caption_length = int(max_caption_length)
        if self.max_caption_length <= 0:
            raise ValueError("max_caption_length must be positive.")
        image_dataset = _VLMImageIndexDataset(
            samples=samples,
            image_size=image_size,
            color_mode=color_mode,
        )
        self.image_pipeline = PatchifiedDataset(
            image_dataset,
            num_channels=num_channels,
            hw=image_size,
            use_rot4_tiling=use_rot4_tiling,
            method=method,
            overlap_percent=overlap_percent,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_tensor, sample_idx = self.image_pipeline[idx]
        sample = self.samples[int(sample_idx)]
        tokenized = self.tokenizer(
            sample.caption,
            max_length=self.max_caption_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        caption_ids = tokenized["input_ids"].squeeze(0).to(dtype=torch.long)
        attention_mask = tokenized["attention_mask"].squeeze(0).to(dtype=torch.long)
        return {
            "image": image_tensor,
            "caption": sample.caption,
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "image_filename": sample.image_filename,
            "image_index": sample.image_index,
            "sample_index": sample.sample_index,
            "caption_index": sample.caption_index,
        }


def _collate_vlm_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "caption": [item["caption"] for item in batch],
        "caption_ids": torch.stack([item["caption_ids"] for item in batch], dim=0),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
        "image_filename": [item["image_filename"] for item in batch],
        "image_index": torch.tensor([item["image_index"] for item in batch], dtype=torch.long),
        "sample_index": torch.tensor([item["sample_index"] for item in batch], dtype=torch.long),
        "caption_index": torch.tensor([item["caption_index"] for item in batch], dtype=torch.long),
    }


class SimpleCaptionTokenizer:
    """
    Lightweight caption tokenizer for offline/local-only runs.

    Default mode (`mode_style='kaggle'`) mimics a TextVectorization-style pipeline:
    lowercase -> standardize -> whitespace tokenize -> fixed vocab ids.
    Legacy mode (`mode_style='legacy'`) preserves prior regex tokenization behavior.
    """

    def __init__(
        self,
        token_to_id: Dict[str, int],
        *,
        pad_token: str = "[PAD]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
        unk_token: str = "[UNK]",
        lowercase: bool = True,
        punctuation_mode: str = "keep",
        mode_style: str = "kaggle",
    ) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {idx: tok for tok, idx in token_to_id.items()}
        self.lowercase = bool(lowercase)
        self.punctuation_mode = str(punctuation_mode).strip().lower()
        if self.punctuation_mode not in {"keep", "remove", "drop"}:
            raise ValueError("punctuation_mode must be one of {'keep', 'remove', 'drop'}.")
        self.mode_style = str(mode_style).strip().lower()
        if self.mode_style not in {"kaggle", "legacy"}:
            raise ValueError("mode_style must be one of {'kaggle', 'legacy'}.")
        self.pad_token = str(pad_token)
        self.bos_token = str(bos_token)
        self.eos_token = str(eos_token)
        self.unk_token = str(unk_token)
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.all_special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.special_token_ids = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
        }

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _ensure_runtime_compat(self) -> None:
        """
        Backward-compat guard for objects created before newer attributes existed.
        """
        if not hasattr(self, "mode_style"):
            self.mode_style = "kaggle"
        if not hasattr(self, "all_special_tokens"):
            self.all_special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        if not hasattr(self, "special_token_ids"):
            self.special_token_ids = {
                self.pad_token: self.pad_token_id,
                self.unk_token: self.unk_token_id,
                self.bos_token: self.bos_token_id,
                self.eos_token: self.eos_token_id,
            }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._ensure_runtime_compat()

    @staticmethod
    def _normalize_unicode_quotes(text: str) -> str:
        quote_map = str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201a": "'",
                "\u201b": "'",
                "\u2032": "'",
                "\u0060": "'",
                "\u00b4": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u201e": '"',
                "\u201f": '"',
                "\u2033": '"',
                "\u00ab": '"',
                "\u00bb": '"',
            }
        )
        return text.translate(quote_map)

    @staticmethod
    def _protect_special_tokens(text: str, special_tokens: List[str]) -> Tuple[str, Dict[str, str]]:
        protected = text
        placeholder_to_token: Dict[str, str] = {}
        # Replace longer tokens first to avoid accidental partial replacements.
        unique_tokens = sorted({str(tok) for tok in special_tokens if str(tok)}, key=len, reverse=True)
        for idx, token in enumerate(unique_tokens):
            placeholder = f"specialtokenplaceholder{idx}"
            protected = protected.replace(token, f" {placeholder} ")
            placeholder_to_token[placeholder] = token
        return protected, placeholder_to_token

    @staticmethod
    def _restore_special_tokens(text: str, placeholder_to_token: Dict[str, str]) -> str:
        restored = text
        for placeholder, token in placeholder_to_token.items():
            restored = re.sub(rf"\b{re.escape(placeholder)}\b", token, restored)
        restored = re.sub(r"\s+", " ", restored).strip()
        return restored

    @classmethod
    def _standardize_kaggle_text(cls, text: str, *, lowercase: bool, special_tokens: List[str]) -> str:
        content = "" if text is None else str(text)
        content = unicodedata.normalize("NFKC", content)
        content = cls._normalize_unicode_quotes(content)
        if lowercase:
            content = content.lower()
        content, placeholders = cls._protect_special_tokens(content, special_tokens)

        # Remove punctuation by replacing each punctuation char with whitespace.
        punctuation_table = str.maketrans({ch: " " for ch in string.punctuation})
        content = content.translate(punctuation_table)
        content = re.sub(r"\s+", " ", content).strip()
        content = cls._restore_special_tokens(content, placeholders)
        return content

    @classmethod
    def _standardize_legacy_text(cls, text: str, *, lowercase: bool) -> str:
        content = "" if text is None else str(text)
        content = unicodedata.normalize("NFKC", content)
        content = cls._normalize_unicode_quotes(content)
        if lowercase:
            content = content.lower()
        return content

    def standardize_text(self, text: str) -> str:
        self._ensure_runtime_compat()
        if self.mode_style == "kaggle":
            return self._standardize_kaggle_text(
                text,
                lowercase=self.lowercase,
                special_tokens=self.all_special_tokens,
            )
        return self._standardize_legacy_text(text, lowercase=self.lowercase)

    @staticmethod
    def _tokenize_legacy_text(
        text: str,
        *,
        lowercase: bool,
        punctuation_mode: str = "keep",
    ) -> List[str]:
        content = SimpleCaptionTokenizer._standardize_legacy_text(text, lowercase=lowercase)
        if punctuation_mode in {"remove", "drop"}:
            return re.findall(r"[a-z0-9']+", content)
        # Keep punctuation as standalone tokens.
        return re.findall(r"[a-z0-9']+|[^\w\s]", content)

    def tokenize(self, text: str) -> List[str]:
        self._ensure_runtime_compat()
        if self.mode_style == "kaggle":
            standardized = self.standardize_text(text)
            return standardized.split() if standardized else []
        return self._tokenize_legacy_text(
            text,
            lowercase=self.lowercase,
            punctuation_mode=self.punctuation_mode,
        )

    @classmethod
    def from_captions(
        cls,
        captions: List[str],
        *,
        min_token_freq: int = 1,
        min_freq: Optional[int] = None,
        lowercase: bool = True,
        punctuation_mode: str = "keep",
        mode_style: str = "kaggle",
        start_token: str = "<start>",
        end_token: str = "<end>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        max_vocab_size: Optional[int] = None,
        print_diagnostics: bool = True,
    ) -> "SimpleCaptionTokenizer":
        if min_freq is not None:
            min_token_freq = int(min_freq)
        if int(min_token_freq) <= 0:
            raise ValueError("min_token_freq must be positive.")
        special_tokens = [str(pad_token), str(unk_token), str(start_token), str(end_token)]
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("Special tokens must be distinct.")
        tokenizer_probe = cls(
            {tok: idx for idx, tok in enumerate(special_tokens)},
            pad_token=str(pad_token),
            bos_token=str(start_token),
            eos_token=str(end_token),
            unk_token=str(unk_token),
            lowercase=bool(lowercase),
            punctuation_mode=str(punctuation_mode),
            mode_style=str(mode_style),
        )
        counter: Counter[str] = Counter()
        lengths_before: List[int] = []
        lengths_after: List[int] = []
        for caption in captions:
            tokens = tokenizer_probe.tokenize(caption)
            counter.update(tokens)
            lengths_before.append(len(tokens))
            lengths_after.append(len(tokens) + 2)  # + <start>, <end>

        base_tokens = [str(pad_token), str(unk_token), str(start_token), str(end_token)]
        vocab_tokens: List[str] = []
        for token, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq >= int(min_token_freq) and token not in base_tokens:
                vocab_tokens.append(token)
        if max_vocab_size is not None:
            max_vocab_size = int(max_vocab_size)
            if max_vocab_size < len(base_tokens):
                raise ValueError(
                    f"max_vocab_size ({max_vocab_size}) must be >= number of special tokens ({len(base_tokens)})."
                )
            keep_tokens = max_vocab_size - len(base_tokens)
            vocab_tokens = vocab_tokens[:keep_tokens]

        ordered_tokens = base_tokens + vocab_tokens
        token_to_id = {token: idx for idx, token in enumerate(ordered_tokens)}
        tokenizer = cls(
            token_to_id,
            pad_token=str(pad_token),
            bos_token=str(start_token),
            eos_token=str(end_token),
            unk_token=str(unk_token),
            lowercase=lowercase,
            punctuation_mode=punctuation_mode,
            mode_style=mode_style,
        )
        if print_diagnostics:
            avg_len_before = float(np.mean(lengths_before)) if lengths_before else 0.0
            avg_len_after = float(np.mean(lengths_after)) if lengths_after else 0.0
            top20 = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
            top20_str = ", ".join([f"{token}:{freq}" for token, freq in top20]) if top20 else "N/A"
            print(
                "[vlm][simple-tokenizer] "
                f"mode={tokenizer.mode_style} vocab_size={len(tokenizer)} "
                f"min_token_freq={int(min_token_freq)} max_vocab_size={max_vocab_size}"
            )
            print(f"[vlm][simple-tokenizer] top20={top20_str}")
            print(
                "[vlm][simple-tokenizer] special_token_ids="
                f"{{'{tokenizer.pad_token}': {tokenizer.pad_token_id}, "
                f"'{tokenizer.unk_token}': {tokenizer.unk_token_id}, "
                f"'{tokenizer.bos_token}': {tokenizer.bos_token_id}, "
                f"'{tokenizer.eos_token}': {tokenizer.eos_token_id}}}"
            )
            print(
                "[vlm][simple-tokenizer] "
                f"avg_len_tokens={avg_len_before:.2f} avg_len_with_bounds={avg_len_after:.2f}"
            )
        return tokenizer

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> List[int]:
        self._ensure_runtime_compat()
        ids = [self.bos_token_id]
        for token in self.tokenize(text):
            ids.append(self.token_to_id.get(token, self.unk_token_id))
        ids.append(self.eos_token_id)
        return ids

    def decode(
        self,
        token_ids: List[int],
        *,
        skip_special_tokens: bool = True,
        stop_at_eos: bool = False,
        skip_pad: bool = True,
    ) -> str:
        self._ensure_runtime_compat()
        pieces: List[str] = []
        special_id_set = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
            self.unk_token_id,
        }
        for token_id in token_ids:
            idx = int(token_id)
            if stop_at_eos and idx == self.eos_token_id:
                break
            if skip_pad and idx == self.pad_token_id:
                continue
            token = self.id_to_token.get(idx, self.unk_token)
            if skip_special_tokens and idx in special_id_set:
                continue
            pieces.append(token)
        text = " ".join(pieces)
        if self.mode_style == "legacy" and self.punctuation_mode not in {"remove", "drop"}:
            text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text.strip()

    def debug_caption(
        self,
        raw_caption: str,
        *,
        max_length: Optional[int] = None,
        print_output: bool = True,
    ) -> Dict[str, Any]:
        self._ensure_runtime_compat()
        standardized = self.standardize_text(raw_caption)
        tokens = self.tokenize(raw_caption)
        token_ids = self.encode(raw_caption)
        padded_ids: Optional[List[int]] = None
        padded_mask: Optional[List[int]] = None
        if max_length is not None:
            packed = self(
                raw_caption,
                max_length=int(max_length),
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_tensors=None,
            )
            padded_ids = [int(x) for x in packed["input_ids"]]
            padded_mask = [int(x) for x in packed["attention_mask"]]
        info = {
            "raw_caption": str(raw_caption),
            "standardized_caption": standardized,
            "tokens": tokens,
            "token_ids": token_ids,
            "token_ids_padded": padded_ids,
            "attention_mask": padded_mask,
        }
        if print_output:
            print("[vlm][simple-tokenizer][debug]")
            print(f"  raw         : {info['raw_caption']}")
            print(f"  standardized: {info['standardized_caption']}")
            print(f"  tokens      : {info['tokens']}")
            print(f"  token_ids   : {info['token_ids']}")
            if padded_ids is not None:
                print(f"  padded_ids  : {padded_ids}")
                print(f"  attn_mask   : {padded_mask}")
        return info

    def __call__(
        self,
        text: str,
        *,
        max_length: int,
        truncation: bool = True,
        padding: str = "max_length",
        return_attention_mask: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, torch.Tensor | List[int]]:
        self._ensure_runtime_compat()
        if int(max_length) <= 0:
            raise ValueError("max_length must be positive.")
        ids = self.encode(text)
        if truncation and len(ids) > int(max_length):
            max_len = int(max_length)
            if max_len == 1:
                ids = [self.bos_token_id]
            else:
                max_inner = max(0, max_len - 2)
                ids = [self.bos_token_id] + ids[1:-1][:max_inner] + [self.eos_token_id]
        if padding == "max_length":
            pad_len = max(0, int(max_length) - len(ids))
            ids = ids + [self.pad_token_id] * pad_len
        if return_attention_mask:
            attention = [0 if token_id == self.pad_token_id else 1 for token_id in ids]
        else:
            attention = []

        if return_tensors == "pt":
            output: Dict[str, torch.Tensor | List[int]] = {
                "input_ids": torch.tensor([ids], dtype=torch.long),
            }
            if return_attention_mask:
                output["attention_mask"] = torch.tensor([attention], dtype=torch.long)
            return output
        output = {"input_ids": ids}
        if return_attention_mask:
            output["attention_mask"] = attention
        return output


def debug_simple_tokenizer_examples(
    tokenizer: SimpleCaptionTokenizer,
    captions: List[str],
    *,
    max_length: int = 32,
) -> List[Dict[str, Any]]:
    debug_rows: List[Dict[str, Any]] = []
    for caption in captions:
        debug_rows.append(
            tokenizer.debug_caption(
                caption,
                max_length=max_length,
                print_output=True,
            )
        )
    return debug_rows


def run_simple_tokenizer_quick_tests() -> None:
    """
    Lightweight sanity checks for tokenizer behavior in kaggle/legacy modes.
    """
    train_caps = [
        "A Dog, RUNS!!!",
        "Two dogs run through shallow water in a bay .",
        "A black dog swimming .",
        "THE cat sits on 2 mats.",
    ]
    tok = SimpleCaptionTokenizer.from_captions(
        train_caps,
        mode_style="kaggle",
        lowercase=True,
        min_token_freq=1,
        max_vocab_size=None,
        print_diagnostics=False,
    )

    # punctuation removal + lowercase
    dbg = tok.debug_caption("A DOG, RUNS!!!", print_output=False)
    assert dbg["standardized_caption"] == "a dog runs", "Kaggle standardization should remove punctuation and lowercase."
    assert dbg["tokens"] == ["a", "dog", "runs"], "Kaggle tokenization should be whitespace-based after standardization."

    # OOV mapping
    unknown_ids = tok.encode("thiswordisnotinvocab")
    assert tok.unk_token_id in unknown_ids, "Unknown tokens must map to <unk>."

    # deterministic vocab ordering with lexical tie-break
    tok_tie = SimpleCaptionTokenizer.from_captions(
        ["banana apple", "apple banana", "cherry"],
        mode_style="kaggle",
        lowercase=True,
        min_token_freq=1,
        print_diagnostics=False,
    )
    assert tok_tie.token_to_id["apple"] < tok_tie.token_to_id["banana"], (
        "Tokens with equal frequency should use lexical tie-break."
    )

    # start/end insertion
    ids = tok.encode("black dog")
    assert ids[0] == tok.bos_token_id and ids[-1] == tok.eos_token_id, "Encoded ids must include <start>/<end>."

    # padding/truncation while preserving <start> and <end> where possible
    packed = tok(
        "one two three four five six",
        max_length=5,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors=None,
    )
    packed_ids = packed["input_ids"]
    assert len(packed_ids) == 5, "Truncated sequence must match max_length."
    assert packed_ids[0] == tok.bos_token_id, "Truncated sequence must start with <start>."
    assert packed_ids[-1] == tok.eos_token_id, "Truncated sequence should end with <end>."

    # decoding option to stop at eos
    decoded = tok.decode(
        [tok.bos_token_id, tok.token_to_id.get("a", tok.unk_token_id), tok.eos_token_id, tok.token_to_id.get("dog", tok.unk_token_id)],
        skip_special_tokens=True,
        stop_at_eos=True,
    )
    assert "dog" not in decoded.split(), "Decoding with stop_at_eos=True should stop before tokens after <end>."


def _load_tokenizer(
    tokenizer_name: str,
    *,
    local_files_only: bool = False,
    fallback_captions: Optional[List[str]] = None,
    simple_min_token_freq: int = 1,
    simple_max_vocab_size: Optional[int] = None,
    simple_lowercase: bool = True,
    simple_mode_style: str = "kaggle",
    simple_print_diagnostics: bool = True,
    start_token: str = "<start>",
    end_token: str = "<end>",
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
    punctuation_mode: str = "keep",
):
    name = str(tokenizer_name).strip()
    use_simple = name.lower() in {"simple", "basic", "whitespace"}
    if use_simple:
        if not fallback_captions:
            raise ValueError("fallback_captions are required for SimpleCaptionTokenizer.")
        print("[vlm] using SimpleCaptionTokenizer (requested by tokenizer_name).")
        return SimpleCaptionTokenizer.from_captions(
            fallback_captions,
            min_token_freq=int(simple_min_token_freq),
            lowercase=bool(simple_lowercase),
            punctuation_mode=str(punctuation_mode),
            mode_style=str(simple_mode_style),
            start_token=str(start_token),
            end_token=str(end_token),
            pad_token=str(pad_token),
            unk_token=str(unk_token),
            max_vocab_size=(None if simple_max_vocab_size is None else int(simple_max_vocab_size)),
            print_diagnostics=bool(simple_print_diagnostics),
        )

    if AutoTokenizer is None:
        if not fallback_captions:
            raise ImportError(
                "transformers is not installed and no fallback captions are available "
                "for SimpleCaptionTokenizer."
            )
        print("[vlm] transformers not found; falling back to SimpleCaptionTokenizer.")
        return SimpleCaptionTokenizer.from_captions(
            fallback_captions,
            min_token_freq=int(simple_min_token_freq),
            lowercase=bool(simple_lowercase),
            punctuation_mode=str(punctuation_mode),
            mode_style=str(simple_mode_style),
            start_token=str(start_token),
            end_token=str(end_token),
            pad_token=str(pad_token),
            unk_token=str(unk_token),
            max_vocab_size=(None if simple_max_vocab_size is None else int(simple_max_vocab_size)),
            print_diagnostics=bool(simple_print_diagnostics),
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            use_fast=True,
            local_files_only=bool(local_files_only),
        )
    except Exception as exc:
        if not fallback_captions:
            raise
        print(f"[vlm] tokenizer load failed ({exc}); falling back to SimpleCaptionTokenizer.")
        return SimpleCaptionTokenizer.from_captions(
            fallback_captions,
            min_token_freq=int(simple_min_token_freq),
            lowercase=bool(simple_lowercase),
            punctuation_mode=str(punctuation_mode),
            mode_style=str(simple_mode_style),
            start_token=str(start_token),
            end_token=str(end_token),
            pad_token=str(pad_token),
            unk_token=str(unk_token),
            max_vocab_size=(None if simple_max_vocab_size is None else int(simple_max_vocab_size)),
            print_diagnostics=bool(simple_print_diagnostics),
        )

    special_update: Dict[str, str] = {}
    if start_token:
        special_update["bos_token"] = str(start_token)
    if end_token:
        special_update["eos_token"] = str(end_token)
    if pad_token:
        special_update["pad_token"] = str(pad_token)
    if unk_token:
        special_update["unk_token"] = str(unk_token)
    if special_update:
        tokenizer.add_special_tokens(special_update)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def build_vlm_caption_loaders(
    *,
    dataset_name: str = "flickr8k",
    dataset_root: str,
    images_dir: Optional[str] = None,
    captions_file: str = "caption.txt",
    train_ratio: float = 0.95,
    val_ratio: float = 0.05,
    split_seed: int = 1337,
    tokenizer_name: str = "bert-base-uncased",
    tokenizer_local_files_only: bool = False,
    simple_mode_style: str = "kaggle",
    simple_tokenizer_min_freq: int = 1,
    simple_min_token_freq: Optional[int] = None,
    simple_max_vocab_size: Optional[int] = None,
    simple_tokenizer_lowercase: bool = True,
    simple_print_diagnostics: bool = True,
    start_token: str = "<start>",
    end_token: str = "<end>",
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
    punctuation_mode: str = "keep",
    max_word: int = 8,
    max_caption_length: int = 32,
    image_size: Tuple[int, int] = (112, 112),
    image_color_mode: str = "rgb",
    batch_size: int = 8,
    num_workers: int = 2,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    shuffle_test: bool = False,
    method: str = "upsampler",
    num_channels: int = 16,
    use_rot4_tiling: bool = False,
    overlap_percent: float = 0.0,
    synthetic_train_samples: Optional[int] = None,
    synthetic_val_samples: Optional[int] = None,
    synthetic_side_gap: int = 2,
    pin_memory: bool = True,
    drop_last_train: bool = False,
) -> Dict[str, Any]:
    """
    Build VLM caption loaders with image-based splits and ONN-compatible images.

    Returns:
        {
            "train_loader": DataLoader,
            "val_loader": DataLoader,
            "test_loader": Optional[DataLoader],
            "tokenizer": tokenizer,
            "split_info": dict,
            "dataset_name": str,
            "paths": dict,
        }
    """
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    dataset_key = str(dataset_name).strip().lower()
    synthetic_dataset_keys = {
        "mnist_fashion",
        "mnist-fashion",
        "mnist_fashion_caption",
        "mnist_fashion_captioning",
        "mnist_fashion_synthetic",
        "mnist_fashion_notmnist",
        "mnist-fashion-notmnist",
        "mnist_fashion_notmnist_caption",
        "mnist_fashion_notmnist_synthetic",
    }
    if not dataset_root_path.is_dir():
        if dataset_key in synthetic_dataset_keys:
            dataset_root_path.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"dataset_root does not exist: {dataset_root_path}")

    image_size_hw = _parse_hw(image_size, default=(112, 112))

    if dataset_key in synthetic_dataset_keys:
        requested_method = str(method).strip().lower()
        if requested_method != "patchify":
            print(
                "[vlm] synthetic mnist+fashion+notmnist dataset forces method='patchify' "
                "for 4x4 channel mapping."
            )
        if image_size_hw[0] % 4 != 0 or image_size_hw[1] % 4 != 0:
            raise ValueError(
                "For mnist+fashion+notmnist synthetic patchify mode, image_size/hw must be divisible by 4 "
                f"(got {image_size_hw})."
            )
        if image_size_hw[0] != image_size_hw[1]:
            raise ValueError(
                "For triangular 3-slot synthetic composition, image_size/hw must be square "
                f"(got {image_size_hw})."
            )

        mnist_train = datasets.MNIST(root=str(dataset_root_path), train=True, download=True)
        mnist_test = datasets.MNIST(root=str(dataset_root_path), train=False, download=True)
        fashion_train = datasets.FashionMNIST(root=str(dataset_root_path), train=True, download=True)
        fashion_test = datasets.FashionMNIST(root=str(dataset_root_path), train=False, download=True)
        notmnist_all_images, notmnist_all_labels = _load_notmnist_aj(dataset_root_path)

        mnist_all_images = torch.cat(
            (
                torch.as_tensor(mnist_train.data),
                torch.as_tensor(mnist_test.data),
            ),
            dim=0,
        )
        fashion_all_images = torch.cat(
            (
                torch.as_tensor(fashion_train.data),
                torch.as_tensor(fashion_test.data),
            ),
            dim=0,
        )
        mnist_all_labels = torch.cat(
            (
                torch.as_tensor(mnist_train.targets, dtype=torch.long),
                torch.as_tensor(mnist_test.targets, dtype=torch.long),
            ),
            dim=0,
        )
        fashion_all_labels = torch.cat(
            (
                torch.as_tensor(fashion_train.targets, dtype=torch.long),
                torch.as_tensor(fashion_test.targets, dtype=torch.long),
            ),
            dim=0,
        )

        mnist_train_idx_np, mnist_val_idx_np = _split_indices_by_ratio(
            int(mnist_all_labels.numel()),
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            seed=int(split_seed),
        )
        fashion_train_idx_np, fashion_val_idx_np = _split_indices_by_ratio(
            int(fashion_all_labels.numel()),
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            seed=int(split_seed) + 1,
        )
        notmnist_train_idx_np, notmnist_val_idx_np = _split_indices_by_ratio(
            int(notmnist_all_labels.numel()),
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            seed=int(split_seed) + 2,
        )
        mnist_train_idx = torch.from_numpy(mnist_train_idx_np).to(dtype=torch.long)
        mnist_val_idx = torch.from_numpy(mnist_val_idx_np).to(dtype=torch.long)
        fashion_train_idx = torch.from_numpy(fashion_train_idx_np).to(dtype=torch.long)
        fashion_val_idx = torch.from_numpy(fashion_val_idx_np).to(dtype=torch.long)
        notmnist_train_idx = torch.from_numpy(notmnist_train_idx_np).to(dtype=torch.long)
        notmnist_val_idx = torch.from_numpy(notmnist_val_idx_np).to(dtype=torch.long)

        mnist_train_images = mnist_all_images.index_select(0, mnist_train_idx)
        mnist_val_images = mnist_all_images.index_select(0, mnist_val_idx)
        fashion_train_images = fashion_all_images.index_select(0, fashion_train_idx)
        fashion_val_images = fashion_all_images.index_select(0, fashion_val_idx)
        notmnist_train_images = notmnist_all_images.index_select(0, notmnist_train_idx)
        notmnist_val_images = notmnist_all_images.index_select(0, notmnist_val_idx)
        mnist_train_labels = mnist_all_labels.index_select(0, mnist_train_idx)
        mnist_val_labels = mnist_all_labels.index_select(0, mnist_val_idx)
        fashion_train_labels = fashion_all_labels.index_select(0, fashion_train_idx)
        fashion_val_labels = fashion_all_labels.index_select(0, fashion_val_idx)
        notmnist_train_labels = notmnist_all_labels.index_select(0, notmnist_train_idx)
        notmnist_val_labels = notmnist_all_labels.index_select(0, notmnist_val_idx)

        default_train_samples = min(
            int(mnist_train_labels.numel()),
            int(fashion_train_labels.numel()),
            int(notmnist_train_labels.numel()),
        )
        default_val_samples = min(
            int(mnist_val_labels.numel()),
            int(fashion_val_labels.numel()),
            int(notmnist_val_labels.numel()),
        )
        train_sample_count = (
            int(default_train_samples)
            if synthetic_train_samples is None
            else int(synthetic_train_samples)
        )
        val_sample_count = (
            int(default_val_samples)
            if synthetic_val_samples is None
            else int(synthetic_val_samples)
        )
        if train_sample_count <= 0 or val_sample_count <= 0:
            raise ValueError("synthetic_train_samples and synthetic_val_samples must be positive.")

        train_samples = _build_tri_family_samples(
            mnist_labels=mnist_train_labels,
            fashion_labels=fashion_train_labels,
            notmnist_labels=notmnist_train_labels,
            num_samples=train_sample_count,
            seed=int(split_seed),
            split_name="train",
        )
        val_samples = _build_tri_family_samples(
            mnist_labels=mnist_val_labels,
            fashion_labels=fashion_val_labels,
            notmnist_labels=notmnist_val_labels,
            num_samples=val_sample_count,
            seed=int(split_seed) + 1,
            split_name="val",
        )

        train_captions_for_vocab = [sample.caption for sample in train_samples]
        if not train_captions_for_vocab:
            raise ValueError("No synthetic training captions available to build tokenizer vocabulary.")
        effective_min_token_freq = (
            int(simple_min_token_freq)
            if simple_min_token_freq is not None
            else int(simple_tokenizer_min_freq)
        )
        if simple_max_vocab_size is not None:
            simple_max_vocab_size = int(simple_max_vocab_size)

        tokenizer = _load_tokenizer(
            tokenizer_name=str(tokenizer_name),
            local_files_only=bool(tokenizer_local_files_only),
            fallback_captions=train_captions_for_vocab,
            simple_min_token_freq=effective_min_token_freq,
            simple_max_vocab_size=simple_max_vocab_size,
            simple_lowercase=bool(simple_tokenizer_lowercase),
            simple_mode_style=str(simple_mode_style),
            simple_print_diagnostics=bool(simple_print_diagnostics),
            start_token=str(start_token),
            end_token=str(end_token),
            pad_token=str(pad_token),
            unk_token=str(unk_token),
            punctuation_mode=str(punctuation_mode),
        )
        tokenizer_info = {
            "vocab_size": int(getattr(tokenizer, "vocab_size", len(tokenizer))),
            "bos_token": getattr(tokenizer, "bos_token", start_token),
            "eos_token": getattr(tokenizer, "eos_token", end_token),
            "pad_token": getattr(tokenizer, "pad_token", pad_token),
            "unk_token": getattr(tokenizer, "unk_token", unk_token),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        }
        if tokenizer_info["bos_token_id"] is None and hasattr(tokenizer, "token_to_id"):
            tokenizer_info["bos_token_id"] = tokenizer.token_to_id.get(tokenizer_info["bos_token"])
        if tokenizer_info["eos_token_id"] is None and hasattr(tokenizer, "token_to_id"):
            tokenizer_info["eos_token_id"] = tokenizer.token_to_id.get(tokenizer_info["eos_token"])
        if tokenizer_info["pad_token_id"] is None and hasattr(tokenizer, "token_to_id"):
            tokenizer_info["pad_token_id"] = tokenizer.token_to_id.get(tokenizer_info["pad_token"])
        if tokenizer_info["unk_token_id"] is None and hasattr(tokenizer, "token_to_id"):
            tokenizer_info["unk_token_id"] = tokenizer.token_to_id.get(tokenizer_info["unk_token"])

        train_ds = SyntheticTriFamilyCaptionDataset(
            samples=train_samples,
            tokenizer=tokenizer,
            max_caption_length=int(max_caption_length),
            mnist_images=mnist_train_images,
            fashion_images=fashion_train_images,
            notmnist_images=notmnist_train_images,
            image_size=image_size_hw,
            num_channels=int(num_channels),
            overlap_percent=float(overlap_percent),
            slot_gap=int(synthetic_side_gap),
        )
        val_ds = SyntheticTriFamilyCaptionDataset(
            samples=val_samples,
            tokenizer=tokenizer,
            max_caption_length=int(max_caption_length),
            mnist_images=mnist_val_images,
            fashion_images=fashion_val_images,
            notmnist_images=notmnist_val_images,
            image_size=image_size_hw,
            num_channels=int(num_channels),
            overlap_percent=float(overlap_percent),
            slot_gap=int(synthetic_side_gap),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=int(batch_size),
            shuffle=bool(shuffle_train),
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            drop_last=bool(drop_last_train),
            collate_fn=_collate_vlm_batch,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(batch_size),
            shuffle=bool(shuffle_val),
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            drop_last=False,
            collate_fn=_collate_vlm_batch,
        )

        split_info = {
            "num_unique_images": len(train_samples) + len(val_samples),
            "train_images_before_filter": len(train_samples),
            "val_images_before_filter": len(val_samples),
            "test_images_before_filter": 0,
            "train_images": len(train_samples),
            "val_images": len(val_samples),
            "test_images": 0,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": 0,
        }
        print(
            "[vlm] synthetic mnist+fashion+notmnist dataset ready: "
            f"samples(train/val)={split_info['train_samples']}/{split_info['val_samples']} "
            f"split(train_ratio/val_ratio)={float(train_ratio):.6f}/{float(val_ratio):.6f} "
            f"image_size={image_size_hw} num_channels={int(num_channels)} method=patchify"
        )
        print(f"[vlm] tokenizer vocab_size(filtered-train)={tokenizer_info['vocab_size']}")
        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": None,
            "tokenizer": tokenizer,
            "tokenizer_info": tokenizer_info,
            "split_info": split_info,
            "dataset_name": str(dataset_name),
            "paths": {
                "dataset_root": str(dataset_root_path),
                "images_dir": "synthetic_mnist_fashion_notmnist_triangular",
                "captions_file": "generated_in_memory",
            },
        }
    images_path = _resolve_vlm_images_dir(dataset_root_path, images_dir)
    captions_path = _resolve_vlm_captions_file(dataset_root_path, images_path, captions_file)
    captions_by_image = _read_flickr_style_captions(captions_path)

    available_images: List[str] = []
    missing_images = 0
    for image_name in sorted(captions_by_image.keys()):
        image_path = (images_path / image_name).resolve()
        if image_path.is_file():
            available_images.append(image_name)
        else:
            missing_images += 1
    if not available_images:
        raise FileNotFoundError(
            "No caption-linked images found. "
            f"Checked image folder: {images_path}"
        )
    if missing_images > 0:
        print(
            f"[vlm] warning: {missing_images} caption entries reference missing images under {images_path}."
        )

    train_images, val_images, test_images = _split_images_by_ratio(
        available_images,
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        seed=int(split_seed),
    )
    effective_max_word = int(max_word)
    if effective_max_word <= 0:
        raise ValueError("max_word must be positive.")

    train_selected = _build_shortest_filtered_caption_map(
        train_images,
        captions_by_image=captions_by_image,
        max_word=effective_max_word,
    )
    val_selected = _build_shortest_filtered_caption_map(
        val_images,
        captions_by_image=captions_by_image,
        max_word=effective_max_word,
    )
    test_selected = _build_shortest_filtered_caption_map(
        test_images,
        captions_by_image=captions_by_image,
        max_word=effective_max_word,
    )

    train_images_kept = [name for name in train_images if name in train_selected]
    val_images_kept = [name for name in val_images if name in val_selected]
    test_images_kept = [name for name in test_images if name in test_selected]
    image_index_map = {name: idx for idx, name in enumerate(sorted(available_images))}

    train_samples = _expand_vlm_samples(
        train_images,
        selected_caption_by_image=train_selected,
        images_dir=images_path,
        image_index_map=image_index_map,
    )
    val_samples = _expand_vlm_samples(
        val_images,
        selected_caption_by_image=val_selected,
        images_dir=images_path,
        image_index_map=image_index_map,
    )
    test_samples = _expand_vlm_samples(
        test_images,
        selected_caption_by_image=test_selected,
        images_dir=images_path,
        image_index_map=image_index_map,
    )

    train_captions_for_vocab = [sample.caption for sample in train_samples]
    if not train_captions_for_vocab:
        raise ValueError("No training captions available to build tokenizer vocabulary.")
    effective_min_token_freq = (
        int(simple_min_token_freq)
        if simple_min_token_freq is not None
        else int(simple_tokenizer_min_freq)
    )
    if simple_max_vocab_size is not None:
        simple_max_vocab_size = int(simple_max_vocab_size)

    tokenizer = _load_tokenizer(
        tokenizer_name=str(tokenizer_name),
        local_files_only=bool(tokenizer_local_files_only),
        fallback_captions=train_captions_for_vocab,
        simple_min_token_freq=effective_min_token_freq,
        simple_max_vocab_size=simple_max_vocab_size,
        simple_lowercase=bool(simple_tokenizer_lowercase),
        simple_mode_style=str(simple_mode_style),
        simple_print_diagnostics=bool(simple_print_diagnostics),
        start_token=str(start_token),
        end_token=str(end_token),
        pad_token=str(pad_token),
        unk_token=str(unk_token),
        punctuation_mode=str(punctuation_mode),
    )

    tokenizer_info = {
        "vocab_size": int(getattr(tokenizer, "vocab_size", len(tokenizer))),
        "bos_token": getattr(tokenizer, "bos_token", start_token),
        "eos_token": getattr(tokenizer, "eos_token", end_token),
        "pad_token": getattr(tokenizer, "pad_token", pad_token),
        "unk_token": getattr(tokenizer, "unk_token", unk_token),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
    }
    if tokenizer_info["bos_token_id"] is None and hasattr(tokenizer, "token_to_id"):
        tokenizer_info["bos_token_id"] = tokenizer.token_to_id.get(tokenizer_info["bos_token"])
    if tokenizer_info["eos_token_id"] is None and hasattr(tokenizer, "token_to_id"):
        tokenizer_info["eos_token_id"] = tokenizer.token_to_id.get(tokenizer_info["eos_token"])
    if tokenizer_info["pad_token_id"] is None and hasattr(tokenizer, "token_to_id"):
        tokenizer_info["pad_token_id"] = tokenizer.token_to_id.get(tokenizer_info["pad_token"])
    if tokenizer_info["unk_token_id"] is None and hasattr(tokenizer, "token_to_id"):
        tokenizer_info["unk_token_id"] = tokenizer.token_to_id.get(tokenizer_info["unk_token"])

    common_kwargs = dict(
        tokenizer=tokenizer,
        max_caption_length=int(max_caption_length),
        image_size=image_size_hw,
        color_mode=str(image_color_mode),
        method=str(method).strip().lower(),
        num_channels=int(num_channels),
        use_rot4_tiling=bool(use_rot4_tiling),
        overlap_percent=float(overlap_percent),
    )
    train_ds = VLMCaptionDataset(samples=train_samples, **common_kwargs)
    val_ds = VLMCaptionDataset(samples=val_samples, **common_kwargs)
    test_ds = VLMCaptionDataset(samples=test_samples, **common_kwargs) if test_samples else None

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle_train),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last_train),
        collate_fn=_collate_vlm_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle_val),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
        collate_fn=_collate_vlm_batch,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=int(batch_size),
            shuffle=bool(shuffle_test),
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            drop_last=False,
            collate_fn=_collate_vlm_batch,
        )

    split_info = {
        "num_unique_images": len(available_images),
        "train_images_before_filter": len(train_images),
        "val_images_before_filter": len(val_images),
        "test_images_before_filter": len(test_images),
        "train_images": len(train_images_kept),
        "val_images": len(val_images_kept),
        "test_images": len(test_images_kept),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
    }
    if len({sample.image_filename for sample in train_samples}) != len(train_samples):
        raise RuntimeError("Train split must contain exactly one filtered caption per image.")
    if len({sample.image_filename for sample in val_samples}) != len(val_samples):
        raise RuntimeError("Validation split must contain exactly one filtered caption per image.")
    if len({sample.image_filename for sample in test_samples}) != len(test_samples):
        raise RuntimeError("Test split must contain exactly one filtered caption per image.")
    print(
        "[vlm] filtered images kept(train/val/test)="
        f"{split_info['train_images']}/{split_info['val_images']}/{split_info['test_images']} "
        "from split identities "
        f"{split_info['train_images_before_filter']}/{split_info['val_images_before_filter']}/{split_info['test_images_before_filter']}"
    )
    print(
        "[vlm] filtered captions(train/val/test)="
        f"{split_info['train_samples']}/{split_info['val_samples']}/{split_info['test_samples']}"
    )
    print(f"[vlm] tokenizer vocab_size(filtered-train)={tokenizer_info['vocab_size']}")
    print(
        "[vlm] dataset ready: "
        f"images(train/val/test)={split_info['train_images']}/{split_info['val_images']}/{split_info['test_images']} "
        f"samples(train/val/test)={split_info['train_samples']}/{split_info['val_samples']}/{split_info['test_samples']}"
    )
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "tokenizer": tokenizer,
        "tokenizer_info": tokenizer_info,
        "split_info": split_info,
        "dataset_name": str(dataset_name),
        "paths": {
            "dataset_root": str(dataset_root_path),
            "images_dir": str(images_path),
            "captions_file": str(captions_path),
        },
    }


def build_vlm_caption_loaders_from_cfg(data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    image_size = _parse_hw(
        data_cfg.get("image_size", data_cfg.get("hw", (112, 112))),
        default=(112, 112),
    )
    max_vocab_value = data_cfg.get("simple_max_vocab_size", None)
    if isinstance(max_vocab_value, str) and max_vocab_value.strip().lower() in {"none", "null", "auto", ""}:
        max_vocab_value = None
    min_freq_value = data_cfg.get("simple_min_token_freq", data_cfg.get("simple_tokenizer_min_freq", 1))
    max_word_value = data_cfg.get("max_word", data_cfg.get("Max_word", 8))
    if isinstance(max_word_value, str):
        max_word_value = int(max_word_value.strip())
    return build_vlm_caption_loaders(
        dataset_name=str(data_cfg.get("dataset_name", data_cfg.get("dataset", "flickr8k"))),
        dataset_root=str(data_cfg.get("dataset_root", data_cfg.get("data_root", "./Data_VLM"))),
        images_dir=data_cfg.get("images_dir"),
        captions_file=str(data_cfg.get("captions_file", "caption.txt")),
        train_ratio=float(data_cfg.get("train_ratio", 0.95)),
        val_ratio=float(data_cfg.get("val_ratio", 0.05)),
        split_seed=int(data_cfg.get("split_seed", data_cfg.get("seed", 1337))),
        tokenizer_name=str(data_cfg.get("tokenizer_name", "bert-base-uncased")),
        tokenizer_local_files_only=bool(data_cfg.get("tokenizer_local_files_only", False)),
        simple_mode_style=str(data_cfg.get("simple_mode_style", "kaggle")),
        simple_tokenizer_min_freq=int(data_cfg.get("simple_tokenizer_min_freq", 1)),
        simple_min_token_freq=int(min_freq_value),
        simple_max_vocab_size=(None if max_vocab_value is None else int(max_vocab_value)),
        simple_tokenizer_lowercase=bool(data_cfg.get("simple_tokenizer_lowercase", True)),
        simple_print_diagnostics=bool(data_cfg.get("simple_print_diagnostics", True)),
        start_token=str(data_cfg.get("start_token", "<start>")),
        end_token=str(data_cfg.get("end_token", "<end>")),
        pad_token=str(data_cfg.get("pad_token", "<pad>")),
        unk_token=str(data_cfg.get("unk_token", "<unk>")),
        punctuation_mode=str(data_cfg.get("punctuation_mode", "keep")),
        max_word=int(max_word_value),
        max_caption_length=int(data_cfg.get("max_caption_length", 32)),
        image_size=image_size,
        image_color_mode=str(data_cfg.get("image_color_mode", "rgb")),
        batch_size=int(data_cfg.get("batch_size", 8)),
        num_workers=int(data_cfg.get("num_workers", 2)),
        shuffle_train=bool(data_cfg.get("shuffle_train", True)),
        shuffle_val=bool(data_cfg.get("shuffle_val", False)),
        shuffle_test=bool(data_cfg.get("shuffle_test", False)),
        method=str(data_cfg.get("method", "upsampler")),
        num_channels=int(data_cfg.get("num_channels", 16)),
        use_rot4_tiling=bool(data_cfg.get("use_rot4_tiling", False)),
        overlap_percent=float(data_cfg.get("overlap_percent", 0.0)),
        synthetic_train_samples=(
            None
            if data_cfg.get("synthetic_train_samples", None) is None
            else int(data_cfg.get("synthetic_train_samples"))
        ),
        synthetic_val_samples=(
            None
            if data_cfg.get("synthetic_val_samples", None) is None
            else int(data_cfg.get("synthetic_val_samples"))
        ),
        synthetic_side_gap=int(data_cfg.get("synthetic_side_gap", 2)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last_train=bool(data_cfg.get("drop_last_train", False)),
    )


def _resolve_dataset_name(name: str) -> str:
    key = str(name).strip().lower()
    if key in {"fashion_mnist", "fashion-mnist", "fmnist"}:
        return "fashion_mnist"
    if key in {"mnist"}:
        return "mnist"
    if key in {"cifar10", "cifar-10", "cifar"}:
        return "cifar10"
    if key in {"facial", "facial_keypoints", "facial-keypoints", "facial_keypoint_detection"}:
        return "facial"
    raise ValueError(f"Unsupported dataset '{name}'.")


def _build_transforms(dataset_name: str, augment: bool, train: bool):
    tfms = []
    if dataset_name == "cifar10":
        tfms.append(transforms.Grayscale(num_output_channels=1))
    if augment and train:
        tfms.append(
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=4,
            )
        )
    tfms.append(transforms.ToTensor())
    if augment and train:
        tfms.append(
            transforms.RandomErasing(
                p=0.15,
                scale=(0.01, 0.10),
                ratio=(0.1, 2.0),
                value=0.0,
                inplace=True,
            )
        )
    return transforms.Compose(tfms)


def _build_base_dataset(dataset_name: str, train: bool, augment: bool, data_root: str):
    name = _resolve_dataset_name(dataset_name)
    tfms = _build_transforms(name, augment=augment, train=train)
    if name == "fashion_mnist":
        return datasets.FashionMNIST(root=data_root, train=train, download=True, transform=tfms)
    if name == "mnist":
        return datasets.MNIST(root=data_root, train=train, download=True, transform=tfms)
    if name == "cifar10":
        return datasets.CIFAR10(root=data_root, train=train, download=True, transform=tfms)
    raise ValueError(f"Unsupported dataset '{dataset_name}'.")


def _resolve_facial_zip_path(data_root: str) -> Path:
    root = Path(data_root).expanduser()
    if root.is_file() and root.suffix.lower() == ".zip":
        return root
    candidate = root / "training.zip"
    if candidate.exists():
        return candidate
    if _FACIAL_DEFAULT_ZIP.exists():
        return _FACIAL_DEFAULT_ZIP
    raise FileNotFoundError(
        "Facial dataset zip not found. Expected training.zip at "
        f"{candidate} or {_FACIAL_DEFAULT_ZIP}."
    )


def _ensure_facial_csv(zip_path: Path) -> Path:
    cache_dir = zip_path.parent / "cache_training"
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "training.csv"
    if not csv_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extract("training.csv", path=cache_dir)
    return csv_path


def _build_facial_datasets(
    *,
    data_root: str,
    eval_split: float = 0.10,
    seed: int = 1337,
) -> Tuple[TensorDataset, TensorDataset]:
    global _FACIAL_SHAPES_LOGGED
    if pd is None:
        raise ImportError("pandas is required for Facial dataset loading.")

    zip_path = _resolve_facial_zip_path(data_root)
    csv_path = _ensure_facial_csv(zip_path)
    df = pd.read_csv(csv_path)
    df = df.dropna().reset_index(drop=True)

    image_series = df["Image"].map(lambda s: np.fromstring(s, sep=" ", dtype=np.float32))
    pixel_count = 96 * 96
    lengths = image_series.map(len).to_numpy()
    if not np.all(lengths == pixel_count):
        bad_count = int(np.sum(lengths != pixel_count))
        raise ValueError(
            f"Facial dataset contains {bad_count} rows with invalid pixel length "
            f"(expected {pixel_count})."
        )
    x_np = np.stack(image_series.to_numpy(), axis=0).astype(np.float32, copy=False)
    x_np = (x_np / 255.0).reshape(-1, 96, 96, 1)
    x_np = np.transpose(x_np, (0, 3, 1, 2)).astype(np.float32, copy=False)

    y_cols = [col for col in df.columns if col != "Image"]
    y_np = df[y_cols].to_numpy(dtype=np.float32, copy=True)
    y_np = (y_np / 96.0).astype(np.float32, copy=False)

    total = x_np.shape[0]
    eval_len = max(1, int(round(total * float(eval_split))))
    if eval_len >= total:
        eval_len = max(1, total - 1)
    rng = np.random.default_rng(seed)
    indices = np.arange(total, dtype=np.int64)
    rng.shuffle(indices)
    eval_idx = indices[:eval_len]
    train_idx = indices[eval_len:]

    x_train = torch.from_numpy(x_np[train_idx].astype(np.float32, copy=False))
    y_train = torch.from_numpy(y_np[train_idx].astype(np.float32, copy=False))
    x_eval = torch.from_numpy(x_np[eval_idx].astype(np.float32, copy=False))
    y_eval = torch.from_numpy(y_np[eval_idx].astype(np.float32, copy=False))

    if not _FACIAL_SHAPES_LOGGED:
        print(
            "[facial] dataset ready: "
            f"X_train={tuple(x_train.shape)}, y_train={tuple(y_train.shape)}, "
            f"X_eval={tuple(x_eval.shape)}, y_eval={tuple(y_eval.shape)}"
        )
        _FACIAL_SHAPES_LOGGED = True

    return TensorDataset(x_train, y_train), TensorDataset(x_eval, y_eval)


def build_patch_loaders(
    dataset_name: str,
    batch_size: int = 1,
    num_channels: int = 16,
    hw: Tuple[int, int] = (112, 112),
    use_rot4_tiling: bool = True,
    method: str = "patchify",
    overlap_percent: float = 0.0,
    augment: bool = False,
    data_root: str = "./data",
    num_workers: int = 2,
    pin_memory: bool = True,
    val_split: float = 0.03,
    drop_last: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val loaders returning patchified or upsampled tensors."""
    resolved_name = _resolve_dataset_name(dataset_name)
    if resolved_name == "facial":
        base_train, base_val = _build_facial_datasets(
            data_root=data_root,
            eval_split=0.10,
            seed=1337,
        )
        train_ds = PatchifiedDataset(
            base_train,
            num_channels=num_channels,
            hw=hw,
            use_rot4_tiling=use_rot4_tiling,
            method=method,
            overlap_percent=overlap_percent,
        )
        val_ds = PatchifiedDataset(
            base_val,
            num_channels=num_channels,
            hw=hw,
            use_rot4_tiling=use_rot4_tiling,
            method=method,
            overlap_percent=overlap_percent,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        return train_loader, val_loader

    base_train = _build_base_dataset(dataset_name, train=True, augment=augment, data_root=data_root)
    val_len = int(len(base_train) * val_split)
    train_len = len(base_train) - val_len
    train_subset, val_subset = random_split(base_train, [train_len, val_len])

    # Override val transform to avoid train-time augs.
    if isinstance(val_subset, Subset):
        val_subset.dataset.transform = _build_transforms(
            _resolve_dataset_name(dataset_name),
            augment=False,
            train=False,
        )

    train_ds = PatchifiedDataset(
        train_subset,
        num_channels=num_channels,
        hw=hw,
        use_rot4_tiling=use_rot4_tiling,
        method=method,
        overlap_percent=overlap_percent,
    )
    val_ds = PatchifiedDataset(
        val_subset,
        num_channels=num_channels,
        hw=hw,
        use_rot4_tiling=use_rot4_tiling,
        method=method,
        overlap_percent=overlap_percent,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return train_loader, val_loader
