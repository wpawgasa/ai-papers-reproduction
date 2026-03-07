"""Oxford-IIIT Pet dataset loading with elastic deformation augmentation.

Wraps torchvision's OxfordIIITPet dataset for binary segmentation
(foreground vs background). Implements elastic deformation augmentation
as described in Section 3 of the U-Net paper.
"""

import random

import numpy as np
import scipy.ndimage
import torch
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def elastic_deformation(
    image: np.ndarray,
    mask: np.ndarray,
    sigma: float = 10.0,
    grid_spacing: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random elastic deformation to image and mask jointly.

    Generates smooth displacement fields from random vectors on a coarse grid,
    interpolated via bicubic splines (Section 3.1 of the paper).

    Args:
        image: Input image of shape (H, W, C) or (H, W).
        mask: Segmentation mask of shape (H, W).
        sigma: Standard deviation of random displacement vectors (in pixels).
        grid_spacing: Number of control points along each axis for the coarse grid.

    Returns:
        Tuple of (deformed_image, deformed_mask).
    """
    h, w = image.shape[:2]

    # Create coarse random displacement grid
    grid_h = np.linspace(0, h - 1, grid_spacing)
    grid_w = np.linspace(0, w - 1, grid_spacing)

    dx_coarse = np.random.randn(grid_spacing, grid_spacing) * sigma
    dy_coarse = np.random.randn(grid_spacing, grid_spacing) * sigma

    # Interpolate to full resolution
    row_coords = np.arange(h)
    col_coords = np.arange(w)

    spline_dx = RectBivariateSpline(grid_h, grid_w, dx_coarse, kx=min(3, grid_spacing - 1), ky=min(3, grid_spacing - 1))
    spline_dy = RectBivariateSpline(grid_h, grid_w, dy_coarse, kx=min(3, grid_spacing - 1), ky=min(3, grid_spacing - 1))

    dx = spline_dx(row_coords, col_coords)
    dy = spline_dy(row_coords, col_coords)

    # Compute displaced coordinates
    row_grid, col_grid = np.meshgrid(row_coords, col_coords, indexing="ij")
    map_row = np.clip(row_grid + dy, 0, h - 1).astype(np.float32)
    map_col = np.clip(col_grid + dx, 0, w - 1).astype(np.float32)

    # Convert to integer indices for nearest-neighbor sampling (for mask)
    map_row_int = np.round(map_row).astype(np.int32)
    map_col_int = np.round(map_col).astype(np.int32)

    # Apply deformation
    # Use interpolated sampling for the image to preserve smooth elastic deformation
    if image.ndim == 3:
        deformed_image = np.empty_like(image)
        for c in range(image.shape[2]):
            deformed_image[..., c] = scipy.ndimage.map_coordinates(
                image[..., c],
                [map_row, map_col],
                order=1,
                mode="reflect",
            )
    else:
        deformed_image = scipy.ndimage.map_coordinates(
            image,
            [map_row, map_col],
            order=1,
            mode="reflect",
        )

    # Keep nearest-neighbor sampling for the mask to preserve discrete labels
    deformed_mask = mask[map_row_int, map_col_int]

    return deformed_image, deformed_mask


class PetSegmentationDataset(Dataset):
    """Oxford-IIIT Pet dataset for binary segmentation.

    Wraps torchvision's OxfordIIITPet and converts the trimap to a binary
    foreground/background mask. Applies optional augmentations including
    elastic deformation.

    Args:
        root: Root directory for dataset download.
        split: "trainval" or "test".
        input_size: Target image size (resized to input_size x input_size).
        augmentation_config: AugmentationConfig for controlling transforms.
        subset_size: If set, randomly subsample to this many images.
    """

    def __init__(
        self,
        root: str = "./data",
        split: str = "trainval",
        input_size: int = 256,
        augmentation_config=None,
        subset_size: int | None = None,
    ):
        self.input_size = input_size
        self.aug_config = augmentation_config
        self.is_train = split == "trainval"

        # Download Oxford-IIIT Pet dataset
        self.dataset = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=True,
        )

        # Handle subsetting
        self.indices = list(range(len(self.dataset)))
        if subset_size is not None and subset_size < len(self.indices):
            rng = random.Random(42)
            self.indices = sorted(rng.sample(self.indices, subset_size))

        # Normalization (ImageNet stats)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        image, trimap = self.dataset[real_idx]

        # Convert PIL to numpy
        image = np.array(image.convert("RGB").resize((self.input_size, self.input_size), Image.BILINEAR))
        trimap = np.array(trimap.resize((self.input_size, self.input_size), Image.NEAREST))

        # Convert trimap to binary mask: 1=foreground, 2=background, 3=boundary
        # Map: 1 -> foreground (1), 2 -> background (0), 3 -> background (0)
        mask = (trimap == 1).astype(np.int64)

        # Apply augmentation (training only)
        if self.is_train and self.aug_config is not None:
            image, mask = self._apply_augmentation(image, mask)

        # Convert to tensors
        image = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        image = self.normalize(image)
        mask = torch.from_numpy(mask.copy()).long()

        return image, mask

    def _apply_augmentation(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation pipeline."""
        # Elastic deformation
        if self.aug_config.elastic_deformation:
            if random.random() > 0.5:
                image, mask = elastic_deformation(
                    image,
                    mask,
                    sigma=self.aug_config.elastic_sigma,
                    grid_spacing=self.aug_config.elastic_grid_spacing,
                )

        # Random horizontal flip
        if self.aug_config.flip:
            if random.random() > 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)

        # Random rotation
        if self.aug_config.rotation:
            if random.random() > 0.5:
                angle = random.uniform(
                    -self.aug_config.max_rotation_degrees,
                    self.aug_config.max_rotation_degrees,
                )
                image = scipy.ndimage.rotate(image, angle, reshape=False, order=1, mode="nearest")
                mask = scipy.ndimage.rotate(mask, angle, reshape=False, order=0, mode="nearest")

        # Gray value variation (random brightness/contrast)
        if self.aug_config.gray_value_variation:
            if random.random() > 0.5:
                image = image.astype(np.float32)
                image += np.random.normal(0, self.aug_config.gray_value_std * 255, image.shape)
                image = np.clip(image, 0, 255).astype(np.uint8)

        return image, mask


_NUMPY_SEED_MAX = 2**32  # NumPy/random seeds must be in [0, 2**32)


def _worker_init_fn(_worker_id: int) -> None:
    """Seed per-worker RNGs so augmentation is reproducible in multi-worker loads."""
    seed = torch.initial_seed() % _NUMPY_SEED_MAX
    random.seed(seed)
    np.random.seed(seed)


def get_dataloaders(config) -> dict[str, DataLoader]:
    """Create train and test DataLoaders from experiment config.

    Args:
        config: ExperimentConfig instance.

    Returns:
        Dict with 'train' and 'test' DataLoader entries.
    """
    train_dataset = PetSegmentationDataset(
        root=config.data.data_dir,
        split="trainval",
        input_size=config.data.input_size,
        augmentation_config=config.augmentation,
        subset_size=config.data.train_subset_size,
    )

    test_dataset = PetSegmentationDataset(
        root=config.data.data_dir,
        split="test",
        input_size=config.data.input_size,
        augmentation_config=None,  # No augmentation for test
        subset_size=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        worker_init_fn=_worker_init_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    return {"train": train_loader, "test": test_loader}
