# U-Net: Convolutional Networks for Biomedical Image Segmentation

Reproduction of the U-Net paper by Ronneberger, Fischer, and Brox (MICCAI 2015).

**Paper**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## Key Ideas

- **Symmetric encoder-decoder with skip connections**: Combines deep semantic features with fine spatial information for precise pixel-level segmentation
- **Elastic deformation augmentation**: Enables effective training from very few annotated images by simulating realistic tissue deformations
- **Overlap-tile strategy**: Allows seamless segmentation of arbitrarily large images

## Experiment

We demonstrate that elastic deformation augmentation is critical for data-efficient segmentation using the Oxford-IIIT Pet dataset (binary foreground/background segmentation):

1. **Full dataset + augmentation** vs **without augmentation**
2. **Small-data regime (30 images)** with vs without augmentation

The key finding: augmentation provides the largest benefit when training data is limited.

## Architecture

```
Input (3, 256, 256)
    |
Encoder:  64 → 128 → 256 → 512 → 1024 channels
    |                                    |
Decoder: 1024 → 512 → 256 → 128 → 64 channels (with skip connections)
    |
Output (2, 256, 256)  [foreground/background]
```

23 convolutional layers + 1×1 final convolution. ~31M parameters.

## Setup

```bash
cd U-net
pip install -e .
```

## Quick Start

```bash
jupyter notebook notebooks/experiment.ipynb
```

For quick testing (5 epochs, 30 images, 128×128):
```python
from unet_segmentation import ExperimentConfig
config = ExperimentConfig.quick_test()
```

## Expected Results

| Experiment | Augmentation | Train Size | Mean IOU | Pixel Acc |
|-----------|-------------|-----------|----------|-----------|
| Full + augment | Elastic + flip + rotate | ~3.7K | >0.85 | >0.90 |
| Full - augment | None | ~3.7K | >0.80 | >0.88 |
| Small + augment | Elastic + flip + rotate | 30 | >0.70 | >0.80 |
| Small - augment | None | 30 | <0.60 | <0.75 |

## Configuration

All settings are controlled via dataclass configs:

```python
from unet_segmentation import ExperimentConfig

# Full training
config = ExperimentConfig.default()

# Quick testing
config = ExperimentConfig.quick_test()

# Custom
config = ExperimentConfig.default(
    epochs=100,
    batch_size=8,
    train_subset_size=50,
    elastic_deformation=True,
)
```

## Project Structure

```
U-net/
├── src/unet_segmentation/
│   ├── config.py              # Dataclass configurations
│   ├── models/unet.py         # U-Net architecture (23 conv layers)
│   ├── data/oxford_pets.py    # Dataset + elastic deformation
│   ├── training/trainer.py    # Training loop
│   ├── evaluation/metrics.py  # IOU, Dice, pixel accuracy
│   └── utils/
│       ├── seed.py            # Reproducibility
│       └── loss.py            # Weighted cross-entropy loss
├── notebooks/experiment.ipynb # Main experiment
├── summary.md                 # Deep paper analysis
└── pyproject.toml             # Dependencies
```

## Citation

```bibtex
@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={MICCAI},
  pages={234--241},
  year={2015},
  publisher={Springer}
}
```
