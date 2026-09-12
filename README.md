# 🐱 Cat Facial Landmark Detection

A lightweight PyTorch implementation of 48-point cat facial landmark regression, built on an EfficientNet-B0 backbone.

<sub>PyTorch · torchvision · EfficientNet-B0 (~5.3M params) · 48 landmarks</sub>

## Why

Reading feline emotion means reading small things: ear angle, whisker-pad tension, eye aperture, mouth curvature. **CatFLW** (*Cat Facial Landmarks in the Wild*) makes that measurable — 2,016 in-the-wild cat faces, each with a bounding box and 48 landmarks chosen to align with cat facial musculature and CatFACS action units [[1]](#references).

Hand-placing 48 points is expensive (~4.2 min/image, ~140 hours total). The CatFLW authors cut that cost with a human-in-the-loop loop: train an EfficientNet-based regressor on what's annotated so far, predict the next batch, and let annotators nudge rather than place. Between the first two batches, average annotation time per image dropped **35%** [[1]](#references).

This repo reimplements that predictor as a clean, modular PyTorch model.

## Architecture

```text
Image [B, 3, 224, 224]
        │
        ▼
  EfficientNet-B0            pretrained on ImageNet-1k
  (MBConv + compound scaling)  →  pooled embedding [B, 1280]
        │
        ▼
  Regression head
    Dropout(p=0.2)           regularization
    Linear(1280 → 96)        48 landmarks × (x, y)
    Sigmoid()                coordinates bounded to [0, 1]
        │
        ▼
  Landmarks [B, 96] → view(-1, 48, 2)
```

**Backbone** — EfficientNet-B0 [[2]](#references) balances depth, width, and resolution under a fixed compute budget, giving rich spatial features at a small footprint. ImageNet weights (`EfficientNet_B0_Weights.DEFAULT`) transfer well to faces with little data.

**Head** — the 1,000-class classifier is swapped for a 96-unit regressor. The `Sigmoid` is the load-bearing choice: it normalizes predictions to the crop's relative coordinate space, which stabilizes training and rules out unbounded coordinate drift.

```python
self.backbone.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=96),
    nn.Sigmoid(),
)
```

## Quickstart

```bash
pip install torch torchvision matplotlib pillow
python models/cat_landmark_model.py   # forward-pass sanity check
```

```python
import torch
from models.cat_landmark_model import CatLandmarkModel

model = CatLandmarkModel(pretrained=True).eval()

with torch.no_grad():
    coords = model(torch.randn(2, 3, 224, 224))   # [2, 96]
    landmarks = coords.view(-1, 48, 2)            # [2, 48, 2]
```

Inspect the dataset — draws landmarks in red, bounding boxes in cyan:

```python
from utils import show_random_cat_labels
show_random_cat_labels(15)
```

## Layout

```text
models/cat_landmark_model.py   CatLandmarkModel + verification routine
utils.py                       CatFLW visualization helpers
orchestrator.ipynb             experimentation notebook
CatFLW dataset/                images/ + labels/  (gitignored)
```

**Label format** — one JSON per image: `labels` is 48 `[x, y]` pairs, `bounding_boxes` is `[left, top, right, bottom]`, both in pixels.

## References

1. G. Martvel, N. Farhat, I. Shimshoni, A. Zamansky. *CatFLW: Cat Facial Landmarks in the Wild Dataset.* arXiv:2305.04232, 2023. <https://arxiv.org/abs/2305.04232>
2. M. Tan, Q. V. Le. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* arXiv:1905.11946, 2019. <https://arxiv.org/abs/1905.11946>

CatFLW is released under CC BY 4.0 and available from its authors on request. This repository is an independent reimplementation, not affiliated with the original authors.
