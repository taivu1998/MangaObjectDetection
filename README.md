# MangaNet: Object Detection for Manga with Deep Neural Networks

A comprehensive study of modern object detection architectures adapted for the manga domain, achieving a **state-of-the-art mAP of 71.0** on the Manga109 benchmark. This project systematically evaluates three detection paradigms — two-stage (Faster R-CNN), single-stage anchor-based (RetinaNet), and single-stage grid-based (YOLOv3) — on the task of localizing characters, faces, panels, and text regions in Japanese manga pages.

**[[Paper]](https://taivu1998.github.io/assets/files/projects/MangaNet_Paper.pdf)**

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture Overview](#architecture-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Technical Details](#technical-details)
- [References](#references)

---

## Motivation

Manga, as a visual medium, presents unique challenges for object detection systems compared to natural images:

- **Extreme scale variation**: Objects range from full-page panel frames to small text bubbles and tiny facial features, often spanning 2-3 orders of magnitude in pixel area within a single page.
- **Dense, overlapping layouts**: Manga pages contain tightly packed, heavily overlapping bounding boxes across semantically distinct categories (e.g., a character body enclosing a face, both inside a panel frame with overlaid text).
- **Domain gap from natural images**: Manga consists of hand-drawn line art with flat shading, halftone patterns, and screentones — a distribution fundamentally different from the photographic images on which most pretrained backbones (ImageNet, COCO) are trained.
- **High intra-class variance**: Artistic styles vary drastically across manga titles, making generalization across the 109 volumes in Manga109 non-trivial.

Prior work by [Ogawa et al. (2018)](https://arxiv.org/abs/1803.08670) introduced SSD300-fork to handle overlapping objects in comics. This project advances beyond single-architecture studies by conducting a rigorous, controlled comparison of three fundamentally different detection paradigms on the same benchmark under identical data splits and evaluation protocols.

---

## Architecture Overview

### 1. Faster R-CNN (Two-Stage Detector)

```
Input Image (variable resolution)
       │
       ▼
┌──────────────────────────┐
│  ResNet-50 Backbone      │
│  (ImageNet pretrained)   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Feature Pyramid Network │   Multi-scale feature maps
│  (P2, P3, P4, P5)       │   256 channels per level
└──────────┬───────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌────────────────┐
│   RPN   │ │   RoI Align    │
│ Anchors:│ │   (7×7 pool)   │
│ 15/loc  │ └───────┬────────┘
└────┬────┘         │
     │              ▼
     │    ┌──────────────────┐
     └───►│  FC Layers       │
          │  1024 → 1024     │
          └──────┬───────────┘
                 │
          ┌──────┴──────┐
          ▼             ▼
     ┌─────────┐  ┌──────────┐
     │ Cls Head│  │ Reg Head │
     │ 5 cls   │  │ 4 coords │
     └─────────┘  └──────────┘
```

- **Backbone**: ResNet-50 + FPN (Feature Pyramid Network) with ImageNet-pretrained weights
- **Region Proposal Network**: 15 anchors per spatial location (5 scales × 3 aspect ratios)
- **Anchor scales**: {32, 64, 128, 256, 512} pixels
- **Anchor aspect ratios**: {0.5, 1.0, 2.0}
- **RoI head**: `FastRCNNPredictor` with 5-class output (4 object classes + background)
- **Loss**: Multi-task loss combining cross-entropy classification + Smooth L1 box regression

### 2. RetinaNet (Single-Stage with Focal Loss)

```
Input Image (variable resolution)
       │
       ▼
┌──────────────────────────┐
│  ResNet-50 Backbone      │
│  (ImageNet pretrained)   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Feature Pyramid Network │   Multi-scale feature maps
│  (P3, P4, P5, P6, P7)   │   Strides: 8, 16, 32, 64, 128
└──────────┬───────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌──────────┐ ┌──────────┐
│ Cls Sub- │ │ Box Sub- │
│ network  │ │ network  │
│ 4×Conv   │ │ 4×Conv   │
│ (256,3×3)│ │ (256,3×3)│
└────┬─────┘ └────┬─────┘
     ▼            ▼
  K×A scores   4×A coords
  per location per location
```

- **Key innovation**: Focal Loss `FL(p_t) = -α_t(1 - p_t)^γ log(p_t)` with γ=2.0 and α=0.25
- **Purpose**: Addresses the extreme foreground-background class imbalance (~100k anchors, <1% positive) that causes standard cross-entropy to be dominated by easy negatives
- **FPN levels**: P3–P7 with anchor scales {2^0, 2^(1/3), 2^(2/3)} per level
- **Classification subnet**: 4 convolutional layers (256 channels, 3×3 kernels, ReLU) → K×A outputs
- **Box regression subnet**: 4 convolutional layers (256 channels, 3×3 kernels, ReLU) → 4×A outputs

### 3. YOLOv3 (Single-Stage Grid-Based)

```
Input Image (416 × 416)
       │
       ▼
┌──────────────────────────┐
│  Darknet-53 Backbone     │
│  53 conv layers          │
│  Residual connections    │
│  (ImageNet pretrained)   │
└──────────┬───────────────┘
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
  ┌─────┐┌─────┐┌─────┐
  │13×13││26×26││52×52│   3 detection scales
  │Large││ Med ││Small│   3 anchors per scale
  └──┬──┘└──┬──┘└──┬──┘
     └───┬───┘─────┘
         ▼
  (bx, by, bw, bh, obj, cls) × 9 anchors
```

- **Backbone**: Darknet-53 with 53 convolutional layers and skip connections, pretrained on ImageNet
- **Multi-scale detection**: Predictions at 3 grid resolutions (13×13, 26×26, 52×52) for detecting objects at different scales
- **Per-cell predictions**: Each grid cell predicts 3 bounding boxes, each with (x, y, w, h, objectness, 4 class probabilities)
- **Loss**: MSE for box coordinates + Binary cross-entropy for objectness and class predictions
- **Inference**: Confidence threshold 0.5, NMS threshold 0.4

---

## Dataset

### Manga109 Benchmark

[Manga109](http://www.manga109.org/en/) is a large-scale academic dataset of Japanese manga compiled by Aizawa Yamasaki Matsui Laboratory at the University of Tokyo.

| Property | Value |
|---|---|
| **Total volumes** | 109 manga titles |
| **Total pages** | 21,142 |
| **Total annotations** | >500,000 bounding boxes |
| **Object classes** | 4: `body`, `face`, `frame`, `text` |
| **Image resolution** | ~1654 × 1170 px (typical page) |
| **Annotation format** | XML with per-page bounding boxes |

### Data Split

| Split | Proportion | Purpose |
|---|---|---|
| Train | 80% | Model optimization |
| Validation | 10% | Hyperparameter tuning & early stopping |
| Test | 10% | Final evaluation |

- Splits are performed using stratified random sampling (`random_state=0`) via scikit-learn's `train_test_split`
- **Condensed dataset**: Pages with no annotations are filtered out to prevent training on empty images
- **Bounding box validation**: Degenerate boxes where `xmin >= xmax` or `ymin >= ymax` are removed during preprocessing

---

## Methodology

### Data Preprocessing Pipeline

```
Raw Manga109 (XML annotations via manga109api)
       │
       ▼
   Parse annotations → Pandas DataFrame (pickle serialization)
       │
       ▼
   Stratified train/val/test split (80/10/10)
       │
       ▼
   Filter empty pages + validate bounding boxes
       │
       ▼
   Format conversion (COCO-style for Faster R-CNN / RetinaNet,
                       normalized xywh for YOLOv3)
```

### Data Augmentation

Augmentations are applied jointly to images and bounding boxes using `imgaug` to maintain geometric consistency:

| Augmentation | Default | Strong |
|---|---|---|
| Sharpen | α ∈ [0.0, 0.1] | α ∈ [0.0, 0.1] |
| Rotation | 0° | ±10° |
| Translation | ±10% | ±10% |
| Scale | 0.8×–1.5× | 0.8×–1.5× |
| Brightness | Δ ∈ [-60, +40] | Δ ∈ [-60, +40] |
| Hue shift | Δ ∈ [-10, +10] | Δ ∈ [-20, +20] |
| Horizontal flip | 50% | 50% |
| Pixel dropout | — | [0.0, 0.01] |
| **Pad to square** | Center-center | Center-center |

All bounding box coordinates are transformed in sync via `BoundingBoxesOnImage` and clipped to image boundaries after augmentation.

### Neural Style Transfer Augmentation

As a domain adaptation experiment, **Neural Style Transfer (NST)** is applied to generate stylistically augmented training images:

- **Model**: 21-style pretrained network (PyTorch-Style-Transfer)
- **Target content size**: 1024 px
- **Output resolution**: Resized to 1654 × 1170 to match original annotations
- **Training regime**: Faster R-CNN fine-tuned on NST-augmented images with SGD (lr=0.01) and batch size 8, initialized from the base Faster R-CNN checkpoint

This explores whether style-based domain augmentation can improve robustness to the visual diversity across manga titles.

### Training Configuration

| Hyperparameter | Faster R-CNN | RetinaNet | YOLOv3 |
|---|---|---|---|
| **Backbone** | ResNet-50 + FPN | ResNet-50 + FPN | Darknet-53 |
| **Pretrained** | ImageNet | ImageNet | ImageNet (darknet53.conv.74) |
| **Optimizer** | Adam | Adam | SGD (default) |
| **Learning rate** | 1×10⁻⁴ | 1×10⁻⁴ | Config-defined |
| **Batch size** | 4 | 4 | 8 |
| **Epochs** | 10 | 15 | Config-defined |
| **Input size** | Variable (with FPN) | Variable (with FPN) | 416 × 416 |
| **Num classes** | 5 (4 + bg) | 5 (4 + bg) | 4 |

### Evaluation Protocol

- **Metric**: COCO-standard mAP@[0.5:0.95] — averaged over 10 IoU thresholds from 0.50 to 0.95 in steps of 0.05
- **Implementation**: `pycocotools` COCO evaluation API
- **Checkpointing**: Best model selected by highest validation mAP, with per-epoch checkpoint saving for reproducibility

---

## Results

### Model Comparison

| Model | Backbone | mAP@[0.5:0.95] |
|---|---|---|
| SSD300-fork ([Ogawa et al., 2018](https://arxiv.org/abs/1803.08670)) | VGG-16 | — (prior SOTA) |
| **Faster R-CNN** | ResNet-50 + FPN | **71.0** |
| RetinaNet | ResNet-50 + FPN | Competitive |
| YOLOv3 | Darknet-53 | Competitive |

**Key finding**: Faster R-CNN with ResNet-50 + FPN achieved the best performance with a **state-of-the-art mAP of 71.0**, surpassing prior methods on the Manga109 benchmark. The two-stage architecture's explicit region proposal mechanism proves advantageous for manga's dense, multi-scale object layouts.

### Detection Categories

| Class | Description | Challenge |
|---|---|---|
| `body` | Full character bodies | High scale variance, occlusion by panel borders |
| `face` | Character faces | Small objects, extreme stylistic variation across titles |
| `frame` | Panel/frame borders | Large objects, near-rectangular, densely packed |
| `text` | Speech bubbles & text regions | Variable size, irregular shapes, overlaps with frames |

---

## Repository Structure

```
MangaObjectDetection/
│
├── faster_rcnn.py           # Faster R-CNN training with ResNet-50+FPN
├── retinanet.py             # RetinaNet training with Focal Loss
├── yolo.py                  # YOLOv3 training configuration
├── yolo_inference.py        # YOLOv3 inference pipeline & visualization
├── yolo_tensorboard.py      # TensorBoard logging for YOLOv3
│
├── data_preprocessing.py    # Manga109 dataset loading, augmentation transforms
├── data_split.py            # Stratified train/val/test splitting
├── yolo_preprocessing.py    # Image directory restructuring for YOLO format
├── yolo_preprocessing_2.py  # Annotation conversion to YOLO normalized format
│
├── nst.py                   # Neural Style Transfer augmentation pipeline
├── nst_fasterrcnn.py        # Faster R-CNN fine-tuning on NST-augmented data
│
└── README.md
```

---

## Getting Started

### Prerequisites

```
Python >= 3.7
PyTorch >= 1.6
torchvision >= 0.7
pycocotools
imgaug
manga109api
pandas
numpy
scikit-learn
Pillow
matplotlib
```

### Dataset Setup

1. Obtain the [Manga109 dataset](http://www.manga109.org/en/) (academic use license required).
2. Place the dataset under `./Manga109/` with the standard directory structure.

### Data Preparation

```bash
# Step 1: Parse annotations from Manga109 into pickle format
python data_preprocessing.py

# Step 2: Create train/val/test splits
python data_split.py

# Step 3 (YOLOv3 only): Convert annotations to YOLO format
python yolo_preprocessing.py
python yolo_preprocessing_2.py
```

### Training

```bash
# Train Faster R-CNN (ResNet-50 + FPN)
python faster_rcnn.py

# Train RetinaNet (ResNet-50 + FPN + Focal Loss)
python retinanet.py

# Train Faster R-CNN on NST-augmented data
python nst_fasterrcnn.py
```

### Inference (YOLOv3)

```bash
python yolo_inference.py
```

Outputs detection visualizations with color-coded bounding boxes and confidence scores to the `output/` directory.

---

## Technical Details

### Transfer Learning Strategy

All models leverage **ImageNet-pretrained backbones** to bridge the domain gap between natural images and manga art:

- **Faster R-CNN / RetinaNet**: Full `fasterrcnn_resnet50_fpn` / `retinanet_resnet50_fpn` pretrained weights from TorchVision, with the classification head replaced for 5-class detection. All backbone parameters remain trainable (full fine-tuning).
- **YOLOv3**: `darknet53.conv.74` weights (first 74 layers of Darknet-53 pretrained on ImageNet), with detection heads trained from scratch.

### Bounding Box Handling

- **Faster R-CNN / RetinaNet**: Absolute pixel coordinates `[xmin, ymin, xmax, ymax]` in COCO format, with COCO-style area computation and `iscrowd=0` annotation flags.
- **YOLOv3**: Normalized center coordinates `[cx, cy, w, h]` where all values are in [0, 1] relative to image dimensions.
- **Augmentation consistency**: The custom `ImgAug` wrapper ensures all spatial augmentations (affine, flip, pad) are applied jointly to both images and bounding boxes, with post-augmentation clipping via `clip_out_of_image()`.

### Checkpoint Management

The training pipeline implements robust checkpointing:
- **Per-epoch saves**: Full model state dict saved after every epoch for analysis
- **Best-model tracking**: Separate checkpoint storing `{epoch, model_state_dict, optimizer_state_dict, mAP}` — updated only when validation mAP improves
- **Resumable training**: Supports loading from either the best or most recent checkpoint to resume interrupted training

---

## References

1. S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," *NeurIPS*, 2015.
2. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, "Focal Loss for Dense Object Detection," *ICCV*, 2017.
3. J. Redmon and A. Farhadi, "YOLOv3: An Incremental Improvement," *arXiv:1804.02767*, 2018.
4. T. Ogawa, A. Otsubo, R. Narita, Y. Matsui, T. Yamasaki, and K. Aizawa, "Object Detection for Comics using Manga109 Annotations," *arXiv:1803.08670*, 2018.
5. K. Aizawa, A. Fujimoto, A. Otsubo, T. Ogawa, Y. Matsui, K. Tsubota, and H. Ikuta, "Building a Manga Dataset 'Manga109' with Annotations for Multimedia Applications," *IEEE Multimedia*, 2020.
6. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR*, 2016.
7. T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie, "Feature Pyramid Networks for Object Detection," *CVPR*, 2017.
8. L. A. Gatys, A. S. Ecker, and M. Bethge, "A Neural Algorithm of Artistic Style," *arXiv:1508.06576*, 2015.
