# Motion Object Encoder

This repository provides a minimal inference-only demo for visualizing the
dense features of our pretrained Swin-H model, together with downstream task
integrations for evaluating the released representation.

## Installation

Install the dependencies for the standalone feature demo:

```bash
pip install -r requirements.txt
```

Downstream tasks may require additional dependencies. See the README in each
downstream directory for its environment and setup instructions.

## Checkpoint

The inference-only Swin-H checkpoint is about 3.0 GB after removing the
optimizer and scheduler states. Place it at:

```text
checkpoints/swin_h.pth
```

The checkpoint uses MMPretrain parameter names and is tracked with Git LFS.
Task-specific conversion instructions, when required, are documented with the
corresponding downstream implementation.

## Feature Visualization

Run the PyTorch-only visualization demo on a single image:

```bash
python tools/feature_visualization.py assets/pic1.png \
    --output assets/pic1_pca.png
```

The script center-crops the input to 16:9, resizes it to 512x288, applies the
reference RGB normalization, and saves a PCA visualization of the dense
features. CUDA is used when available; pass `--device cpu` or `--device cuda`
to select the device explicitly.

The PCA colors are computed independently for each image. They indicate
within-image feature similarity and are not semantic labels.

## Downstream Tasks

Each downstream integration is kept self-contained under `downstream/`. Its
README documents task-specific dependencies, datasets, checkpoint adaptation,
training, and evaluation.

| Task | Framework | Dataset | Documentation |
| --- | --- | --- | --- |
| Monocular depth estimation | DCDepth | KITTI | [DCDepth README](downstream/DCDepth/README.md) |
| 3D object detection | BEVFormer V2 | nuScenes | Coming soon |
| 3D occupancy prediction | SparseOcc | nuScenes | Coming soon |

The main release provides the pretrained representation checkpoint, not a
complete model for every downstream task. Evaluation may therefore require a
separately trained task-specific checkpoint containing the corresponding
decoder, prediction heads, or other task modules.

## Notebook

An interactive version of the feature visualization demo is available at
`notebooks/feature_visualization.ipynb`:

```bash
jupyter notebook notebooks/feature_visualization.ipynb
```

The notebook loads two example driving images and renders one PCA feature map
for each image.
