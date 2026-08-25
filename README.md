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

The inference-only Swin-H and Swin-L checkpoints are about 3.0 GB and 759 MB,
respectively, after removing the optimizer, schedulers, message hub, and full
training metadata. Place them at:

```text
checkpoints/swin_h.pth
checkpoints/swin_l.pth
```

The checkpoint uses MMPretrain parameter names and is tracked with Git LFS.
Task-specific conversion instructions, when required, are documented with the
corresponding downstream implementation.

The standalone PyTorch model supports all five released Swin configurations.
Select the checkpoint architecture explicitly when loading the model:

| Variant | Embedding | Stage depths | Attention heads | FPN input channels |
| --- | ---: | --- | --- | --- |
| H / huge | 384 | 2, 2, 18, 2 | 12, 24, 48, 96 | 384, 768, 1536, 3072 |
| L / large | 192 | 2, 2, 18, 2 | 6, 12, 24, 48 | 192, 384, 768, 1536 |
| B / base | 128 | 2, 2, 18, 2 | 4, 8, 16, 32 | 128, 256, 512, 1024 |
| S / small | 96 | 2, 2, 18, 2 | 3, 6, 12, 24 | 96, 192, 384, 768 |
| T / tiny | 96 | 2, 2, 6, 2 | 3, 6, 12, 24 | 96, 192, 384, 768 |

To export the distilled student backbone, neck, and head from a trusted MMEngine
training checkpoint, retain only minimal provenance metadata:

```bash
python tools/export_inference_checkpoint.py \
    /path/to/epoch_10.pth \
    /tmp/swin_l.pth \
    --model-name FlowSeg-421M-SwinL-Distilled \
    --trust-source
```

The exporter maps `student_backbone.*`, `student_neck.*`, and `student_head.*`
to `backbone.*`, `neck.*`, and `head.*`, respectively. It discards all other
model and training state, verifies every exported tensor, and prints the output
size and SHA-256. Run it in the trusted source training environment so MMEngine
checkpoint objects can be imported. Use `--force` to replace an existing
destination.

## Feature Visualization

Run the PyTorch-only visualization demo on a single image:

```bash
python tools/feature_visualization.py assets/pic1.png \
    --arch huge \
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
| 3D object detection | BEVFormer V2 | nuScenes | [BEVFormer V2 README](downstream/BEVFormerV2/README.md) |
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
