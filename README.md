<h1 align="center">Object Concepts Emerge from Motion</h1>

<p align="center">
  <a href="https://tj12342.github.io/object-concepts-from-motion/">Project Page</a>
  &nbsp;|&nbsp;
  <span>&#128196; Paper (coming soon)</span>
  &nbsp;|&nbsp;
  <a href="https://huggingface.co/tj111/object-concepts-from-motion">Models</a>
</p>

This repository provides a minimal inference-only demo for visualizing the
dense features of our pretrained Swin-H model, together with downstream task
integrations for evaluating the released representation.

## Installation

The root `requirements.txt` is only for loading the released checkpoints,
running the standalone PyTorch forward pass, and using the feature
visualization demo:

```bash
pip install -r requirements.txt
```

It is not a complete environment for the downstream tasks. Each downstream
integration has its own framework versions and dependency constraints; do not
try to combine all of them in one Python environment.



The checkpoint uses MMPretrain parameter names and is stored on
Hugging Face. Task-specific conversion instructions, when required, are
documented with the corresponding downstream implementation.

The standalone PyTorch model supports all five released Swin configurations.
Select the checkpoint architecture explicitly when loading the model:

| Variant | Embedding | Stage depths | Attention heads | FPN input channels |
| --- | ---: | --- | --- | --- |
| H / huge | 384 | 2, 2, 18, 2 | 12, 24, 48, 96 | 384, 768, 1536, 3072 |
| L / large | 192 | 2, 2, 18, 2 | 6, 12, 24, 48 | 192, 384, 768, 1536 |
| B / base | 128 | 2, 2, 18, 2 | 4, 8, 16, 32 | 128, 256, 512, 1024 |
| S / small | 96 | 2, 2, 18, 2 | 3, 6, 12, 24 | 96, 192, 384, 768 |
| T / tiny | 96 | 2, 2, 6, 2 | 3, 6, 12, 24 | 96, 192, 384, 768 |

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

Downstream projects are maintained in their own repositories and have
independent, often incompatible environments. The original upstream README is
preserved in each corresponding `downstream/*/README.md`; read it first and
follow that project's installation, dataset, training, and evaluation steps.
Installing the root requirements alone is not sufficient for downstream
evaluation.

The `FLOWSEG.md` file next to each upstream README contains only the
FlowSeg-specific integration details: checkpoint paths, adapted configurations,
and commands for using this release. Use a separate Python environment for
each downstream project.

| Task | Upstream project and README | Dataset | FlowSeg integration |
| --- | --- | --- | --- |
| Monocular depth estimation | [DCDepth](https://github.com/w2kun/DCDepth#readme) | KITTI | [Local FlowSeg adapter](downstream/DCDepth/FLOWSEG.md) |
| 3D object detection | [BEVFormer](https://github.com/fundamentalvision/BEVFormer#readme) | nuScenes | [Local FlowSeg adapter](downstream/BEVFormerV2/FLOWSEG.md) |
| 3D occupancy prediction | [SparseOcc](https://github.com/MCG-NJU/SparseOcc#readme) | nuScenes | [Local FlowSeg adapter](downstream/SparseOcc/FLOWSEG.md) |

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
