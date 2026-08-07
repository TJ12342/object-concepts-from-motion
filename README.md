# Motion Object Encoder: Minimal Feature Demo

This repository provides a minimal inference-only demo for visualizing the
dense features of the 421M-frame Swin-H model. The model is implemented in
plain PyTorch; OpenMMLab packages are not required. The repository does not
include the pretraining pipeline, video data, optical flow files,
pseudo-labels, or downstream-task training checkpoints. An optional DCDepth
downstream source tree is included under `downstream/DCDepth`.

## Requirements

Install the public runtime dependencies:

```bash
pip install -r requirements.txt
```

## Checkpoint

The inference-only checkpoint is about 3.0 GB after removing optimizer and
scheduler states. Download it to `checkpoints/swin_h.pth`; the file is tracked
with Git LFS.

## Convert for DCDepth

The released checkpoint uses the MMPretrain Swin-H parameter names. To use
this encoder with DCDepth's older Swin v1 implementation, run the bundled
PyTorch-only converter after downloading the release checkpoint:

```bash
python tools/mmlab_to_swinv1.py \
    checkpoints/swin_h.pth \
    downstream/DCDepth/checkpoints/swin_h.pth
```

The converter renames the Swin stages and patch embedding, and applies the
patch-merging channel-order permutation required by DCDepth. It writes a
direct state dict compatible with DCDepth's `pretrained` argument and prints
the output SHA-256. The destination directory is already part of the
repository and the generated file is ignored by Git. It does not require
MMEngine, MMPretrain, or the original training repository. Add `--force` to
overwrite an existing destination.

## Single-image inference

```bash
python tools/feature_visualization.py assets/pic1.png \
    --output assets/pic1_pca.png
```

The command center-crops the input to a 16:9 field of view, resizes it to
512x288, applies the reference model's RGB mean/std normalization, prints the
dense feature shape, and saves a 512x288 PCA RGB image. The default device is
CUDA when available; use `--device cpu` or `--device cuda` to select one
explicitly.

## Notebook

```bash
cd /path/to/open_source
jupyter notebook notebooks/feature_visualization.ipynb
```

The notebook loads two example driving images and renders one PCA feature map
for each image. PCA colors are computed independently per image and indicate
within-image feature similarity; they are not semantic labels.
