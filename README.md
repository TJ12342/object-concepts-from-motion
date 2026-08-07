# Motion Object Encoder: Minimal Feature Demo

This repository provides a minimal inference-only demo for visualizing the
dense features of the 421M-frame Swin-H model. The model is implemented in
plain PyTorch; OpenMMLab packages are not required. The repository does not
include the pretraining pipeline, video data, optical flow files,
pseudo-labels, or downstream-task code.

## Requirements

Install the public runtime dependencies:

```bash
pip install -r requirements.txt
```

## Checkpoint

The inference-only checkpoint is about 3.0 GB after removing optimizer and
scheduler states. It is stored at `checkpoints/flowseg_421m_swinh.pth` and is
tracked with Git LFS.

## Single-image inference

```bash
python tools/feature_visualization.py assets/kitti_highway.jpg \
    --output assets/kitti_highway_pca.png
```

The command prints the dense feature shape and saves a PCA RGB image. The
default device is CUDA when available and CPU otherwise; use `--device cpu`
or `--device cuda` to select one explicitly.

## Notebook

```bash
cd /path/to/open_source
jupyter notebook notebooks/feature_visualization.ipynb
```

The notebook loads two example driving images and renders one PCA feature map
for each image. PCA colors are computed independently per image and indicate
within-image feature similarity; they are not semantic labels.
