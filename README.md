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

The five Eigen configs use the standard downstream filenames
`checkpoints/swin_b.pth`, `swin_s.pth`, `swin_t.pth`, `swin_l.pth`, and
`swin_h.pth`. This release provides the Swin-H weight; the other four paths
are reserved for matching converted checkpoints.

## Depth Task (DCDepth)

The `downstream/DCDepth` directory contains the DCDepth source and Eigen
configuration. The released Swin-H file is only the encoder. A depth
prediction also needs a complete, trained DCDepth checkpoint containing the
CRF, decoder, and depth-update head. That downstream checkpoint is not
included in this repository because it is large and is a separate task
artifact.

### 1. Prepare the files

After downloading the release checkpoint, the repository should contain:

```text
checkpoints/swin_h.pth
downstream/DCDepth/checkpoints/
```

Convert the MMPretrain encoder once from the repository root:

```bash
python tools/mmlab_to_swinv1.py \
    checkpoints/swin_h.pth \
    downstream/DCDepth/checkpoints/swin_h.pth
```

Then place the complete DCDepth Eigen checkpoint at:

```text
downstream/DCDepth/checkpoints/dcdepth_eigen_huge.ckpt
```

The converted encoder is ignored by Git, as are downstream task checkpoints
and evaluation outputs.

### 2. Prepare KITTI Eigen data

The default [eigen_base.yaml](downstream/DCDepth/configs/eigen_base.yaml)
expects this layout relative to `downstream/DCDepth`:

```text
downstream/DCDepth/data/kitti/
├── data_rgb/
└── data_depth_annotated/
```

The split files under `downstream/DCDepth/data_splits` provide the relative
RGB and ground-truth paths. If KITTI is stored elsewhere, edit these four
fields in `eigen_base.yaml`:

```yaml
dataset:
  data_path: '/your/path/to/kitti/'
  data_path_eval: '/your/path/to/kitti/'
  gt_path: '/your/path/to/kitti/data_depth_annotated/'
  gt_path_eval: '/your/path/to/kitti/data_depth_annotated/'
```

No other path in the default huge configuration needs to be changed when the
converted encoder is stored at `downstream/DCDepth/checkpoints/swin_h.pth`.

### 3. Install Depth dependencies

The root `requirements.txt` covers the standalone feature demo. The DCDepth
evaluator has a separate dependency list:

```bash
pip install -r downstream/DCDepth/requirements.txt
```

The standard `test.py` path uses MMEngine for YAML config inheritance and the
local model/data registries. It does not require MMCV. The legacy
`networks/checkpoint.py` helper still contains optional MMCV-based OpenMMLab
checkpoint-loading code, but it is not imported by the documented Eigen
evaluation command.

### 4. Run Eigen evaluation

Run from the DCDepth directory because its original scripts resolve
`configs/`, `data_splits/`, and `checkpoints/` relative to the current working
directory:

```bash
cd downstream/DCDepth
CUDA_VISIBLE_DEVICES=0 \
python test.py dct_eigen_pff_huge \
    checkpoints/dcdepth_eigen_huge.ckpt \
    --vis
```

The `dct_eigen_pff_huge.yaml` configuration selects the `huge07` Swin-H
backbone and reads the converted encoder from `checkpoints/swin_h.pth`. The
full task checkpoint passed as the second argument must match this model
configuration.

Metrics are written under:

```text
downstream/DCDepth/checkpoints/dct_eigen_pff_huge/result.csv
downstream/DCDepth/checkpoints/dct_eigen_pff_huge/result_avg.csv
```

With `--vis`, predicted depth visualizations are written to the corresponding
`vis/` directory. The bundled `test.py` evaluates the KITTI Eigen split; it is
not a single-image inference script.

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
