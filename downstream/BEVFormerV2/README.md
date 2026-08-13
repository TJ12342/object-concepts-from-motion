# BEVFormer V2: 3D Object Detection

This directory contains the BEVFormer V2 integration used to evaluate our
pretrained Swin representations on the nuScenes 3D object detection task. The
experiments use two input frames and a 24-epoch downstream schedule.

The code is adapted from the official
[BEVFormer repository](https://github.com/fundamentalvision/BEVFormer). See
`LICENSE` for its license terms.

## Configurations

One configuration is provided for each supported backbone size:

| Backbone | Pretraining variant | Config | Checkpoint path |
| --- | --- | --- | --- |
| Swin-T | Distilled from Cycle 2 Swin-H | `configs/swin_t.py` | `../../checkpoints/swin_t.pth` |
| Swin-S | Distilled from Cycle 2 Swin-H | `configs/swin_s.py` | `../../checkpoints/swin_s.pth` |
| Swin-B | Distilled from Cycle 2 Swin-H | `configs/swin_b.py` | `../../checkpoints/swin_b.pth` |
| Swin-L | Distilled from Cycle 2 Swin-H | `configs/swin_l.py` | `../../checkpoints/swin_l.pth` |
| Swin-H | Cycle 2 | `configs/swin_h.py` | `../../checkpoints/swin_h.pth` |

The paths are resolved from this directory. Place each representation
checkpoint that you intend to use in the root repository's `checkpoints/`
directory before training. The current release includes Swin-H; the other
configuration paths are ready for their matching compact checkpoints.

## Environment

The integration follows the legacy OpenMMLab stack used by BEVFormer V2:

```text
Python 3.8
PyTorch 1.9.1
CUDA 11.1
MMCV-full 1.4.0
MMDetection 2.14.0
MMSegmentation 0.14.1
MMDetection3D 0.17.1
```

Install the OpenMMLab packages following the
[MMDetection3D 0.17.1 instructions](https://github.com/open-mmlab/mmdetection3d/tree/v0.17.1),
then install the additional dependencies:

```bash
pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Detectron2 must be built against the selected PyTorch and CUDA versions.

## Dataset

Download nuScenes v1.0 and the CAN bus expansion. From this directory, arrange
them as follows:

```text
data/
├── can_bus/
└── nuscenes/
    ├── maps/
    ├── samples/
    ├── sweeps/
    ├── v1.0-test/
    └── v1.0-trainval/
```

Generate the temporal annotation files used by the configs:

```bash
python tools/create_data.py \
    --root-path data/nuscenes \
    --out-dir data/nuscenes \
    --canbus data \
    --version v1.0
```

This creates `nuscenes_infos_temporal_train.pkl`,
`nuscenes_infos_temporal_val.pkl`, and
`nuscenes_infos_temporal_test.pkl` under `data/nuscenes/`.

## Training

Run from this directory because the configs use paths relative to it. For
example, train Swin-H on 8 GPUs with:

```bash
./tools/dist_train.sh configs/swin_h.py 8 \
    --work-dir work_dirs/swin_h
```

Replace `swin_h.py` with the configuration for another backbone size as
needed.

## Evaluation

A released representation checkpoint initializes only the image backbone. To
evaluate detection, pass a complete BEVFormer V2 checkpoint produced by the
corresponding training configuration:

```bash
./tools/dist_test.sh \
    configs/swin_h.py \
    work_dirs/swin_h/latest.pth \
    8
```

The command evaluates 3D bounding boxes on the nuScenes validation set.
Single-GPU evaluation is also supported by setting the last argument to `1`.

## Citation

```bibtex
@inproceedings{yang2023bevformer,
  title={{BEVFormer} v2: Adapting modern image backbones to bird's-eye-view recognition via perspective supervision},
  author={Yang, Chenyu and Chen, Yuntao and Tian, Hao and Tao, Chenxin and Zhu, Xizhou and Zhang, Zhaoxiang and Huang, Gao and Li, Hongyang and Qiao, Yu and Lu, Lewei and others},
  booktitle={CVPR},
  year={2023}
}
```
