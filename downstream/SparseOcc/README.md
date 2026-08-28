# SparseOcc: 3D Occupancy Prediction

This directory contains the SparseOcc integration for semantic occupancy
prediction on nuScenes. The code is vendored from the official
[SparseOcc repository](https://github.com/MCG-NJU/SparseOcc) (ECCV 2024,
arXiv:2312.17118) and is kept self-contained like the other downstream tasks
in this release.

The model reconstructs a sparse 3D representation from eight camera frames and
predicts semantic occupancy with the RayIoU evaluation metric. The released
representation checkpoint initializes the image backbone; a complete
SparseOcc checkpoint is still required for evaluation.

## Configurations

The recommended configs use the local `SwinTransformer2` implementation and
the representation checkpoints in the repository root:

| Backbone | Config | Representation checkpoint |
| --- | --- | --- |
| Swin-T | `configs/sparseocc_swin_t.py` | `../../checkpoints/swin_t.pth` |
| Swin-S | `configs/sparseocc_swin_s.py` | `../../checkpoints/swin_s.pth` |
| Swin-B | `configs/sparseocc_swin_b.py` | `../../checkpoints/swin_b.pth` |
| Swin-L | `configs/sparseocc_swin_l.py` | `../../checkpoints/swin_l.pth` |
| Swin-H | `configs/sparseocc_swin_h.py` | `../../checkpoints/swin_h.pth` |

Only the Swin-H and Swin-L representation checkpoints are included in the
current release. The other paths are ready for their matching checkpoints.
The official R50 model-zoo configs are
`configs/r50_nuimg_704x256_8f.py`, `configs/r50_nuimg_704x256_8f_60e.py`,
and `configs/r50_nuimg_704x256_8f_pano.py`.

## Environment

SparseOcc follows the upstream environment:

```text
Python 3.8
PyTorch 2.0.0
CUDA 11.8
MMCV-full 1.6.0
MMDetection 2.28.2
MMSegmentation 0.30.0
MMDetection3D 1.0.0rc6
MMClassification 0.25.0
```

Create the environment and install the task dependencies:

```bash
conda create -n sparseocc python=3.8
conda activate sparseocc
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install openmim
mim install mmcv-full==1.6.0
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
mim install mmdet3d==1.0.0rc6
mim install mmcls==0.25.0
pip install -r requirements.txt
```

The root `requirements.txt` targets the standalone feature demo and requests a
newer PyTorch version; use the versions above for this legacy OpenMMLab task.

Compile the multi-scale sampling extension before training or evaluation:

```bash
cd downstream/SparseOcc/models/csrc
python setup.py build_ext --inplace
cd ../../../..
```

The RayIoU evaluator also builds the small DVR CUDA extension on first import.
This requires a CUDA compiler and a compatible host compiler.

## Dataset

Download nuScenes v1.0 and the CAN bus data. Prepare the standard MMDetection3D
nuScenes info files, then generate the temporal sweep annotations:

```bash
cd downstream/SparseOcc
python gen_sweep_info.py --data-root data/nuscenes
```

Download the Occ3D-nuScenes occupancy labels and place them under
`data/nuscenes/occ3d`. The expected layout is:

```text
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-test/
├── v1.0-trainval/
├── nuscenes_infos_train_sweep.pkl
├── nuscenes_infos_val_sweep.pkl
├── nuscenes_infos_test_sweep.pkl
└── occ3d/<scene>/<sample-token>/labels.npz
```

The `gen_instance_info.py` utility can optionally create panoptic labels under
`data/nuscenes/occ3d_panoptic`.

The R50 nuImages configs expect the official nuImages semantic-pretraining
checkpoint at `pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth`.
Download it from the upstream SparseOcc release instructions before using the
R50 configs; the Swin configs use the root checkpoints listed above instead.

## Training

Run commands from this directory so that `data/` and checkpoint paths resolve
as documented. For example, train the Swin-H configuration on eight GPUs:

```bash
cd downstream/SparseOcc
torchrun --nproc_per_node 8 train.py --config configs/sparseocc_swin_h.py
```

The global batch size is defined by `batch_size` in the config and is divided
across workers automatically.

## Evaluation

Evaluation expects a complete task checkpoint containing the SparseOcc decoder
and prediction heads, not only a backbone representation checkpoint:

```bash
cd downstream/SparseOcc
python val.py \
    --config configs/sparseocc_swin_h.py \
    --weights /path/to/sparseocc_checkpoint.pth
```

Use `torchrun --nproc_per_node N val.py ... --world_size N` for distributed
evaluation. `timing.py` measures single-GPU throughput and `viz_prediction.py`
renders qualitative occupancy predictions.

## Standalone RayIoU

To evaluate saved semantic predictions independently, save one compressed
`uint8` array with shape `200x200x16` per sample token:

```python
np.savez_compressed(
    "prediction/my_model/<sample-token>.npz",
    pred=semantic_prediction,
)
```

Then run:

```bash
python ray_metrics.py \
    --data-root data/nuscenes \
    --pred-dir prediction/my_model \
    --data-type occ3d
```

The evaluator reports RayIoU at depth tolerances 1, 2, and 4. Panoptic outputs
can additionally be scored with the RayPQ path used by the dataset evaluator.

## License and Citation

See [LICENSE](LICENSE) for the upstream license terms. If you use this code,
please cite:

```bibtex
@article{liu2023fully,
  title={Fully sparse 3d panoptic occupancy prediction},
  author={Liu, Haisong and Wang, Haiguang and Chen, Yang and Yang, Zetong and Zeng, Jia and Chen, Li and Wang, Limin},
  journal={arXiv preprint arXiv:2312.17118},
  year={2023}
}
```
