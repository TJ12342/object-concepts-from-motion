# FlowSeg integration: BEVFormer

This document contains only the FlowSeg-specific integration details. The
upstream BEVFormer documentation, including its reference environment,
dataset preparation, and general workflow, is preserved in
[README.md](README.md). Follow that README first.

Upstream project: [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer)

## Configurations

The FlowSeg adapter adds one configuration per released Swin checkpoint:

| Backbone | Pretraining variant | Config | Representation checkpoint |
| --- | --- | --- | --- |
| Swin-T | Distilled from Cycle 2 Swin-H | `configs/swin_t.py` | `../../checkpoints/swin_t.pth` |
| Swin-S | Distilled from Cycle 2 Swin-H | `configs/swin_s.py` | `../../checkpoints/swin_s.pth` |
| Swin-B | Distilled from Cycle 2 Swin-H | `configs/swin_b.py` | `../../checkpoints/swin_b.pth` |
| Swin-L | Distilled from Cycle 2 Swin-H | `configs/swin_l.py` | `../../checkpoints/swin_l.pth` |
| Swin-H | Cycle 2 | `configs/swin_h.py` | `../../checkpoints/swin_h.pth` |

The paths are resolved from this directory. The representation checkpoint
initializes the image backbone; a complete BEVFormer task checkpoint is still
required for detection evaluation.

Run the upstream training or evaluation commands from the directory containing
this adapter, using one of the configurations above. Keep the BEVFormer
environment separate from the root feature-demo environment because the
OpenMMLab versions are different.

## License

See [LICENSE](LICENSE) for the upstream license terms.
