# FlowSeg integration: SparseOcc

This document contains only the FlowSeg-specific integration details. The
upstream SparseOcc documentation, including its reference environment, dataset
preparation, and general workflow, is preserved in [README.md](README.md).
Follow that README first.

Upstream project: [MCG-NJU/SparseOcc](https://github.com/MCG-NJU/SparseOcc)

## Configurations

The FlowSeg adapter adds one configuration per released Swin checkpoint:

| Backbone | Config | Representation checkpoint |
| --- | --- | --- |
| Swin-T | `configs/sparseocc_swin_t.py` | `../../checkpoints/swin_t.pth` |
| Swin-S | `configs/sparseocc_swin_s.py` | `../../checkpoints/swin_s.pth` |
| Swin-B | `configs/sparseocc_swin_b.py` | `../../checkpoints/swin_b.pth` |
| Swin-L | `configs/sparseocc_swin_l.py` | `../../checkpoints/swin_l.pth` |
| Swin-H | `configs/sparseocc_swin_h.py` | `../../checkpoints/swin_h.pth` |

The paths are resolved from this directory. The representation checkpoint
initializes the image backbone; a complete SparseOcc task checkpoint is still
required for occupancy evaluation.

Use the upstream SparseOcc training and evaluation commands from
[README.md](README.md), selecting one of the FlowSeg configurations above.
Compile the CUDA extensions described by the upstream README before running
the task. Keep this environment separate from the root feature-demo
environment.

## Dependencies

Install the versions documented by the upstream README first. The additional
packages needed by this adapter are listed in
[`requirements.txt`](requirements.txt).

## License

See [LICENSE](LICENSE) for the upstream license terms.
