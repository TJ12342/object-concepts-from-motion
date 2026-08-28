# FlowSeg integration: DCDepth

This document contains only the FlowSeg-specific integration details. The
upstream DCDepth documentation, including its reference environment, dataset
preparation, and general workflow, is preserved in [README.md](README.md).
Follow that README first.

Upstream project: [w2kun/DCDepth](https://github.com/w2kun/DCDepth)

## Configurations

The FlowSeg adapter provides Eigen-evaluation configurations for all released
Swin checkpoints:

| Backbone | Config | Representation checkpoint |
| --- | --- | --- |
| Swin-T | `configs/dct_eigen_pff_tiny.yaml` | `checkpoints/swin_t.pth` |
| Swin-S | `configs/dct_eigen_pff_small.yaml` | `checkpoints/swin_s.pth` |
| Swin-B | `configs/dct_eigen_pff_base.yaml` | `checkpoints/swin_b.pth` |
| Swin-L | `configs/dct_eigen_pff_large.yaml` | `checkpoints/swin_l.pth` |
| Swin-H | `configs/dct_eigen_pff_huge.yaml` | `checkpoints/swin_h.pth` |

The paths are resolved by the DCDepth evaluator from its working directory.
Place or symlink the released representation checkpoints under the
configuration's expected `checkpoints/` directory before running evaluation.
The representation checkpoint initializes the Swin encoder; a complete
task-specific DCDepth checkpoint is still required for depth evaluation.

Use the upstream DCDepth commands from [README.md](README.md), selecting one
of the FlowSeg configurations above. Keep the DCDepth environment separate
from the root feature-demo environment.

## Dependencies

Install the versions documented by the upstream README first. The additional
packages needed by this adapter are listed in
[`requirements.txt`](requirements.txt).

## License

See the upstream DCDepth repository for its license terms.
