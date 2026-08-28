"""DINOv3 backbone adapter for SparseOcc.

The reference DINOv3 implementation lives outside this repository.  This
adapter keeps the integration local to the backbone registry while preserving
the feature format expected by the existing FPN necks.
"""

import sys
from pathlib import Path

import torch
from mmdet.models.builder import BACKBONES

from .base_backbone import BaseBackbone


@BACKBONES.register_module()
class DINOv3VisionTransformer(BaseBackbone):
    """Load a local DINOv3 ViT and return intermediate patch feature maps."""

    def __init__(self,
                 repo_dir,
                 checkpoint,
                 arch='large',
                 out_indices=(5, 11, 17, 23),
                 frozen=False,
                 init_cfg=None,
                 **kwargs):
        # The DINOv3 weights are loaded explicitly below.  Keeping init_cfg
        # empty prevents the runner from trying to interpret the raw DINO
        # checkpoint as an MMCV checkpoint a second time.
        super().__init__(init_cfg=None)

        repo_dir = Path(repo_dir).expanduser().resolve()
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not repo_dir.is_dir():
            raise FileNotFoundError(f'DINOv3 repository not found: {repo_dir}')
        if not checkpoint.is_file():
            raise FileNotFoundError(f'DINOv3 checkpoint not found: {checkpoint}')

        # The reference package is intentionally kept outside SparseOcc to
        # avoid vendoring its large dependency tree and model sources.
        repo_dir_str = str(repo_dir)
        if repo_dir_str not in sys.path:
            sys.path.insert(0, repo_dir_str)
        from dinov3.hub.backbones import (
            dinov3_vitb16,
            dinov3_vitl16,
            dinov3_vits16,
        )

        backbone_builders = {
            'small': dinov3_vits16,
            'base': dinov3_vitb16,
            'large': dinov3_vitl16,
        }
        if arch not in backbone_builders:
            raise ValueError(
                f'Unsupported DINOv3 architecture {arch!r}; '
                f'expected one of {tuple(backbone_builders)}')
        self.arch = arch
        self.backbone = backbone_builders[arch](pretrained=False)
        state_dict = torch.load(str(checkpoint), map_location='cpu')
        if isinstance(state_dict, dict) and 'teacher' in state_dict:
            state_dict = state_dict['teacher']
        if not isinstance(state_dict, dict):
            raise TypeError(f'Unsupported DINOv3 checkpoint type: {type(state_dict)!r}')
        state_dict = {
            key.replace('module.', '').replace('backbone.', ''): value
            for key, value in state_dict.items()
        }
        msg = self.backbone.load_state_dict(state_dict, strict=True)
        if msg.missing_keys or msg.unexpected_keys:
            raise RuntimeError(
                f'DINOv3 checkpoint does not match the ViT-{arch} architecture: '
                f'missing={msg.missing_keys}, unexpected={msg.unexpected_keys}')

        self.out_indices = tuple(out_indices)
        if not self.out_indices:
            raise ValueError('out_indices must contain at least one transformer block')
        if min(self.out_indices) < 0 or max(self.out_indices) >= len(self.backbone.blocks):
            raise ValueError(
                f'out_indices {self.out_indices} are invalid for '
                f'{len(self.backbone.blocks)} DINOv3 blocks')

        self.embed_dims = self.backbone.embed_dim
        self.frozen = frozen
        if frozen:
            self._freeze_backbone()

    def _freeze_backbone(self):
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def init_weights(self):
        # Weights are loaded in __init__; the runner calls init_weights after
        # model construction, so this must remain a no-op.
        return None

    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                features = self.backbone.get_intermediate_layers(
                    x, n=self.out_indices, reshape=True)
        else:
            features = self.backbone.get_intermediate_layers(
                x, n=self.out_indices, reshape=True)
        return list(features)
