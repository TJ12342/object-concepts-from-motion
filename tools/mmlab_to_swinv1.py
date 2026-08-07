"""Convert the released MMPretrain Swin-H checkpoint for DCDepth.

The released checkpoint uses the MMPretrain/MMEngine Swin-H names and patch
merging order.  DCDepth uses the older Swin v1 implementation, so its
backbone keys need to be renamed and the patch-merging weights need a fixed
permutation.  This script is intentionally pure PyTorch and does not import
MMEngine or any other OpenMMLab package.

The output is a direct state dict, matching the output of the original
``NeWCRFs/mmlab_to_swinv1.py`` script.  DCDepth's ``load_checkpoint`` accepts
this file and removes the ``backbone.`` prefix before loading it.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch


def correct_unfold_reduction_order(weight: torch.Tensor) -> torch.Tensor:
    """Reorder a MMPretrain patch-merging reduction matrix for DCDepth."""

    out_channels, in_channels = weight.shape
    if in_channels % 4:
        raise ValueError(
            f"Patch-merging reduction has {in_channels} input channels; "
            "the channel count must be divisible by 4"
        )
    weight = weight.reshape(out_channels, in_channels // 4, 4).transpose(1, 2)
    weight = weight[:, [0, 2, 1, 3], :]
    return weight.reshape(out_channels, in_channels)


def correct_unfold_norm_order(weight: torch.Tensor) -> torch.Tensor:
    """Reorder a MMPretrain patch-merging norm vector for DCDepth."""

    in_channels = weight.shape[0]
    if in_channels % 4:
        raise ValueError(
            f"Patch-merging norm has {in_channels} channels; "
            "the channel count must be divisible by 4"
        )
    weight = weight.reshape(in_channels // 4, 4).transpose(0, 1)
    weight = weight[[0, 2, 1, 3], :]
    return weight.reshape(in_channels)


def swin_converter(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict:
    """Convert MMPretrain Swin-H keys and patch-merging parameters.

    Non-backbone entries are retained for compatibility with the original
    converter.  DCDepth will select the entries beginning with ``backbone.``
    when it loads the resulting file.
    """

    converted = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        new_value = value

        if key.startswith("backbone.stages"):
            if "attn.w_msa." in key:
                new_key = key.replace("attn.w_msa.", "attn.")
            elif "ffn.layers.0.0." in key:
                new_key = key.replace("ffn.layers.0.0.", "mlp.fc1.")
            elif "ffn.layers.1." in key:
                new_key = key.replace("ffn.layers.1.", "mlp.fc2.")
            elif "ffn." in key:
                new_key = key.replace("ffn.", "mlp.")
            elif "downsample.reduction." in key:
                new_value = correct_unfold_reduction_order(value)
            elif "downsample.norm." in key:
                new_value = correct_unfold_norm_order(value)
            new_key = new_key.replace("backbone.stages", "backbone.layers", 1)
        elif key.startswith("backbone.patch_embed.projection"):
            new_key = key.replace(
                "backbone.patch_embed.projection",
                "backbone.patch_embed.proj",
                1,
            )

        if new_key in converted:
            raise ValueError(f"Key collision while converting {key!r} -> {new_key!r}")
        converted[new_key] = new_value

    return converted


def load_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    """Load either a release wrapper or a direct state dict."""

    checkpoint = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Expected a checkpoint mapping, got {type(checkpoint)!r}")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Checkpoint does not contain a non-empty state_dict")
    return state_dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_checkpoint(source: Path, destination: Path, force: bool = False) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source}")
    if destination.exists() and not force:
        raise FileExistsError(
            f"Destination already exists: {destination}; use --force to overwrite"
        )
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must be different files")

    state_dict = load_state_dict(source)
    converted = swin_converter(state_dict)

    converted_backbone = [
        key for key in converted if key.startswith("backbone.")
    ]
    if not converted_backbone:
        raise ValueError("No backbone.* keys found; this is not an MMPretrain Swin checkpoint")
    if any(key.startswith("backbone.stages") for key in converted):
        raise ValueError("Conversion left backbone.stages keys behind")
    if any(key.startswith("backbone.patch_embed.projection") for key in converted):
        raise ValueError("Conversion left patch_embed.projection keys behind")

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, destination)

    values = sum(value.numel() for value in converted.values())
    print(f"saved: {destination}")
    print(f"state_dict tensors: {len(converted)}")
    print(f"state_dict values: {values}")
    print(f"sha256: {sha256(destination)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the released Swin-H checkpoint for DCDepth."
    )
    parser.add_argument("source", type=Path, help="Downloaded release checkpoint")
    parser.add_argument("destination", type=Path, help="DCDepth checkpoint path")
    parser.add_argument("--force", action="store_true", help="Overwrite destination")
    args = parser.parse_args()
    convert_checkpoint(args.source, args.destination, force=args.force)


if __name__ == "__main__":
    main()
