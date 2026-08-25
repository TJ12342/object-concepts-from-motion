"""Export distilled student weights from a trusted MMEngine checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch


STATE_DICT_PREFIX_RENAMES = (
    ("student_backbone.", "backbone."),
    ("student_neck.", "neck."),
    ("student_head.", "head."),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_checkpoint(path: Path) -> Mapping:
    """Load a trusted training checkpoint, including MMEngine metadata."""

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The source checkpoint references packages from its training "
            "environment. Run this exporter in that environment."
        ) from error

    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            f"Expected a checkpoint mapping, got {type(checkpoint)!r}"
        )
    return checkpoint


def extract_state_dict(checkpoint: Mapping) -> OrderedDict[str, torch.Tensor]:
    """Select student model tensors and remove the student prefix."""

    source_state_dict = checkpoint.get("state_dict")
    if not isinstance(source_state_dict, Mapping) or not source_state_dict:
        raise ValueError("Source checkpoint has no non-empty state_dict")

    state_dict = OrderedDict()
    prefix_counts = {
        source_prefix: 0 for source_prefix, _ in STATE_DICT_PREFIX_RENAMES
    }
    non_tensors = []
    for source_key, value in source_state_dict.items():
        if not isinstance(source_key, str):
            continue
        for source_prefix, destination_prefix in STATE_DICT_PREFIX_RENAMES:
            if not source_key.startswith(source_prefix):
                continue

            if not torch.is_tensor(value):
                non_tensors.append(source_key)
                break

            destination_key = (
                destination_prefix + source_key[len(source_prefix):]
            )
            if destination_key in state_dict:
                raise ValueError(
                    f"Multiple source parameters map to {destination_key!r}"
                )
            state_dict[destination_key] = value
            prefix_counts[source_prefix] += 1
            break

    if non_tensors:
        preview = ", ".join(map(str, non_tensors[:5]))
        raise ValueError(
            f"Selected state_dict contains non-tensor values: {preview}"
        )

    missing_prefixes = [
        prefix for prefix, count in prefix_counts.items() if count == 0
    ]
    if missing_prefixes:
        preview = ", ".join(f"{prefix}*" for prefix in missing_prefixes)
        raise ValueError(f"state_dict contains no parameters matching: {preview}")

    source_metadata = getattr(source_state_dict, "_metadata", None)
    if isinstance(source_metadata, Mapping):
        metadata = OrderedDict()
        for source_key, value in source_metadata.items():
            if not isinstance(source_key, str):
                continue
            for source_prefix, destination_prefix in STATE_DICT_PREFIX_RENAMES:
                source_root = source_prefix[:-1]
                if source_key != source_root and not source_key.startswith(
                    source_prefix
                ):
                    continue
                destination_root = destination_prefix[:-1]
                destination_key = (
                    destination_root + source_key[len(source_root):]
                )
                metadata[destination_key] = value
                break
        state_dict._metadata = metadata

    return state_dict


def source_epoch(checkpoint: Mapping, override: int | None) -> int:
    if override is not None:
        return override

    meta = checkpoint.get("meta")
    epoch = meta.get("epoch") if isinstance(meta, Mapping) else None
    if not isinstance(epoch, int):
        raise ValueError(
            "Source checkpoint has no integer meta.epoch; pass --source-epoch"
        )
    return epoch


def validate_export(
    path: Path,
    source_state_dict: Mapping[str, torch.Tensor],
    expected_meta: Mapping,
) -> None:
    exported = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(exported, Mapping):
        raise ValueError("Exported checkpoint is not a mapping")
    if list(exported) != ["state_dict", "meta"]:
        raise ValueError(
            f"Unexpected exported fields: {list(exported)}"
        )
    if exported["meta"] != expected_meta:
        raise ValueError(
            f"Unexpected exported metadata: {exported['meta']!r}"
        )

    exported_state_dict = exported["state_dict"]
    if list(exported_state_dict) != list(source_state_dict):
        raise ValueError("Exported state_dict keys or key order differ")
    if getattr(exported_state_dict, "_metadata", None) != getattr(
        source_state_dict, "_metadata", None
    ):
        raise ValueError("Exported state_dict metadata differs")

    for key, source_value in source_state_dict.items():
        exported_value = exported_state_dict[key]
        if exported_value.shape != source_value.shape:
            raise ValueError(f"Shape differs for {key}")
        if exported_value.dtype != source_value.dtype:
            raise ValueError(f"Dtype differs for {key}")
        if not torch.equal(exported_value, source_value):
            raise ValueError(f"Tensor values differ for {key}")


def export_checkpoint(
    source: Path,
    destination: Path,
    model_name: str,
    epoch_override: int | None = None,
    force: bool = False,
    expected_sha256: str | None = None,
    archive_name: str | None = None,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must be different files")
    if destination.exists() and not force:
        raise FileExistsError(
            f"Destination already exists: {destination}; use --force to overwrite"
        )
    if not model_name.strip():
        raise ValueError("Model name must not be empty")

    save_name = archive_name or destination.name
    if not save_name or Path(save_name).name != save_name:
        raise ValueError("Archive name must be a filename without directories")

    checkpoint = load_training_checkpoint(source)
    state_dict = extract_state_dict(checkpoint)
    source_entry_count = len(checkpoint["state_dict"])
    epoch = source_epoch(checkpoint, epoch_override)
    meta = {"model": model_name, "source_epoch": epoch}
    release = {"state_dict": state_dict, "meta": meta}

    removed_fields = [key for key in checkpoint if key != "state_dict"]
    del checkpoint

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temp_dir:
        temp_path = Path(temp_dir) / save_name
        torch.save(release, temp_path)
        validate_export(temp_path, state_dict, meta)

        digest = sha256(temp_path)
        if (
            expected_sha256 is not None
            and digest.lower() != expected_sha256.lower()
        ):
            raise ValueError(
                f"SHA-256 mismatch: expected {expected_sha256}, got {digest}"
            )
        os.replace(temp_path, destination)

    tensor_count = len(state_dict)
    value_count = sum(value.numel() for value in state_dict.values())
    print(f"source: {source}")
    print(f"discarded source fields: {', '.join(map(str, removed_fields))}")
    print(
        "state_dict mapping: "
        "student_backbone.* -> backbone.*, student_neck.* -> neck.*, "
        "student_head.* -> head.*"
    )
    print(f"discarded state_dict entries: {source_entry_count - tensor_count}")
    print(f"saved: {destination}")
    print(f"meta: {meta}")
    print(f"state_dict tensors: {tensor_count}")
    print(f"state_dict values: {value_count}")
    print(f"size: {destination.stat().st_size}")
    print(f"sha256: {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the renamed student backbone, neck, and head with minimal "
            "metadata from a trusted MMEngine training checkpoint."
        )
    )
    parser.add_argument("source", type=Path, help="Trusted training checkpoint")
    parser.add_argument("destination", type=Path, help="Release checkpoint")
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model identifier stored in release metadata",
    )
    parser.add_argument(
        "--source-epoch",
        type=int,
        default=None,
        help="Override source meta.epoch",
    )
    parser.add_argument(
        "--archive-name",
        default=None,
        help="Internal torch archive filename for byte-level reproduction",
    )
    parser.add_argument(
        "--expected-sha256",
        default=None,
        help="Fail unless the exported file has this SHA-256",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite destination")
    parser.add_argument(
        "--trust-source",
        action="store_true",
        help="Confirm that arbitrary Python objects in the source are trusted",
    )
    args = parser.parse_args()

    if not args.trust_source:
        parser.error(
            "--trust-source is required because training checkpoints may "
            "contain arbitrary pickled Python objects"
        )

    export_checkpoint(
        source=args.source,
        destination=args.destination,
        model_name=args.model_name,
        epoch_override=args.source_epoch,
        force=args.force,
        expected_sha256=args.expected_sha256,
        archive_name=args.archive_name,
    )


if __name__ == "__main__":
    main()
