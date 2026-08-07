from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))


def load_model(checkpoint_path=None, device="cuda"):
    from pytorch_model import FlowSegModel

    checkpoint_path = checkpoint_path or REPO_ROOT / "checkpoints" / "flowseg_421m_swinh.pth"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Build on meta to avoid allocating a second 3 GB copy before moving to GPU.
    with torch.device("meta"):
        model = FlowSegModel()
    model.load_state_dict(state_dict, strict=True, assign=True)
    model = model.to(device).eval()
    return model


def extract_feature_map(model, image_path, device="cuda", size=(512, 288)):
    image = Image.open(image_path).convert("RGB")
    resampling = getattr(Image, "Resampling", Image)
    image = image.resize(size, resampling.BILINEAR)
    # Match the RGB input convention used by the released inference pipeline.
    image_rgb = np.asarray(image, dtype=np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    tensor = torch.from_numpy((image_rgb - mean) / std)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.inference_mode():
        features = model(tensor)
        features = torch.nn.functional.interpolate(
            features, scale_factor=2, mode="bilinear", align_corners=False)
        features = torch.nn.functional.normalize(features.float(), dim=1)
    return image_rgb.astype(np.uint8), features[0].cpu()


def pca_rgb(feature_map):
    channels, height, width = feature_map.shape
    flat = feature_map.numpy().reshape(channels, -1).T
    flat = flat - flat.mean(axis=0, keepdims=True)
    left, singular_values, right = np.linalg.svd(flat, full_matrices=False)
    signs = np.sign(right[np.arange(3), np.abs(right[:3]).argmax(axis=1)])
    signs[signs == 0] = 1
    projected = left[:, :3] * singular_values[:3] * signs
    projected = projected.reshape(height, width, 3)
    lo = projected.min(axis=(0, 1), keepdims=True)
    hi = projected.max(axis=(0, 1), keepdims=True)
    projected = (projected - lo) / np.maximum(hi - lo, 1e-8)
    return (projected * 255).clip(0, 255).astype(np.uint8)


def show_feature_map(model, image_path, device="cuda"):
    image, features = extract_feature_map(model, image_path, device=device)
    visualization = pca_rgb(features)
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[1].imshow(visualization)
    axes[1].set_title("PCA feature map")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    return image, visualization


def main():
    parser = argparse.ArgumentParser(
        description="Run one image through the released model and save PCA RGB.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output = args.output or args.image.with_name(f"{args.image.stem}_pca.png")
    model = load_model(args.checkpoint, device=args.device)
    _, feature = extract_feature_map(model, args.image, device=args.device)
    visualization = pca_rgb(feature)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visualization).save(output)
    print(f"feature: {tuple(feature.shape)}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
