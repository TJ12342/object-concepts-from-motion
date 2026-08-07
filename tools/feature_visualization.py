from pathlib import Path
import argparse
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))


def load_model(checkpoint_path=None, device="cuda"):
    from pytorch_model import FlowSegModel

    checkpoint_path = checkpoint_path or REPO_ROOT / "checkpoints" / "swin_h.pth"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Build on meta to avoid allocating a second 3 GB copy before moving to GPU.
    with torch.device("meta"):
        model = FlowSegModel()
    model.load_state_dict(state_dict, strict=True, assign=True)
    model = model.to(device).eval()
    return model


def center_aspect_resize(image_rgb, out_wh=(512, 288)):
    """Center-crop to the target aspect ratio, then resize without distortion."""
    if len(out_wh) != 2:
        raise ValueError(f"out_wh must contain width and height, got {out_wh}")
    out_w, out_h = (int(value) for value in out_wh)
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"out_wh values must be positive, got {out_wh}")

    height, width = image_rgb.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Input image has an invalid spatial size")

    target_aspect = out_w / out_h
    source_aspect = width / height
    if source_aspect >= target_aspect:
        crop_h = height
        crop_w = int(round(height * target_aspect))
    else:
        crop_w = width
        crop_h = int(round(width / target_aspect))

    x1 = max((width - crop_w) // 2, 0)
    y1 = max((height - crop_h) // 2, 0)
    cropped = image_rgb[y1:y1 + crop_h, x1:x1 + crop_w]

    scale = max(out_w / crop_w, out_h / crop_h)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(cropped, (out_w, out_h), interpolation=interpolation)


def extract_feature_map(model, image_path, device="cuda", size=(512, 288)):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = center_aspect_resize(image_rgb, size)

    # Runner injects the top-level SelfSupDataPreprocessor from the reference
    # config, which normalizes RGB values before the model forward.
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    tensor = torch.from_numpy(
        (image_rgb.astype(np.float32) - mean) / std
    ).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device)

    with torch.inference_mode():
        features = model(tensor)
        features = torch.nn.functional.interpolate(features, scale_factor=2)
        features = torch.nn.functional.normalize(features.float(), dim=1)
    return image_rgb, features[0].cpu()


def pca_rgb(feature_map):
    channels, height, width = feature_map.shape
    flat = feature_map.numpy().reshape(channels, -1).T
    projected = PCA(
        n_components=3, svd_solver="randomized", random_state=42
    ).fit_transform(flat)
    projected = projected.reshape(height, width, 3)
    lo = projected.min()
    hi = projected.max()
    projected = (projected - lo) / max(hi - lo, 1e-8)
    projected = cv2.resize(projected, (width * 2, height * 2))
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
