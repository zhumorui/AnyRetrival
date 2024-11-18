import os
from typing import Literal

import einops as ein
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms
from tqdm import tqdm

_DINO_V2_MODELS = Literal[
    "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"
]
_DINO_FACETS = Literal["query", "key", "value", "token"]


class DinoV2FeatureExtractor:
    """
    Extracts features from intermediate layers in Dino-v2 models.
    """

    def __init__(
        self,
        model_type: _DINO_V2_MODELS,
        layer: int,
        facet: _DINO_FACETS = "token",
        use_cls: bool = False,
        normalize: bool = True,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = (
            torch.hub.load("facebookresearch/dinov2", model_type).eval().to(self.device)
        )
        self.layer = layer
        self.facet = facet
        self.use_cls = use_cls
        self.normalize = normalize
        self._hook_output = None
        self._register_hook()

    def _register_hook(self):
        """Registers a forward hook based on the selected facet."""
        if self.facet == "token":
            self.hook = self.model.blocks[self.layer].register_forward_hook(
                self._hook_fn
            )
        else:
            self.hook = self.model.blocks[self.layer].attn.qkv.register_forward_hook(
                self._hook_fn
            )

    def _hook_fn(self, module, inputs, output):
        self._hook_output = output

    def set_facet(self, facet: _DINO_FACETS):
        """Updates the facet and re-registers the hook."""
        self.hook.remove()
        self.facet = facet
        self._register_hook()

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Extracts features for the input image tensor."""
        with torch.no_grad():
            _ = self.model(img)
            features = (
                self._hook_output[:, 1:, ...] if not self.use_cls else self._hook_output
            )

            if self.facet in ["query", "key", "value"]:
                dim = features.shape[2] // 3
                idx = {"query": 0, "key": 1, "value": 2}[self.facet]
                features = features[:, :, idx * dim : (idx + 1) * dim]

            if self.normalize:
                features = F.normalize(features, dim=-1)

        self._hook_output = None  # Reset for the next call
        return features

    def __del__(self):
        self.hook.remove()


def process_image(
    img_path: str, extractor, device: str = "cuda", resize_to: tuple = (224, 224)
) -> torch.Tensor:
    """Processes a single image, resizes it to a fixed size, and extracts its features."""
    # 打开图像并转换为 PIL.Image
    img = Image.open(img_path).convert("RGB")  # 确保转换为 RGB 格式
    transform = transforms.Compose(
        [
            transforms.Resize(resize_to),  # 将图片调整到指定大小
            transforms.ToTensor(),
            transforms.CenterCrop(
                [(resize_to[0] // 14) * 14, (resize_to[1] // 14) * 14]
            ),  # 保证裁剪后大小可以被14整除
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # 应用 transforms
    img_tensor = transform(img)[None, ...].to(device)
    features = extractor(img_tensor).cpu()
    patches_h, patches_w = img_tensor.shape[-2] // 14, img_tensor.shape[-1] // 14

    return ein.rearrange(features[0], "(h w) d -> h w d", h=patches_h, w=patches_w)


def extract_folder_features(
    folder: str,
    model_type: _DINO_V2_MODELS = "dinov2_vitg14",
    layer: int = 31,
    facet: _DINO_FACETS = "value",
    device: str = "cuda",
    limit: int = None,
) -> torch.Tensor:
    """Extracts features from all images in a folder."""
    extractor = DinoV2FeatureExtractor(model_type, layer, facet=facet, device=device)
    image_files = [
        f
        for f in sorted(os.listdir(folder))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if limit:
        image_files = image_files[:limit]

    features = []
    for img_file in tqdm(image_files, desc="Extracting features", unit="image"):
        img_path = os.path.join(folder, img_file)
        features.append(process_image(img_path, extractor, device=device))
        print(features[-1].shape)

    return torch.stack(features)


def save_features(
    folder: str,
    output_path: str,
    model_type: _DINO_V2_MODELS = "dinov2_vitg14",
    layer: int = 31,
    facet: _DINO_FACETS = "value",
    device: str = "cuda",
    limit: int = None,
):
    """Extracts and saves features from a folder of images."""
    features = extract_folder_features(
        folder,
        model_type=model_type,
        layer=layer,
        facet=facet,
        device=device,
        limit=limit,
    )
    torch.save(features, output_path)
    print(f"Features saved with shape {features.shape} to '{output_path}'")


def main():
    """Main entry point for the feature extraction pipeline."""
    folder = "data/datasets/roxford5k/jpg/"  # Input folder
    output_path = "features.pt"  # Output file
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_features(folder, output_path, facet="query", device=device, limit=20)


if __name__ == "__main__":
    main()
