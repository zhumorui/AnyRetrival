import os

import einops as ein
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.model.dinov2extractor import DinoV2FeatureExtractor
from src.utils.dataset import configdataset
from src.utils.download import download_datasets
from src.utils.evaluate import compute_map

def gem_pooling(x, p=3, eps=1e-6):
    """
    GeM pooling implementation.
    Args:
        x (torch.Tensor): Input tensor of shape (H, W, D).
        p (float): GeM pooling parameter (default: 3).
        eps (float): Small value to avoid numerical issues (default: 1e-6).
    Returns:
        torch.Tensor: Pooled feature of shape (D,).
    """
    return torch.pow(torch.mean(torch.pow(x.clamp(min=eps), p), dim=(0, 1)), 1.0 / p)

def process_image(img_path, extractor, device, q=3):
    """
    Processes a single image and applies GeM pooling to extract features.
    Args:
        img_path (str): Path to the input image.
        extractor: Feature extractor.
        device (torch.device): Device to run the computation.
        q (float): GeM pooling parameter (default: 3).
    Returns:
        torch.Tensor: Extracted and pooled feature vector.
    """
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(img)[None, ...].to(device)
    features = extractor(img_tensor).cpu()
    patches_h, patches_w = img_tensor.shape[-2] // 14, img_tensor.shape[-1] // 14
    reshaped_features = ein.rearrange(
        features[0], "(h w) d -> h w d", h=patches_h, w=patches_w
    )
    # Apply GeM pooling
    return gem_pooling(reshaped_features, p=q)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = DinoV2FeatureExtractor(
    model_type="dinov2_vitg14", layer=31, facet="value", device=device
)


def extract_features_with_dino(image_paths):
    features_list = []
    for image_path in tqdm(image_paths, desc="Extracting Features"):
        features = process_image(image_path, extractor, device=device).numpy()
        features /= np.linalg.norm(features, keepdims=True)
        features_list.append(features)
    return np.vstack(features_list)


data_root = "data"

download_datasets(data_root)

test_dataset = "rparis6k"

cfg = configdataset(test_dataset, os.path.join(data_root, "datasets"))

query_images = [
    os.path.join(data_root, "datasets", test_dataset, "jpg", f + ".jpg")
    for f in cfg["qimlist"]
]
database_images = [
    os.path.join(data_root, "datasets", test_dataset, "jpg", f + ".jpg")
    for f in cfg["imlist"]
]

print(f">> {test_dataset}: Extracting query features...")
Q = extract_features_with_dino(query_images).T

print(f">> {test_dataset}: Extracting database features...")
X = extract_features_with_dino(database_images).T

print(f">> {test_dataset}: Retrieval...")
sim = np.dot(X.T, Q)
ranks = np.argsort(-sim, axis=0)

gnd = cfg["gnd"]

ks = [1, 5, 10]

gnd_t = []
for i in range(len(gnd)):
    g = {
        "ok": np.concatenate([gnd[i]["easy"]]),
        "junk": np.concatenate([gnd[i]["junk"], gnd[i]["hard"]]),
    }
    gnd_t.append(g)
mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)

gnd_t = []
for i in range(len(gnd)):
    g = {
        "ok": np.concatenate([gnd[i]["easy"], gnd[i]["hard"]]),
        "junk": np.concatenate([gnd[i]["junk"]]),
    }
    gnd_t.append(g)
mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

gnd_t = []
for i in range(len(gnd)):
    g = {
        "ok": np.concatenate([gnd[i]["hard"]]),
        "junk": np.concatenate([gnd[i]["junk"], gnd[i]["easy"]]),
    }
    gnd_t.append(g)
mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

print(
    f">> {test_dataset}: mAP E: {np.around(mapE*100, decimals=2)}, M: {np.around(mapM*100, decimals=2)}, H: {np.around(mapH*100, decimals=2)}"
)
print(
    f">> {test_dataset}: mP@k{ks} E: {np.around(mprE*100, decimals=2)}, M: {np.around(mprM*100, decimals=2)}, H: {np.around(mprH*100, decimals=2)}"
)
