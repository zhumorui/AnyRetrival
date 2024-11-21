import os
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm

from src.utils.dataset import configdataset
from src.utils.download import download_datasets
from src.utils.evaluate import compute_map

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Function to extract image features
def extract_features(image_paths):
    features_list = []
    for image_path in tqdm(image_paths, desc="Extracting Features"):
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image).cpu().numpy()
        # Normalize features
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        features_list.append(features)
    return np.vstack(features_list)

# Set data folder
data_root = "data"

# Download datasets
download_datasets(data_root)

# Set test dataset
test_dataset = "roxford5k"

# Load dataset configuration
cfg = configdataset(test_dataset, os.path.join(data_root, "datasets"))

# Load query and database image paths
query_images = [
    os.path.join(data_root, "datasets", test_dataset, "jpg", f + ".jpg")
    for f in cfg["qimlist"]
]
database_images = [
    os.path.join(data_root, "datasets", test_dataset, "jpg", f + ".jpg")
    for f in cfg["imlist"]
]

# Extract features
print(f">> {test_dataset}: Extracting query features...")
Q = extract_features(query_images).T

print(f">> {test_dataset}: Extracting database features...")
X = extract_features(database_images).T

# Perform retrieval
print(f">> {test_dataset}: Retrieval...")
sim = np.dot(X.T, Q)
ranks = np.argsort(-sim, axis=0)

# Load ground truth
gnd = cfg["gnd"]

# Evaluate rankings
ks = [1, 5, 10]

# Easy level evaluation
gnd_t = []
for i in range(len(gnd)):
    g = {
        "ok": np.concatenate([gnd[i]["easy"]]),
        "junk": np.concatenate([gnd[i]["junk"], gnd[i]["hard"]]),
    }
    gnd_t.append(g)
mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)

# Medium level evaluation
gnd_t = []
for i in range(len(gnd)):
    g = {
        "ok": np.concatenate([gnd[i]["easy"], gnd[i]["hard"]]),
        "junk": np.concatenate([gnd[i]["junk"]]),
    }
    gnd_t.append(g)
mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

# Hard level evaluation
gnd_t = []
for i in range(len(gnd)):
    g = {
        "ok": np.concatenate([gnd[i]["hard"]]),
        "junk": np.concatenate([gnd[i]["junk"], gnd[i]["easy"]]),
    }
    gnd_t.append(g)
mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

# Print results
print(
    f">> {test_dataset}: mAP E: {np.around(mapE*100, decimals=2)}, M: {np.around(mapM*100, decimals=2)}, H: {np.around(mapH*100, decimals=2)}"
)
print(
    f">> {test_dataset}: mP@k{ks} E: {np.around(mprE*100, decimals=2)}, M: {np.around(mprM*100, decimals=2)}, H: {np.around(mprH*100, decimals=2)}"
)