import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import einops as ein
from src.utils.dataset import configdataset
from src.utils.download import download_datasets
from src.utils.evaluate import compute_map
from src.model.dinov2extractor import DinoV2FeatureExtractor

def process_image(img_path, extractor, device):
    """Processes image and extracts features."""

    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img)[None, ...].to(device)
    features = extractor(img_tensor).cpu()
    patches_h, patches_w = img_tensor.shape[-2] // 14, img_tensor.shape[-1] // 14
    return ein.rearrange(features[0], "(h w) d -> h w d", h=patches_h, w=patches_w).flatten(1).mean(dim=0)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 提取图像特征的函数
extractor = DinoV2FeatureExtractor(model_type="dinov2_vits14", layer=11, facet="value", device=device)

def extract_features_with_dino(image_paths):
    """利用DINOv2模型提取特征"""
    features_list = []
    for image_path in tqdm(image_paths, desc="Extracting Features"):
        features = process_image(image_path, extractor, device=device).numpy()
        # 归一化特征向量
        features /= np.linalg.norm(features, keepdims=True)
        features_list.append(features)
    return np.vstack(features_list)

#---------------------------------------------------------------------
# Set data folder and testing parameters
#---------------------------------------------------------------------
# 设置数据文件夹，修改为适合你系统的路径
data_root = ('data')

# 检查并下载数据集和相关文件
download_datasets(data_root)

# 设置测试数据集: roxford5k | rparis6k
test_dataset = 'rparis6k'

#---------------------------------------------------------------------
# Evaluate
#---------------------------------------------------------------------
print('>> {}: Evaluating test dataset...'.format(test_dataset)) 

# 加载数据集配置
cfg = configdataset(test_dataset, os.path.join(data_root, 'datasets'))

# 加载查询图像和数据库图像的路径
query_images = [os.path.join(data_root, 'datasets', test_dataset, 'jpg', f + '.jpg') for f in cfg['qimlist']]
database_images = [os.path.join(data_root, 'datasets', test_dataset, 'jpg', f+'.jpg') for f in cfg['imlist']]

# 提取查询图像和数据库图像的特征
print('>> {}: Extracting query features...'.format(test_dataset))
Q = extract_features_with_dino(query_images).T  # 转置以适应相似性计算

print('>> {}: Extracting database features...'.format(test_dataset))
X = extract_features_with_dino(database_images).T  # 转置以适应相似性计算

# 执行检索
print('>> {}: Retrieval...'.format(test_dataset))
sim = np.dot(X.T, Q)  # 计算相似度
ranks = np.argsort(-sim, axis=0)  # 根据相似度进行排序

# 加载地面真值
gnd = cfg['gnd']

# 评估排名
ks = [1, 5, 10]

# Easy难度的评估
gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['easy']])
    g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
    gnd_t.append(g)
mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)

# Medium难度的评估
gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
    g['junk'] = np.concatenate([gnd[i]['junk']])
    gnd_t.append(g)
mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

# Hard难度的评估
gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['hard']])
    g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
    gnd_t.append(g)
mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

# 输出评估结果
print('>> {}: mAP E: {}, M: {}, H: {}'.format(test_dataset, np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(test_dataset, np.array(ks), np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))
