import torch
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import einops as ein
from src.model.dinov2extractor import DinoV2FeatureExtractor
from torchvision import transforms
from matplotlib.patches import ConnectionPatch
import random

@torch.no_grad()
def get_sims(src_img, tgt_img, pix_loc, model_type="dinov2_vitg14", 
             layer=31, device="cuda"):
    sim_res = {}
    
    # Calculate patch sizes based on model type
    patch_size = 14 if "14" in model_type else 16
    
    # Function to make dimensions divisible by patch_size
    def pad_to_patch_size(img, patch_size):
        h, w = img.shape[:2]
        new_h = ((h + patch_size - 1) // patch_size) * patch_size
        new_w = ((w + patch_size - 1) // patch_size) * patch_size
        pad_h = new_h - h
        pad_w = new_w - w
        
        # Create padded image
        padded_img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 
                           mode='constant', constant_values=0)
        return padded_img
    
    # Pad images if needed
    src_img_padded = pad_to_patch_size(src_img, patch_size)
    tgt_img_padded = pad_to_patch_size(tgt_img, patch_size)
    
    # Create transformations for both images
    def create_transform(img):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    tf_src = create_transform(src_img_padded)
    tf_tgt = create_transform(tgt_img_padded)
    
    # Transform images
    src_pt = tf_src(src_img_padded)[None, ...].to(device)
    tgt_pt = tf_tgt(tgt_img_padded)[None, ...].to(device)
    
    # Initialize DinoV2
    dino = DinoV2FeatureExtractor(model_type, layer, device=device)
    
    for facet in ["key", "query", "token", "value"]:
        dino.set_facet(facet)
        res_s = dino(src_pt).cpu()
        res_t = dino(tgt_pt).cpu()
        
        # Calculate feature map dimensions
        src_p_h = src_pt.shape[-2] // patch_size
        src_p_w = src_pt.shape[-1] // patch_size
        tgt_p_h = tgt_pt.shape[-2] // patch_size
        tgt_p_w = tgt_pt.shape[-1] // patch_size
        
        # Reshape and interpolate source features
        res_s_img = ein.rearrange(res_s[0], "(p_h p_w) d -> d p_h p_w", 
                                 p_h=src_p_h, p_w=src_p_w)[None, ...]
        res_s_img = F.interpolate(res_s_img, mode='nearest',
                                 size=(src_img.shape[0], src_img.shape[1]))
        
        # Reshape and interpolate target features
        res_t_img = ein.rearrange(res_t[0], "(p_h p_w) d -> d p_h p_w", 
                                 p_h=tgt_p_h, p_w=tgt_p_w)[None, ...]
        res_t_img = F.interpolate(res_t_img, mode='nearest',
                                 size=(tgt_img.shape[0], tgt_img.shape[1]))
        
        # Extract features at the specified pixel location
        s_pix = res_s_img[[0], ..., pix_loc[1], pix_loc[0]]
        s_pix = ein.repeat(s_pix, "1 d -> 1 d h w", 
                          h=tgt_img.shape[0], w=tgt_img.shape[1])
        
        # Calculate similarity
        sim = F.cosine_similarity(res_t_img, s_pix, dim=1)
        sim_res[facet] = sim[0].cpu().numpy()
    
    return sim_res

def sample_random_points(img, n_points=5, margin=50):
    height, width = img.shape[:2]
    points = []
    
    x_min, x_max = margin, width - margin
    y_min, y_max = margin, height - margin
    
    while len(points) < n_points:
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        
        is_far_enough = True
        for px, py in points:
            if np.sqrt((x - px)**2 + (y - py)**2) < margin:
                is_far_enough = False
                break
                
        if is_far_enough:
            points.append((x, y))
    
    return points

def find_top_matches(similarity_map, n_top=5, min_distance=50):
    """
    Find top N matching points in the similarity map with minimum distance constraint.
    """
    # Convert to numpy array if it's not already
    sim_map = np.array(similarity_map)
    
    # Initialize list for top matches
    top_matches = []
    
    # Create a copy of the similarity map
    temp_map = sim_map.copy()
    
    while len(top_matches) < n_top:
        # Find the maximum value in the temporary map
        max_val = np.max(temp_map)
        if max_val <= 0:  # No more valid matches
            break
            
        # Find the coordinates of the maximum value
        y, x = np.unravel_index(np.argmax(temp_map), temp_map.shape)
        
        # Check if this point is far enough from existing points
        valid_point = True
        for prev_y, prev_x in top_matches:
            if np.sqrt((y - prev_y)**2 + (x - prev_x)**2) < min_distance:
                valid_point = False
                break
                
        if valid_point:
            top_matches.append((y, x))
            
        # Zero out the region around the current maximum
        y_min = max(0, y - min_distance//2)
        y_max = min(temp_map.shape[0], y + min_distance//2)
        x_min = max(0, x - min_distance//2)
        x_max = min(temp_map.shape[1], x + min_distance//2)
        temp_map[y_min:y_max, x_min:x_max] = 0
        
    return top_matches

def visualize_and_save_multi(src_img, tgt_img, sample_points, sims_list, 
                           save_path="sparse_matching.png", n_matches=1, 
                           match_type="key"):

    fig = plt.figure(figsize=(15, 5))
    
    ax1 = plt.subplot(131)
    ax1.imshow(src_img)

    for idx, pix_loc in enumerate(sample_points):
        ax1.plot(pix_loc[0], pix_loc[1], 'ro', markersize=8, 
                label=f'Source {idx+1}')
    ax1.set_title("Source Image", pad=10)
    ax1.axis('off')
    ax1.legend()
    
    ax2 = plt.subplot(132)
    ax2.imshow(tgt_img)
    ax2.set_title("Target Image", pad=10)
    ax2.axis('off')
    
    ax3 = plt.subplot(133)
    sim = sims_list[-1][match_type]
    im = ax3.imshow(sim, cmap="jet", vmin=0, vmax=1)
    ax3.set_title(f"{match_type.capitalize()} Similarity", pad=10)
    ax3.axis('off')
    plt.colorbar(im, ax=ax3)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(sample_points)))
    for idx, (pix_loc, sims) in enumerate(zip(sample_points, sims_list)):
        sim = sims[match_type]
        
        top_matches = find_top_matches(sim, n_top=n_matches)
        
        for match_idx, (y, x) in enumerate(top_matches):
            ax2.plot(x, y, 'o', color=colors[idx], markersize=8, 
                    label=f'Match {idx+1}' if match_idx == 0 else "")
            ax3.plot(x, y, 'o', color=colors[idx], markersize=8)
            
            con = ConnectionPatch(
                xyA=(pix_loc[0], pix_loc[1]),  
                xyB=(x, y),  
                coordsA="data", coordsB="data",
                axesA=ax1, axesB=ax2,
                color=colors[idx], linestyle='--', alpha=0.6
            )
            fig.add_artist(con)
    
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def main():
    src_img = np.array(Image.open("data/datasets/roxford5k/jpg/all_souls_000015.jpg"))
    tgt_img = np.array(Image.open("data/datasets/roxford5k/jpg/all_souls_000126.jpg"))
    
    sample_points = sample_random_points(src_img, n_points=6, margin=50)
    
    sims_list = []
    for point in sample_points:
        sims = get_sims(src_img, tgt_img, point, layer=31, model_type="dinov2_vitg14")
        sims_list.append(sims)
    
    visualize_and_save_multi(src_img, tgt_img, sample_points, sims_list, 
                           n_matches=1, match_type="value")

if __name__ == "__main__":
    main()