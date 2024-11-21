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

def visualize_and_save(src_img, tgt_img, pix_loc, sims, save_path="sim_map.png", n_matches=1, match_type="key"):
    # Calculate the figure size based on image aspect ratios
    base_height = 4
    src_aspect = src_img.shape[1] / src_img.shape[0]
    tgt_aspect = tgt_img.shape[1] / tgt_img.shape[0]
    total_width = base_height * (src_aspect + tgt_aspect + len(sims))
    
    fig, axes = plt.subplots(1, 2 + len(sims), figsize=(total_width, base_height))
    
    # Source image
    axes[0].imshow(src_img)
    axes[0].plot(pix_loc[0], pix_loc[1], 'ro', markersize=8)
    axes[0].set_title("Source Image", pad=10)
    axes[0].axis('off')
    
    # Target image
    axes[1].imshow(tgt_img)
    axes[1].set_title("Target Image", pad=10)
    axes[1].axis('off')
    
    # Process each similarity map
    for i, (facet, sim) in enumerate(sims.items(), 2):
        # Display similarity map
        im = axes[i].imshow(sim, cmap="jet", vmin=0, vmax=1)
        axes[i].set_title(f"{facet.capitalize()}", pad=10)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])
        
        # Only find and plot matches for the specified facet type
        if facet == match_type:
            # Find and plot top matching points
            top_matches = find_top_matches(sim, n_top=n_matches)
            
            # Plot matches on target image
            for idx, (y, x) in enumerate(top_matches):
                # Plot point on target image
                axes[1].plot(x, y, 'go', markersize=8)
                
                # Create connection patch
                con = ConnectionPatch(
                    xyA=(pix_loc[0], pix_loc[1]), # source point
                    xyB=(x, y), # target point
                    coordsA="data", coordsB="data",
                    axesA=axes[0], axesB=axes[1],
                    color='g', linestyle='--', alpha=0.6
                )
                fig.add_artist(con)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

# Example usage
def main():
    # Load images of different sizes
    src_img = np.array(Image.open("data/datasets/roxford5k/jpg/all_souls_000015.jpg"))
    tgt_img = np.array(Image.open("data/datasets/roxford5k/jpg/all_souls_000126.jpg"))
    
    pix_loc = (700, 150)  # Make sure this point exists in your source image
    
    # Get similarity maps and visualize
    sims = get_sims(src_img, tgt_img, pix_loc, layer=8, model_type="dinov2_vits14")
    visualize_and_save(src_img, tgt_img, pix_loc, sims, n_matches=1, match_type="key")

if __name__ == "__main__":
    main()