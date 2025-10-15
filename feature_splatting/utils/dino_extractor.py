import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm

# Load DINOv2 model
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').eval()

# Image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def load_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img, transform(img).unsqueeze(0)

def extract_feature_map(img_tensor):
    with torch.no_grad():
        feats = dinov2.forward_features(img_tensor)['x_norm_patchtokens']  # [1, 196, C]
    return feats

def visualize_feature_overlay(orig_img, feats, save_path=None):
    # feats: [1, N_tokens, C]
    token_feats = feats[0].mean(dim=-1)  # shape: [N_tokens]

    # Get square size
    N = token_feats.shape[0]
    side_len = int(N ** 0.5)
    if side_len * side_len != N:
        raise ValueError(f"Token count {N} is not a perfect square")

    feat_map = token_feats.reshape(side_len, side_len)
    feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
    feat_map = cv2.resize(feat_map.cpu().numpy(), (orig_img.shape[1], orig_img.shape[0]))

    plt.imshow(orig_img)
    plt.imshow(feat_map, cmap='jet', alpha=0.5)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

# Set paths
img_dir = 'input/path/dir'
out_dir = 'output/path/dir'
os.makedirs(out_dir, exist_ok=True)

# Process each image
for fname in tqdm(os.listdir(img_dir)):
    if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(img_dir, fname)
        img, img_tensor = load_image(img_path)
        feats = extract_feature_map(img_tensor)
        save_path = os.path.join(out_dir, fname)
        visualize_feature_overlay(img, feats, save_path)