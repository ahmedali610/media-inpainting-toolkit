import torch
import lpips
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import numpy as np
import os
import cv2

# Initialize LPIPS model globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net='alex').to(device)


# Preprocessing function for LPIPS
def load_image(path, size=(256, 256)):
    image = Image.open(path).convert('RGB').resize(size)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(image).unsqueeze(0)

# SSIM computation (grayscale)
def compute_ssim(img_path1, img_path2, size=(256, 256)):
    img1 = Image.open(img_path1).convert('L').resize(size)
    img2 = Image.open(img_path2).convert('L').resize(size)
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    return ssim(img1_np, img2_np)

# Evaluate all inpainted images vs originals
def evaluate_folder(original_dir, inpainted_dir):
    lpips_scores = []
    ssim_scores = []

    original_files = sorted([f for f in os.listdir(original_dir) if f.endswith(('.png', '.jpg'))])
    for file in original_files:
        orig_path = os.path.join(original_dir, file)
        inpaint_path = os.path.join(inpainted_dir, file)

        if not os.path.exists(inpaint_path):
            print(f"Missing inpainted image for: {file}")
            continue

        # Load and evaluate LPIPS
        img1 = load_image(orig_path).to(device)
        img2 = load_image(inpaint_path).to(device)

        lpips_score = lpips_model(img1, img2).item()

        # Evaluate SSIM
        ssim_score = compute_ssim(orig_path, inpaint_path)

        lpips_scores.append(lpips_score)
        ssim_scores.append(ssim_score)

        print(f"{file}: LPIPS = {lpips_score:.4f}, SSIM = {ssim_score:.4f}")

    avg_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0
    avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0
    print(f"\nAverage LPIPS: {avg_lpips:.4f}")
    print(f"Average SSIM : {avg_ssim:.4f}")



def evaluate_video_quality(original_video_path, inpainted_video_path):
    """Evaluates the quality of inpainted video using SSIM and LPIPS."""

    def preprocess_frame(frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)  # LPIPS expects input in [-1, 1]
        ])
        return transform(image).unsqueeze(0)

    def load_video(path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    original_frames = load_video(original_video_path)
    inpainted_frames = load_video(inpainted_video_path)

    if len(original_frames) != len(inpainted_frames):
        raise ValueError("Videos must have the same number of frames")

    ssim_scores = []
    lpips_scores = []

    for orig_frame, inpaint_frame in zip(original_frames, inpainted_frames):
        orig_resized = cv2.resize(orig_frame, (256, 256))
        inpaint_resized = cv2.resize(inpaint_frame, (256, 256))

        # SSIM
        orig_gray = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2GRAY)
        inpaint_gray = cv2.cvtColor(inpaint_resized, cv2.COLOR_BGR2GRAY)
        ssim_value = ssim(orig_gray, inpaint_gray)
        ssim_scores.append(ssim_value)

        # LPIPS
        orig_tensor = preprocess_frame(orig_resized).to(device)
        inpaint_tensor = preprocess_frame(inpaint_resized).to(device)
        lpips_value = lpips_model(orig_tensor, inpaint_tensor)
        lpips_scores.append(lpips_value.item())

    avg_ssim = np.mean(ssim_scores)
    avg_lpips = np.mean(lpips_scores)

    print(f"\n📊 Evaluation Results (Video):")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
