# Media Inpainting Toolkit

A versatile Python toolkit for image and video inpainting and object removal using [LAMA](https://github.com/saic-mdal/lama) and [XMem](https://github.com/hkchengrex/XMem).

---

## Description
Media Inpainting Toolkit is a command-line Python application that enables seamless removal of unwanted objects from images and videos by leveraging COCO-format masks, state-of-the-art LAMA image inpainting, and XMem video mask propagation.

---

## Table of Contents

1. [Features](#features)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
3. [Usage](#usage)
   - [A. Image Inpainting](#a-image-inpainting)
   - [B. Video Inpainting](#b-video-inpainting)
4. [Project Structure](#project-structure)
5. [Models and Dependencies](#models-and-dependencies)
6. [Test-Results](#Test-Results)
7. [Evaluation](#Evaluation)

---

## Features

- **Image Inpainting**: Remove objects from images using LAMA with COCO-style pixel masks.
- **Video Inpainting**: Remove objects from videos by propagating an initial COCO mask through frames via XMem and inpainting each frame with LAMA.
- Easy CLI-based interface and configurable paths
- Supports batch processing and common video formats (MP4, AVI, etc.)

---

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended) or CPU (fallback)
- [Git](https://git-scm.com/) for version control

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmedali610/media-inpainting-toolkit.git
   
   cd media-inpainting-toolkit
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download pre-trained model weights for LAMA and XMem and place them in `models/`:
   ```text
   models/
   ├── lama_ckpt.pt
   └── xmem_ckpt.pth
   ```

---

## Usage

### A. Image Inpainting

1. Prepare your input images in the `input_images/` folder.
2. Provide a COCO-format JSON (`json_file.json`) that contains segmentation masks for each image.
3. Run the inpainting script:
   ```bash
   python scripts/inpaint_images.py \
     --input_dir input_images/ \
     --mask_json path/to/json_file.json \
     --output_dir outputs/lama_output_images/
   ```
4. The inpainted images will be saved under `outputs/lama_output_images/`.

### B. Video Inpainting

1. Put your input video file in `input_video/` (e.g., `input_video/video.mp4`).
2. Provide a COCO-format JSON (`json_file.json`) containing the mask for the first frame.
3. Run the video inpainting script:
   ```bash
   python scripts/inpaint_video.py \
     --input_video input_video/video.mp4 \
     --mask_json path/to/json_file.json \
     --output_video outputs/video_inpainted.mp4 \
     --temp_frames_dir temp/frames/ \
     --temp_masks_dir temp/masks/
   ```
4. The final inpainted video will be saved at `outputs/video_inpainted.mp4`.

---

---

## Models and Dependencies

- **LAMA**: Large Masked Autoencoder for conditional image generation
- **XMem**: Memory-based video object segmentation for mask propagation
- **Others**: `opencv-python`, `torch`, `numpy`, `pycocotools`, etc.

---

## Test Results

![image1](https://github.com/user-attachments/assets/d47a52a6-fab2-4cb9-a118-7b11ff4d83c7)


![image6](https://github.com/user-attachments/assets/60ee35c5-14dd-4f71-9863-08c6a46896e2)


---

## Evaluation

### Metrics

The toolkit computes two quantitative performance metrics for inpainting quality:

- **Average SSIM (Structural Similarity Index):** Measures perceptual similarity between original and inpainted images. Values range from 0 to 1, with values closer to **1** indicating higher similarity and better inpainting quality.
- **Average LPIPS (Learned Perceptual Image Patch Similarity):** A learned metric capturing perceptual differences. Values are typically between 0 and 1, with values closer to **0** indicating less perceptual difference and better inpainting performance.


### Image Evaluation Results

<div align="center">
  <img src="https://github.com/user-attachments/assets/7270de18-fde9-4070-b5bd-1a82e13ecbf0" alt="Image Evaluation" />
</div>

### Video Evaluation Results

<div align="center">
  <img src="https://github.com/user-attachments/assets/978a589e-bf14-40b8-9a78-35070aa8eadc" alt="Video Evaluation" />
</div>



