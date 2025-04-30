# media-inpainting-toolkit
 Media Inpainting Toolkit is a command-line Python application that enables seamless removal of unwanted objects from images and videos by leveraging COCO-format masks, state-of-the-art LAMA image inpainting, and XMem video mask propagation.

 Table of Contents

Features

Getting Started

Prerequisites

Installation

Usage

A. Image Inpainting

B. Video Inpainting

Project Structure

Models and Dependencies

Contributing

License

Features

Image Inpainting: Remove objects from images using LAMA with COCO-style pixel masks.

Video Inpainting: Remove objects from videos by propagating an initial COCO mask through frames via XMem and inpainting each frame with LAMA.

Easy CLI-based interface and configurable paths

Supports batch processing and common video formats (MP4, AVI, etc.)

Getting Started

Prerequisites

Python 3.8+

CUDA-enabled GPU (recommended) or CPU (fallback)

Git for version control

Installation

Clone the repository:

git clone https://github.com/<your-username>/media-inpainting-toolkit.git
cd media-inpainting-toolkit

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Download pre-trained model weights for LAMA and XMem and place them in models/:

models/
├── lama_ckpt.pt
└── xmem_ckpt.pth

Usage

A. Image Inpainting

Prepare your input images in the input_images/ folder.

Provide a COCO-format JSON (json_file.json) that contains segmentation masks for each image.

Run the inpainting script:

python scripts/inpaint_images.py \
  --input_dir input_images/ \
  --mask_json path/to/json_file.json \
  --output_dir outputs/lama_output_images/

The inpainted images will be saved under outputs/lama_output_images/.

B. Video Inpainting

Put your input video file in input_video/ (e.g., input_video/video.mp4).

Provide a COCO-format JSON (json_file.json) containing the mask for the first frame.

Run the video inpainting script:

python scripts/inpaint_video.py \
  --input_video input_video/video.mp4 \
  --mask_json path/to/json_file.json \
  --output_video outputs/video_inpainted.mp4 \
  --temp_frames_dir temp/frames/ \
  --temp_masks_dir temp/masks/

The final inpainted video will be saved at outputs/video_inpainted.mp4.

Project Structure

media-inpainting-toolkit/
├── input_images/               # Place source images here
├── input_video/                # Place source video here
├── outputs/
│   ├── lama_output_images/     # Image inpainting results
│   └── video_inpainted.mp4     # Final video output
├── models/                     # Download pretrained LAMA & XMem weights
├── scripts/
│   ├── inpaint_images.py       # Image inpainting pipeline
│   └── inpaint_video.py        # Video inpainting pipeline
├── requirements.txt            # Python dependencies
└── README.md                   # This file

Models and Dependencies

LAMA: Large Masked Autoencoder for conditional image generation

XMem: Memory-based video object segmentation for mask propagation

Others: opencv-python, torch, numpy, pycocotools etc.

Contributing

Contributions are welcome! Please:

Fork the repo

Create a feature branch (git checkout -b feature-name)

Commit your changes (git commit -m "Add new feature")

Push to the branch (git push origin feature-name)

Open a Pull Request

License

This project is licensed under the MIT License. See the LICENSE file for details.
