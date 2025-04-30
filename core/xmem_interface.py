import os
import logging
from .file_utils import run_subprocess
from . import config # To get repo path

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_xmem_propagation(
    data_root: str, # Should contain JPEGImages/video1 and Annotations/video1/frame_0000.png
    output_dir: str, # Where to save the propagated masks
    xmem_repo_path: str,  #= config.XMEM_REPO_PATH
    model_path: str = config.XMEM_MODEL_PATH):
    """Runs the XMem evaluation script for mask propagation."""

    eval_script = os.path.join(xmem_repo_path, "eval.py")

    if not os.path.isdir(xmem_repo_path):
        logging.error(f"XMem repository not found at: {xmem_repo_path}")
        return False
    if not os.path.isfile(model_path):
        logging.error(f"XMem model not found at: {model_path}")
        return False
    if not os.path.exists(eval_script):
        logging.error(f"XMem eval script not found: {eval_script}")
        return False
    if not os.path.isdir(data_root):
        logging.error(f"XMem data root directory not found: {data_root}")
        return False
    # Check for initial mask? The script might handle this, but good practice.
    # initial_mask = os.path.join(data_root, "Annotations", "video1", "frame_0000.png") # Assuming default name
    # if not os.path.isfile(initial_mask):
    #      logging.error(f"Initial mask for XMem not found at: {initial_mask}")
    #      return False


    # Construct the command
    # Run from the XMem repo directory is usually required for relative imports in their code.
    command = [
        "python",
        os.path.join("eval.py"), # Relative path from repo root
        f"--model", os.path.abspath(model_path),
        f"--generic_path", os.path.abspath(data_root), # Path to frames/ dir
        "--dataset", "G", # Generic dataset flag
        f"--output", os.path.abspath(output_dir),
        "--save_all" # Save masks for all frames
    ]

    logging.info("Starting XMem mask propagation...")
    # Run from within the XMem directory
    success = run_subprocess(command, working_dir=xmem_repo_path)

    if success:
        logging.info(f"XMem propagation finished. Output masks in: {output_dir}")
    else:
        logging.error("XMem propagation failed.")
    return success