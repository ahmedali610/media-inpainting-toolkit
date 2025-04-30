import os
import logging
from .file_utils import run_subprocess
from . import config # To get repo path and env vars

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_lama_prediction(input_dir: str, output_dir: str, lama_repo_path: str = config.LAMA_REPO_PATH, model_dir: str = config.LAMA_MODEL_DIR):
    """Runs the LaMa prediction script."""
    predict_script = os.path.join(lama_repo_path, "bin", "predict.py")
    model_ckpt_path = os.path.join(model_dir, "models", "best.ckpt") # LaMa expects model dir, not ckpt directly in arg? Check docs. Usually path points to dir containing config.yaml & models/best.ckpt

    if not os.path.isdir(lama_repo_path):
        logging.error(f"LaMa repository not found at: {lama_repo_path}")
        return False
    if not os.path.isdir(model_dir):
         logging.error(f"LaMa model directory not found at: {model_dir}")
         return False
    if not os.path.exists(predict_script):
        logging.error(f"LaMa predict script not found: {predict_script}")
        return False

    # Construct the command
    # Note: LaMa uses Hydra, which changes the working directory.
    # Running from the lama repo root might be necessary, or adjusting paths in the command.
    # Let's assume running from repo root is safer.
    command = [
        "python",
        os.path.join("bin", "predict.py"), # Relative path from repo root
        f"model.path={os.path.abspath(model_dir)}", # Absolute path to model dir
        f"indir={os.path.abspath(input_dir)}",     # Absolute path to input
        f"outdir={os.path.abspath(output_dir)}"    # Absolute path to output
    ]

    logging.info("Starting LaMa prediction...")
    # Run from within the lama directory, passing necessary env vars
    success = run_subprocess(command, working_dir=lama_repo_path, env=config.LAMA_ENV)

    if success:
        for filename in os.listdir(output_dir):
            if '_mask' in filename:  # Check if '_mask' is in the file name
                old_path = os.path.join(output_dir, filename)
                new_filename = filename.replace('_mask', '')  # Remove '_mask' from the filename
                new_path = os.path.join(output_dir, new_filename)
                
                # Rename the file
                os.rename(old_path, new_path)

        logging.info(f"LaMa prediction finished. Output in: {output_dir}")
    else:
        logging.error("LaMa prediction failed.")
    return success