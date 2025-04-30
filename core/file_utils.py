import os
import shutil
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_directory(directory_path):
    """Deletes all files and subdirectories inside the specified directory."""
    if not os.path.isdir(directory_path):
        logging.warning(f"Directory not found, cannot clean: {directory_path}")
        return
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.error(f'Failed to delete {file_path}. Reason: {e}')
    logging.info(f"Cleaned directory: {directory_path}")


def ensure_dir_exists(directory_path):
    """Creates a directory if it doesn't exist."""
    os.makedirs(directory_path, exist_ok=True)
    logging.debug(f"Ensured directory exists: {directory_path}")


def prepare_lama_input(image_src_dir, mask_src_dir, lama_input_dir, mask_suffix="_mask.png"):
    """
    Copies images and masks to the LaMa input directory with expected naming.
    Images: frame_xxxx.png
    Masks: frame_xxxx_mask.png
    """
    ensure_dir_exists(lama_input_dir)
    clean_directory(lama_input_dir) # Start fresh

    logging.info(f"Preparing LaMa input in: {lama_input_dir}")

    # Copy and rename images
    logging.info(f"Copying images from {image_src_dir}...")
    for filename in sorted(os.listdir(image_src_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(filename)[0]
            img_dst_name = f"{base_name}.png" # Ensure consistent extension if needed
            src_path = os.path.join(image_src_dir, filename)
            dst_path = os.path.join(lama_input_dir, img_dst_name)
            shutil.copy(src_path, dst_path)

    # Copy and rename masks
    logging.info(f"Copying masks from {mask_src_dir}...")
    for filename in sorted(os.listdir(mask_src_dir)):
        if filename.lower().endswith('.png'):
            base_name = os.path.splitext(filename)[0]
            mask_dst_name = f"{base_name}{mask_suffix}"
            src_path = os.path.join(mask_src_dir, filename)
            dst_path = os.path.join(lama_input_dir, mask_dst_name)
            shutil.copy(src_path, dst_path)

    logging.info("LaMa input preparation complete.")


def run_subprocess(command, working_dir=None, env=None):
    """Runs a command as a subprocess and logs output."""
    logging.info(f"Running command: {' '.join(command)}")
    if working_dir:
        logging.info(f"Working directory: {working_dir}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Redirect stderr to stdout
            text=True,
            cwd=working_dir,
            env=env
        )
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logging.info(output.strip())
        rc = process.poll()
        if rc != 0:
            logging.error(f"Subprocess command failed with exit code {rc}: {' '.join(command)}")
            return False
        logging.info(f"Subprocess command finished successfully: {' '.join(command)}")
        return True
    except Exception as e:
        logging.error(f"Failed to run subprocess command: {' '.join(command)}. Error: {e}")
        return False