import os
import argparse
import logging
import shutil
from core import config
from core.file_utils import ensure_dir_exists, prepare_lama_input
from core.video_utils import extract_frames, create_video_from_frames
from core.mask_utils import generate_masks_from_coco_json
from core.lama_interface import run_lama_prediction
from core.xmem_interface import run_xmem_propagation
from core.evaluation import evaluate_folder, evaluate_video_quality


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    """Runs the video inpainting pipeline."""

    # --- Setup Paths ---
    base_work_dir = args.output_dir
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]

    # Frame and initial mask directories following XMem expected structure
    frames_root = os.path.join(base_work_dir, "frames")
    images_dir = args.images_path
    coco_json_path = "json_file\mask.json"
    lama_input_images_dir = os.path.join(base_work_dir, "lama_input_images")
    XMem_dir =  "XMem"
    frames_dir = os.path.join(frames_root, "JPEGImages", video_name)
    annotations_root = os.path.join(frames_root, "Annotations")
    initial_mask_dir = os.path.join(annotations_root, video_name) # Dir for the first mask from COCO
    output_masks = os.path.join(base_work_dir, "masks")
    # XMem output directory for propagated masks
    propagated_mask_dir_rel = os.path.join("Annotations", video_name, video_name) # Relative path within XMem output structure
    propagated_mask_dir_abs = os.path.join(annotations_root, video_name, video_name) # Absolute path where masks will be saved by XMem

    # LaMa input/output directories
    lama_input_pass1_dir = os.path.join(base_work_dir, "lama_input_pass")
    lama_output_pass1_dir = os.path.join(base_work_dir, "lama_output_pass")
    lama_output_images_dir = os.path.join(base_work_dir, "lama_output_images")

    # Final video output paths
    output_video = os.path.join(base_work_dir, f"{video_name}_inpainted.mp4")

    # --- Create Directories ---
    ensure_dir_exists(base_work_dir)
    ensure_dir_exists(frames_dir)
    ensure_dir_exists(initial_mask_dir)
    ensure_dir_exists(lama_input_images_dir)
    # ensure_dir_exists(propagated_mask_dir_abs) # XMem script should create this output path
    ensure_dir_exists(lama_input_pass1_dir)
    ensure_dir_exists(lama_output_pass1_dir)
    ensure_dir_exists(lama_output_images_dir)
    ensure_dir_exists(output_masks)


    # --- Pipeline Steps ---
    if args.is_image:
        # Step 1: Generate binary masks from COCO JSON
        logging.info("Step 1: Generating binary masks from COCO JSON...")
        if not generate_masks_from_coco_json(coco_json_path, images_dir, output_masks):
            logging.error("Failed to generate binary masks. Exiting.")
            return

        # Step 2: Prepare input for LaMa
        logging.info("Step 2: Preparing LaMa input directory...")
        prepare_lama_input(images_dir, output_masks, lama_input_images_dir)

        # Step 3: Run LaMa Inpainting
        logging.info("Step 3: Running LaMa Inpainting...")
        if not run_lama_prediction(lama_input_images_dir, lama_output_images_dir):
            logging.error("LaMa inpainting failed. Exiting.")
            return
        
        # Step 4: Evaluate inpainted images
        logging.info("Step 4: Evaluating inpainted results...")
        evaluate_folder(images_dir, lama_output_images_dir)

        logging.info("Pipeline completed successfully!")
        logging.info(f"Final inpainted images are saved in: {lama_output_images_dir}")


    if args.is_video:
        # 1. Extract Frames
        logging.info("Step 1: Extracting video frames...")
        num_frames = extract_frames(args.video_path, frames_dir)
        if num_frames == 0:
            logging.error("Frame extraction failed. Exiting.")
            return

        # 2. Generate Initial Mask from COCO JSON
        # This requires mapping JSON filenames to the extracted frame paths
        logging.info("Step 2: Generating initial mask from COCO JSON...")
        image_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        # Assuming JSON file_names match extracted frame names (e.g., "frame_0000.png")
        image_name_to_path = {f: os.path.join(frames_dir, f) for f in image_files}
        if not generate_masks_from_coco_json(args.coco_json_path, image_name_to_path, initial_mask_dir):
            logging.warning("Failed to generate initial masks from COCO JSON. Attempting XMem without it if possible, or exiting.")

            # Check if the specific first frame mask was created:
            first_mask_name = os.path.splitext(image_files[0])[0] + ".png"
            if not os.path.exists(os.path.join(initial_mask_dir, first_mask_name)):
                logging.error("Initial mask generation failed and first mask is missing. Exiting.")
                return
            else:
                logging.info("Initial mask possibly created, continuing to XMem.")


        # 3. Propagate Masks with XMem
        logging.info("Step 3: Propagating masks with XMem...")
        if not run_xmem_propagation(data_root=frames_root, output_dir=initial_mask_dir, xmem_repo_path=XMem_dir, model_path='XMem/saves/XMem.pth'):
            logging.error("XMem mask propagation failed. Exiting.")
            return

        # Check if propagated masks exist after XMem run
        if not os.path.isdir(propagated_mask_dir_abs) or not os.listdir(propagated_mask_dir_abs):
            logging.error(f"XMem finished but no masks found in expected output directory: {propagated_mask_dir_abs}. Exiting.")
            return


        # 4. Prepare Input for LaMa
        logging.info("Step 4: Preparing input for LaMa (Pass 1)...")
        prepare_lama_input(frames_dir, propagated_mask_dir_abs, lama_input_pass1_dir)

        # 5. Run LaMa Inpainting 
        logging.info("Step 5: Running LaMa Inpainting (Pass 1)...")
        if not run_lama_prediction(lama_input_pass1_dir, lama_output_pass1_dir):
            logging.error("LaMa Pass 1 failed. Exiting.")
            return

        # 6. Create Video from Output
        logging.info("Step 6: Creating video from LaMa Pass 1 output...")
        create_video_from_frames(lama_output_pass1_dir, output_video, args.fps, frame_pattern="frame_*_mask.png") # LaMa output includes _mask suffix

        # Step 7: Evaluate inpainting quality on the final video
        logging.info("Step 7: Evaluating inpainted video quality...")
        evaluate_video_quality(args.video_path, output_video)


        logging.info("Pipeline completed successfully!")
        logging.info(f"Final inpainted video: {output_video}")


if __name__ == "__main__":
    class Args:
        video_path = r"input_video\video.mp4"
        images_path = r"input_images"
        coco_json_path = r"json_file\mask.json"
        output_dir = r"outputs"
        fps = 25
        is_image = True
        is_video = True

    args = Args()

    # Basic validation
    if not os.path.isfile(args.video_path):
        raise FileNotFoundError(f"Input video not found: {args.video_path}")
    if not os.path.isfile(args.coco_json_path):
        raise FileNotFoundError(f"COCO JSON file not found: {args.coco_json_path}")

    main(args)