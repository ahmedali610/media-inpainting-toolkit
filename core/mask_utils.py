import os
import json
import numpy as np
import cv2
import logging
from pycocotools import mask as maskUtils # Ensure pycocotools is installed
from .file_utils import ensure_dir_exists

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_masks_from_coco_json(json_path: str, image_name_to_path_map: dict, mask_save_dir: str):
    """
    Given a COCO JSON file and a map of image filenames to their full paths,
    generates binary masks and saves them.

    Args:
        json_path: Path to the COCO JSON file.
        image_name_to_path_map: Dictionary mapping 'file_name' from JSON to actual image path.
                                This isn't strictly needed for mask generation itself, but
                                helps ensure we only process relevant images and get dimensions.
        mask_save_dir: Path to save binary mask PNGs.
    """
    ensure_dir_exists(mask_save_dir)

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"COCO JSON file not found: {json_path}")
        return False
    except json.JSONDecodeError:
        logging.error(f"Error decoding COCO JSON file: {json_path}")
        return False

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id is not None:
            annotations_by_image.setdefault(img_id, []).append(ann)

    processed_count = 0
    # Process each image defined in the JSON
    for img_info in data.get("images", []):
        img_id = img_info.get("id")
        file_name = img_info.get("file_name")

        if img_id is None or file_name is None:
            logging.warning(f"Skipping image info due to missing id or file_name: {img_info}")
            continue

        # Use provided dimensions, fall back to reading image if necessary (though COCO should have it)
        width, height = img_info.get("width"), img_info.get("height")
        if not (width and height) and file_name in image_name_to_path_map:
             try:
                  img = cv2.imread(image_name_to_path_map[file_name])
                  if img is not None:
                       height, width = img.shape[:2]
                  else:
                       logging.warning(f"Could not read image {file_name} to get dimensions.")
                       continue
             except Exception as e:
                  logging.warning(f"Error reading image {file_name} for dimensions: {e}")
                  continue
        elif not (width and height):
             logging.warning(f"Skipping image {file_name}: Dimensions not found in JSON or image map.")
             continue


        # Create blank mask
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        # Apply all annotations for this image
        for ann in annotations_by_image.get(img_id, []):
            if "segmentation" in ann:
                try:
                    rle = maskUtils.frPyObjects(ann["segmentation"], height, width)
                    ann_mask = maskUtils.decode(rle)
                    # Handle potential multiple segmentations for one annotation
                    if len(ann_mask.shape) == 3:
                        # Check if it's multi-channel (usually instances) or multi-part segmentation
                        if ann_mask.shape[2] > 1: # Multiple RLEs combined
                             ann_mask = np.any(ann_mask, axis=2).astype(np.uint8)
                        else: # Single RLE resulted in 3D somehow? Take first channel.
                            ann_mask = ann_mask[:,:,0].astype(np.uint8)

                    # Combine with the main mask for the image
                    combined_mask = np.maximum(combined_mask, ann_mask)
                except Exception as e:
                    logging.error(f"Error processing segmentation for image {img_id}, ann {ann.get('id', 'N/A')}: {e}")

        # Convert mask to binary (0 or 255)
        final_mask = (combined_mask * 255).astype(np.uint8)

        # Save mask with the same base name as the image but in the mask directory
        base_name = os.path.splitext(file_name)[0]
        save_path = os.path.join(mask_save_dir, base_name + ".png")
        try:
            cv2.imwrite(save_path, final_mask)
            logging.debug(f"Mask saved: {save_path}")
            processed_count += 1
        except Exception as e:
            logging.error(f"Failed to save mask {save_path}. Reason: {e}")

    logging.info(f"Generated {processed_count} masks in {mask_save_dir}")
    return True