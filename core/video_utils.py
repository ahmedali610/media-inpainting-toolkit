import cv2
import os
import logging
from .file_utils import ensure_dir_exists

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_frames(video_path, output_dir, frame_prefix="frame_", file_ext=".png"):
    """Extracts all frames from a video file."""
    ensure_dir_exists(output_dir)

    if not os.path.isfile(video_path):
        logging.error(f"Video file not found: {video_path}")
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        return 0

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"{frame_prefix}{frame_id:04d}{file_ext}")
        cv2.imwrite(frame_filename, frame)
        frame_id += 1

    cap.release()
    logging.info(f"Extracted {frame_id} frames from {video_path} to {output_dir}")
    return frame_id

def create_video_from_frames(frame_dir, output_video_path, fps=25, frame_pattern=None):
    """Creates a video from a sequence of image frames."""
    if not os.path.isdir(frame_dir):
        logging.error(f"Frame directory not found: {frame_dir}")
        return False
    
    frames = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if frame_pattern: # Simple pattern matching if needed, e.g., "frame_*.png"
        from fnmatch import fnmatch
        frames = sorted([f for f in frames if fnmatch(f, frame_pattern)])

    if not frames:
        logging.error(f"No image frames found in {frame_dir} matching pattern.")
        return False

    # Read the first frame to get dimensions
    try:
        first_frame_path = os.path.join(frame_dir, frames[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            logging.error(f"Could not read first frame: {first_frame_path}")
            return False
        height, width, layers = first_frame.shape
        
    except Exception as e:
        logging.error(f"Error reading first frame {first_frame_path}: {e}")
        return False

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or use 'XVID', 'MJPG', etc.
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not video.isOpened():
         logging.error(f"Could not open video writer for path: {output_video_path}")
         return False

    logging.info(f"Creating video {output_video_path} from {len(frames)} frames at {fps} FPS...")

    for frame_name in frames:
        frame_path = os.path.join(frame_dir, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            logging.warning(f"Could not read frame: {frame_path}. Skipping.")
            continue
        # Resize frame if necessary (shouldn't be if all frames are consistent)
        if frame.shape[0] != height or frame.shape[1] != width:
             logging.warning(f"Frame {frame_name} has different dimensions. Resizing.")
             frame = cv2.resize(frame, (width, height))
        video.write(frame)

    video.release()
    logging.info(f"Video saved successfully at {output_video_path}")
    return True