import os

# --- Core Paths ---
# Assume setup.sh cloned repos into the same directory as this project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Parent dir of 'core'
LAMA_REPO_PATH = os.path.join(PROJECT_ROOT, "lama")
XMEM_REPO_PATH = os.path.join(PROJECT_ROOT, "XMem")

# --- Model Paths ---
LAMA_MODEL_DIR = os.path.join(LAMA_REPO_PATH, "big-lama")
XMEM_MODEL_PATH = os.path.join(XMEM_REPO_PATH, "saves/XMem.pth")

# --- Default Input/Output ---
# These should ideally be passed as arguments to main_pipeline.py
DEFAULT_INPUT_VIDEO = "/content/vi.mp4" # Example placeholder
DEFAULT_COCO_JSON = "/content/labels_my-project-name_2025-04-29-01-02-32.json" # Example placeholder
DEFAULT_BASE_OUTPUT_DIR = "/content/pipeline_output" # Example placeholder

# --- Video Processing ---
DEFAULT_VIDEO_FPS = 30

# --- LaMa Specific Environment (Optional but potentially needed) ---
# LaMa might need its directory in PYTHONPATH and a TORCH_HOME
LAMA_ENV = os.environ.copy()
LAMA_ENV['PYTHONPATH'] = f"{LAMA_REPO_PATH}{os.pathsep}{LAMA_ENV.get('PYTHONPATH', '')}"
LAMA_ENV['TORCH_HOME'] = LAMA_REPO_PATH # LaMa downloads some torch models here