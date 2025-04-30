#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
INSTALL_DIR="." # Directory where repos will be cloned (current directory)
LAMA_REPO_NAME="lama"
XMEM_REPO_NAME="XMem"
LAMA_MODEL_ZIP="big-lama.zip"
LAMA_MODEL_URL="https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"

# --- Functions ---
clone_repo() {
  local repo_url=$1
  local repo_name=$2
  if [ ! -d "$INSTALL_DIR/$repo_name" ]; then
    echo "Cloning $repo_name..."
    git clone "$repo_url" "$INSTALL_DIR/$repo_name"
  else
    echo "$repo_name directory already exists. Skipping clone."
  fi
}

# --- Main Setup Logic ---

echo "Starting environment setup..."

# 1. Install Python Dependencies
echo "Installing Python requirements..."
pip install -r requirements.txt
# Handle special thinplate install if needed (uncomment if not handled by requirements.txt)
# echo "Installing thinplate spline..."
# pip install git+https://github.com/cheind/py-thin-plate-spline

# 2. Clone Repositories
clone_repo "https://github.com/advimman/lama.git" "$LAMA_REPO_NAME"
clone_repo "https://github.com/hkchengrex/XMem.git" "$XMEM_REPO_NAME"

# 3. Download and Setup LaMa Model
LAMA_PATH="$INSTALL_DIR/$LAMA_REPO_NAME"
if [ ! -f "$LAMA_PATH/big-lama/models/best.ckpt" ]; then
  echo "Downloading LaMa model..."
  # Use curl within the target directory
  (cd "$LAMA_PATH" && curl -LJO "$LAMA_MODEL_URL")
  echo "Unzipping LaMa model..."
  unzip "$LAMA_PATH/$LAMA_MODEL_ZIP" -d "$LAMA_PATH/"
  # Optional: Remove zip file after extraction
  # rm "$LAMA_PATH/$LAMA_MODEL_ZIP"
else
  echo "LaMa model already downloaded. Skipping."
fi

# 4. Download XMem Models
XMEM_PATH="$INSTALL_DIR/$XMEM_REPO_NAME"
if [ ! -f "$XMEM_PATH/saves/XMem.pth" ]; then
    echo "Downloading XMem models..."
    (cd "$XMEM_PATH" && ./scripts/download_models.sh)
else
    echo "XMem models already downloaded. Skipping."
fi


echo "Environment setup complete."
echo "LaMa repository is in: $LAMA_PATH"
echo "XMem repository is in: $XMEM_PATH"