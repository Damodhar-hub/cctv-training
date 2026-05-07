#!/bin/bash
# Setup RunPod for FGVD classifier training.
# Run once after pod creation.
#
# Required env var:
#   FGVD_TOKEN_URL  — download URL from IDD portal
#
# Usage:
#   export FGVD_TOKEN_URL="https://idd.insaan.iiit.ac.in/dataset/download/..."
#   bash cctv-training/setup_classifier_pod.sh
set -e

if [ -z "$FGVD_TOKEN_URL" ]; then
  echo "ERROR: Set FGVD_TOKEN_URL environment variable first."
  echo "  Get the token URL from: https://idd.insaan.iiit.ac.in/dataset/details/"
  exit 1
fi

cd /workspace

# ---------------------------------------------------------------------------
# 1. Install dependencies
# ---------------------------------------------------------------------------
echo "Installing classifier dependencies..."
pip install -r cctv-training/requirements.txt
pip install -r cctv-training/requirements_classifier.txt

# ---------------------------------------------------------------------------
# 2. Download FGVD dataset
# ---------------------------------------------------------------------------
mkdir -p datasets

# Skip download if already present and > 1MB
if [ -f datasets/fgvd.tar.gz ] && [ $(stat -c%s datasets/fgvd.tar.gz 2>/dev/null || echo 0) -gt 1000000 ]; then
  echo "FGVD archive already exists, skipping download."
else
  # Clean up bad partial downloads
  [ -f datasets/fgvd.tar.gz ] && rm datasets/fgvd.tar.gz
  python cctv-training/download_fgvd.py \
    --url "$FGVD_TOKEN_URL" \
    --output datasets/fgvd.tar.gz
fi

# ---------------------------------------------------------------------------
# 3. Extract archive
# ---------------------------------------------------------------------------
cd datasets

if [ -d "FGVD" ] || ls -d FGVD* 2>/dev/null | head -1 > /dev/null 2>&1; then
  echo "FGVD folder already extracted, skipping."
else
  echo "Extracting FGVD archive..."
  if file fgvd.tar.gz | grep -q "gzip"; then
    tar -xzf fgvd.tar.gz
  elif file fgvd.tar.gz | grep -q "Zip"; then
    unzip fgvd.tar.gz
  else
    echo "Trying tar..."
    tar -xf fgvd.tar.gz
  fi
fi

# Find the extracted folder
EXTRACTED=$(ls -d FGVD* 2>/dev/null | head -1)
if [ -z "$EXTRACTED" ]; then
  echo "WARNING: Could not find FGVD folder. Listing contents:"
  ls -la
  echo "Set EXTRACTED manually and re-run the conversion step."
  exit 1
fi
echo "Extracted folder: $EXTRACTED"

# ---------------------------------------------------------------------------
# 4. Convert to classification crops
# ---------------------------------------------------------------------------
cd /workspace
echo "Converting FGVD to classification crops..."

python cctv-training/convert_fgvd_to_crops.py \
  --input "datasets/$EXTRACTED" \
  --output "datasets/fgvd_crops" \
  --min-samples 5 \
  --padding 0.1

# ---------------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------------
echo ""
echo "=== Dataset Stats ==="
cat datasets/fgvd_crops/dataset_stats.json
echo ""
echo "Train classes: $(ls datasets/fgvd_crops/train/ | wc -l)"
echo "Val classes:   $(ls datasets/fgvd_crops/val/ 2>/dev/null | wc -l)"
echo ""
echo "Setup complete! Next steps:"
echo "  1. bash cctv-training/sanity_check_classifier.sh"
echo "  2. bash cctv-training/train_classifier.sh"
