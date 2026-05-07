#!/bin/bash
set -e

# Expect IDD_TOKEN_URL as environment variable (passed at runtime, NOT hardcoded)
if [ -z "$IDD_TOKEN_URL" ]; then
  echo "ERROR: Set IDD_TOKEN_URL env var before running"
  echo "  export IDD_TOKEN_URL='https://idd.insaan.iiit.ac.in/...your-token-url...'"
  exit 1
fi

cd /workspace

# Install deps
pip install -r cctv-training/requirements.txt

# Delete any bad partial downloads (e.g. HTML login page from plain wget)
if [ -f datasets/idd_detection.tar.gz ]; then
  SIZE=$(stat -c%s datasets/idd_detection.tar.gz 2>/dev/null || echo "0")
  if [ "$SIZE" -lt 1000000 ]; then
    echo "Removing bad partial file (${SIZE} bytes)..."
    rm -f datasets/idd_detection.tar.gz
  fi
fi

# Download dataset with authenticated session
mkdir -p datasets
python cctv-training/download_idd.py \
  --url "$IDD_TOKEN_URL" \
  --output datasets/idd_detection.tar.gz

# Extract (auto-detect format)
cd datasets
FILETYPE=$(file idd_detection.tar.gz)
echo "Detected file type: $FILETYPE"

if [[ "$FILETYPE" == *"gzip"* ]]; then
  echo "Extracting tar.gz..."
  tar -xzf idd_detection.tar.gz
elif [[ "$FILETYPE" == *"Zip"* ]]; then
  echo "Extracting zip..."
  unzip idd_detection.tar.gz
else
  echo "ERROR: Unknown archive format"
  exit 1
fi

# Find extracted folder name dynamically
EXTRACTED=$(ls -d IDD_Detection* 2>/dev/null || ls -d idd* 2>/dev/null | head -1)
if [ -z "$EXTRACTED" ]; then
  echo "ERROR: Could not find extracted IDD folder. Contents:"
  ls -la
  exit 1
fi
echo "Found extracted folder: $EXTRACTED"

# Convert to YOLO format
cd /workspace
python cctv-training/convert_idd_to_yolo.py \
  --input "datasets/$EXTRACTED" \
  --output "datasets/cctv_dataset"

# Verify
echo ""
echo "=========================================="
echo "=== Conversion stats ==="
echo "=========================================="
cat datasets/cctv_dataset/conversion_stats.json
echo ""
echo "=== Train images count ==="
ls datasets/cctv_dataset/images/train/ | wc -l
echo "=== Val images count ==="
ls datasets/cctv_dataset/images/val/ | wc -l
echo ""
echo "Setup complete. Run sanity_check.sh next."
