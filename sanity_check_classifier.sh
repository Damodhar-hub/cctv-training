#!/bin/bash
# Quick 2-epoch sanity check for the classifier.
# Verifies data loading, augmentation, and loss convergence.
# Takes ~5 minutes on RTX 4090.
set -e
cd /workspace

echo "=== Classifier Sanity Check (2 epochs) ==="

python cctv-training/train_classifier.py \
  --data datasets/fgvd_crops \
  --project test_run_cls \
  --name sanity \
  --backbone efficientnet_b0 \
  --epochs 2 \
  --batch-size 64 \
  --imgsz 224 \
  --device 0 \
  --workers 8 \
  --patience 99

echo ""
echo "Sanity check complete!"
echo "If loss decreased across epochs, data pipeline is working."
