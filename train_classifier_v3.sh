#!/bin/bash
# V3 training: merged classes + EfficientNet-B2 + balanced augmentation
#
# Run:
#   nohup bash cctv-training/train_classifier_v3.sh > /workspace/train_v3_log.txt 2>&1 &
#   tail -20 /workspace/train_v3_log.txt
set -e
cd /workspace

# Step 1: Prepare V3 dataset (merge rare classes)
echo "=== Preparing V3 Dataset ==="
python cctv-training/prepare_v3_dataset.py \
  --input datasets/fgvd_crops \
  --output datasets/fgvd_crops_v3 \
  --min-samples 30 \
  --min-manufacturer 15

echo ""

# Step 2: Train with EfficientNet-B2
echo "=== FGVD Classifier V3 Training ==="
echo "Backbone:       EfficientNet-B2 (260px)"
echo "Epochs:         80 (early stopping patience=15)"
echo "Batch size:     48 (larger model needs more memory)"
echo "Dropout:        0.4"
echo "LR:             head=5e-4, backbone=5e-5"
echo ""

python cctv-training/train_classifier.py \
  --data datasets/fgvd_crops_v3 \
  --project classifier_v3 \
  --name indian_fgvd_run3 \
  --backbone efficientnet_b2 \
  --epochs 80 \
  --batch-size 48 \
  --lr 5e-4 \
  --backbone-lr-factor 0.1 \
  --imgsz 260 \
  --device 0 \
  --patience 15 \
  --workers 8 \
  --label-smoothing 0.1 \
  --warmup-epochs 5 \
  --freeze-backbone-epochs 8 \
  --dropout 0.4

echo ""
echo "V3 Training complete!"
echo "Best model: classifier_v3/indian_fgvd_run3/best.pt"
echo "Log:        classifier_v3/indian_fgvd_run3/training_log.json"
