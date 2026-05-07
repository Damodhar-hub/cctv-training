#!/bin/bash
# Full fine-grained classifier training (60 epochs).
#
# Run in tmux/screen so it survives SSH disconnects:
#   tmux new -s cls
#   bash cctv-training/train_classifier.sh
#   Ctrl+B then D  (to detach)
#   tmux attach -t cls  (to reattach)
set -e
cd /workspace

echo "=== FGVD Fine-Grained Classifier Training ==="
echo "Backbone:   EfficientNet-B0"
echo "Epochs:     60 (early stopping patience=12)"
echo "Batch size: 64"
echo "Image size: 224"
echo ""

python cctv-training/train_classifier.py \
  --data datasets/fgvd_crops \
  --project classifier_v1 \
  --name indian_fgvd_run1 \
  --backbone efficientnet_b0 \
  --epochs 60 \
  --batch-size 64 \
  --lr 1e-3 \
  --backbone-lr-factor 0.1 \
  --imgsz 224 \
  --device 0 \
  --patience 12 \
  --workers 8 \
  --label-smoothing 0.1 \
  --warmup-epochs 5

echo ""
echo "Training complete!"
echo "Best model: classifier_v1/indian_fgvd_run1/best.pt"
echo "Log:        classifier_v1/indian_fgvd_run1/training_log.json"
