#!/bin/bash
# Full fine-grained classifier training (v2 — anti-overfitting).
#
# Changes from v1:
#   - Backbone frozen for first 10 epochs (head learns first)
#   - Higher dropout (0.5) + deeper head (BN + 512-dim hidden)
#   - Stronger augmentation (GaussianBlur, heavier erasing/crop)
#   - Lower LR (3e-4) for more stable convergence
#   - Larger image size (288) for fine-grained detail
#   - More epochs (100) with patience=15
#
# Run in background:
#   nohup bash cctv-training/train_classifier.sh > /workspace/train_log.txt 2>&1 &
#   tail -20 /workspace/train_log.txt
set -e
cd /workspace

echo "=== FGVD Fine-Grained Classifier Training (v2) ==="
echo "Backbone:       EfficientNet-B0 (frozen 10 epochs)"
echo "Epochs:         100 (early stopping patience=15)"
echo "Batch size:     64"
echo "Image size:     288"
echo "Dropout:        0.5"
echo "LR:             head=3e-4, backbone=3e-5"
echo ""

python cctv-training/train_classifier.py \
  --data datasets/fgvd_crops \
  --project classifier_v2 \
  --name indian_fgvd_run2 \
  --backbone efficientnet_b0 \
  --epochs 100 \
  --batch-size 64 \
  --lr 3e-4 \
  --backbone-lr-factor 0.1 \
  --imgsz 288 \
  --device 0 \
  --patience 15 \
  --workers 8 \
  --label-smoothing 0.1 \
  --warmup-epochs 5 \
  --freeze-backbone-epochs 10 \
  --dropout 0.5

echo ""
echo "Training complete!"
echo "Best model: classifier_v2/indian_fgvd_run2/best.pt"
echo "Log:        classifier_v2/indian_fgvd_run2/training_log.json"
