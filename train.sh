#!/bin/bash
set -e
cd /workspace
echo "Starting full training (50 epochs)..."
echo "Checkpoints saved every 5 epochs. Patience=15 for early stopping."
echo "Detach tmux: Ctrl+B then D"
echo ""
yolo train \
  data=datasets/cctv_dataset/data.yaml \
  model=yolov8s.pt \
  epochs=50 \
  imgsz=640 \
  batch=32 \
  device=0 \
  project=cctv_v2 \
  name=indian_run1 \
  save_period=5 \
  patience=15 \
  cos_lr=True \
  augment=True \
  workers=8
echo ""
echo "Training complete!"
echo "Best model: cctv_v2/indian_run1/weights/best.pt"
echo "Last model: cctv_v2/indian_run1/weights/last.pt"
