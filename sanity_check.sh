#!/bin/bash
set -e
cd /workspace
echo "Running 2-epoch sanity check..."
yolo train \
  data=datasets/cctv_dataset/data.yaml \
  model=yolov8s.pt \
  epochs=2 \
  imgsz=640 \
  batch=32 \
  device=0 \
  project=test_run \
  name=sanity \
  workers=8
echo ""
echo "Sanity check complete. Check test_run/sanity/ for results."
echo "If no errors, proceed with: tmux new -s training && bash train.sh"
