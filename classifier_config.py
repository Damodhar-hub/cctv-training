"""
Central configuration for the IDD-FGVD fine-grained vehicle classifier.

Change these values via CLI args in train_classifier.py, or edit defaults here.
"""

# ---------------------------------------------------------------------------
# Paths (RunPod defaults — override via CLI)
# ---------------------------------------------------------------------------
FGVD_RAW_DIR = "/workspace/datasets/FGVD"
FGVD_CROPS_DIR = "/workspace/datasets/fgvd_crops"
YOLO_WEIGHTS = "/workspace/cctv_v2/indian_run1/weights/best.pt"
CLASSIFIER_PROJECT = "/workspace/classifier_v1"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BACKBONE = "efficientnet_b0"  # timm model name
INPUT_SIZE = 224
PRETRAINED = True

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
NUM_EPOCHS = 60
LEARNING_RATE = 1e-3           # classifier head LR
BACKBONE_LR_FACTOR = 0.1      # backbone LR = LEARNING_RATE * this
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
WARMUP_EPOCHS = 5
MIN_LR = 1e-6

# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
PATIENCE = 12

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
MIN_SAMPLES_PER_CLASS = 5      # classes below this are dropped
CROP_PADDING = 0.1             # 10% bbox padding when cropping
CROP_QUALITY = 95              # JPEG save quality for crops

# ---------------------------------------------------------------------------
# YOLO detector vehicle classes (indices from the 9-class model)
# Only these detections are forwarded to the classifier.
# ---------------------------------------------------------------------------
YOLO_VEHICLE_CLASSES = {
    2: "motorcycle",
    3: "car",
    4: "autorickshaw",
    5: "bus",
    6: "truck",
    7: "vehicle_other",
}

# ---------------------------------------------------------------------------
# FGVD vehicle type hierarchy (top level)
# ---------------------------------------------------------------------------
FGVD_VEHICLE_TYPES = [
    "autorickshaw", "bus", "car", "motorcycle", "scooter", "truck", "other",
]
