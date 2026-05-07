"""
Two-stage vehicle detection + fine-grained classification.

Stage 1: YOLOv8 detector (9-class Indian vehicle model)
Stage 2: EfficientNet classifier (IDD-FGVD make/model)

Usage:
    # Single image
    python inference_two_stage.py \
        --yolo-weights best.pt \
        --cls-weights classifier_v1/run1/best.pt \
        --source image.jpg --output result.jpg

    # Video
    python inference_two_stage.py \
        --yolo-weights best.pt \
        --cls-weights classifier_v1/run1/best.pt \
        --source video.mp4 --output result.mp4

    # Directory of images
    python inference_two_stage.py \
        --yolo-weights best.pt \
        --cls-weights classifier_v1/run1/best.pt \
        --source images_dir/ --output results_dir/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torchvision import transforms
from ultralytics import YOLO

from train_classifier import FGVDClassifier
from classifier_config import YOLO_VEHICLE_CLASSES


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Two-stage vehicle detection + classification")
    p.add_argument("--yolo-weights", required=True, help="Path to YOLOv8 weights (best.pt)")
    p.add_argument("--cls-weights", required=True, help="Path to classifier checkpoint")
    p.add_argument("--source", required=True, help="Input image, video, or directory")
    p.add_argument("--output", default="output", help="Output path")
    p.add_argument("--device", default="0", help="CUDA device or 'cpu'")
    p.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    p.add_argument("--topk", type=int, default=3, help="Top-K classifier predictions to show")
    p.add_argument("--no-display", action="store_true", help="Skip display window")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Two-Stage Detector
# ---------------------------------------------------------------------------
class TwoStageDetector:
    def __init__(self, yolo_weights: str, cls_weights: str, device: str = "cuda:0"):
        self.device = torch.device(device)

        # Load YOLO
        print(f"Loading YOLO: {yolo_weights}")
        self.yolo = YOLO(yolo_weights)

        # Load classifier
        print(f"Loading classifier: {cls_weights}")
        checkpoint = torch.load(cls_weights, map_location=self.device)
        config = checkpoint["config"]

        self.classifier = FGVDClassifier(
            backbone_name=config["backbone"],
            num_classes=config["num_classes"],
            pretrained=False,
        )
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()

        self.class_names = checkpoint.get("class_names", [])
        self.class_mapping = checkpoint.get("class_mapping", {})
        input_size = config.get("input_size", 224)

        # Val transform for crops
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(int(input_size * 256 / 224)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"Classifier: {config['backbone']}, {config['num_classes']} classes")

    def _crop_vehicle(self, image: np.ndarray, bbox, padding: float = 0.1) -> np.ndarray:
        """Crop a vehicle from the image with padding."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * padding)
        pad_y = int(bh * padding)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        return image[y1:y2, x1:x2]

    def _classify_batch(self, crops: list[np.ndarray], topk: int = 3) -> list[list[dict]]:
        """Classify a batch of vehicle crops. Returns top-K predictions per crop."""
        if not crops:
            return []

        # Transform all crops to tensors
        tensors = []
        for crop in crops:
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(crop_rgb))

        if not tensors:
            return []

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad(), autocast():
            logits = self.classifier(batch)
            probs = torch.softmax(logits, dim=1)

        results = []
        k = min(topk, probs.size(1))
        top_probs, top_indices = probs.topk(k, dim=1)

        for i in range(len(tensors)):
            preds = []
            for j in range(k):
                idx = top_indices[i, j].item()
                conf = top_probs[i, j].item()
                name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
                preds.append({"class": name, "confidence": round(conf, 4)})
            results.append(preds)

        return results

    def predict(self, image: np.ndarray, conf: float = 0.3, topk: int = 3) -> list[dict]:
        """Run two-stage detection on a single image.

        Returns list of detections, each with:
          bbox, yolo_class, yolo_conf, fine_grained (list of top-K predictions)
        """
        # Stage 1: YOLO detection
        results = self.yolo(image, conf=conf, verbose=False)

        detections = []
        vehicle_indices = []
        vehicle_crops = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                cls_id = int(boxes.cls[i].item())
                yolo_conf = float(boxes.conf[i].item())

                det = {
                    "bbox": bbox.tolist(),
                    "yolo_class_id": cls_id,
                    "yolo_conf": round(yolo_conf, 4),
                    "fine_grained": None,
                }
                detections.append(det)

                # Only classify vehicle classes
                if cls_id in YOLO_VEHICLE_CLASSES:
                    crop = self._crop_vehicle(image, bbox)
                    if crop.size > 0:
                        vehicle_indices.append(i)
                        vehicle_crops.append(crop)

        # Stage 2: Batch classify all vehicle crops
        if vehicle_crops:
            cls_results = self._classify_batch(vehicle_crops, topk=topk)
            for idx, cls_preds in zip(vehicle_indices, cls_results):
                detections[idx]["fine_grained"] = cls_preds

        return detections

    def draw(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw detections on image with fine-grained labels."""
        vis = image.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cls_id = det["yolo_class_id"]
            yolo_conf = det["yolo_conf"]

            # Color: green for vehicles with fine-grained, blue for others
            if det["fine_grained"]:
                color = (0, 200, 0)
                top_pred = det["fine_grained"][0]
                # Format: "car_MarutiSuzuki_Ciaz" -> "MarutiSuzuki Ciaz"
                label_parts = top_pred["class"].split("_", 1)
                display_name = label_parts[1].replace("_", " ") if len(label_parts) > 1 else top_pred["class"]
                label = f"{display_name} {top_pred['confidence']:.2f}"
            else:
                color = (200, 100, 0)
                yolo_name = YOLO_VEHICLE_CLASSES.get(cls_id, f"cls{cls_id}")
                label = f"{yolo_name} {yolo_conf:.2f}"

            # Draw bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return vis


# ---------------------------------------------------------------------------
# Process image
# ---------------------------------------------------------------------------
def process_image(detector, image_path, output_path, conf, topk):
    """Process a single image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  WARN: Cannot read {image_path}")
        return

    detections = detector.predict(img, conf=conf, topk=topk)
    vis = detector.draw(img, detections)

    cv2.imwrite(output_path, vis)

    # Print detections
    vehicles = [d for d in detections if d["fine_grained"]]
    if vehicles:
        for d in vehicles:
            top = d["fine_grained"][0]
            print(f"  {top['class']} ({top['confidence']:.2f})")


# ---------------------------------------------------------------------------
# Process video
# ---------------------------------------------------------------------------
def process_video(detector, video_path, output_path, conf, topk):
    """Process a video frame by frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Processing video: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.predict(frame, conf=conf, topk=topk)
        vis = detector.draw(frame, detections)
        writer.write(vis)

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            fps_actual = frame_idx / elapsed
            print(f"  Frame {frame_idx}/{total_frames} ({fps_actual:.1f} fps)")

    cap.release()
    writer.release()

    elapsed = time.time() - t0
    print(f"Done: {frame_idx} frames in {elapsed:.1f}s ({frame_idx / elapsed:.1f} fps)")
    print(f"Output: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    detector = TwoStageDetector(args.yolo_weights, args.cls_weights, device=device)

    source = args.source
    output = args.output

    if os.path.isfile(source):
        ext = Path(source).suffix.lower()
        if ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            # Video
            process_video(detector, source, output, args.conf, args.topk)
        else:
            # Single image
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
            print(f"Processing: {source}")
            process_image(detector, source, output, args.conf, args.topk)
            print(f"Saved: {output}")

    elif os.path.isdir(source):
        # Directory of images
        os.makedirs(output, exist_ok=True)
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted(
            f for f in Path(source).iterdir()
            if f.suffix.lower() in image_exts
        )
        print(f"Processing {len(image_files)} images from {source}")

        for img_file in image_files:
            out_path = os.path.join(output, img_file.name)
            print(f"  {img_file.name}")
            process_image(detector, str(img_file), out_path, args.conf, args.topk)

        print(f"\nResults saved to {output}")

    else:
        print(f"ERROR: Source not found: {source}")
        sys.exit(1)


if __name__ == "__main__":
    main()
