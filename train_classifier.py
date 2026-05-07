"""
Train a fine-grained vehicle classifier on IDD-FGVD crops.

Uses EfficientNet-B0 (via timm) with mixed-precision training,
cosine annealing LR with warmup, early stopping, and weighted
cross-entropy for class imbalance.

Input:  ImageFolder structure from convert_fgvd_to_crops.py
Output: best.pt checkpoint with embedded class mapping + config

Usage:
    python train_classifier.py \
        --data /workspace/datasets/fgvd_crops \
        --project /workspace/classifier_v1 \
        --name run1 \
        --backbone efficientnet_b0 \
        --epochs 60 --batch-size 64 --lr 1e-3 \
        --imgsz 224 --device 0 --patience 12 --workers 8
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train FGVD fine-grained vehicle classifier")
    p.add_argument("--data", required=True, help="Path to fgvd_crops (ImageFolder layout)")
    p.add_argument("--project", default="classifier_v1", help="Output project directory")
    p.add_argument("--name", default="run1", help="Experiment name")
    p.add_argument("--backbone", default="efficientnet_b0", help="timm model name")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="Head learning rate")
    p.add_argument("--backbone-lr-factor", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--imgsz", type=int, default=224, help="Input image size")
    p.add_argument("--device", default="0", help="CUDA device (e.g. '0' or 'cpu')")
    p.add_argument("--patience", type=int, default=12, help="Early stopping patience")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FGVDClassifier(nn.Module):
    """EfficientNet backbone + dropout + linear head."""

    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------
def build_transforms(imgsz: int):
    """Return (train_transform, val_transform)."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(imgsz, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(imgsz * 256 / 224)),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


# ---------------------------------------------------------------------------
# Class weights (inverse frequency, capped)
# ---------------------------------------------------------------------------
def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """Compute inverse-frequency class weights from an ImageFolder dataset."""
    counts = defaultdict(int)
    for _, label in dataset.samples:
        counts[label] += 1

    num_classes = len(dataset.classes)
    weight_list = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        weight_list.append(1.0 / c)

    weights = torch.tensor(weight_list, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes  # normalize so mean = 1
    weights = weights.clamp(max=10.0)  # cap extreme weights
    return weights


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        # Top-1
        _, pred = logits.max(1)
        correct += pred.eq(labels).sum().item()

        # Top-5
        if logits.size(1) >= 5:
            _, top5_pred = logits.topk(5, dim=1)
            correct_top5 += top5_pred.eq(labels.unsqueeze(1)).any(1).sum().item()
        else:
            correct_top5 += pred.eq(labels).sum().item()

    avg_loss = running_loss / total
    top1 = correct / total
    top5 = correct_top5 / total
    return avg_loss, top1, top5


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        _, pred = logits.max(1)
        correct += pred.eq(labels).sum().item()

        if logits.size(1) >= 5:
            _, top5_pred = logits.topk(5, dim=1)
            correct_top5 += top5_pred.eq(labels.unsqueeze(1)).any(1).sum().item()
        else:
            correct_top5 += pred.eq(labels).sum().item()

    avg_loss = running_loss / total
    top1 = correct / total
    top5 = correct_top5 / total
    return avg_loss, top1, top5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Device
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    print(f"Device: {device}")

    # Output directory
    run_dir = os.path.join(args.project, args.name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output: {run_dir}")

    # Transforms
    train_tf, val_tf = build_transforms(args.imgsz)

    # Datasets
    train_dir = os.path.join(args.data, "train")
    val_dir = os.path.join(args.data, "val")

    if not os.path.isdir(train_dir):
        print(f"ERROR: Training directory not found: {train_dir}")
        sys.exit(1)
    if not os.path.isdir(val_dir):
        print(f"ERROR: Validation directory not found: {val_dir}")
        sys.exit(1)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tf)

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes  # sorted folder names
    print(f"Classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # Class mapping (name -> index)
    class_mapping = {name: idx for idx, name in enumerate(class_names)}

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Model
    model = FGVDClassifier(args.backbone, num_classes, pretrained=True)
    model = model.to(device)
    print(f"Model: {args.backbone} -> {num_classes} classes")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    # Class weights + loss
    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    # Optimizer (differential LR)
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_factor},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    # Scheduler: linear warmup -> cosine annealing
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=args.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[args.warmup_epochs],
    )

    # Mixed precision
    scaler = GradScaler()

    # Resume
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    # Training log
    log = []
    patience_counter = 0

    # Save config
    config = {
        "backbone": args.backbone,
        "num_classes": num_classes,
        "input_size": args.imgsz,
        "lr": args.lr,
        "backbone_lr_factor": args.backbone_lr_factor,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
    }

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  LR: head={args.lr}, backbone={args.lr * args.backbone_lr_factor}")
    print(f"  Early stopping: patience={args.patience}")
    print()

    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Current LR
        current_lr = optimizer.param_groups[1]["lr"]  # head LR

        # Train
        train_loss, train_top1, train_top5 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
        )

        # Validate
        val_loss, val_top1, val_top5 = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_top1": round(train_top1, 4),
            "train_top5": round(train_top5, 4),
            "val_loss": round(val_loss, 4),
            "val_top1": round(val_top1, 4),
            "val_top5": round(val_top5, 4),
            "lr": round(current_lr, 8),
            "epoch_time_s": round(epoch_time, 1),
        }
        log.append(entry)

        # Print
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} top1={train_top1:.4f} top5={train_top5:.4f} | "
            f"val_loss={val_loss:.4f} top1={val_top1:.4f} top5={val_top5:.4f} | "
            f"lr={current_lr:.6f} | {epoch_time:.0f}s"
        )

        # Save best
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "class_mapping": class_mapping,
                "class_names": class_names,
                "config": config,
            }
            best_path = os.path.join(run_dir, "best.pt")
            torch.save(checkpoint, best_path)
            print(f"  -> New best! val_top1={val_top1:.4f}  Saved to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

        # Save last
        last_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "class_mapping": class_mapping,
            "class_names": class_names,
            "config": config,
        }
        torch.save(last_checkpoint, os.path.join(run_dir, "last.pt"))

        # Save log
        with open(os.path.join(run_dir, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2)

    # Final summary
    total_time = time.time() - t0
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time:      {total_time / 60:.1f} min")
    print(f"  Best val top-1:  {best_val_acc:.4f}")
    print(f"  Best checkpoint: {os.path.join(run_dir, 'best.pt')}")
    print(f"  Training log:    {os.path.join(run_dir, 'training_log.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
