"""
Prepare V3 dataset: merge rare FGVD classes into manufacturer-level groups.

Strategy:
  1. Classes with >= min_samples: keep as-is (e.g., car_MarutiSuzuki_Swift)
  2. Classes with < min_samples but same manufacturer has other classes:
     merge into "{type}_{manufacturer}_Other" (e.g., car_MarutiSuzuki_Other)
  3. Manufacturers with total < min_samples across all models: drop entirely

This dramatically reduces class count while preserving most training data.

Usage:
    python prepare_v3_dataset.py \
        --input /workspace/datasets/fgvd_crops \
        --output /workspace/datasets/fgvd_crops_v3 \
        --min-samples 30
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Merge rare FGVD classes for V3 training")
    p.add_argument("--input", required=True, help="Path to fgvd_crops (from convert_fgvd_to_crops.py)")
    p.add_argument("--output", required=True, help="Path to write merged dataset")
    p.add_argument("--min-samples", type=int, default=30,
                    help="Min samples per class to keep as standalone (default: 30)")
    p.add_argument("--min-manufacturer", type=int, default=15,
                    help="Min total samples per manufacturer group (default: 15)")
    return p.parse_args()


def parse_class_name(class_name: str) -> dict:
    """Parse 'car_MarutiSuzuki_Swift' -> {type, manufacturer, model}."""
    parts = class_name.split("_", 2)
    result = {"type": "", "manufacturer": "", "model": "", "raw": class_name}
    if len(parts) >= 1:
        result["type"] = parts[0]
    if len(parts) >= 2:
        result["manufacturer"] = parts[1]
    if len(parts) >= 3:
        result["model"] = parts[2]
    return result


def analyze_dataset(input_dir: str) -> dict:
    """Analyze class distribution across all splits."""
    stats = {}  # {class_name: {split: count}}

    for split in ("train", "val", "test"):
        split_dir = os.path.join(input_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            count = len([f for f in os.listdir(class_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            if class_name not in stats:
                stats[class_name] = {}
            stats[class_name][split] = count

    return stats


def build_merge_plan(stats: dict, min_samples: int, min_manufacturer: int) -> dict:
    """Decide how to merge/keep/drop each class.

    Returns {old_class_name: new_class_name} mapping.
    None value means drop.
    """
    # Compute total samples per class
    totals = {}
    for cls, splits in stats.items():
        totals[cls] = sum(splits.values())

    # Parse hierarchy
    parsed = {cls: parse_class_name(cls) for cls in stats}

    # Group by manufacturer
    manufacturer_classes = defaultdict(list)  # {(type, mfr): [class_names]}
    for cls, info in parsed.items():
        key = (info["type"], info["manufacturer"])
        manufacturer_classes[key].append(cls)

    # Build merge plan
    merge_plan = {}
    kept_standalone = 0
    merged_into_other = 0
    dropped = 0

    for (vtype, mfr), classes in manufacturer_classes.items():
        # Total samples for this manufacturer
        mfr_total = sum(totals[c] for c in classes)

        if mfr_total < min_manufacturer:
            # Drop entire manufacturer — too few samples
            for cls in classes:
                merge_plan[cls] = None
                dropped += 1
            continue

        # Check each model within manufacturer
        other_name = f"{vtype}_{mfr}_Other"
        other_candidates = []

        for cls in classes:
            if totals[cls] >= min_samples:
                merge_plan[cls] = cls  # keep as-is
                kept_standalone += 1
            else:
                other_candidates.append(cls)

        # Merge small classes into "Other"
        if other_candidates:
            other_total = sum(totals[c] for c in other_candidates)
            if other_total >= min_manufacturer:
                for cls in other_candidates:
                    merge_plan[cls] = other_name
                    merged_into_other += 1
            else:
                # Even merged they're too small — drop
                for cls in other_candidates:
                    merge_plan[cls] = None
                    dropped += 1

    print(f"Merge plan summary:")
    print(f"  Kept standalone:    {kept_standalone}")
    print(f"  Merged into Other:  {merged_into_other}")
    print(f"  Dropped entirely:   {dropped}")

    # Count unique output classes
    output_classes = set(v for v in merge_plan.values() if v is not None)
    print(f"  Output classes:     {len(output_classes)}")

    return merge_plan


def apply_merge(input_dir: str, output_dir: str, merge_plan: dict, stats: dict):
    """Copy/merge files according to merge plan."""
    total_copied = 0
    total_dropped = 0

    for split in ("train", "val", "test"):
        split_in = os.path.join(input_dir, split)
        split_out = os.path.join(output_dir, split)

        if not os.path.isdir(split_in):
            continue

        for old_class in tqdm(os.listdir(split_in), desc=f"Merging {split}"):
            old_dir = os.path.join(split_in, old_class)
            if not os.path.isdir(old_dir):
                continue

            new_class = merge_plan.get(old_class)
            if new_class is None:
                # Drop
                count = len(os.listdir(old_dir))
                total_dropped += count
                continue

            # Copy files to new class folder
            new_dir = os.path.join(split_out, new_class)
            os.makedirs(new_dir, exist_ok=True)

            for fname in os.listdir(old_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                src = os.path.join(old_dir, fname)
                # Prefix with old class name to avoid filename collisions when merging
                if old_class != new_class:
                    dst_name = f"{old_class}__{fname}"
                else:
                    dst_name = fname
                dst = os.path.join(new_dir, dst_name)
                shutil.copy2(src, dst)
                total_copied += 1

    print(f"\nFiles copied: {total_copied}, dropped: {total_dropped}")
    return total_copied


def generate_metadata(output_dir: str):
    """Generate class_mapping.json and dataset_stats.json for merged dataset."""
    all_classes = set()
    split_stats = {}

    for split in ("train", "val", "test"):
        split_dir = os.path.join(output_dir, split)
        if not os.path.isdir(split_dir):
            continue
        split_stats[split] = {}
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            count = len([f for f in os.listdir(class_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            split_stats[split][class_name] = count
            all_classes.add(class_name)

    # Class mapping
    class_mapping = {name: idx for idx, name in enumerate(sorted(all_classes))}
    with open(os.path.join(output_dir, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f, indent=2)

    # Dataset stats
    total_crops = 0
    ds_stats = {"num_classes": len(class_mapping), "splits": {}}
    for split, counts in split_stats.items():
        split_total = sum(counts.values())
        total_crops += split_total
        ds_stats["splits"][split] = {"total": split_total, "class_counts": counts}
    ds_stats["total_crops"] = total_crops

    with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
        json.dump(ds_stats, f, indent=2)

    # Hierarchy
    hierarchy = {}
    for cls in sorted(all_classes):
        hierarchy[cls] = parse_class_name(cls)
    with open(os.path.join(output_dir, "hierarchy.json"), "w") as f:
        json.dump(hierarchy, f, indent=2)

    return ds_stats


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input not found: {input_dir}")
        sys.exit(1)

    # Step 1: Analyze
    print("Analyzing dataset...")
    stats = analyze_dataset(input_dir)
    print(f"Found {len(stats)} classes")

    # Show distribution
    totals = {cls: sum(splits.values()) for cls, splits in stats.items()}
    sorted_classes = sorted(totals.items(), key=lambda x: -x[1])

    print(f"\nTop 10 classes:")
    for cls, count in sorted_classes[:10]:
        print(f"  {cls}: {count}")
    print(f"\nBottom 10 classes:")
    for cls, count in sorted_classes[-10:]:
        print(f"  {cls}: {count}")

    bins = {">=100": 0, "50-99": 0, "30-49": 0, "15-29": 0, "<15": 0}
    for cls, count in totals.items():
        if count >= 100:
            bins[">=100"] += 1
        elif count >= 50:
            bins["50-99"] += 1
        elif count >= 30:
            bins["30-49"] += 1
        elif count >= 15:
            bins["15-29"] += 1
        else:
            bins["<15"] += 1
    print(f"\nClass size distribution:")
    for label, count in bins.items():
        print(f"  {label:>6s} samples: {count} classes")

    # Step 2: Build merge plan
    print(f"\nBuilding merge plan (min_samples={args.min_samples}, "
          f"min_manufacturer={args.min_manufacturer})...")
    merge_plan = build_merge_plan(stats, args.min_samples, args.min_manufacturer)

    # Step 3: Apply
    print(f"\nApplying merge to {output_dir}...")
    apply_merge(input_dir, output_dir, merge_plan, stats)

    # Step 4: Generate metadata
    print("\nGenerating metadata...")
    ds_stats = generate_metadata(output_dir)

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("V3 DATASET READY")
    print("=" * 60)
    print(f"  Output:      {output_dir}")
    print(f"  Classes:     {ds_stats['num_classes']}")
    print(f"  Total crops: {ds_stats['total_crops']}")
    for split, info in ds_stats["splits"].items():
        print(f"  {split:>5s}:       {info['total']}")
    print()
    print(f"  class_mapping.json:  {os.path.join(output_dir, 'class_mapping.json')}")
    print(f"  dataset_stats.json:  {os.path.join(output_dir, 'dataset_stats.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
