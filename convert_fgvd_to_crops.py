"""
Convert IDD-FGVD dataset to classification-ready cropped images.

Reads FGVD annotations (Pascal VOC XML with fine-grained labels like
'car_MarutiSuzuki_Ciaz'), crops each vehicle from the scene image,
and organises crops into an ImageFolder structure:

    output/
      train/
        car_MarutiSuzuki_Ciaz/
          img001_crop0.jpg
          ...
      val/
        ...
      test/
        ...
      class_mapping.json
      hierarchy.json
      dataset_stats.json

Usage:
    python convert_fgvd_to_crops.py \
        --input /workspace/datasets/FGVD \
        --output /workspace/datasets/fgvd_crops \
        --min-samples 5 --padding 0.1
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
from lxml import etree
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Convert FGVD annotations to cropped vehicle images",
    )
    p.add_argument("--input", required=True, help="Path to extracted FGVD folder")
    p.add_argument("--output", required=True, help="Path to write crop dataset")
    p.add_argument(
        "--min-samples", type=int, default=5,
        help="Drop classes with fewer samples than this (default: 5)",
    )
    p.add_argument(
        "--padding", type=float, default=0.1,
        help="Bbox padding ratio on each side (default: 0.1 = 10%%)",
    )
    p.add_argument(
        "--quality", type=int, default=95,
        help="JPEG save quality (default: 95)",
    )
    p.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Auto-detect dataset structure
# ---------------------------------------------------------------------------
def discover_structure(input_dir: str) -> dict:
    """Detect how FGVD is organized on disk.

    Supports two layouts:
      A) Split-based:  {input}/train/annotations/, {input}/val/annotations/, ...
      B) Flat:         {input}/Annotations/, {input}/JPEGImages/
         with optional ImageSets/Main/train.txt, val.txt, test.txt

    Returns dict with 'layout' ('split' or 'flat') and discovered paths.
    """
    root = Path(input_dir)
    info = {"root": input_dir, "layout": None, "splits": {}}

    # Check layout A: per-split folders
    for split in ("train", "val", "test"):
        annot_dir = None
        image_dir = None

        # Try common patterns
        for ann_name in ("annotations", "Annotations"):
            d = root / split / ann_name
            if d.is_dir():
                annot_dir = str(d)
                break
        for img_name in ("images", "JPEGImages"):
            d = root / split / img_name
            if d.is_dir():
                image_dir = str(d)
                break

        if annot_dir and image_dir:
            info["splits"][split] = {
                "annotations": annot_dir,
                "images": image_dir,
            }

    if info["splits"]:
        info["layout"] = "split"
        return info

    # Check layout B: single Annotations + JPEGImages
    for ann_name in ("annotations", "Annotations"):
        d = root / ann_name
        if d.is_dir():
            for img_name in ("images", "JPEGImages"):
                id_ = root / img_name
                if id_.is_dir():
                    info["layout"] = "flat"
                    info["annotations_dir"] = str(d)
                    info["images_dir"] = str(id_)
                    # Look for split files
                    for split in ("train", "val", "test"):
                        for sets_path in root.rglob(f"{split}.txt"):
                            if "ImageSets" in str(sets_path) or "imagesets" in str(sets_path).lower():
                                info.setdefault("split_files", {})[split] = str(sets_path)
                                break
                    return info

    # Fallback: search recursively for any XML + image pairs
    xmls = list(root.rglob("*.xml"))
    if xmls:
        info["layout"] = "recursive"
        info["xml_files"] = [str(x) for x in xmls]
        return info

    return info


# ---------------------------------------------------------------------------
# Parse FGVD hierarchy from label string
# ---------------------------------------------------------------------------
def parse_label(label: str) -> dict:
    """Parse a FGVD label like 'car_MarutiSuzuki_Ciaz' into components.

    Returns {'raw': ..., 'vehicle_type': ..., 'manufacturer': ..., 'model': ...}
    """
    parts = label.split("_", 2)
    result = {"raw": label, "vehicle_type": "", "manufacturer": "", "model": ""}

    if len(parts) >= 1:
        result["vehicle_type"] = parts[0].lower()
    if len(parts) >= 2:
        result["manufacturer"] = parts[1]
    if len(parts) >= 3:
        result["model"] = parts[2]

    return result


# ---------------------------------------------------------------------------
# Parse a single XML annotation file
# ---------------------------------------------------------------------------
def parse_xml(xml_path: str) -> list[dict]:
    """Parse a Pascal VOC XML file and return list of object dicts.

    Each dict: {label, xmin, ymin, xmax, ymax}
    """
    try:
        tree = etree.parse(xml_path)
    except Exception as e:
        print(f"  WARN: Cannot parse {xml_path}: {e}")
        return []

    root = tree.getroot()
    objects = []

    for obj in root.iter("object"):
        name = obj.findtext("name", "").strip()
        if not name:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        try:
            xmin = int(float(bbox.findtext("xmin", "0")))
            ymin = int(float(bbox.findtext("ymin", "0")))
            xmax = int(float(bbox.findtext("xmax", "0")))
            ymax = int(float(bbox.findtext("ymax", "0")))
        except (ValueError, TypeError):
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        objects.append({
            "label": name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })

    return objects


# ---------------------------------------------------------------------------
# Crop a single image's objects
# ---------------------------------------------------------------------------
def process_one_image(args: tuple) -> dict:
    """Worker function: load image, crop all objects, save to class folders.

    args = (xml_path, image_path, output_dir, split, padding, quality)
    Returns stats dict.
    """
    xml_path, image_path, output_dir, split, padding, quality = args

    stats = {"total": 0, "saved": 0, "errors": 0, "classes": defaultdict(int)}

    objects = parse_xml(xml_path)
    if not objects:
        return stats

    img = cv2.imread(image_path)
    if img is None:
        stats["errors"] += 1
        return stats

    img_h, img_w = img.shape[:2]
    img_stem = Path(image_path).stem

    for idx, obj in enumerate(objects):
        stats["total"] += 1
        label = obj["label"]

        # Compute padded bbox
        bw = obj["xmax"] - obj["xmin"]
        bh = obj["ymax"] - obj["ymin"]
        pad_x = int(bw * padding)
        pad_y = int(bh * padding)

        x1 = max(0, obj["xmin"] - pad_x)
        y1 = max(0, obj["ymin"] - pad_y)
        x2 = min(img_w, obj["xmax"] + pad_x)
        y2 = min(img_h, obj["ymax"] + pad_y)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            stats["errors"] += 1
            continue

        # Sanitize label for use as folder name
        safe_label = label.replace("/", "_").replace("\\", "_").replace(" ", "_")

        # Save
        class_dir = os.path.join(output_dir, split, safe_label)
        os.makedirs(class_dir, exist_ok=True)
        crop_name = f"{img_stem}_crop{idx}.jpg"
        crop_path = os.path.join(class_dir, crop_name)

        cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
        stats["saved"] += 1
        stats["classes"][safe_label] += 1

    return stats


# ---------------------------------------------------------------------------
# Find image for an XML file
# ---------------------------------------------------------------------------
def find_image_for_xml(xml_path: str, images_dir: str) -> str | None:
    """Given an XML annotation path, find the corresponding image."""
    stem = Path(xml_path).stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Build work items from discovered structure
# ---------------------------------------------------------------------------
def build_work_items(structure: dict, output_dir: str, padding: float, quality: int) -> list[tuple]:
    """Build list of (xml_path, image_path, output_dir, split, padding, quality) tuples."""
    items = []

    if structure["layout"] == "split":
        for split, paths in structure["splits"].items():
            annot_dir = paths["annotations"]
            image_dir = paths["images"]
            for xml_file in Path(annot_dir).rglob("*.xml"):
                img_path = find_image_for_xml(str(xml_file), image_dir)
                if img_path:
                    items.append((str(xml_file), img_path, output_dir, split, padding, quality))

    elif structure["layout"] == "flat":
        annot_dir = structure["annotations_dir"]
        image_dir = structure["images_dir"]
        split_files = structure.get("split_files", {})

        # Load split assignments
        stem_to_split = {}
        for split, txt_path in split_files.items():
            with open(txt_path) as f:
                for line in f:
                    stem = line.strip().split()[0] if line.strip() else ""
                    if stem:
                        stem_to_split[stem] = split

        for xml_file in Path(annot_dir).rglob("*.xml"):
            img_path = find_image_for_xml(str(xml_file), image_dir)
            if not img_path:
                continue
            stem = xml_file.stem
            split = stem_to_split.get(stem, "train")  # default to train
            items.append((str(xml_file), img_path, output_dir, split, padding, quality))

    elif structure["layout"] == "recursive":
        # Best-effort: find images next to XMLs
        for xml_str in structure["xml_files"]:
            xml_p = Path(xml_str)
            # Try sibling directories
            parent = xml_p.parent
            for img_dir_name in ("JPEGImages", "images", "."):
                img_dir = parent.parent / img_dir_name if img_dir_name != "." else parent
                if img_dir.is_dir():
                    img_path = find_image_for_xml(xml_str, str(img_dir))
                    if img_path:
                        items.append((xml_str, img_path, output_dir, "train", padding, quality))
                        break

    return items


# ---------------------------------------------------------------------------
# Post-processing: filter rare classes, build metadata
# ---------------------------------------------------------------------------
def postprocess(output_dir: str, min_samples: int) -> dict:
    """Filter rare classes and generate metadata files.

    Returns dataset statistics dict.
    """
    all_classes = defaultdict(lambda: defaultdict(int))  # {split: {class: count}}
    hierarchy = {}

    # Count crops per class per split
    for split in ("train", "val", "test"):
        split_dir = os.path.join(output_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            count = len([f for f in os.listdir(class_dir) if f.endswith((".jpg", ".png"))])
            all_classes[split][class_name] = count

            if class_name not in hierarchy:
                hierarchy[class_name] = parse_label(class_name)

    # Compute total samples per class across all splits
    total_per_class = defaultdict(int)
    for split_counts in all_classes.values():
        for cls, cnt in split_counts.items():
            total_per_class[cls] += cnt

    # Filter rare classes
    dropped_classes = []
    for cls, total in total_per_class.items():
        if total < min_samples:
            dropped_classes.append((cls, total))
            # Remove the folders
            for split in ("train", "val", "test"):
                class_dir = os.path.join(output_dir, split, cls)
                if os.path.isdir(class_dir):
                    import shutil
                    shutil.rmtree(class_dir)

    # Rebuild counts after filtering
    kept_classes = sorted(set(total_per_class.keys()) - {c for c, _ in dropped_classes})

    # Build class mapping (sorted alphabetically)
    class_mapping = {name: idx for idx, name in enumerate(kept_classes)}

    # Recount
    final_stats = {"splits": {}, "num_classes": len(kept_classes), "total_crops": 0}
    for split in ("train", "val", "test"):
        split_dir = os.path.join(output_dir, split)
        if not os.path.isdir(split_dir):
            continue
        split_total = 0
        split_class_counts = {}
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            count = len([f for f in os.listdir(class_dir) if f.endswith((".jpg", ".png"))])
            split_class_counts[class_name] = count
            split_total += count
        final_stats["splits"][split] = {
            "total": split_total,
            "class_counts": split_class_counts,
        }
        final_stats["total_crops"] += split_total

    final_stats["dropped_classes"] = [
        {"name": name, "count": cnt} for name, cnt in dropped_classes
    ]

    # Save metadata files
    mapping_path = os.path.join(output_dir, "class_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=2)

    hierarchy_path = os.path.join(output_dir, "hierarchy.json")
    filtered_hierarchy = {k: v for k, v in hierarchy.items() if k in class_mapping}
    with open(hierarchy_path, "w") as f:
        json.dump(filtered_hierarchy, f, indent=2)

    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(final_stats, f, indent=2)

    return final_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    # Step 1: discover dataset structure
    print("Discovering FGVD dataset structure...")
    structure = discover_structure(input_dir)

    if structure["layout"] is None:
        print("ERROR: Could not detect FGVD dataset layout.")
        print("Expected one of:")
        print("  A) {input}/train/annotations/ + {input}/train/images/")
        print("  B) {input}/Annotations/ + {input}/JPEGImages/")
        print(f"Contents of {input_dir}:")
        for item in sorted(os.listdir(input_dir)):
            print(f"  {item}")
        sys.exit(1)

    print(f"Detected layout: {structure['layout']}")
    if structure["layout"] == "split":
        for split, paths in structure["splits"].items():
            print(f"  {split}: {paths['annotations']}")
    elif structure["layout"] == "flat":
        print(f"  Annotations: {structure['annotations_dir']}")
        print(f"  Images: {structure['images_dir']}")
        if "split_files" in structure:
            for split, path in structure["split_files"].items():
                print(f"  Split file ({split}): {path}")

    # Step 2: build work items
    print("\nBuilding work list...")
    work_items = build_work_items(structure, output_dir, args.padding, args.quality)
    print(f"Found {len(work_items)} image-annotation pairs")

    if not work_items:
        print("ERROR: No valid image-annotation pairs found.")
        sys.exit(1)

    # Count per split
    split_counts = defaultdict(int)
    for _, _, _, split, _, _ in work_items:
        split_counts[split] += 1
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count} images")

    # Step 3: process in parallel
    num_workers = args.workers or min(cpu_count(), 8)
    print(f"\nCropping vehicles with {num_workers} workers...")

    total_stats = {"total": 0, "saved": 0, "errors": 0}

    with Pool(num_workers) as pool:
        for stats in tqdm(
            pool.imap_unordered(process_one_image, work_items, chunksize=32),
            total=len(work_items),
            desc="Cropping",
        ):
            total_stats["total"] += stats["total"]
            total_stats["saved"] += stats["saved"]
            total_stats["errors"] += stats["errors"]

    print(f"\nCropping done: {total_stats['saved']} crops saved, "
          f"{total_stats['errors']} errors, {total_stats['total']} objects total")

    # Step 4: post-process — filter rare classes, build metadata
    print(f"\nPost-processing (min_samples={args.min_samples})...")
    final_stats = postprocess(output_dir, args.min_samples)

    # Step 5: summary
    print("\n" + "=" * 60)
    print("FGVD CROP CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Output:       {output_dir}")
    print(f"  Num classes:  {final_stats['num_classes']}")
    print(f"  Total crops:  {final_stats['total_crops']}")
    for split, info in final_stats["splits"].items():
        print(f"  {split:>5s} crops:  {info['total']}")
    if final_stats["dropped_classes"]:
        print(f"  Dropped {len(final_stats['dropped_classes'])} rare classes "
              f"(< {args.min_samples} samples)")
    print()
    print(f"  class_mapping.json:  {os.path.join(output_dir, 'class_mapping.json')}")
    print(f"  hierarchy.json:      {os.path.join(output_dir, 'hierarchy.json')}")
    print(f"  dataset_stats.json:  {os.path.join(output_dir, 'dataset_stats.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
