"""
Convert IDD Detection dataset (Pascal VOC XML) to YOLO format.

Usage:
    python convert_idd_to_yolo.py --input /path/to/IDD_Detection --output /path/to/yolo_dataset
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

from lxml import etree
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Class mapping: IDD label name -> YOLO class ID (None = skip)
# ---------------------------------------------------------------------------
CLASS_MAPPING = {
    # Persons
    'person': 0,
    'rider': 0,
    # Two-wheelers
    'bicycle': 1,
    'motorcycle': 2,
    # 4-wheelers
    'car': 3,
    'autorickshaw': 4,
    'auto rickshaw': 4,
    'auto-rickshaw': 4,
    'bus': 5,
    'truck': 6,
    'caravan': 7,
    'trailer': 7,
    'vehicle fallback': 7,
    # Other
    'animal': 8,
    # Skip these
    'traffic light': None,
    'traffic sign': None,
    'billboard': None,
    'train': None,
}

CLASS_NAMES = [
    'person', 'bicycle', 'motorcycle', 'car',
    'autorickshaw', 'bus', 'truck', 'vehicle_other', 'animal',
]

NUM_CLASSES = len(CLASS_NAMES)

# Floating-point clamp epsilon
EPS = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert IDD Detection (VOC XML) to YOLO format',
    )
    parser.add_argument(
        '--input', required=True,
        help='Path to extracted IDD_Detection folder',
    )
    parser.add_argument(
        '--output', required=True,
        help='Path to write YOLO-format dataset',
    )
    parser.add_argument(
        '--val-split', type=float, default=0.15,
        help='Validation split ratio (only used if IDD val.txt is missing)',
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Number of parallel workers (default: cpu_count)',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Discovery: find all XML files and their matching images
# ---------------------------------------------------------------------------
def discover_annotations(input_dir: str) -> list[tuple[str, str]]:
    """Walk input_dir, find .xml files in **/Annotations/ folders,
    match each to its .jpg in the corresponding JPEGImages folder.
    Returns list of (xml_path, image_path) tuples."""

    pairs = []
    input_path = Path(input_dir)

    for xml_file in input_path.rglob('*.xml'):
        # Only consider XMLs inside an Annotations folder
        if 'Annotations' not in xml_file.parts:
            continue

        # Build corresponding image path: replace "Annotations" with "JPEGImages"
        parts = list(xml_file.parts)
        # Find the last occurrence of "Annotations" in the path
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == 'Annotations':
                parts[i] = 'JPEGImages'
                break

        img_path = Path(*parts).with_suffix('.jpg')

        if img_path.exists():
            pairs.append((str(xml_file), str(img_path)))
        else:
            # Try .png as fallback
            img_png = img_path.with_suffix('.png')
            if img_png.exists():
                pairs.append((str(xml_file), str(img_png)))
            # else: will be logged later as missing image

    return pairs


# ---------------------------------------------------------------------------
# Load IDD train/val splits if available
# ---------------------------------------------------------------------------
def load_idd_splits(input_dir: str) -> tuple[set[str] | None, set[str] | None]:
    """Look for ImageSets/Main/train.txt and val.txt.
    Returns (train_stems, val_stems) or (None, None) if not found."""

    input_path = Path(input_dir)
    train_stems = None
    val_stems = None

    # Search for ImageSets/Main recursively (IDD may nest these)
    for txt_file in input_path.rglob('train.txt'):
        if 'ImageSets' in str(txt_file):
            train_stems = set()
            for line in txt_file.read_text().strip().splitlines():
                stem = line.strip().split()[0]  # some VOC files have "name flag"
                if stem:
                    train_stems.add(stem)
            break

    for txt_file in input_path.rglob('val.txt'):
        if 'ImageSets' in str(txt_file):
            val_stems = set()
            for line in txt_file.read_text().strip().splitlines():
                stem = line.strip().split()[0]
                if stem:
                    val_stems.add(stem)
            break

    if train_stems and val_stems:
        print(f"Found IDD splits: {len(train_stems)} train, {len(val_stems)} val stems")
        return train_stems, val_stems

    return None, None


# ---------------------------------------------------------------------------
# Parse a single XML annotation
# ---------------------------------------------------------------------------
def parse_single_xml(args: tuple) -> dict | None:
    """Parse one (xml_path, image_path) pair. Returns a dict with results
    or None on failure."""

    xml_path, image_path = args

    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        return {'error': f'Malformed XML: {xml_path} ({e})'}

    # Get image dimensions from XML
    size_el = root.find('size')
    if size_el is None:
        return {'error': f'No <size> element: {xml_path}'}

    try:
        img_w = int(size_el.findtext('width', '0'))
        img_h = int(size_el.findtext('height', '0'))
    except ValueError:
        return {'error': f'Invalid size values: {xml_path}'}

    if img_w <= 0 or img_h <= 0:
        return {'error': f'Invalid dimensions {img_w}x{img_h}: {xml_path}'}

    # Get the filename/stem for split matching
    xml_stem = Path(xml_path).stem

    # Parse objects
    annotations = []
    unmapped = defaultdict(int)
    class_counts = defaultdict(int)

    for obj in root.iter('object'):
        name = obj.findtext('name', '').strip().lower()

        if name in CLASS_MAPPING:
            cls_id = CLASS_MAPPING[name]
            if cls_id is None:
                continue  # explicitly skipped class
        else:
            unmapped[name] += 1
            continue

        bbox = obj.find('bndbox')
        if bbox is None:
            continue

        try:
            xmin = float(bbox.findtext('xmin', '0'))
            ymin = float(bbox.findtext('ymin', '0'))
            xmax = float(bbox.findtext('xmax', '0'))
            ymax = float(bbox.findtext('ymax', '0'))
        except ValueError:
            continue

        # Validate bbox
        if xmax <= xmin or ymax <= ymin:
            continue
        if xmin < 0 or ymin < 0:
            continue

        # Convert to YOLO normalized format
        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # Clamp to [0, 1]
        x_center = max(EPS, min(1.0 - EPS, x_center))
        y_center = max(EPS, min(1.0 - EPS, y_center))
        w = max(EPS, min(1.0 - EPS, w))
        h = max(EPS, min(1.0 - EPS, h))

        annotations.append((cls_id, x_center, y_center, w, h))
        class_counts[cls_id] += 1

    return {
        'xml_path': xml_path,
        'image_path': image_path,
        'stem': xml_stem,
        'annotations': annotations,
        'class_counts': dict(class_counts),
        'unmapped': dict(unmapped),
    }


# ---------------------------------------------------------------------------
# Build flat filename from image path relative to input dir
# ---------------------------------------------------------------------------
def make_flat_name(image_path: str, input_dir: str) -> str:
    """Replace path separators with underscores to create a flat filename."""
    rel = os.path.relpath(image_path, input_dir)
    # Replace both / and \ with underscore
    flat = rel.replace(os.sep, '_').replace('/', '_').replace('\\', '_')
    return flat


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

    # Step 1: Discover all XML-image pairs
    print("Discovering annotations...")
    pairs = discover_annotations(input_dir)
    print(f"Found {len(pairs)} XML-image pairs")

    if not pairs:
        print("ERROR: No annotation pairs found. Check --input path.")
        sys.exit(1)

    # Step 2: Load IDD splits
    train_stems, val_stems = load_idd_splits(input_dir)
    use_idd_splits = train_stems is not None and val_stems is not None

    # Step 3: Parse all XMLs in parallel
    num_workers = args.workers or min(cpu_count(), 8)
    print(f"Parsing {len(pairs)} annotations with {num_workers} workers...")

    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(parse_single_xml, pairs, chunksize=64),
            total=len(pairs),
            desc='Parsing XMLs',
        ):
            results.append(result)

    # Separate errors from valid results
    errors = [r for r in results if r is None or 'error' in r]
    valid = [r for r in results if r is not None and 'error' not in r]

    for e in errors:
        if e and 'error' in e:
            print(f"  SKIP: {e['error']}")

    print(f"Parsed {len(valid)} valid annotations, {len(errors)} errors")

    # Step 4: Filter out images with zero valid annotations
    with_annots = [r for r in valid if len(r['annotations']) > 0]
    skipped_empty = len(valid) - len(with_annots)
    print(f"Images with annotations: {len(with_annots)}, skipped (empty): {skipped_empty}")

    # Step 5: Assign train/val splits
    if use_idd_splits:
        train_items = []
        val_items = []
        unmatched = []

        # Build a combined set of all known stems for matching
        all_idd_stems = train_stems | val_stems

        for item in with_annots:
            stem = item['stem']
            # Try direct match first
            if stem in train_stems:
                train_items.append(item)
            elif stem in val_stems:
                val_items.append(item)
            else:
                # IDD stems might include relative paths; try matching just filename
                unmatched.append(item)

        # Put unmatched into train by default
        if unmatched:
            print(f"  {len(unmatched)} images not in IDD splits, adding to train")
            train_items.extend(unmatched)
    else:
        import random
        random.seed(42)
        random.shuffle(with_annots)
        split_idx = int(len(with_annots) * (1.0 - args.val_split))
        train_items = with_annots[:split_idx]
        val_items = with_annots[split_idx:]
        print(f"Random split: {len(train_items)} train, {len(val_items)} val")

    print(f"Final split: {len(train_items)} train, {len(val_items)} val")

    # Step 6: Create output directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # Step 7: Write images and labels
    all_unmapped = defaultdict(int)
    train_class_counts = defaultdict(int)
    val_class_counts = defaultdict(int)

    def write_split(items, split_name, class_counter):
        print(f"Writing {split_name} split ({len(items)} images)...")
        for item in tqdm(items, desc=f'Writing {split_name}'):
            flat_name = make_flat_name(item['image_path'], input_dir)
            flat_stem = Path(flat_name).stem

            # Copy image
            dst_img = os.path.join(output_dir, 'images', split_name, flat_name)
            shutil.copy2(item['image_path'], dst_img)

            # Write label
            label_name = flat_stem + '.txt'
            dst_label = os.path.join(output_dir, 'labels', split_name, label_name)
            with open(dst_label, 'w') as f:
                for cls_id, xc, yc, w, h in item['annotations']:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            # Accumulate class counts
            for cls_id, count in item['class_counts'].items():
                class_counter[cls_id] += count

            # Accumulate unmapped
            for name, count in item['unmapped'].items():
                all_unmapped[name] += count

    write_split(train_items, 'train', train_class_counts)
    write_split(val_items, 'val', val_class_counts)

    # Step 8: Generate data.yaml
    data_yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {NUM_CLASSES}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    # Step 9: Generate conversion_stats.json
    stats = {
        'total_images_processed': len(valid),
        'total_skipped_no_annotations': skipped_empty,
        'total_skipped_errors': len(errors),
        'train_images': len(train_items),
        'val_images': len(val_items),
        'train_class_counts': {
            CLASS_NAMES[k]: v for k, v in sorted(train_class_counts.items())
        },
        'val_class_counts': {
            CLASS_NAMES[k]: v for k, v in sorted(val_class_counts.items())
        },
        'unmapped_classes': dict(all_unmapped),
        'used_idd_splits': use_idd_splits,
    }

    stats_path = os.path.join(output_dir, 'conversion_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Step 10: Print summary
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Output:       {output_dir}")
    print(f"  Train images: {len(train_items)}")
    print(f"  Val images:   {len(val_items)}")
    print(f"  Skipped (no annotations): {skipped_empty}")
    print(f"  Skipped (errors):         {len(errors)}")
    print()
    print("Per-class counts (train):")
    for cls_id in range(NUM_CLASSES):
        name = CLASS_NAMES[cls_id]
        count = train_class_counts.get(cls_id, 0)
        print(f"  [{cls_id}] {name:>15s}: {count:>8,}")
    print()
    if all_unmapped:
        print("Unmapped classes encountered (skipped):")
        for name, count in sorted(all_unmapped.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count}")
    print()
    print(f"data.yaml:           {data_yaml_path}")
    print(f"conversion_stats:    {stats_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
