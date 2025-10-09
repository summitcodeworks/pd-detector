"""
Prepare Dataset from Uploads Folder

This script collects images from uploads/ folder structure and prepares them for training.
It handles the specific structure: uploads/train/ and uploads/test/ subdirectories.
"""

import os
import sys
import shutil
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mobile_panel_detector.detector.panel_detector import AdvancedPanelDetector


def convert_to_yolo_format(bbox, img_width, img_height):
    """Convert bbox (x, y, w, h) to YOLO format (class, x_center, y_center, width, height) normalized"""
    x, y, w, h = bbox
    
    # Calculate center point
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    
    # Normalize width and height
    width = w / img_width
    height = h / img_height
    
    # Clamp to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def collect_images_from_uploads(uploads_dir: str) -> List[Path]:
    """Recursively collect all images from uploads directory"""
    uploads_path = Path(uploads_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    all_images = []
    
    for ext in image_extensions:
        # Find all images recursively
        all_images.extend(uploads_path.rglob(f"*{ext}"))
    
    return all_images


def process_and_label_images(
    image_files: List[Path],
    output_images_dir: Path,
    output_labels_dir: Path,
    confidence_threshold: float = 0.25
) -> int:
    """Process images and generate YOLO labels"""
    
    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    print("Loading detector for auto-labeling...")
    detector = AdvancedPanelDetector()
    
    labeled_count = 0
    skipped_count = 0
    total = len(image_files)
    
    for idx, img_file in enumerate(image_files, 1):
        print(f"[{idx}/{total}] Processing: {img_file.name}...", end=" ")
        
        # Read image
        image = cv2.imread(str(img_file))
        if image is None:
            print("FAILED to read")
            skipped_count += 1
            continue
        
        height, width = image.shape[:2]
        
        # Detect panels
        detections, status_info = detector.detect(image)
        
        # Filter only panel detections (not display issues)
        panel_detections = [d for d in detections if 'issue_type' not in d and 'bbox' in d]
        
        # Filter by confidence
        panel_detections = [d for d in panel_detections if d.get('confidence', 0) >= confidence_threshold]
        
        if not panel_detections:
            print("No panels detected (skipped)")
            skipped_count += 1
            continue
        
        # Generate unique filename to avoid conflicts
        # Use original filename with parent directory to make it unique
        parent_name = img_file.parent.name
        unique_name = f"{parent_name}_{img_file.name}"
        
        # Copy image to output
        output_img_path = output_images_dir / unique_name
        cv2.imwrite(str(output_img_path), image)
        
        # Create label file
        label_file = output_labels_dir / f"{Path(unique_name).stem}.txt"
        with open(label_file, 'w') as f:
            for detection in panel_detections:
                bbox = detection['bbox']
                yolo_label = convert_to_yolo_format(bbox, width, height)
                f.write(yolo_label + '\n')
        
        print(f"OK ({len(panel_detections)} panels)")
        labeled_count += 1
    
    return labeled_count, skipped_count


def split_dataset(images_dir: Path, labels_dir: Path, train_ratio: float = 0.8):
    """Split dataset into train/val sets"""
    import random
    
    # Get all image files
    all_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_images)
    
    # Split
    split_idx = int(len(all_images) * train_ratio)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Create train/val directories
    train_images_dir = images_dir.parent / "train"
    val_images_dir = images_dir.parent / "val"
    train_labels_dir = labels_dir.parent / "train"
    val_labels_dir = labels_dir.parent / "val"
    
    for d in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Move files
    print("\nSplitting dataset into train/val...")
    
    for img in train_images:
        shutil.copy(img, train_images_dir / img.name)
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            shutil.copy(label, train_labels_dir / label.name)
    
    for img in val_images:
        shutil.copy(img, val_images_dir / img.name)
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            shutil.copy(label, val_labels_dir / label.name)
    
    # Remove temp directories
    shutil.rmtree(images_dir)
    shutil.rmtree(labels_dir)
    
    print(f"Train: {len(train_images)} images")
    print(f"Val: {len(val_images)} images")
    
    return len(train_images), len(val_images)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare dataset from uploads folder")
    parser.add_argument("--uploads", default="uploads", help="Path to uploads directory")
    parser.add_argument("--output", default="training/datasets/panel_dataset", help="Output directory")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold (0-1)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/val split ratio")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("DATASET PREPARATION FOR PANEL DETECTION")
    print("="*80 + "\n")
    
    # Collect images
    print(f"Collecting images from: {args.uploads}")
    print("-"*80)
    
    image_files = collect_images_from_uploads(args.uploads)
    
    if not image_files:
        print(f"\nERROR: No images found in {args.uploads}")
        print("Please check that the uploads folder contains images.")
        sys.exit(1)
    
    print(f"\nFound {len(image_files)} images")
    print("\nImage sources:")
    
    # Count by subdirectory
    from collections import Counter
    sources = Counter([str(img.parent.relative_to(args.uploads)) for img in image_files])
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count} images")
    
    # Process and label
    output_path = Path(args.output)
    temp_images_dir = output_path / "temp_images"
    temp_labels_dir = output_path / "temp_labels"
    
    print(f"\n{'='*80}")
    print("AUTO-LABELING IMAGES")
    print("="*80)
    print(f"Using detector to automatically label panels...")
    print(f"Confidence threshold: {args.confidence}\n")
    
    labeled_count, skipped_count = process_and_label_images(
        image_files,
        temp_images_dir,
        temp_labels_dir,
        args.confidence
    )
    
    if labeled_count == 0:
        print("\nERROR: No images were successfully labeled!")
        print("Try lowering the confidence threshold with --confidence 0.1")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Successfully labeled: {labeled_count} images")
    print(f"Skipped (no detections): {skipped_count} images")
    print("="*80)
    
    # Split into train/val
    train_count, val_count = split_dataset(
        temp_images_dir, 
        temp_labels_dir, 
        args.train_ratio
    )
    
    # Create data.yaml
    print("\nCreating dataset configuration...")
    
    yaml_content = f"""# Panel Detection Dataset
# Auto-generated from uploads folder
# Images: {labeled_count} (train: {train_count}, val: {val_count})

path: {output_path.absolute()}
train: train
val: val

# Classes
names:
  0: mobile_panel

nc: 1
"""
    
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created: {yaml_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("DATASET READY FOR TRAINING!")
    print("="*80)
    print(f"\nDataset location: {output_path}")
    print(f"Configuration: {yaml_path}")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Total labeled: {labeled_count}")
    
    print("\nNOTE:")
    print("The labels were generated automatically using the current detector.")
    print("They may not be perfect, but they provide a good starting point.")
    print("The trained YOLO model will likely perform better than the current detector!")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"\n1. Train the model:")
    print(f"   python training/train_yolo.py --data {yaml_path} --epochs 100 --batch 16")
    
    print(f"\n2. Or use the quick training script:")
    print(f"   python training/quick_train.py")
    
    print("\n")
    
    return yaml_path


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

