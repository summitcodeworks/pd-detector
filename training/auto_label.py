"""
Automatic Image Labeling

Uses the current detector to automatically label images for training.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

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


def auto_label_images(input_dir, output_images_dir, output_labels_dir, confidence_threshold=0.25):
    """Automatically label images using current detector"""
    
    input_path = Path(input_dir)
    images_path = Path(output_images_dir)
    labels_path = Path(output_labels_dir)
    
    # Create output directories
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    print("Loading detector...")
    detector = AdvancedPanelDetector()
    
    # Get all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(input_path.glob(ext))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return 0
    
    print(f"Found {len(image_files)} images to label")
    print(f"Confidence threshold: {confidence_threshold}")
    print()
    
    labeled_count = 0
    skipped_count = 0
    
    for img_file in image_files:
        print(f"Processing: {img_file.name}...", end=" ")
        
        # Read image
        image = cv2.imread(str(img_file))
        if image is None:
            print("❌ Failed to read")
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
            print(f"⚠️  No panels detected (skipped)")
            skipped_count += 1
            continue
        
        # Copy image to output
        output_img_path = images_path / img_file.name
        cv2.imwrite(str(output_img_path), image)
        
        # Create label file
        label_file = labels_path / f"{img_file.stem}.txt"
        with open(label_file, 'w') as f:
            for detection in panel_detections:
                bbox = detection['bbox']
                yolo_label = convert_to_yolo_format(bbox, width, height)
                f.write(yolo_label + '\n')
        
        print(f"✅ Labeled ({len(panel_detections)} panels)")
        labeled_count += 1
    
    print()
    print(f"{'='*80}")
    print(f"Auto-labeling complete!")
    print(f"{'='*80}")
    print(f"✅ Successfully labeled: {labeled_count} images")
    print(f"⚠️  Skipped (no detections): {skipped_count} images")
    print(f"📁 Images saved to: {images_path}")
    print(f"📁 Labels saved to: {labels_path}")
    
    return labeled_count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-label images using current detector")
    parser.add_argument("--input", default="uploads", help="Input directory with images")
    parser.add_argument("--output", default="training/datasets/auto_labeled", help="Output directory")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold (0-1)")
    
    args = parser.parse_args()
    
    print("\n🤖 AUTOMATIC IMAGE LABELING")
    print("="*80)
    print("Using current detector to automatically label your images")
    print("This creates a starting dataset that you can improve later!")
    print("="*80 + "\n")
    
    output_path = Path(args.output)
    total_labeled = 0
    
    # Label images from input folder (default: uploads/)
    print(f"Labeling images from {args.input}/...")
    print("-"*80)
    count = auto_label_images(
        args.input,
        output_path / "images" / "train",
        output_path / "labels" / "train",
        args.confidence
    )
    total_labeled += count
    
    if total_labeled == 0:
        print("\n❌ No images were labeled!")
        print(f"Make sure you have images in {args.input}/ folder")
        sys.exit(1)
    
    # Create data.yaml
    print("\nCreating dataset configuration...")
    yaml_content = f"""# Auto-labeled Mobile Panel Dataset
# Created by auto_label.py

path: {output_path.absolute()}
train: images/train
val: images/train  # Using same for now, split manually if needed
test: images/train

# Classes
names:
  0: mobile_panel

nc: 1
"""
    
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created: {yaml_path}")
    
    print("\n" + "="*80)
    print("🎉 SUCCESS! Dataset ready for training")
    print("="*80)
    print(f"\nTotal images labeled: {total_labeled}")
    print(f"Dataset location: {output_path}")
    print(f"Configuration: {yaml_path}")
    
    print("\n📝 IMPORTANT NOTES:")
    print("1. These labels are from the CURRENT detector (which uses color detection)")
    print("2. They may not be perfect - review and correct them if needed")
    print("3. This gives you a starting point to train a BETTER model")
    print("4. The trained model will be better than the labeler!\n")
    
    print("Next steps:")
    print(f"  1. Review labels (optional): Check {output_path}/labels/train/")
    print(f"  2. Train model: python training/train_yolo.py --data {yaml_path}")
    print(f"  3. Or use simple script: python training/simple_train.py")
    
    return total_labeled


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(1)
