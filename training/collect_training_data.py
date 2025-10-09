"""
Training Data Collection Helper

This script helps collect training data from your API uploads and processed images.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def collect_from_uploads(
    uploads_dir: str = "uploads",
    processed_dir: str = "processed",
    output_dir: str = "training/datasets/collected_data",
    copy_mode: str = "copy"
):
    """Collect images from uploads and processed folders"""
    
    uploads_path = Path(uploads_dir)
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    print("Collecting training data...")
    print(f"Source: {uploads_path} and {processed_path}")
    print(f"Destination: {output_path}")
    
    collected_images = []
    
    # Collect from uploads
    if uploads_path.exists():
        print(f"\nScanning {uploads_path}...")
        image_files = list(uploads_path.glob("*.jpg")) + \
                     list(uploads_path.glob("*.jpeg")) + \
                     list(uploads_path.glob("*.png"))
        
        for img_file in image_files:
            dest_file = images_dir / img_file.name
            
            if copy_mode == "copy":
                shutil.copy2(img_file, dest_file)
            else:  # move
                shutil.move(str(img_file), str(dest_file))
            
            collected_images.append(str(dest_file))
            print(f"  Collected: {img_file.name}")
    
    # Collect from processed
    if processed_path.exists():
        print(f"\nScanning {processed_path}...")
        image_files = list(processed_path.glob("*.jpg")) + \
                     list(processed_path.glob("*.jpeg")) + \
                     list(processed_path.glob("*.png"))
        
        for img_file in image_files:
            # Skip already marked images (they have detection boxes drawn)
            if "_marked" in img_file.name or "processed" in img_file.name:
                continue
            
            dest_file = images_dir / img_file.name
            
            if copy_mode == "copy":
                shutil.copy2(img_file, dest_file)
            else:  # move
                shutil.move(str(img_file), str(dest_file))
            
            collected_images.append(str(dest_file))
            print(f"  Collected: {img_file.name}")
    
    print(f"\nTotal images collected: {len(collected_images)}")
    
    # Create collection report
    report = {
        'collection_date': datetime.now().isoformat(),
        'total_images': len(collected_images),
        'sources': [str(uploads_path), str(processed_path)],
        'destination': str(output_path),
        'images': [str(Path(img).name) for img in collected_images]
    }
    
    report_path = output_path / "collection_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nCollection report saved: {report_path}")
    
    # Create README for labeling
    readme_content = f"""# Collected Training Data

## Collection Info
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Total Images: {len(collected_images)}
- Location: {output_path.absolute()}

## Next Steps

### 1. Label These Images

Use one of these tools to label the images:

#### Option A: Roboflow (Recommended)
1. Go to https://roboflow.com
2. Create a project
3. Upload all images from `images/` folder
4. Draw bounding boxes around mobile panels
5. Label them as "mobile_panel"
6. Export as YOLOv8 format
7. Download and extract to your training folder

#### Option B: LabelImg
```bash
pip install labelImg
labelImg {images_dir}
```
- Change format to "YOLO"
- Draw boxes around panels
- Save labels

#### Option C: CVAT
1. Go to https://cvat.ai
2. Create project and upload images
3. Annotate with bounding boxes
4. Export as YOLO format

### 2. Organize for Training

After labeling, organize like this:
```
training/datasets/your_dataset/
├── images/
│   ├── train/  (70% of images)
│   ├── val/    (20% of images)
│   └── test/   (10% of images)
├── labels/
│   ├── train/  (corresponding labels)
│   ├── val/    (corresponding labels)
│   └── test/   (corresponding labels)
└── data.yaml
```

### 3. Train Your Model
```bash
python training/train_yolo.py --data path/to/data.yaml --epochs 100
```

## Labeling Guidelines

1. **What to Label**: 
   - Mobile device panels/displays
   - Include entire visible screen area
   - Include partial panels at image edges

2. **What NOT to Label**:
   - TVs or monitors
   - Printed images of phones
   - Phone cases without screens

3. **Bounding Box Tips**:
   - Make boxes tight around the screen
   - Include screen protectors/glass
   - Be consistent across all images

4. **Quality Control**:
   - Review labels before training
   - Ensure all panels are labeled
   - Check for accidental mislabels
"""
    
    readme_path = output_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Instructions saved: {readme_path}")
    
    print("\n" + "=" * 80)
    print("Data collection complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Label the images in: {images_dir}")
    print(f"2. Follow instructions in: {readme_path}")
    print(f"3. Train your model when ready")
    
    return collected_images


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1
):
    """Split labeled dataset into train/val/test sets"""
    
    import random
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    
    # Get all images with corresponding labels
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(images_path.glob(f"*{ext}"))
    
    # Filter to only images that have labels
    paired_files = []
    for img_file in image_files:
        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            paired_files.append((img_file, label_file))
    
    print(f"Found {len(paired_files)} image-label pairs")
    
    # Shuffle
    random.shuffle(paired_files)
    
    # Calculate split indices
    total = len(paired_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': paired_files[:train_end],
        'val': paired_files[train_end:val_end],
        'test': paired_files[val_end:]
    }
    
    # Create directory structure
    for split_name in ['train', 'val', 'test']:
        (output_path / 'images' / split_name).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split_name).mkdir(parents=True, exist_ok=True)
    
    # Copy files
    for split_name, files in splits.items():
        print(f"\n{split_name.capitalize()} set: {len(files)} images")
        for img_file, label_file in files:
            # Copy image
            dest_img = output_path / 'images' / split_name / img_file.name
            shutil.copy2(img_file, dest_img)
            
            # Copy label
            dest_label = output_path / 'labels' / split_name / label_file.name
            shutil.copy2(label_file, dest_label)
    
    # Create data.yaml
    yaml_content = f"""# Mobile Panel Detection Dataset
# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

path: {output_path.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: mobile_panel

nc: 1
"""
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nDataset split complete!")
    print(f"Output: {output_path}")
    print(f"Config: {yaml_path}")
    print(f"\nTrain: {len(splits['train'])} images")
    print(f"Val:   {len(splits['val'])} images")
    print(f"Test:  {len(splits['test'])} images")
    
    return str(yaml_path)


def main():
    parser = argparse.ArgumentParser(
        description="Collect training data from uploads and processed folders"
    )
    
    parser.add_argument(
        "--uploads",
        default="uploads",
        help="Path to uploads directory"
    )
    
    parser.add_argument(
        "--processed",
        default="processed",
        help="Path to processed directory"
    )
    
    parser.add_argument(
        "--output",
        default="training/datasets/collected_data",
        help="Output directory for collected images"
    )
    
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="Copy or move files (copy recommended)"
    )
    
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split dataset into train/val/test after collection (requires labels)"
    )
    
    parser.add_argument(
        "--images-dir",
        help="Images directory for splitting (if different from collected)"
    )
    
    parser.add_argument(
        "--labels-dir",
        help="Labels directory for splitting (required if --split is used)"
    )
    
    parser.add_argument(
        "--split-output",
        default="training/datasets/split_dataset",
        help="Output directory for split dataset"
    )
    
    args = parser.parse_args()
    
    # Collect images
    collected = collect_from_uploads(
        args.uploads,
        args.processed,
        args.output,
        args.mode
    )
    
    # Split dataset if requested
    if args.split:
        if not args.labels_dir:
            print("\nError: --labels-dir required for splitting")
            print("First label your images, then run with --split")
            sys.exit(1)
        
        images_dir = args.images_dir or f"{args.output}/images"
        
        print("\n" + "=" * 80)
        print("Splitting dataset...")
        print("=" * 80)
        
        yaml_path = split_dataset(
            images_dir,
            args.labels_dir,
            args.split_output
        )
        
        print(f"\nReady to train!")
        print(f"Run: python training/train_yolo.py --data {yaml_path}")


if __name__ == "__main__":
    main()
