"""
Dataset Downloader for Mobile Panel Detection

This script downloads public datasets for mobile phone manufacturing and quality control.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_msd_dataset(output_dir: str):
    """Download Mobile Phone Screen Surface Defect (MSD) Dataset"""
    print("Downloading MSD Dataset...")
    print("Repository: https://github.com/jianzhang96/MSD")
    
    output_path = Path(output_dir) / "MSD"
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Clone the repository
        subprocess.run([
            "git", "clone", 
            "https://github.com/jianzhang96/MSD.git",
            str(output_path)
        ], check=True)
        
        print(f"Successfully downloaded MSD dataset to {output_path}")
        print(f"Dataset contains 1,200 images with defect annotations")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading MSD dataset: {e}")
        return False


def download_roboflow_dataset(api_key: str, workspace: str, project: str, version: int, output_dir: str):
    """Download dataset from Roboflow"""
    print("Downloading dataset from Roboflow...")
    
    try:
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project)
        dataset = project.version(version).download("yolov8", location=output_dir)
        
        print(f"Successfully downloaded Roboflow dataset to {output_dir}")
        return True
    except ImportError:
        print("Error: roboflow package not installed. Run: pip install roboflow")
        return False
    except Exception as e:
        print(f"Error downloading from Roboflow: {e}")
        return False


def download_kaggle_dataset(dataset_slug: str, output_dir: str):
    """Download dataset from Kaggle"""
    print(f"Downloading {dataset_slug} from Kaggle...")
    
    try:
        # Check if kaggle.json exists
        kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_config.exists():
            print("Error: Kaggle credentials not found!")
            print("Please download kaggle.json from https://www.kaggle.com/settings")
            print(f"And place it at: {kaggle_config}")
            return False
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", dataset_slug,
            "-p", str(output_path),
            "--unzip"
        ], check=True)
        
        print(f"Successfully downloaded Kaggle dataset to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading from Kaggle: {e}")
        return False


def create_sample_dataset(output_dir: str, num_images: int = 50):
    """Create a sample dataset structure for manual labeling"""
    print(f"Creating sample dataset structure in {output_dir}...")
    
    output_path = Path(output_dir) / "custom_dataset"
    
    # Create directory structure
    dirs = [
        output_path / "images" / "train",
        output_path / "images" / "val",
        output_path / "images" / "test",
        output_path / "labels" / "train",
        output_path / "labels" / "val",
        output_path / "labels" / "test",
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create data.yaml
    yaml_content = f"""# Mobile Panel Detection Dataset
# Created: {Path(__file__).stem}

path: {output_path.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (relative to 'path')

# Classes
names:
  0: mobile_panel
  1: mobile_display
  2: panel_with_defect

# Number of classes
nc: 3
"""
    
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    # Create README
    readme_content = """# Custom Mobile Panel Dataset

## Directory Structure
- images/train/ - Training images
- images/val/ - Validation images
- images/test/ - Test images
- labels/train/ - Training labels (YOLO format)
- labels/val/ - Validation labels (YOLO format)
- labels/test/ - Test labels (YOLO format)

## Label Format (YOLO)
Each .txt file should contain one line per object:
```
class_id center_x center_y width height
```

All values should be normalized (0-1):
- class_id: 0 (mobile_panel), 1 (mobile_display), 2 (panel_with_defect)
- center_x, center_y: Center of bounding box (relative to image width/height)
- width, height: Size of bounding box (relative to image width/height)

## How to Label

### Option 1: LabelImg
1. Install: pip install labelImg
2. Run: labelImg
3. Open images folder
4. Draw bounding boxes
5. Save as YOLO format

### Option 2: Roboflow (Recommended)
1. Upload images to https://roboflow.com
2. Label images in browser
3. Export as YOLOv8 format
4. Download and replace this dataset

### Option 3: CVAT
1. Use https://cvat.ai
2. Create project
3. Upload images
4. Annotate
5. Export as YOLO format

## Recommended Split
- Training: 70% (420 images if you have 600)
- Validation: 20% (120 images)
- Test: 10% (60 images)

## Next Steps
1. Add your images to images/train/, images/val/, images/test/
2. Label them using one of the tools above
3. Place label .txt files in corresponding labels/ folders
4. Run: python training/train_yolo.py --data {yaml_path}
"""
    
    readme_path = output_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Created sample dataset structure at {output_path}")
    print(f"Configuration file: {yaml_path}")
    print(f"Instructions: {readme_path}")
    print("\nNext steps:")
    print(f"1. Copy your manufacturing images to: {output_path}/images/train/")
    print(f"2. Label them using LabelImg, Roboflow, or CVAT")
    print(f"3. Run training: python training/train_yolo.py --data {yaml_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for mobile panel detection training"
    )
    
    parser.add_argument(
        "--source",
        choices=["msd", "roboflow", "kaggle", "custom"],
        default="custom",
        help="Dataset source"
    )
    
    parser.add_argument(
        "--output",
        default="training/datasets",
        help="Output directory for datasets"
    )
    
    # Roboflow options
    parser.add_argument("--roboflow-api-key", help="Roboflow API key")
    parser.add_argument("--roboflow-workspace", help="Roboflow workspace")
    parser.add_argument("--roboflow-project", help="Roboflow project name")
    parser.add_argument("--roboflow-version", type=int, default=1, help="Dataset version")
    
    # Kaggle options
    parser.add_argument("--kaggle-dataset", help="Kaggle dataset slug (e.g., 'username/dataset-name')")
    
    # Custom dataset options
    parser.add_argument("--num-samples", type=int, default=50, help="Number of sample images for custom dataset")
    
    args = parser.parse_args()
    
    success = False
    
    if args.source == "msd":
        success = download_msd_dataset(args.output)
        
    elif args.source == "roboflow":
        if not all([args.roboflow_api_key, args.roboflow_workspace, args.roboflow_project]):
            print("Error: Roboflow credentials required!")
            print("Usage: --source roboflow --roboflow-api-key YOUR_KEY --roboflow-workspace WORKSPACE --roboflow-project PROJECT")
            sys.exit(1)
        success = download_roboflow_dataset(
            args.roboflow_api_key,
            args.roboflow_workspace,
            args.roboflow_project,
            args.roboflow_version,
            args.output
        )
        
    elif args.source == "kaggle":
        if not args.kaggle_dataset:
            print("Error: Kaggle dataset slug required!")
            print("Usage: --source kaggle --kaggle-dataset 'username/dataset-name'")
            sys.exit(1)
        success = download_kaggle_dataset(args.kaggle_dataset, args.output)
        
    elif args.source == "custom":
        success = create_sample_dataset(args.output, args.num_samples)
    
    if success:
        print("\nDataset download/setup completed successfully!")
        print(f"Dataset location: {args.output}")
    else:
        print("\nDataset download/setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
