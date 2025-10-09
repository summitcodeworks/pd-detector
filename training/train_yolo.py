"""
YOLOv8 Training Script for Mobile Panel Detection

This script trains a custom YOLOv8 model on your mobile panel dataset.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


def validate_dataset(data_yaml_path: str) -> bool:
    """Validate dataset structure and configuration"""
    print("Validating dataset...")
    
    data_path = Path(data_yaml_path)
    if not data_path.exists():
        print(f"Error: Data config file not found: {data_yaml_path}")
        return False
    
    # Load data.yaml
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Check required fields
    required_fields = ['path', 'train', 'val', 'names', 'nc']
    missing_fields = [field for field in required_fields if field not in data_config]
    
    if missing_fields:
        print(f"Error: Missing required fields in data.yaml: {missing_fields}")
        return False
    
    # Check if directories exist
    dataset_path = Path(data_config['path'])
    train_path = dataset_path / data_config['train']
    val_path = dataset_path / data_config['val']
    
    if not train_path.exists():
        print(f"Error: Training images directory not found: {train_path}")
        return False
    
    if not val_path.exists():
        print(f"Error: Validation images directory not found: {val_path}")
        return False
    
    # Count images
    train_images = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
    val_images = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))
    
    if len(train_images) == 0:
        print(f"Error: No training images found in {train_path}")
        return False
    
    if len(val_images) == 0:
        print(f"Warning: No validation images found in {val_path}")
        print("Consider splitting your dataset into train/val sets")
    
    print(f"Dataset validation successful!")
    print(f"  Training images: {len(train_images)}")
    print(f"  Validation images: {len(val_images)}")
    print(f"  Classes: {data_config['nc']}")
    print(f"  Class names: {data_config['names']}")
    
    return True


def train_model(
    data_yaml: str,
    model_size: str = "n",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "auto",
    pretrained: bool = True,
    project: str = "training/runs/detect",
    name: str = "mobile_panel_train",
    patience: int = 50,
    save_period: int = 10,
    workers: int = 8,
    optimizer: str = "auto",
    lr0: float = 0.01,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    augment: bool = True,
    mosaic: float = 1.0,
    mixup: float = 0.0,
    copy_paste: float = 0.0,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    flipud: float = 0.0,
    fliplr: float = 0.5,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
):
    """Train YOLOv8 model"""
    
    # Validate dataset first
    if not validate_dataset(data_yaml):
        print("Dataset validation failed. Please fix the issues and try again.")
        sys.exit(1)
    
    # Select device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nTraining Configuration:")
    print(f"  Model: YOLOv8{model_size}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Workers: {workers}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Learning rate: {lr0} -> {lrf}")
    print(f"  Data augmentation: {augment}")
    
    # Load model
    model_name = f"yolov8{model_size}.pt" if pretrained else f"yolov8{model_size}.yaml"
    print(f"\nLoading model: {model_name}")
    model = YOLO(model_name)
    
    # Training arguments
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'project': project,
        'name': name,
        'patience': patience,
        'save_period': save_period,
        'workers': workers,
        'optimizer': optimizer,
        'lr0': lr0,
        'lrf': lrf,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'warmup_epochs': warmup_epochs,
        'augment': augment,
        'mosaic': mosaic,
        'mixup': mixup,
        'copy_paste': copy_paste,
        'degrees': degrees,
        'translate': translate,
        'scale': scale,
        'shear': shear,
        'perspective': perspective,
        'flipud': flipud,
        'fliplr': fliplr,
        'hsv_h': hsv_h,
        'hsv_s': hsv_s,
        'hsv_v': hsv_v,
        'exist_ok': True,
        'pretrained': pretrained,
        'verbose': True,
        'save': True,
        'plots': True,
    }
    
    print("\nStarting training...")
    print("=" * 80)
    
    # Train the model
    results = model.train(**train_args)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    
    # Get best model path
    best_model_path = Path(project) / name / "weights" / "best.pt"
    last_model_path = Path(project) / name / "weights" / "last.pt"
    
    print(f"\nModel files:")
    print(f"  Best model: {best_model_path}")
    print(f"  Last model: {last_model_path}")
    print(f"  Results: {Path(project) / name}")
    
    # Validate the trained model
    print("\nValidating trained model...")
    metrics = model.val()
    
    print("\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    # Save training summary
    summary_path = Path(project) / name / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Mobile Panel Detection - Training Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: YOLOv8{model_size}\n")
        f.write(f"Dataset: {data_yaml}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Image size: {img_size}\n")
        f.write(f"Device: {device}\n\n")
        f.write("Validation Metrics:\n")
        f.write(f"  mAP50: {metrics.box.map50:.4f}\n")
        f.write(f"  mAP50-95: {metrics.box.map:.4f}\n")
        f.write(f"  Precision: {metrics.box.mp:.4f}\n")
        f.write(f"  Recall: {metrics.box.mr:.4f}\n\n")
        f.write(f"Best model: {best_model_path}\n")
        f.write(f"Last model: {last_model_path}\n")
    
    print(f"\nTraining summary saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("Next steps:")
    print(f"1. Review training results: {Path(project) / name}")
    print(f"2. Test the model: python training/evaluate_model.py --model {best_model_path}")
    print(f"3. Deploy the model: python training/deploy_model.py --model {best_model_path}")
    
    return str(best_model_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for mobile panel detection"
    )
    
    # Required arguments
    parser.add_argument(
        "--data",
        required=True,
        help="Path to data.yaml file"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-size",
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)"
    )
    
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights (transfer learning)"
    )
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="auto", help="Training device (auto, cpu, cuda, 0, 1, etc.)")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--save-period", type=int, default=10, help="Save checkpoint every N epochs")
    
    # Output configuration
    parser.add_argument("--project", default="training/runs/detect", help="Project directory")
    parser.add_argument("--name", default="mobile_panel_train", help="Run name")
    
    # Optimizer settings
    parser.add_argument("--optimizer", default="auto", help="Optimizer (auto, SGD, Adam, AdamW)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs")
    
    # Data augmentation
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.0, help="Mixup augmentation probability")
    parser.add_argument("--copy-paste", type=float, default=0.0, help="Copy-paste augmentation probability")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation augmentation (degrees)")
    parser.add_argument("--translate", type=float, default=0.1, help="Translation augmentation")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale augmentation")
    parser.add_argument("--shear", type=float, default=0.0, help="Shear augmentation (degrees)")
    parser.add_argument("--perspective", type=float, default=0.0, help="Perspective augmentation")
    parser.add_argument("--flipud", type=float, default=0.0, help="Vertical flip probability")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability")
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV hue augmentation")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV saturation augmentation")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV value augmentation")
    
    args = parser.parse_args()
    
    # Train the model
    best_model = train_model(
        data_yaml=args.data,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        pretrained=args.pretrained,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save_period=args.save_period,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        augment=not args.no_augment,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
    )
    
    print(f"\nTrained model saved at: {best_model}")


if __name__ == "__main__":
    main()
