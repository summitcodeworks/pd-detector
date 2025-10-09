"""
Automated Training Pipeline

This script automates the entire training process from dataset preparation to deployment.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "=" * 80)
    print(f"STEP: {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n{description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{description} - FAILED")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Automated training pipeline for mobile panel detection"
    )
    
    # Pipeline steps
    parser.add_argument("--download-dataset", action="store_true", help="Download dataset")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--deploy", action="store_true", help="Deploy model")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    # Dataset options
    parser.add_argument("--dataset-source", default="custom", help="Dataset source (msd, custom)")
    parser.add_argument("--dataset-dir", default="training/datasets", help="Dataset directory")
    
    # Training options
    parser.add_argument("--data-yaml", help="Path to data.yaml (auto-detected if not provided)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--model-size", default="n", help="Model size (n, s, m, l, x)")
    parser.add_argument("--device", default="auto", help="Training device")
    
    # Deployment options
    parser.add_argument("--model-path", help="Path to trained model (auto-detected if not provided)")
    parser.add_argument("--deploy-path", default="models/mobile_panel_custom.pt", help="Deployment path")
    
    args = parser.parse_args()
    
    # If --all is specified, enable all steps
    if args.all:
        args.download_dataset = True
        args.train = True
        args.evaluate = True
        args.deploy = True
    
    # Check if at least one step is selected
    if not any([args.download_dataset, args.train, args.evaluate, args.deploy]):
        print("Error: No pipeline steps selected!")
        print("Use --download-dataset, --train, --evaluate, --deploy, or --all")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("AUTOMATED TRAINING PIPELINE - Mobile Panel Detection")
    print("=" * 80)
    
    # Step 1: Download/Setup Dataset
    if args.download_dataset:
        cmd = [
            "python", "training/download_dataset.py",
            "--source", args.dataset_source,
            "--output", args.dataset_dir
        ]
        
        if not run_command(cmd, "Dataset Download/Setup"):
            print("\nPipeline failed at dataset download step")
            sys.exit(1)
        
        # Auto-detect data.yaml
        if not args.data_yaml:
            if args.dataset_source == "custom":
                args.data_yaml = f"{args.dataset_dir}/custom_dataset/data.yaml"
            elif args.dataset_source == "msd":
                # You'll need to create data.yaml for MSD dataset
                print("\nNote: For MSD dataset, you need to create data.yaml manually")
                print("See training/datasets/MSD for dataset structure")
    
    # Step 2: Train Model
    if args.train:
        if not args.data_yaml:
            print("\nError: data.yaml path required for training")
            print("Provide --data-yaml or run with --download-dataset first")
            sys.exit(1)
        
        cmd = [
            "python", "training/train_yolo.py",
            "--data", args.data_yaml,
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--model-size", args.model_size,
            "--device", args.device
        ]
        
        if not run_command(cmd, "Model Training"):
            print("\nPipeline failed at training step")
            sys.exit(1)
        
        # Auto-detect trained model path
        if not args.model_path:
            args.model_path = "training/runs/detect/mobile_panel_train/weights/best.pt"
    
    # Step 3: Evaluate Model
    if args.evaluate:
        if not args.model_path:
            print("\nError: model path required for evaluation")
            print("Provide --model-path or run with --train first")
            sys.exit(1)
        
        cmd = [
            "python", "training/evaluate_model.py",
            "--model", args.model_path
        ]
        
        if args.data_yaml:
            cmd.extend(["--data", args.data_yaml])
        
        if not run_command(cmd, "Model Evaluation"):
            print("\nPipeline failed at evaluation step")
            sys.exit(1)
    
    # Step 4: Deploy Model
    if args.deploy:
        if not args.model_path:
            print("\nError: model path required for deployment")
            print("Provide --model-path or run with --train first")
            sys.exit(1)
        
        cmd = [
            "python", "training/deploy_model.py",
            "--model", args.model_path,
            "--destination", args.deploy_path,
            "--test",
            "--info"
        ]
        
        if not run_command(cmd, "Model Deployment"):
            print("\nPipeline failed at deployment step")
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    print("\nSummary:")
    if args.download_dataset:
        print(f"  Dataset: {args.dataset_dir}")
        if args.data_yaml:
            print(f"  Config: {args.data_yaml}")
    
    if args.train:
        print(f"  Trained model: {args.model_path}")
        print(f"  Training results: training/runs/detect/mobile_panel_train/")
    
    if args.deploy:
        print(f"  Deployed model: {args.deploy_path}")
    
    print("\nNext steps:")
    print("  1. Restart your Flask application: python main.py")
    print("  2. Test the API: curl -X POST -F 'file=@test_image.jpg' http://localhost:5000/detect")
    print("  3. Monitor performance and retrain if needed")


if __name__ == "__main__":
    main()
