"""
Quick Training Script

Combines dataset preparation and training into one simple command.
This is the easiest way to train a model from the uploads folder.
"""

import os
import sys
from pathlib import Path

def main():
    # Make sure we're in the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print("\n" + "="*80)
    print("QUICK TRAINING - PANEL DETECTION MODEL")
    print("="*80 + "\n")
    
    print("This script will:")
    print("1. Collect all images from uploads/ folder")
    print("2. Auto-label them using the current detector")
    print("3. Train a custom YOLO model")
    print("4. Deploy the trained model to production")
    print()
    
    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        sys.exit(0)
    
    # Step 1: Prepare dataset
    print("\n" + "="*80)
    print("STEP 1: PREPARING DATASET")
    print("="*80 + "\n")
    
    result = os.system("python training/prepare_dataset.py --uploads uploads --output training/datasets/panel_dataset")
    
    if result != 0:
        print("\nERROR: Dataset preparation failed!")
        sys.exit(1)
    
    # Step 2: Train model
    print("\n" + "="*80)
    print("STEP 2: TRAINING MODEL")
    print("="*80 + "\n")
    
    print("Training configuration:")
    print("  Model: YOLOv8n (nano - fastest)")
    print("  Epochs: 100")
    print("  Batch: 16")
    print("  Image size: 640")
    print()
    
    response = input("Start training? (yes/no): ")
    if response.lower() != 'yes':
        print("Training cancelled. Dataset is ready at: training/datasets/panel_dataset")
        sys.exit(0)
    
    data_yaml = "training/datasets/panel_dataset/data.yaml"
    
    result = os.system(f"python training/train_yolo.py --data {data_yaml} --epochs 100 --batch 16 --model-size n")
    
    if result != 0:
        print("\nERROR: Training failed!")
        print("\nCommon issues:")
        print("- Out of memory: Try --batch 8 or --batch 4")
        print("- No GPU: Training will be slower but should work")
        sys.exit(1)
    
    # Step 3: Deploy model
    print("\n" + "="*80)
    print("STEP 3: DEPLOYING MODEL")
    print("="*80 + "\n")
    
    best_model = "training/runs/detect/mobile_panel_train/weights/best.pt"
    
    if not Path(best_model).exists():
        print(f"ERROR: Trained model not found at {best_model}")
        sys.exit(1)
    
    response = input("Deploy to production? (yes/no): ")
    if response.lower() == 'yes':
        result = os.system(f"python training/deploy_model.py --model {best_model} --test")
        
        if result == 0:
            print("\n" + "="*80)
            print("SUCCESS! MODEL DEPLOYED")
            print("="*80)
            print("\nYour custom-trained model is now active!")
            print("\nRestart the API server to use it:")
            print("  python main.py")
            print("\nOr if using Docker:")
            print("  docker-compose restart")
    else:
        print(f"\nYou can deploy later with:")
        print(f"  python training/deploy_model.py --model {best_model}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel location: {best_model}")
    print(f"Training results: training/runs/detect/mobile_panel_train/")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        sys.exit(1)

