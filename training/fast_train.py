"""
Fast Training Script - Optimized for Speed

Trains with reduced settings for faster results.
Good for testing and quick iterations.
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
    print("FAST TRAINING - PANEL DETECTION MODEL")
    print("="*80 + "\n")
    
    print("This will train FASTER with:")
    print("  - 50 epochs (instead of 100)")
    print("  - Batch size 8 (faster iterations)")
    print("  - Image size 416 (smaller, faster)")
    print("  - Reduced augmentation")
    print()
    print("Note: Slightly lower accuracy but 2-3x faster!")
    print()
    
    # Check if dataset exists
    data_yaml = "training/datasets/panel_dataset/data.yaml"
    
    if not Path(data_yaml).exists():
        print("Dataset not found! Running preparation first...")
        result = os.system("python training/prepare_dataset.py --uploads uploads --output training/datasets/panel_dataset")
        if result != 0:
            print("\nERROR: Dataset preparation failed!")
            sys.exit(1)
    
    response = input("Start fast training? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        sys.exit(0)
    
    # Train with fast settings
    print("\n" + "="*80)
    print("TRAINING MODEL (FAST MODE)")
    print("="*80 + "\n")
    
    result = os.system(
        f"python training/train_yolo.py "
        f"--data {data_yaml} "
        f"--epochs 50 "
        f"--batch 8 "
        f"--img-size 416 "
        f"--model-size n "
        f"--patience 25 "
        f"--workers 4"
    )
    
    if result != 0:
        print("\nERROR: Training failed!")
        print("\nTry:")
        print("  python training/train_yolo.py --data training/datasets/panel_dataset/data.yaml --epochs 30 --batch 4")
        sys.exit(1)
    
    # Deploy
    print("\n" + "="*80)
    print("DEPLOYING MODEL")
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
            print("\nRestart the API server:")
            print("  python main.py")
    else:
        print(f"\nDeploy later with:")
        print(f"  python training/deploy_model.py --model {best_model}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel: {best_model}")
    print(f"Results: training/runs/detect/mobile_panel_train/")
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

