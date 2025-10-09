"""
Model Deployment Script

Deploys a trained YOLOv8 model to the production system.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def deploy_model(
    model_path: str,
    destination: str = "models/mobile_panel_custom.pt",
    backup_existing: bool = True,
    update_config: bool = True
):
    """Deploy trained model to production"""
    
    model_path = Path(model_path)
    destination_path = Path(destination)
    
    # Validate model file
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return False
    
    if not model_path.suffix == '.pt':
        print(f"Error: Invalid model file format: {model_path.suffix}")
        print("Expected: .pt file")
        return False
    
    print(f"Deploying model: {model_path}")
    print(f"Destination: {destination_path}")
    
    # Create destination directory if it doesn't exist
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup existing model
    if destination_path.exists() and backup_existing:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = destination_path.parent / f"{destination_path.stem}_backup_{timestamp}{destination_path.suffix}"
        print(f"Backing up existing model to: {backup_path}")
        shutil.copy2(destination_path, backup_path)
    
    # Copy model to destination
    print("Copying model...")
    shutil.copy2(model_path, destination_path)
    
    # Update configuration
    if update_config:
        config_path = Path("config.py")
        if config_path.exists():
            print("Updating configuration...")
            
            # Read current config
            with open(config_path, 'r') as f:
                config_content = f.read()
            
            # Update model path in config (if applicable)
            # This is a placeholder - adjust based on your actual config structure
            print("Note: You may need to manually update config.py with the new model path")
    
    # Create deployment info file
    deployment_info = {
        'model_path': str(destination_path),
        'source_model': str(model_path),
        'deployment_time': datetime.now().isoformat(),
        'deployed_by': 'deploy_model.py'
    }
    
    info_path = destination_path.parent / "deployment_info.json"
    with open(info_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"\nDeployment successful!")
    print(f"Model deployed to: {destination_path}")
    print(f"Deployment info: {info_path}")
    
    # Print integration instructions
    print("\n" + "=" * 80)
    print("Integration Instructions:")
    print("=" * 80)
    print("\n1. Update your detector initialization to use the custom model:")
    print(f"   detector = AdvancedPanelDetector(yolo_model_path='{destination_path}')")
    print("\n2. Or update src/mobile_panel_detector/detector/panel_detector.py:")
    print("   Replace 'yolov8n.pt' with the path to your custom model")
    print("\n3. Restart your Flask application:")
    print("   python main.py")
    print("\n4. Test the deployment:")
    print("   curl -X POST -F 'file=@test_image.jpg' http://localhost:5000/detect")
    
    # Update panel_detector.py automatically
    detector_file = Path("src/mobile_panel_detector/detector/panel_detector.py")
    if detector_file.exists():
        print("\n5. Automatically updating panel_detector.py...")
        
        with open(detector_file, 'r') as f:
            detector_content = f.read()
        
        # Check if we can find the model path
        if "def __init__(self, yolo_model_path: str = 'yolov8n.pt')" in detector_content:
            # Update default model path
            updated_content = detector_content.replace(
                "def __init__(self, yolo_model_path: str = 'yolov8n.pt')",
                f"def __init__(self, yolo_model_path: str = '{destination_path}')"
            )
            
            # Create backup
            backup_detector = detector_file.parent / f"{detector_file.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{detector_file.suffix}"
            shutil.copy2(detector_file, backup_detector)
            print(f"   Created backup: {backup_detector}")
            
            # Write updated content
            with open(detector_file, 'w') as f:
                f.write(updated_content)
            
            print(f"   Updated default model path in {detector_file}")
            print("   Your detector will now use the custom model by default!")
        else:
            print("   Could not automatically update panel_detector.py")
            print("   Please manually update the default model path")
    
    print("\n" + "=" * 80)
    
    return True


def test_deployed_model(model_path: str, test_image: str = None):
    """Test the deployed model"""
    
    print(f"Testing deployed model: {model_path}")
    
    from ultralytics import YOLO
    import cv2
    import numpy as np
    
    # Load model
    try:
        model = YOLO(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Test with image or generate dummy image
    if test_image and Path(test_image).exists():
        print(f"Testing with image: {test_image}")
        results = model(test_image)
    else:
        print("Testing with dummy image...")
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy_image)
    
    print("Inference successful!")
    print(f"Model classes: {model.names}")
    
    return True


def create_model_info(model_path: str):
    """Create information file about the model"""
    
    model_path = Path(model_path)
    
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    
    info = {
        'model_path': str(model_path),
        'model_size_mb': model_path.stat().st_size / (1024 * 1024),
        'classes': model.names,
        'num_classes': len(model.names),
        'created': datetime.now().isoformat()
    }
    
    info_path = model_path.parent / f"{model_path.stem}_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Model info saved to: {info_path}")
    
    print("\nModel Information:")
    print(f"  Size: {info['model_size_mb']:.2f} MB")
    print(f"  Classes: {info['num_classes']}")
    print(f"  Class names: {info['classes']}")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy trained YOLOv8 model to production"
    )
    
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model (.pt file)"
    )
    
    parser.add_argument(
        "--destination",
        default="models/mobile_panel_custom.pt",
        help="Destination path for deployed model"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't backup existing model"
    )
    
    parser.add_argument(
        "--no-update-config",
        action="store_true",
        help="Don't update configuration files"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the deployed model"
    )
    
    parser.add_argument(
        "--test-image",
        help="Test image for model testing"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Create model information file"
    )
    
    args = parser.parse_args()
    
    # Deploy the model
    success = deploy_model(
        args.model,
        args.destination,
        backup_existing=not args.no_backup,
        update_config=not args.no_update_config
    )
    
    if not success:
        print("Deployment failed!")
        sys.exit(1)
    
    # Create model info
    if args.info:
        create_model_info(args.destination)
    
    # Test the deployed model
    if args.test:
        print("\n" + "=" * 80)
        print("Testing deployed model...")
        print("=" * 80)
        test_success = test_deployed_model(args.destination, args.test_image)
        if not test_success:
            print("Model testing failed!")
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Deployment completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
