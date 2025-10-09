"""
SIMPLIFIED TRAINING SCRIPT
Just run this and follow the prompts!
"""

import os
import sys
from pathlib import Path

def print_step(step_num, title):
    print("\n" + "="*80)
    print(f"STEP {step_num}: {title}")
    print("="*80 + "\n")

def main():
    # Make sure we're in the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print("\n" + "🚀 SIMPLE MOBILE PANEL DETECTOR TRAINING" + "\n")
    print("This script will guide you through training a custom model.")
    print("It's much simpler than it looks - just follow along!\n")
    
    # Step 1: Check for images
    print_step(1, "Check for Training Images")
    
    # Check uploads folder (use original images, not processed ones)
    uploads_dir = Path("uploads")
    
    upload_images = []
    if uploads_dir.exists():
        upload_images = list(uploads_dir.glob("*.jpg")) + list(uploads_dir.glob("*.jpeg")) + list(uploads_dir.glob("*.png")) + \
                       list(uploads_dir.glob("*.JPG")) + list(uploads_dir.glob("*.JPEG")) + list(uploads_dir.glob("*.PNG"))
    
    total_images = len(upload_images)
    
    print(f"Found {total_images} images in uploads/")
    print("(Note: Using uploads/ folder for training, not processed/)")
    
    # Step 2: Download public dataset
    print_step(2, "Download Public Dataset (Optional)")
    
    print("Would you like to download a public mobile phone dataset?")
    print("This gives you 1,200 pre-labeled images to start with!\n")
    print("You can combine this with your own images for better results.\n")
    
    download_public = input("Download MSD dataset? (yes/no): ")
    
    if download_public.lower() == 'yes':
        # Check if already downloaded
        msd_path = Path("training/datasets/MSD")
        if msd_path.exists():
            print("✅ MSD dataset already exists!")
            msd_images = list(msd_path.rglob("*.jpg")) + list(msd_path.rglob("*.png"))
            print(f"   Found {len(msd_images)} images in MSD dataset")
        else:
            print("\nDownloading MSD dataset (1,200 images)...")
            print("This may take a few minutes...\n")
            result = os.system("python training/download_dataset.py --source msd --output training/datasets")
            if result == 0:
                print("✅ Public dataset downloaded!")
                msd_images = list(msd_path.rglob("*.jpg")) + list(msd_path.rglob("*.png"))
                print(f"   Added {len(msd_images)} images from MSD dataset")
            else:
                print("⚠️  Download failed, continuing with your images only...")
    
    # Step 3: Collect your images
    print_step(3, "Collect Your Production Images")
    
    if total_images == 0:
        print("❌ NO IMAGES FOUND in uploads/")
        print("\nPlease add some images and run this script again.")
        print("Copy your manufacturing line images to: uploads/ folder")
        print("(Don't use processed/ - those have detection boxes already drawn)\n")
        sys.exit(1)
    
    print(f"Found {total_images} images from your uploads/ folder")
    
    if total_images < 50:
        print(f"⚠️  Only {total_images} images - recommend adding more for best results")
    else:
        print(f"✅ Good! {total_images} images is a decent starting point")
    
    # Step 4: Labeling instructions
    print_step(4, "Label Your Images")
    
    print("Now you need to draw boxes around mobile panels in your images.")
    print("NOTE: If you downloaded MSD dataset, it's already labeled!")
    print("      You only need to label YOUR images from uploads/.\n")
    
    print("🤖 AUTOMATIC LABELING (EASIEST!)")
    print("   Use the current detector to automatically label your images!")
    print("   Quick, easy, and good enough to train a better model.\n")
    
    auto_label = input("Use automatic labeling? (yes/no): ")
    
    if auto_label.lower() == 'yes':
        print("\nRunning automatic labeling...")
        result = os.system("python training/auto_label.py")
        if result == 0:
            print("\n✅ Auto-labeling complete!")
            print("Your images are now labeled and ready for training!")
            input("\nPress ENTER to continue to training...")
        else:
            print("\n⚠️  Auto-labeling failed. You can still label manually.")
            input("Press ENTER to see manual labeling options...")
    
    print("\n📦 MANUAL LABELING OPTIONS (if you skipped auto-labeling):\n")
    print("1. ROBOFLOW (Easiest - Recommended)")
    print("   - Go to: https://roboflow.com")
    print("   - Create free account")
    print("   - Upload images from: uploads/")
    print("   - Draw boxes around mobile panels")
    print("   - Export as 'YOLOv8' format")
    print("   - Download and extract to: training/datasets/\n")
    
    print("2. CVAT (Web-based)")
    print("   - Go to: https://cvat.ai")
    print("   - Upload and label images")
    print("   - Export as 'YOLO' format\n")
    
    print("3. LabelImg (Desktop app)")
    print("   - Run: pip install pyqt5 lxml labelImg")
    print("   - Run: labelImg")
    print("   - Select YOLO format and label images\n")
    
    print("WHAT TO LABEL:")
    print("- Draw boxes around mobile device screens/panels")
    print("- Label them as 'mobile_panel'")
    print("- Be consistent!\n")
    
    input("Press ENTER when you've finished labeling...")
    
    # Step 5: Find the data.yaml or merge datasets
    print_step(5, "Prepare Final Dataset")
    
    print("Now let's choose which dataset to train on:\n")
    
    # Look for data.yaml files
    data_yamls = list(Path("training/datasets").rglob("data.yaml"))
    
    if data_yamls:
        print("Found these datasets:\n")
        for i, yaml_file in enumerate(data_yamls, 1):
            # Try to identify what kind of dataset this is
            yaml_str = str(yaml_file)
            if "MSD" in yaml_str:
                desc = "MSD public dataset (1,200 pre-labeled images)"
            elif "simple_dataset" in yaml_str or "my_" in yaml_str:
                desc = "Your production images (needs labeling)"
            else:
                desc = "Custom dataset"
            print(f"{i}. {yaml_file}")
            print(f"   {desc}\n")
        
        print("RECOMMENDATION:")
        print("- Use MSD dataset for quick start (already labeled)")
        print("- Or use YOUR images for best accuracy on YOUR setup")
        print("- Advanced: Merge both datasets for best results\n")
        
        choice = input(f"Which dataset? (1-{len(data_yamls)}): ")
        try:
            data_yaml = str(data_yamls[int(choice) - 1])
        except:
            print("Invalid choice. Using first one.")
            data_yaml = str(data_yamls[0])
    else:
        print("❌ No data.yaml found!")
        print("\nIf you labeled with Roboflow:")
        print("1. Download the dataset ZIP")
        print("2. Extract it to training/datasets/")
        print("3. Run this script again\n")
        data_yaml = input("Or enter path to data.yaml manually: ")
    
    print(f"\n✅ Using dataset: {data_yaml}")
    
    # Step 6: Training parameters
    print_step(6, "Training Settings")
    
    print("Let's configure training. Default values work well for most cases.\n")
    
    epochs = input("Number of training epochs (default: 100): ").strip() or "100"
    batch_size = input("Batch size (default: 16, lower if out of memory): ").strip() or "16"
    model_size = input("Model size - n/s/m/l/x (default: n for fastest): ").strip() or "n"
    
    print(f"\nTraining with:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch: {batch_size}")
    print(f"  - Model: YOLOv8{model_size}")
    print(f"  - Dataset: {data_yaml}")
    
    response = input("\nStart training? (yes/no): ")
    if response.lower() != 'yes':
        print("Okay, cancelled.")
        sys.exit(0)
    
    # Step 7: Train
    print_step(7, "Training Model")
    
    print("Starting training... This will take 1-3 hours with GPU, 5-15 hours without GPU.")
    print("You can stop anytime with Ctrl+C. The best model will be saved.\n")
    
    cmd = f"python training/train_yolo.py --data {data_yaml} --epochs {epochs} --batch {batch_size} --model-size {model_size}"
    print(f"Command: {cmd}\n")
    
    result = os.system(cmd)
    
    if result != 0:
        print("\n❌ Training failed! Check the errors above.")
        print("Common issues:")
        print("- Not enough memory: Lower batch size")
        print("- No images: Make sure data.yaml points to correct folders")
        print("- No labels: Make sure you labeled the images")
        sys.exit(1)
    
    # Step 8: Deploy
    print_step(8, "Deploy Trained Model")
    
    best_model = "training/runs/detect/mobile_panel_train/weights/best.pt"
    
    if not Path(best_model).exists():
        print(f"❌ Could not find trained model at {best_model}")
        sys.exit(1)
    
    print(f"✅ Training complete! Model saved at: {best_model}\n")
    
    response = input("Deploy to production? (yes/no): ")
    if response.lower() == 'yes':
        print("\nDeploying model...")
        os.system(f"python training/deploy_model.py --model {best_model} --test")
        
        print("\n" + "="*80)
        print("🎉 SUCCESS! Your custom model is now deployed!")
        print("="*80)
        print("\nRestart your API to use the new model:")
        print("  python main.py")
        print("\nYour API will now use the custom-trained model automatically!")
    else:
        print(f"\nYou can deploy later with:")
        print(f"  python training/deploy_model.py --model {best_model}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(1)
