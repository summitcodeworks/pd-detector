# Troubleshooting Guide

Common issues and solutions for training panel detection models.

## Dataset Issues

### No Images Found
```bash
# Check if images exist
ls uploads/

# Should show train/ and test/ directories with images
```

### No Panels Detected During Auto-Labeling
```bash
# Lower confidence threshold
python training/prepare_dataset.py --confidence 0.1

# Or check that images actually contain panels
```

### Not Enough Training Data
You need at least 200 labeled images. Your uploads/ folder has 820+ images which is excellent.

## Training Issues

### Out of Memory (CUDA OOM)
```bash
# Option 1: Reduce batch size
python training/train_yolo.py --data ... --batch 4

# Option 2: Use smaller model
python training/train_yolo.py --data ... --model-size n

# Option 3: Reduce image size
python training/train_yolo.py --data ... --img-size 416

# Option 4: Use CPU
python training/train_yolo.py --data ... --device cpu
```

### Training Very Slow
- Use GPU if available (10-50x faster)
- Reduce batch size: `--batch 8`
- Use smaller model: `--model-size n`
- Reduce image size: `--img-size 416`
- Reduce workers: `--workers 4`

### Loss is NaN
```bash
# Lower learning rate
python training/train_yolo.py --data ... --lr0 0.001

# Check that labels are valid (values between 0 and 1)
```

### Model Not Improving
1. Check if labels are correct
2. Collect more diverse training data
3. Increase epochs: `--epochs 200`
4. Try larger model: `--model-size s`
5. Adjust learning rate: `--lr0 0.005`

## Performance Issues

### Low mAP (<0.5)

**Cause 1: Not enough data**
- Solution: Use all 820+ images from uploads/

**Cause 2: Incorrect labels**
- Solution: Review auto-generated labels or manually label images

**Cause 3: Model too small**
```bash
python training/train_yolo.py --data ... --model-size s
```

**Cause 4: Need more training**
```bash
python training/train_yolo.py --data ... --epochs 200
```

### High Training Accuracy, Low Validation Accuracy (Overfitting)

Solutions:
1. Collect more training data
2. Enable more augmentation:
```bash
python training/train_yolo.py --mosaic 1.0 --mixup 0.15 --degrees 10
```
3. Use early stopping (enabled by default with `--patience 50`)

### Too Many False Positives

Solutions:
1. Increase confidence threshold in detector
2. Add negative examples (images without panels)
3. Train with more diverse backgrounds

### Missing Panels (Low Recall)

Solutions:
1. Lower confidence threshold in detector
2. Collect more examples of difficult cases
3. Ensure all panels are labeled in training data
4. Train for more epochs
5. Try larger model

## Deployment Issues

### Model File Not Found
```bash
# Check if model exists
ls training/runs/detect/mobile_panel_train/weights/best.pt

# Copy manually if needed
cp training/runs/detect/mobile_panel_train/weights/best.pt models/
```

### API Still Using Old Model
```bash
# Clear cache
rm -rf __pycache__
rm -rf src/**/__pycache__

# Restart API
python main.py
```

### Deployed Model Performs Worse

Causes:
- Different image quality/format
- Different confidence threshold
- Test images very different from training data

Solutions:
1. Test on same images used during training first
2. Check confidence threshold matches
3. Collect and retrain with production images

## Data Issues

### Images Are Too Different from Production

**Critical:** Train on actual production images!

Your uploads/ folder already has production images:
- 20 good training images
- 400 oil defect images
- 400 scratch defect images

This is perfect! Just run:
```bash
python training/quick_train.py
```

### Dataset Structure Wrong

Should look like:
```
training/datasets/panel_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

The prepare_dataset.py script creates this automatically.

## Evaluation Issues

### Cannot Evaluate Model

Make sure you have test images:
```bash
ls training/datasets/panel_dataset/val/
```

If empty, the dataset preparation failed. Re-run:
```bash
python training/prepare_dataset.py
```

### Evaluation Metrics Don't Match Production

Your test set must be representative of production. Use real production images from uploads/.

## Integration Issues

### Model Works in Training but Not in Application

Debug steps:
```python
# Test model loading
from ultralytics import YOLO
model = YOLO('models/mobile_panel_custom.pt')
print(model.names)

# Test inference
import cv2
img = cv2.imread('test_image.jpg')
results = model(img)
print(f"Detections: {len(results[0].boxes)}")
```

### Different Results Between Training and Production

Check:
1. Image preprocessing is same
2. Confidence threshold matches
3. Image size/format is same
4. Test with exact same image in both environments

## Resource Issues

### Not Enough Disk Space
```bash
# Clean old training runs
rm -rf training/runs/detect/old_run*/

# Keep only best model
rm training/runs/detect/mobile_panel_train/weights/epoch*.pt
```

### Not Enough RAM
- Reduce batch size: `--batch 4`
- Reduce workers: `--workers 2`
- Reduce image size: `--img-size 416`
- Close other applications

## Quick Reference

### Minimum Requirements
- **Data:** 200+ labeled images (you have 820+!)
- **Storage:** 5 GB free space
- **RAM:** 8 GB (CPU) or 4 GB (GPU)
- **Time:** 1-15 hours depending on GPU/CPU

### Good Performance Metrics
- mAP50: > 0.90
- Precision: > 0.85
- Recall: > 0.80

### Common Commands

**Out of memory:**
```bash
python training/train_yolo.py --data ... --batch 4
```

**Too slow:**
```bash
python training/train_yolo.py --data ... --epochs 50 --model-size n
```

**Poor accuracy:**
```bash
python training/train_yolo.py --data ... --epochs 200 --model-size s
```

## Getting Help

1. Check training logs: `training/runs/detect/mobile_panel_train/`
2. View training curves: `results.png`
3. Review predictions: `val_batch*_pred.jpg`
4. YOLOv8 docs: https://docs.ultralytics.com

## Still Stuck?

Run the quick training script which handles most issues automatically:
```bash
python training/quick_train.py
```

It will:
- Automatically collect images
- Handle labeling
- Set optimal parameters
- Deploy when done
