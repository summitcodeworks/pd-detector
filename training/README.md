# Panel Detection Training

Train a custom YOLO model for panel detection using your images in the `uploads/` folder.

## Quick Start (Recommended)

You have **820+ images** in your uploads folder. Train a model in one command:

```bash
python training/quick_train.py
```

This will:
1. Collect all images from `uploads/` (train/ and test/ subdirectories)
2. Auto-label panels using the current detector
3. Train a YOLOv8 model (1-3 hours with GPU, 5-15 hours without)
4. Deploy the trained model to production

Just follow the prompts!

## Step-by-Step Training

If you want more control:

### Step 1: Prepare Dataset
```bash
python training/prepare_dataset.py --uploads uploads --output training/datasets/panel_dataset
```

This collects all images recursively from uploads/, auto-labels them, and creates train/val splits.

### Step 2: Train Model
```bash
python training/train_yolo.py --data training/datasets/panel_dataset/data.yaml --epochs 100 --batch 16
```

Options:
- `--epochs INT` - Training iterations (default: 100)
- `--batch INT` - Batch size (default: 16, lower if out of memory)
- `--model-size {n,s,m,l,x}` - Model size (default: n for fastest)
- `--device {auto,cpu,cuda}` - Training device

### Step 3: Deploy
```bash
python training/deploy_model.py --model training/runs/detect/mobile_panel_train/weights/best.pt --test
```

Then restart your API:
```bash
python main.py
```

## Your Images

Your `uploads/` folder structure:
```
uploads/
├── train/good/         # 20 training images
└── test/
    ├── ground_truth/   # Ground truth images
    ├── oil/            # 400 oil defect images
    ├── scratch/        # 400 scratch defect images
    └── stain/          # Stain defect images
```

**Total: 820+ images** - Excellent for training!

## Why Train a Custom Model?

Your current detector uses simple color detection. A trained YOLO model will:
- Learn actual visual patterns of panels
- Be more accurate and robust
- Handle various lighting conditions better
- Detect panels faster

## Troubleshooting

### Out of Memory
```bash
python training/train_yolo.py --data ... --batch 4
```

### Training Too Slow
```bash
python training/train_yolo.py --data ... --epochs 50
```

### No Panels Detected During Auto-Labeling
```bash
python training/prepare_dataset.py --confidence 0.1
```

### Model Not Deployed
- Check that `best.pt` exists in `training/runs/detect/mobile_panel_train/weights/`
- Run deploy script with `--test` flag
- Restart API server

## Expected Results

Good model performance:
- mAP50: > 0.90 (90%+ accuracy)
- Precision: > 0.85 (85% of detections are correct)
- Recall: > 0.80 (80% of panels are found)

If lower:
- Add more diverse images
- Train longer (200-300 epochs)
- Use larger model (`--model-size s` or `m`)

## Training Files

- `prepare_dataset.py` - Collects and labels images from uploads/
- `train_yolo.py` - Main training script
- `evaluate_model.py` - Evaluate trained model
- `deploy_model.py` - Deploy model to production
- `quick_train.py` - All-in-one automated training
- `auto_label.py` - Auto-label images using current detector
- `simple_train.py` - Interactive training with prompts

## Advanced Options

### Manual Labeling

If auto-labeling doesn't work well:
1. Use [Roboflow](https://roboflow.com) - Upload images, draw boxes, export as YOLOv8
2. Use [LabelImg](https://github.com/heartexlabs/labelImg) - Desktop app for labeling
3. Use [CVAT](https://cvat.ai) - Professional annotation tool

### Training Parameters

```bash
python training/train_yolo.py \
    --data your_data.yaml \
    --epochs 100 \
    --batch 16 \
    --model-size n \
    --patience 50 \
    --lr0 0.01 \
    --img-size 640
```

### Resume Training

```bash
python training/train_yolo.py \
    --data your_data.yaml \
    --model training/runs/detect/mobile_panel_train/weights/best.pt \
    --epochs 50
```

## Getting Help

- Check training logs: `training/runs/detect/mobile_panel_train/`
- View training curves: `results.png` in run directory
- Review validation predictions: `val_batch*_pred.jpg` files
- YOLOv8 docs: https://docs.ultralytics.com

## Ready?

```bash
python training/quick_train.py
```

That's it! The script will guide you through everything.
