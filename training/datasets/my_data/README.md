# Collected Training Data

## Collection Info
- Date: 2025-10-08 21:48:01
- Total Images: 3
- Location: /Users/debarunlahiri/Development/PythonProjects/pd-backend/training/datasets/my_data

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
labelImg training/datasets/my_data/images
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
