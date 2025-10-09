# Custom Mobile Panel Dataset

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
