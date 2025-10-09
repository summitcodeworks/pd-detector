"""
Model Evaluation Script

Evaluates a trained YOLOv8 model on test images and generates performance metrics.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


def evaluate_model(model_path: str, data_yaml: str = None, test_dir: str = None, conf_threshold: float = 0.25):
    """Evaluate trained model"""
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    results_dict = {
        'model_path': model_path,
        'conf_threshold': conf_threshold,
        'results': []
    }
    
    # Option 1: Evaluate on dataset using data.yaml
    if data_yaml:
        print(f"\nEvaluating on dataset: {data_yaml}")
        metrics = model.val(data=data_yaml, conf=conf_threshold)
        
        print("\nValidation Metrics:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        print(f"  F1 Score: {2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr):.4f}")
        
        results_dict['validation_metrics'] = {
            'map50': float(metrics.box.map50),
            'map50_95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'f1_score': float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr))
        }
    
    # Option 2: Evaluate on test directory
    if test_dir:
        print(f"\nEvaluating on test images: {test_dir}")
        test_path = Path(test_dir)
        
        if not test_path.exists():
            print(f"Error: Test directory not found: {test_dir}")
            return
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(test_path.glob(f"*{ext}"))
            image_files.extend(test_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"Error: No images found in {test_dir}")
            return
        
        print(f"Found {len(image_files)} test images")
        
        total_detections = 0
        images_with_detections = 0
        confidences = []
        
        for img_path in image_files:
            results = model(str(img_path), conf=conf_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cls_name = model.names[cls]
                    
                    detections.append({
                        'class': cls_name,
                        'confidence': conf,
                        'bbox': box.xyxy[0].tolist()
                    })
                    
                    confidences.append(conf)
                    total_detections += 1
            
            if detections:
                images_with_detections += 1
            
            results_dict['results'].append({
                'image': str(img_path.name),
                'num_detections': len(detections),
                'detections': detections
            })
        
        print(f"\nTest Set Results:")
        print(f"  Total images: {len(image_files)}")
        print(f"  Images with detections: {images_with_detections}")
        print(f"  Total detections: {total_detections}")
        print(f"  Average detections per image: {total_detections / len(image_files):.2f}")
        
        if confidences:
            print(f"  Average confidence: {np.mean(confidences):.4f}")
            print(f"  Min confidence: {np.min(confidences):.4f}")
            print(f"  Max confidence: {np.max(confidences):.4f}")
        
        results_dict['test_set_summary'] = {
            'total_images': len(image_files),
            'images_with_detections': images_with_detections,
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(image_files),
            'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'min_confidence': float(np.min(confidences)) if confidences else 0.0,
            'max_confidence': float(np.max(confidences)) if confidences else 0.0
        }
    
    # Save results to JSON
    output_path = Path(model_path).parent.parent / "evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nEvaluation results saved to: {output_path}")
    
    return results_dict


def visualize_predictions(model_path: str, image_path: str, output_path: str = None, conf_threshold: float = 0.25):
    """Visualize model predictions on a single image"""
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Running inference on: {image_path}")
    results = model(image_path, conf=conf_threshold)
    
    # Get the result image with annotations
    annotated_image = results[0].plot()
    
    if output_path is None:
        output_path = Path(image_path).parent / f"predicted_{Path(image_path).name}"
    
    cv2.imwrite(str(output_path), annotated_image)
    print(f"Saved annotated image to: {output_path}")
    
    # Print detection details
    for result in results:
        boxes = result.boxes
        print(f"\nDetected {len(boxes)} objects:")
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = model.names[cls]
            bbox = box.xyxy[0].tolist()
            
            print(f"  {i+1}. Class: {cls_name}, Confidence: {conf:.4f}, BBox: {bbox}")


def compare_models(model_paths: List[str], data_yaml: str, conf_threshold: float = 0.25):
    """Compare performance of multiple models"""
    
    print(f"Comparing {len(model_paths)} models...")
    
    comparison_results = []
    
    for model_path in model_paths:
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {model_path}")
        print(f"{'=' * 80}")
        
        model = YOLO(model_path)
        metrics = model.val(data=data_yaml, conf=conf_threshold)
        
        comparison_results.append({
            'model': model_path,
            'map50': float(metrics.box.map50),
            'map50_95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        })
    
    print(f"\n{'=' * 80}")
    print("Model Comparison")
    print(f"{'=' * 80}")
    print(f"{'Model':<50} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print(f"{'-' * 80}")
    
    for result in comparison_results:
        model_name = Path(result['model']).name
        print(f"{model_name:<50} {result['map50']:>8.4f} {result['map50_95']:>10.4f} "
              f"{result['precision']:>10.4f} {result['recall']:>8.4f}")
    
    # Save comparison results
    output_path = "training/model_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\nComparison results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLOv8 model for mobile panel detection"
    )
    
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model (.pt file)"
    )
    
    parser.add_argument(
        "--data",
        help="Path to data.yaml for validation"
    )
    
    parser.add_argument(
        "--test-dir",
        help="Path to directory with test images"
    )
    
    parser.add_argument(
        "--visualize",
        help="Path to single image for visualization"
    )
    
    parser.add_argument(
        "--output",
        help="Output path for visualized image"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections"
    )
    
    parser.add_argument(
        "--compare-models",
        nargs='+',
        help="Compare multiple models (provide multiple model paths)"
    )
    
    args = parser.parse_args()
    
    if args.compare_models:
        if not args.data:
            print("Error: --data required for model comparison")
            sys.exit(1)
        compare_models(args.compare_models, args.data, args.conf)
    
    elif args.visualize:
        visualize_predictions(args.model, args.visualize, args.output, args.conf)
    
    else:
        if not args.data and not args.test_dir:
            print("Error: Provide either --data or --test-dir for evaluation")
            sys.exit(1)
        
        evaluate_model(args.model, args.data, args.test_dir, args.conf)


if __name__ == "__main__":
    main()
