"""
Training Module for Panel Detection

This module provides tools for training custom YOLO models for panel detection.

Quick Start:
    python training/quick_train.py

Or step by step:
    1. python training/prepare_dataset.py
    2. python training/train_yolo.py --data training/datasets/panel_dataset/data.yaml
    3. python training/deploy_model.py --model training/runs/detect/mobile_panel_train/weights/best.pt
"""

__version__ = "2.0.0"
__author__ = "Panel Detection Team"

# Import main functions for convenience
try:
    from .train_yolo import train_model
    from .evaluate_model import evaluate_model, visualize_predictions
    from .deploy_model import deploy_model
    from .prepare_dataset import collect_images_from_uploads, process_and_label_images
except ImportError:
    # Dependencies not installed
    pass

__all__ = [
    'train_model',
    'evaluate_model',
    'visualize_predictions',
    'deploy_model',
    'collect_images_from_uploads',
    'process_and_label_images'
]
