"""
Image Utilities Module

This module contains utility functions for image processing, marking, and file handling.
"""

import cv2
import numpy as np
import os
import uuid
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from flask import current_app


def mark_image(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Mark detected panels - handles rotated panels"""
    marked = image.copy()
    height, width = marked.shape[:2]
    
    if not detections:
        # No panel detected - show OK
        overlay = marked.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 180, 0), -1)
        marked = cv2.addWeighted(overlay, 0.3, marked, 0.7, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "OK - No Panel Detected"
        font_scale = 1.2
        thickness = 3
        
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (width - text_width) // 2
        y = 50
        
        cv2.putText(marked, text, (x+2, y+2), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(marked, text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        # Checkmark
        check_size = 40
        check_x = x - check_size - 20
        check_y = y - check_size//2
        cv2.line(marked, (check_x, check_y), (check_x+check_size//3, check_y+check_size//2), (255, 255, 255), 4)
        cv2.line(marked, (check_x+check_size//3, check_y+check_size//2), (check_x+check_size, check_y-check_size//2), (255, 255, 255), 4)
        
    else:
        # Panel detected - show NG
        for idx, detection in enumerate(detections, 1):
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            method = detection.get('method', 'Unknown')
            
            # If we have rotated rect points, draw them too
            if 'rotated_rect' in detection:
                box_points = detection['rotated_rect']
                cv2.drawContours(marked, [box_points], 0, (0, 100, 255), 2)  # Blue outline for rotation
            
            # Draw red rectangle
            cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 0, 255), 4)
            
            # Draw corners
            corner_length = 30
            corner_thickness = 6
            cv2.line(marked, (x, y), (x + corner_length, y), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x, y), (x, y + corner_length), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x + w, y), (x + w - corner_length, y), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x + w, y), (x + w, y + corner_length), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x, y + h), (x + corner_length, y + h), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x, y + h), (x, y + h - corner_length), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x + w, y + h), (x + w - corner_length, y + h), (0, 0, 255), corner_thickness)
            cv2.line(marked, (x + w, y + h), (x + w, y + h - corner_length), (0, 0, 255), corner_thickness)
            
            # NG Label
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = "NG"
            font_scale = 2.5
            thickness = 4
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            label_y = max(y - 10, text_height + 20)
            label_x = x
            padding = 15
            
            cv2.rectangle(marked, 
                         (label_x - padding, label_y - text_height - padding),
                         (label_x + text_width + padding, label_y + baseline + padding),
                         (0, 0, 255), -1)
            
            cv2.rectangle(marked, 
                         (label_x - padding, label_y - text_height - padding),
                         (label_x + text_width + padding, label_y + baseline + padding),
                         (255, 255, 255), 2)
            
            cv2.putText(marked, label, (label_x + 2, label_y + 2),
                       font, font_scale, (0, 0, 0), thickness + 1)
            cv2.putText(marked, label, (label_x, label_y),
                       font, font_scale, (255, 255, 255), thickness)
            
            # Info text with angle if available
            angle_info = ""
            if 'angle' in detection:
                angle_info = f" {int(detection['angle'])}°"
            info_text = f"#{idx} {confidence*100:.1f}%{angle_info} ({method})"
            info_font_scale = 0.5
            info_thickness = 2
            
            (info_width, info_height), _ = cv2.getTextSize(info_text, font, info_font_scale, info_thickness)
            info_y = y + h + 25
            
            cv2.rectangle(marked,
                         (x, info_y - info_height - 5),
                         (x + info_width + 10, info_y + 5),
                         (0, 0, 0), -1)
            cv2.rectangle(marked,
                         (x, info_y - info_height - 5),
                         (x + info_width + 10, info_y + 5),
                         (0, 0, 255), 2)
            
            cv2.putText(marked, info_text, (x + 5, info_y),
                       font, info_font_scale, (255, 255, 255), info_thickness)
        
        # Warning banner
        overlay = marked.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 255), -1)
        marked = cv2.addWeighted(overlay, 0.3, marked, 0.7, 0)
        
        warning_text = f"PANEL DETECTED - NG ({len(detections)} found)"
        font_scale = 1.0
        thickness = 3
        (text_width, text_height), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = (width - text_width) // 2
        text_y = 50
        
        # Warning X icon
        icon_size = 35
        icon_x = text_x - icon_size - 20
        icon_y = text_y - icon_size//2
        cv2.line(marked, (icon_x, icon_y), (icon_x+icon_size, icon_y+icon_size), (255, 255, 255), 5)
        cv2.line(marked, (icon_x+icon_size, icon_y), (icon_x, icon_y+icon_size), (255, 255, 255), 5)
        
        cv2.putText(marked, warning_text, (text_x+2, text_y+2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(marked, warning_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return marked


def save_processed_image(image: np.ndarray, detections: List[Dict], 
                        original_filename: Optional[str] = None, 
                        base_url: str = "http://localhost:6000") -> Tuple[str, str]:
    """
    Save processed image and return URL
    
    Args:
        image: Processed image to save
        detections: List of detections
        original_filename: Original filename (optional)
        base_url: Base URL for the API
        
    Returns:
        Tuple of (image_url, saved_filename)
    """
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    if original_filename:
        name, ext = os.path.splitext(original_filename)
        filename = f"{name}_{timestamp}_{unique_id}{ext}"
    else:
        filename = f"processed_{timestamp}_{unique_id}.jpg"
    
    # Get processed folder from Flask config
    processed_folder = current_app.config.get('PROCESSED_FOLDER', 'processed')
    
    # Save the processed image
    filepath = os.path.join(processed_folder, filename)
    cv2.imwrite(filepath, image)
    
    # Generate full URL
    image_url = f"{base_url}/processed/{filename}"
    
    return image_url, filename


def decode_image_from_request(request_data: bytes) -> Optional[np.ndarray]:
    """
    Decode image from request data
    
    Args:
        request_data: Raw image data
        
    Returns:
        Decoded image as numpy array or None if failed
    """
    try:
        nparr = np.frombuffer(request_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


def validate_image(image: np.ndarray) -> bool:
    """
    Validate if image is valid for processing
    
    Args:
        image: Image to validate
        
    Returns:
        True if valid, False otherwise
    """
    if image is None:
        return False
    
    if len(image.shape) != 3:
        return False
    
    height, width = image.shape[:2]
    if height < 100 or width < 100:
        return False
    
    return True
