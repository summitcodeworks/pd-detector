from flask import Flask, request, jsonify, send_file, url_for
from flask_cors import CORS
import cv2
import numpy as np
from io import BytesIO
import base64
from PIL import Image
import os
from ultralytics import YOLO
import torch
import uuid
from datetime import datetime
from config import config

app = Flask(__name__)
CORS(app)

# Load configuration
config_name = os.getenv('FLASK_CONFIG', 'default')
app.config.from_object(config[config_name])

# Configure upload folders
app.config['UPLOAD_FOLDER'] = config[config_name].UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = config[config_name].PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config[config_name].MAX_CONTENT_LENGTH

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Initialize YOLO model (will download automatically on first run)
try:
    model = YOLO('yolov8n.pt')
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load YOLO model: {e}")
    model = None


class AdvancedPanelDetector:
    """Hybrid panel detector - Handles rotated/sideways panels"""
    
    def __init__(self):
        # Lower confidence for better detection of angled panels
        self.min_yolo_confidence = 0.35
        self.min_contour_confidence = 0.45
        
    def detect_with_yolo(self, image):
        """Detect using YOLOv8 - specifically for mobile devices only"""
        if model is None:
            return []
        
        results = model(image, verbose=False)
        detections = []
        height, width = image.shape[:2]
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Only accept cell phone class (mobile devices)
                if confidence > self.min_yolo_confidence and class_name == 'cell phone':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    
                    # Mobile device size constraints (not too large like TVs/monitors)
                    # Typical mobile devices are smaller than 15% of image width/height
                    max_mobile_width = width * 0.15
                    max_mobile_height = height * 0.15
                    
                    if (w >= 60 and h >= 60 and 
                        w <= max_mobile_width and h <= max_mobile_height):
                        
                        # Additional check: aspect ratio should be reasonable for mobile devices
                        aspect_ratio = w / h if h > 0 else 0
                        if 0.3 <= aspect_ratio <= 3.0:  # Mobile devices typically 0.5-2.0
                            detections.append({
                                'bbox': (x1, y1, w, h),
                                'confidence': confidence,
                                'method': 'YOLO-Mobile',
                                'class': class_name
                            })
        
        return detections
    
    def get_rotated_rect_bbox(self, rotated_rect):
        """Get bounding box from rotated rectangle"""
        box = cv2.boxPoints(rotated_rect)
        box = np.int32(box)  # Use int32 instead of deprecated int0
        x, y, w, h = cv2.boundingRect(box)
        # Convert to regular Python integers to avoid JSON serialization issues
        return (int(x), int(y), int(w), int(h)), box
    
    def has_screen_characteristics(self, image, bbox):
        """Check if region has mobile display characteristics - excludes TVs/monitors and dark bars"""
        x, y, w, h = bbox
        
        # Ensure coordinates are within image bounds
        img_h, img_w = image.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            return False
        
        # Extract region
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return False
        
        # Convert to different color spaces
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check 1: Not too dark (exclude dark bars/black screens)
        mean_brightness = np.mean(gray)
        not_too_dark = mean_brightness > 30  # Exclude very dark regions
        
        # Check 2: Not too bright (exclude bright lights/reflections)
        not_too_bright = mean_brightness < 240
        
        # Check 3: Standard deviation (screens have content variation)
        std_dev = np.std(gray)
        has_content = std_dev > 15  # Higher threshold for mobile displays
        
        # Check 4: Edge density (mobile screens have text/icons)
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / edges.size
        has_edges = edge_density > 0.05  # Higher threshold for mobile displays
        
        # Check 5: Color variety (mobile screens have diverse colors)
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        h_variety = np.count_nonzero(h_hist > 10)
        has_colors = h_variety > 6  # Higher threshold for mobile displays
        
        # Check 6: Not uniform (exclude uniform surfaces)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        dominant_value = np.max(hist)
        not_uniform = dominant_value < (gray.size * 0.6)  # Stricter for mobile displays
        
        # Check 7: Texture analysis (mobile screens have fine texture)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()
        has_texture = texture_variance > 80  # Higher threshold for mobile displays
        
        # Check 8: Size constraint for mobile devices (not TV/monitor sized)
        img_area = img_h * img_w
        roi_area = w * h
        mobile_size = roi_area < (img_area * 0.1)  # Mobile devices < 10% of image
        
        # Check 9: Aspect ratio for mobile devices
        aspect_ratio = w / h if h > 0 else 0
        mobile_aspect = 0.4 <= aspect_ratio <= 2.5  # Mobile device aspect ratios
        
        # Score the region (need at least 6 out of 9 for mobile display)
        score = sum([not_too_dark, not_too_bright, has_content, has_edges, 
                    has_colors, not_uniform, has_texture, mobile_size, mobile_aspect])
        
        return score >= 6
    
    def detect_with_contours(self, image):
        """Detect rectangular objects - specifically for mobile displays"""
        # Preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Multiple edge detection approaches
        edges1 = cv2.Canny(blurred, 30, 100)
        edges2 = cv2.Canny(blurred, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        height, width = image.shape[:2]
        img_area = height * width
        
        # Mobile device size constraints (much smaller than TVs/monitors)
        min_area = img_area * 0.01  # 1% minimum (mobile devices)
        max_area = img_area * 0.15  # 15% maximum (exclude large displays)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area < area < max_area:
                # Get minimum area rectangle (handles rotation)
                rect = cv2.minAreaRect(contour)
                (cx, cy), (rect_w, rect_h), angle = rect
                
                # Swap w/h if needed to get correct aspect ratio
                if rect_w < rect_h:
                    rect_w, rect_h = rect_h, rect_w
                    angle = angle + 90
                
                # Calculate aspect ratio
                aspect_ratio = float(rect_w) / rect_h if rect_h > 0 else 0
                
                # Mobile device aspect ratio constraints
                # Typical mobile devices: 0.5 (portrait) to 2.0 (landscape)
                mobile_aspect = 0.4 <= aspect_ratio <= 2.5
                
                # Size constraints for mobile devices
                mobile_width = rect_w <= width * 0.2  # Max 20% of image width
                mobile_height = rect_h <= height * 0.2  # Max 20% of image height
                min_size = rect_w >= 60 and rect_h >= 60
                
                if mobile_aspect and mobile_width and mobile_height and min_size:
                    # Get bounding box for the rotated rectangle
                    bbox, box_points = self.get_rotated_rect_bbox(rect)
                    
                    # Check if this looks like a mobile display
                    if self.has_screen_characteristics(image, bbox):
                        # Calculate confidence
                        perimeter = cv2.arcLength(contour, True)
                        epsilon = 0.02 * perimeter
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        rectangularity = 1.0 - abs(len(approx) - 4) * 0.08
                        confidence = max(0.45, min(0.85, rectangularity))
                        
                        detections.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'method': f'Contour-Mobile (angle:{int(angle)}°)',
                            'aspect_ratio': aspect_ratio,
                            'angle': angle,
                            'rotated_rect': box_points
                        })
        
        return detections
    
    def detect_with_color(self, image):
        """Detect based on mobile screen colors - excludes dark bars and large displays"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multiple color ranges for mobile screens (exclude very dark regions)
        masks = []
        
        # Range 1: Blue screens (mobile apps often have blue themes)
        masks.append(cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255])))
        
        # Range 2: White/bright screens (but not too bright)
        masks.append(cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 30, 220])))
        
        # Range 3: Cyan/Light blue (mobile UI elements)
        masks.append(cv2.inRange(hsv, np.array([85, 40, 80]), np.array([105, 255, 255])))
        
        # Range 4: Green screens (mobile apps)
        masks.append(cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255])))
        
        # Combine all masks
        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_or(mask, m)
        
        # Exclude very dark regions (dark bars, black screens)
        dark_mask = cv2.inRange(gray, 0, 50)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(dark_mask))
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        height, width = image.shape[:2]
        img_area = height * width
        
        # Mobile device size constraints
        min_area = img_area * 0.01  # 1% minimum
        max_area = img_area * 0.15  # 15% maximum (exclude large displays)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Use minAreaRect for rotation handling
                rect = cv2.minAreaRect(contour)
                bbox, box_points = self.get_rotated_rect_bbox(rect)
                x, y, w, h = bbox
                
                # Mobile device size and aspect ratio constraints
                aspect_ratio = w / h if h > 0 else 0
                mobile_aspect = 0.4 <= aspect_ratio <= 2.5
                mobile_width = w <= width * 0.2
                mobile_height = h <= height * 0.2
                min_size = w >= 60 and h >= 60
                
                if mobile_aspect and mobile_width and mobile_height and min_size:
                    # Additional check: not too dark
                    roi = image[y:y+h, x:x+w]
                    if roi.size > 0:
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        mean_brightness = np.mean(roi_gray)
                        
                        # Exclude dark bars and very dark regions
                        if mean_brightness > 40:  # Higher threshold to exclude dark bars
                            confidence = min(0.75, area / (img_area * 0.1))
                            detections.append({
                                'bbox': bbox,
                                'confidence': confidence,
                                'method': 'Color-Mobile',
                                'rotated_rect': box_points
                            })
        
        return detections
    
    def merge_detections(self, detections):
        """Merge overlapping detections"""
        if not detections:
            return []
        
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        areas = boxes[:, 2] * boxes[:, 3]
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            inds = np.where(iou <= 0.35)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
    def detect(self, image):
        """Hybrid mobile panel detection: YOLO + Contours + Color - excludes TVs/monitors and dark bars"""
        all_detections = []
        
        # Method 1: YOLO (specifically for mobile devices)
        yolo_detections = self.detect_with_yolo(image)
        all_detections.extend(yolo_detections)
        
        # Method 2: Contour detection (mobile-sized objects only)
        contour_detections = self.detect_with_contours(image)
        all_detections.extend(contour_detections)
        
        # Method 3: Color-based detection (excludes dark bars)
        color_detections = self.detect_with_color(image)
        all_detections.extend(color_detections)
        
        # Merge overlapping detections
        final_detections = self.merge_detections(all_detections)
        
        return final_detections


def mark_image(image, detections):
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


detector = AdvancedPanelDetector()


def save_processed_image(image, detections, original_filename=None):
    """Save processed image and return URL"""
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    if original_filename:
        name, ext = os.path.splitext(original_filename)
        filename = f"{name}_{timestamp}_{unique_id}{ext}"
    else:
        filename = f"processed_{timestamp}_{unique_id}.jpg"
    
    # Save the processed image
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    cv2.imwrite(filepath, image)
    
    # Generate full URL
    base_url = config[config_name].BASE_URL
    image_url = f"{base_url}/processed/{filename}"
    
    return image_url, filename


@app.route('/processed/<filename>')
def serve_processed_image(filename):
    """Serve processed images"""
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename))


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Mobile Panel Detection API",
        "yolo_loaded": model is not None,
        "version": "2.4 - Mobile-Specific Detection",
        "yolo_confidence": detector.min_yolo_confidence,
        "detection_mode": "mobile_panels_only",
        "targets": "mobile devices (phones/tablets)",
        "excludes": "TVs, monitors, dark bars, large displays",
        "supports": "all orientations including sideways/rotated panels"
    })


@app.route('/detect', methods=['POST'])
def detect_panel():
    """Main detection endpoint"""
    try:
        original_filename = None
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            original_filename = file.filename
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        elif request.is_json and 'image' in request.json:
            base64_image = request.json['image']
            if ',' in base64_image:
                base64_image = base64_image.split(',')[1]
            
            image_bytes = base64.b64decode(base64_image)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return jsonify({"error": "No image provided"}), 400
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        # Detect panels (all orientations)
        detections = detector.detect(image)
        
        # Mark the image
        marked_image = mark_image(image, detections)
        
        # Save processed image and get URL
        image_url, saved_filename = save_processed_image(marked_image, detections, original_filename)
        
        avg_confidence = sum(d['confidence'] for d in detections) / len(detections) if detections else 0.0
        
        response = {
            "detected": len(detections) > 0,
            "result": "NG - Panel Detected" if detections else "OK - No Panel Detected",
            "panel_count": len(detections),
            "confidence": round(avg_confidence, 3),
            "detections": [
                {
                    "id": idx + 1,
                    "bbox": d['bbox'],
                    "confidence": round(d['confidence'], 3),
                    "method": d.get('method', 'Unknown'),
                    "angle": round(d.get('angle', 0), 1) if 'angle' in d else None,
                    "area": d['bbox'][2] * d['bbox'][3]
                }
                for idx, d in enumerate(detections)
            ],
            "processed_image_url": image_url,
            "saved_filename": saved_filename,
            "message": f"Detected {len(detections)} mobile panel(s)" if detections else "No mobile panels detected",
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "detection_mode": "mobile_panels_only"
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/detect/image', methods=['POST'])
def detect_panel_return_image():
    """Returns marked image directly and saves it"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        original_filename = file.filename
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        detections = detector.detect(image)
        marked_image = mark_image(image, detections)
        
        # Save processed image and get URL
        image_url, saved_filename = save_processed_image(marked_image, detections, original_filename)
        
        _, buffer = cv2.imencode('.jpg', marked_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        byte_io = BytesIO(buffer)
        byte_io.seek(0)
        
        status = "NG" if detections else "OK"
        
        # Return both the file and URL information
        response = send_file(
            byte_io,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'marked_{status}_{len(detections)}panels.jpg'
        )
        
        # Add custom headers with URL information
        response.headers['X-Processed-Image-URL'] = image_url
        response.headers['X-Saved-Filename'] = saved_filename
        response.headers['X-Panel-Count'] = str(len(detections))
        response.headers['X-Status'] = status
        
        return response
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/detect/batch', methods=['POST'])
def detect_batch():
    """Batch processing with image saving"""
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files provided"}), 400
        
        results = []
        for file in files:
            if file.filename:
                original_filename = file.filename
                image_bytes = file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    detections = detector.detect(image)
                    marked_image = mark_image(image, detections)
                    
                    # Save processed image and get URL
                    image_url, saved_filename = save_processed_image(marked_image, detections, original_filename)
                    
                    avg_confidence = sum(d['confidence'] for d in detections) / len(detections) if detections else 0.0
                    
                    results.append({
                        "filename": file.filename,
                        "detected": len(detections) > 0,
                        "panel_count": len(detections),
                        "confidence": round(avg_confidence, 3),
                        "result": "NG" if detections else "OK",
                        "processed_image_url": image_url,
                        "saved_filename": saved_filename
                    })
        
        return jsonify({
            "total_images": len(results),
            "ng_count": sum(1 for r in results if r['detected']),
            "ok_count": sum(1 for r in results if not r['detected']),
            "results": results
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("Mobile Panel Detection API v2.4 - MOBILE-SPECIFIC DETECTION")
    print("=" * 70)
    print("YOLO Model:", "Loaded" if model else "Not available")
    print("Detection: YOLO + Mobile Contours + Color")
    print("YOLO Confidence:", detector.min_yolo_confidence)
    print("Target: Mobile devices (phones/tablets) ONLY")
    print("Excludes: TVs, monitors, dark bars, large displays")
    print("Supports: ALL orientations (0°, 45°, 90°, 180°, 270°, etc.)")
    print("Size Limits: Max 20% of image dimensions")
    print("Brightness: Excludes dark bars (brightness > 30)")
    print("\nEndpoints:")
    print("  - POST /detect          (JSON)")
    print("  - POST /detect/image    (Image file)")
    print("  - POST /detect/batch    (Multiple)")
    print("  - GET  /health          (Status)")
    print("\nServer: http://0.0.0.0:5000")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)