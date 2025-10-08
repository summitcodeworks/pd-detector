"""
Mobile Panel Detector Module

This module contains the core detection logic for identifying mobile device panels
while excluding TVs, monitors, and dark bars.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional


class AdvancedPanelDetector:
    """Hybrid panel detector - Handles rotated/sideways panels, mobile devices only"""
    
    def __init__(self, yolo_model_path: str = 'yolov8n.pt'):
        """
        Initialize the panel detector.
        
        Args:
            yolo_model_path: Path to YOLO model file
        """
        # Higher confidence for better selectivity - focus on main smartphones
        self.min_yolo_confidence = 0.40
        self.min_contour_confidence = 0.50
        
        # Initialize YOLO model
        try:
            self.model = YOLO(yolo_model_path)
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            self.model = None
        
    def detect_with_yolo(self, image: np.ndarray) -> List[Dict]:
        """Detect using YOLOv8 - specifically for mobile devices only"""
        if self.model is None:
            return []
        
        results = self.model(image, verbose=False)
        detections = []
        height, width = image.shape[:2]
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # Only accept cell phone class (mobile devices)
                if confidence > self.min_yolo_confidence and class_name == 'cell phone':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    
                    # Mobile device size constraints (not too large like TVs/monitors)
                    # Typical mobile devices are smaller than 25% of image width/height
                    max_mobile_width = width * 0.25
                    max_mobile_height = height * 0.25
                    
                    if (w >= 80 and h >= 80 and 
                        w <= max_mobile_width and h <= max_mobile_height):
                        
                        # Additional check: aspect ratio should be reasonable for mobile devices
                        aspect_ratio = w / h if h > 0 else 0
                        if 0.5 <= aspect_ratio <= 2.2:  # Mobile devices typically 0.5-2.2
                            # Boost confidence for larger mobile devices (main phones vs small objects)
                            area = w * h
                            img_area = width * height
                            area_ratio = area / img_area
                            
                            # Give higher confidence to devices that are reasonably sized (not too small)
                            if area_ratio > 0.01:  # At least 1% of image area
                                confidence_boost = min(0.2, area_ratio * 2)  # Up to 0.2 boost
                                confidence = min(1.0, confidence + confidence_boost)
                            
                            detections.append({
                                'bbox': (x1, y1, w, h),
                                'confidence': confidence,
                                'method': 'YOLO-Mobile',
                                'class': class_name
                            })
        
        return detections
    
    def get_rotated_rect_bbox(self, rotated_rect: Tuple) -> Tuple[Tuple, np.ndarray]:
        """Get bounding box from rotated rectangle"""
        box = cv2.boxPoints(rotated_rect)
        box = np.int32(box)  # Use int32 instead of deprecated int0
        x, y, w, h = cv2.boundingRect(box)
        # Convert to regular Python integers to avoid JSON serialization issues
        return (int(x), int(y), int(w), int(h)), box
    
    def has_screen_characteristics(self, image: np.ndarray, bbox: Tuple) -> bool:
        """Check if region has display/screen characteristics"""
        x, y, w, h = bbox
        
        # Extract region
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return False
        
        # Convert to different color spaces
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check 1: Standard deviation (screens have content variation)
        std_dev = np.std(gray)
        has_content = std_dev > 15  # Screens have varying content
        
        # Check 2: Edge density (screens have text/icons)
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / edges.size
        has_edges = edge_density > 0.05  # At least 5% edges
        
        # Check 3: Color variety (not uniform like lights/tables)
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        h_variety = np.count_nonzero(h_hist > 10)  # Multiple color hues
        has_colors = h_variety > 5
        
        # Check 4: Brightness (not too bright like lights)
        mean_brightness = np.mean(gray)
        not_too_bright = mean_brightness < 240  # Not a pure light
        
        # Check 5: Not uniform (tables/walls are uniform)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        dominant_value = np.max(hist)
        not_uniform = dominant_value < (gray.size * 0.8)  # Not 80% same color
        
        # Score the region
        score = sum([has_content, has_edges, has_colors, not_too_bright, not_uniform])
        
        return score >= 3  # Need at least 3 out of 5 characteristics
    
    def detect_with_contours(self, image: np.ndarray) -> List[Dict]:
        """Detect rectangular objects with screen characteristics"""
        # Preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        height, width = image.shape[:2]
        min_area = (height * width) * 0.04
        max_area = (height * width) * 0.85
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Check aspect ratio (phones + partial panels)
                valid_aspect = (0.3 <= aspect_ratio <= 0.9) or (1.1 <= aspect_ratio <= 3.0)
                
                if valid_aspect and w >= 80 and h >= 80:
                    # Check if this looks like a screen/display
                    if self.has_screen_characteristics(image, (x, y, w, h)):
                        # Calculate confidence based on rectangularity
                        perimeter = cv2.arcLength(contour, True)
                        epsilon = 0.02 * perimeter
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        rectangularity = 1.0 - abs(len(approx) - 4) * 0.1
                        confidence = max(0.5, min(0.85, rectangularity))
                        
                        detections.append({
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'method': 'Contour+Screen',
                            'aspect_ratio': aspect_ratio
                        })
        
        return detections
    
    def detect_with_color(self, image: np.ndarray) -> List[Dict]:
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
        
        # Range 5: Dark screens (mobile devices with dark/black screens)
        masks.append(cv2.inRange(gray, 5, 100))  # Dark but not completely black
        
        # Combine all masks
        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_or(mask, m)
        
        # Include dark regions for mobile screens (but exclude completely black)
        dark_mask = cv2.inRange(gray, 0, 5)  # Only exclude completely black
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(dark_mask))
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        height, width = image.shape[:2]
        img_area = height * width
        
        # Mobile device size constraints - prioritize larger devices
        min_area = img_area * 0.002  # 0.2% minimum (allow main smartphones)
        max_area = img_area * 0.30   # 30% maximum (exclude large displays)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Use minAreaRect for rotation handling
                rect = cv2.minAreaRect(contour)
                bbox, box_points = self.get_rotated_rect_bbox(rect)
                x, y, w, h = bbox
                
                # Mobile device size and aspect ratio constraints
                aspect_ratio = w / h if h > 0 else 0
                mobile_aspect = 0.5 <= aspect_ratio <= 2.2
                mobile_width = w <= width * 0.15
                mobile_height = h <= height * 0.15
                min_size = w >= 80 and h >= 80
                
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
    
    def merge_detections(self, detections: List[Dict]) -> List[Dict]:
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
            
            inds = np.where(iou <= 0.4)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
    def detect_cables_across_screen(self, image: np.ndarray, panel_detections: List[Dict]) -> List[Dict]:
        """Detect cables or wires crossing the mobile display panel"""
        cable_detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for panel in panel_detections:
            x, y, w, h = panel['bbox']
            
            # Extract panel region
            panel_roi = gray[y:y+h, x:x+w]
            if panel_roi.size == 0:
                continue
            
            # Detect thin linear structures (cables)
            # Use Hough Line Transform to detect lines
            edges = cv2.Canny(panel_roi, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                                   minLineLength=max(20, w//4), maxLineGap=10)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Check if line crosses significant portion of panel
                    line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    panel_diagonal = np.sqrt(w**2 + h**2)
                    
                    if line_length > panel_diagonal * 0.3:  # At least 30% of panel diagonal
                        # Calculate line angle
                        angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                        angle = abs(angle) if angle < 0 else angle
                        
                        # Cables can be at various angles but typically not perfectly horizontal/vertical
                        if not (abs(angle) < 10 or abs(angle - 90) < 10 or abs(angle - 180) < 10):
                            cable_detections.append({
                                'bbox': (x + x1, y + y1, x2-x1, y2-y1),
                                'confidence': min(0.8, line_length / panel_diagonal),
                                'method': 'Cable-Detection',
                                'issue_type': 'cable_across_screen',
                                'line_length': line_length,
                                'angle': angle,
                                'panel_id': len(cable_detections)
                            })
            
            # Alternative: Detect thin elongated contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Minimum area for cable
                    # Get bounding rectangle
                    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
                    
                    # Check if it's thin and elongated (cable-like)
                    aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h) if min(rect_w, rect_h) > 0 else 0
                    if aspect_ratio > 5 and min(rect_w, rect_h) < 8:  # Thin and long
                        cable_detections.append({
                            'bbox': (int(x + rect_x), int(y + rect_y), int(rect_w), int(rect_h)),
                            'confidence': min(0.7, aspect_ratio / 10),
                            'method': 'Cable-Contour',
                            'issue_type': 'cable_across_screen',
                            'aspect_ratio': aspect_ratio,
                            'panel_id': len(cable_detections)
                        })
        
        return cable_detections

    def detect_earbuds_on_display(self, image: np.ndarray, panel_detections: List[Dict]) -> List[Dict]:
        """Detect earbuds or small circular objects on the mobile display"""
        earbud_detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for panel in panel_detections:
            x, y, w, h = panel['bbox']
            
            # Extract panel region
            panel_roi = gray[y:y+h, x:x+w]
            if panel_roi.size == 0:
                continue
            
            # Detect circles (earbuds are typically circular)
            circles = cv2.HoughCircles(panel_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                     param1=50, param2=30, minRadius=5, maxRadius=min(w,h)//4)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (cx, cy, r) in circles:
                    # Check if circle is reasonably sized for earbuds
                    if 8 <= r <= min(w,h)//6:  # Earbud size range
                        earbud_detections.append({
                            'bbox': (int(x + cx - r), int(y + cy - r), int(2*r), int(2*r)),
                            'confidence': 0.75,
                            'method': 'Earbud-Circle',
                            'issue_type': 'earbuds_on_display',
                            'radius': r,
                            'center': (x + cx, y + cy),
                            'panel_id': len(earbud_detections)
                        })
            
            # Alternative: Detect small dark objects (earbuds are often dark)
            _, thresh = cv2.threshold(panel_roi, 100, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 <= area <= 2000:  # Earbud size range
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.6:  # Reasonably circular
                            x_cont, y_cont, w_cont, h_cont = cv2.boundingRect(contour)
                            earbud_detections.append({
                                'bbox': (int(x + x_cont), int(y + y_cont), int(w_cont), int(h_cont)),
                                'confidence': min(0.7, circularity),
                                'method': 'Earbud-Contour',
                                'issue_type': 'earbuds_on_display',
                                'circularity': circularity,
                                'area': area,
                                'panel_id': len(earbud_detections)
                            })
        
        return earbud_detections

    def detect_protective_case_overhang(self, image: np.ndarray, panel_detections: List[Dict]) -> List[Dict]:
        """Detect protective case overhang extending beyond screen boundaries"""
        case_detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for panel in panel_detections:
            x, y, w, h = panel['bbox']
            
            # Expand search area around the panel
            margin = 20
            search_x = max(0, x - margin)
            search_y = max(0, y - margin)
            search_w = min(image.shape[1] - search_x, w + 2*margin)
            search_h = min(image.shape[0] - search_y, h + 2*margin)
            
            # Extract expanded region
            search_roi = gray[search_y:search_y+search_h, search_x:search_x+search_w]
            if search_roi.size == 0:
                continue
            
            # Detect edges around the panel area
            edges = cv2.Canny(search_roi, 30, 100)
            
            # Look for rectangular structures around the panel
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Case overhang should be substantial
                    # Get bounding rectangle
                    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
                    
                    # Check if this rectangle extends beyond the panel
                    panel_in_search = (x - search_x, y - search_y, w, h)
                    
                    # Calculate overlap and extension
                    overlap_x = max(0, min(panel_in_search[0] + panel_in_search[2], rect_x + rect_w) - 
                                   max(panel_in_search[0], rect_x))
                    overlap_y = max(0, min(panel_in_search[1] + panel_in_search[3], rect_y + rect_h) - 
                                   max(panel_in_search[1], rect_y))
                    
                    if overlap_x > 0 and overlap_y > 0:
                        # Check if case extends beyond panel
                        case_extends = (rect_x < panel_in_search[0] or 
                                      rect_y < panel_in_search[1] or
                                      rect_x + rect_w > panel_in_search[0] + panel_in_search[2] or
                                      rect_y + rect_h > panel_in_search[1] + panel_in_search[3])
                        
                        if case_extends:
                            case_detections.append({
                                'bbox': (int(search_x + rect_x), int(search_y + rect_y), int(rect_w), int(rect_h)),
                                'confidence': min(0.8, area / (w * h)),
                                'method': 'Case-Overhang',
                                'issue_type': 'protective_case_overhang',
                                'overlap_area': overlap_x * overlap_y,
                                'panel_id': len(case_detections)
                            })
        
        return case_detections

    def detect_screen_protector_misalignment(self, image: np.ndarray, panel_detections: List[Dict]) -> List[Dict]:
        """Detect screen protector misalignment or edges around the screen"""
        protector_detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for panel in panel_detections:
            x, y, w, h = panel['bbox']
            
            # Expand search area to include edges around panel
            margin = 15
            search_x = max(0, x - margin)
            search_y = max(0, y - margin)
            search_w = min(image.shape[1] - search_x, w + 2*margin)
            search_h = min(image.shape[0] - search_y, h + 2*margin)
            
            # Extract expanded region
            search_roi = gray[search_y:search_y+search_h, search_x:search_x+search_w]
            if search_roi.size == 0:
                continue
            
            # Detect edges
            edges = cv2.Canny(search_roi, 30, 100)
            
            # Look for rectangular frames around the panel
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 200 < area < w * h * 0.5:  # Reasonable size for screen protector
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check if it's roughly rectangular (4 corners)
                    if len(approx) >= 4:
                        # Get bounding rectangle
                        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
                        
                        # Check if this frame is around the panel
                        panel_in_search = (x - search_x, y - search_y, w, h)
                        
                        # Calculate if frame surrounds panel
                        frame_center_x = rect_x + rect_w // 2
                        frame_center_y = rect_y + rect_h // 2
                        panel_center_x = panel_in_search[0] + panel_in_search[2] // 2
                        panel_center_y = panel_in_search[1] + panel_in_search[3] // 2
                        
                        # Check if frame is larger than panel and centered around it
                        if (rect_w > panel_in_search[2] * 1.1 and rect_h > panel_in_search[3] * 1.1 and
                            abs(frame_center_x - panel_center_x) < panel_in_search[2] * 0.3 and
                            abs(frame_center_y - panel_center_y) < panel_in_search[3] * 0.3):
                            
                            protector_detections.append({
                                'bbox': (int(search_x + rect_x), int(search_y + rect_y), int(rect_w), int(rect_h)),
                                'confidence': min(0.75, len(approx) / 8),  # More corners = higher confidence
                                'method': 'Protector-Misalignment',
                                'issue_type': 'screen_protector_misalignment',
                                'corners': len(approx),
                                'panel_id': len(protector_detections)
                            })
        
        return protector_detections

    def detect_stickers_on_panel(self, image: np.ndarray, panel_detections: List[Dict]) -> List[Dict]:
        """Detect stickers or small rectangular objects on the mobile display"""
        sticker_detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for panel in panel_detections:
            x, y, w, h = panel['bbox']
            
            # Extract panel region
            panel_roi = gray[y:y+h, x:x+w]
            if panel_roi.size == 0:
                continue
            
            # Detect small rectangular objects (stickers)
            # Use adaptive threshold to handle varying lighting
            thresh = cv2.adaptiveThreshold(panel_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 <= area <= 5000:  # Sticker size range
                    # Get bounding rectangle
                    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
                    
                    # Check if it's roughly rectangular
                    aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h) if min(rect_w, rect_h) > 0 else 0
                    if 1.0 <= aspect_ratio <= 4.0:  # Reasonable aspect ratio for stickers
                        # Check rectangularity
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            rectangularity = area / (rect_w * rect_h)
                            if rectangularity > 0.7:  # Fairly rectangular
                                sticker_detections.append({
                                    'bbox': (int(x + rect_x), int(y + rect_y), int(rect_w), int(rect_h)),
                                    'confidence': min(0.8, rectangularity * aspect_ratio / 4),
                                    'method': 'Sticker-Detection',
                                    'issue_type': 'stickers_on_panel',
                                    'rectangularity': rectangularity,
                                    'aspect_ratio': aspect_ratio,
                                    'area': area,
                                    'panel_id': len(sticker_detections)
                                })
        
        return sticker_detections

    def detect_debris_covering_screen(self, image: np.ndarray, panel_detections: List[Dict]) -> List[Dict]:
        """Detect debris, dust, or irregular particles covering the screen"""
        debris_detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for panel in panel_detections:
            x, y, w, h = panel['bbox']
            
            # Extract panel region
            panel_roi = gray[y:y+h, x:x+w]
            if panel_roi.size == 0:
                continue
            
            # Detect small irregular objects (debris)
            # Use morphological operations to enhance small particles
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            enhanced = cv2.morphologyEx(panel_roi, cv2.MORPH_TOPHAT, kernel)
            
            # Threshold to find dark particles
            _, thresh = cv2.threshold(enhanced, 30, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            debris_clusters = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 10 <= area <= 500:  # Debris size range
                    # Check if it's irregular (not too circular or rectangular)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # Get bounding rectangle for rectangularity check
                        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
                        rectangularity = area / (rect_w * rect_h) if rect_w * rect_h > 0 else 0
                        
                        # Debris is typically irregular (not too circular or rectangular)
                        if 0.3 <= circularity <= 0.8 and 0.4 <= rectangularity <= 0.9:
                            debris_clusters.append({
                                'bbox': (int(x + rect_x), int(y + rect_y), int(rect_w), int(rect_h)),
                                'area': area,
                                'circularity': circularity,
                                'rectangularity': rectangularity,
                                'contour': contour
                            })
            
            # Group nearby debris into clusters
            if debris_clusters:
                # Simple clustering based on proximity
                clusters = []
                for debris in debris_clusters:
                    added_to_cluster = False
                    for cluster in clusters:
                        # Check if debris is close to any item in cluster
                        for item in cluster:
                            dist = np.sqrt((debris['bbox'][0] - item['bbox'][0])**2 + 
                                         (debris['bbox'][1] - item['bbox'][1])**2)
                            if dist < 30:  # Within 30 pixels
                                cluster.append(debris)
                                added_to_cluster = True
                                break
                        if added_to_cluster:
                            break
                    
                    if not added_to_cluster:
                        clusters.append([debris])
                
                # Create detections for each cluster
                for cluster in clusters:
                    if len(cluster) >= 2:  # At least 2 pieces of debris
                        # Calculate cluster bounding box
                        min_x = min(item['bbox'][0] for item in cluster)
                        min_y = min(item['bbox'][1] for item in cluster)
                        max_x = max(item['bbox'][0] + item['bbox'][2] for item in cluster)
                        max_y = max(item['bbox'][1] + item['bbox'][3] for item in cluster)
                        
                        total_area = sum(item['area'] for item in cluster)
                        
                        debris_detections.append({
                            'bbox': (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)),
                            'confidence': min(0.7, len(cluster) / 10 + total_area / 1000),
                            'method': 'Debris-Detection',
                            'issue_type': 'debris_covering_screen',
                            'debris_count': len(cluster),
                            'total_area': total_area,
                            'panel_id': len(debris_detections)
                        })
        
        return debris_detections

    def detect_external_mounts_holders(self, image: np.ndarray, panel_detections: List[Dict]) -> List[Dict]:
        """Detect external mounts, holders, or larger objects attached to the device"""
        mount_detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for panel in panel_detections:
            x, y, w, h = panel['bbox']
            
            # Expand search area significantly around the panel
            margin = 50
            search_x = max(0, x - margin)
            search_y = max(0, y - margin)
            search_w = min(image.shape[1] - search_x, w + 2*margin)
            search_h = min(image.shape[0] - search_y, h + 2*margin)
            
            # Extract expanded region
            search_roi = gray[search_y:search_y+search_h, search_x:search_x+search_w]
            if search_roi.size == 0:
                continue
            
            # Detect larger objects (mounts/holders)
            edges = cv2.Canny(search_roi, 30, 100)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # Mounts/holders are typically larger
                    # Get bounding rectangle
                    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
                    
                    # Check if this object is adjacent to or overlapping with the panel
                    panel_in_search = (x - search_x, y - search_y, w, h)
                    
                    # Calculate distance from panel
                    panel_center_x = panel_in_search[0] + panel_in_search[2] // 2
                    panel_center_y = panel_in_search[1] + panel_in_search[3] // 2
                    obj_center_x = rect_x + rect_w // 2
                    obj_center_y = rect_y + rect_h // 2
                    
                    distance = np.sqrt((panel_center_x - obj_center_x)**2 + 
                                     (panel_center_y - obj_center_y)**2)
                    
                    # Check if object is close to panel or overlaps
                    max_distance = max(panel_in_search[2], panel_in_search[3]) * 0.8
                    
                    if distance < max_distance:
                        # Check if object is larger than panel (typical for mounts)
                        obj_area = rect_w * rect_h
                        panel_area = panel_in_search[2] * panel_in_search[3]
                        
                        if obj_area > panel_area * 0.5:  # At least 50% of panel size
                            mount_detections.append({
                                'bbox': (int(search_x + rect_x), int(search_y + rect_y), int(rect_w), int(rect_h)),
                                'confidence': min(0.8, area / (panel_area * 2)),
                                'method': 'Mount-Holder-Detection',
                                'issue_type': 'external_mounts_holders',
                                'distance_from_panel': distance,
                                'area_ratio': obj_area / panel_area,
                                'panel_id': len(mount_detections)
                            })
        
        return mount_detections

    def detect_misplaced_accessories(self, image: np.ndarray, panel_detections: List[Dict]) -> List[Dict]:
        """Detect various misplaced accessories blocking the display"""
        accessory_detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for panel in panel_detections:
            x, y, w, h = panel['bbox']
            
            # Extract panel region
            panel_roi = gray[y:y+h, x:x+w]
            if panel_roi.size == 0:
                continue
            
            # Detect various objects that might be accessories
            # Use multiple detection methods
            
            # Method 1: Detect small to medium objects
            contours, _ = cv2.findContours(panel_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 200 <= area <= 3000:  # Accessory size range
                    # Get bounding rectangle
                    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
                    
                    # Calculate various shape characteristics
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        rectangularity = area / (rect_w * rect_h) if rect_w * rect_h > 0 else 0
                        aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h) if min(rect_w, rect_h) > 0 else 0
                        
                        # Check if it could be an accessory (various shapes)
                        is_accessory = (
                            (0.4 <= circularity <= 0.9) or  # Could be circular accessory
                            (0.6 <= rectangularity <= 0.95) or  # Could be rectangular accessory
                            (1.5 <= aspect_ratio <= 6.0)  # Could be elongated accessory
                        )
                        
                        if is_accessory:
                            # Calculate confidence based on how much it blocks the panel
                            coverage_ratio = area / (w * h)
                            
                            accessory_detections.append({
                                'bbox': (int(x + rect_x), int(y + rect_y), int(rect_w), int(rect_h)),
                                'confidence': min(0.75, coverage_ratio * 2),
                                'method': 'Accessory-Detection',
                                'issue_type': 'misplaced_accessories_blocking_display',
                                'coverage_ratio': coverage_ratio,
                                'circularity': circularity,
                                'rectangularity': rectangularity,
                                'aspect_ratio': aspect_ratio,
                                'area': area,
                                'panel_id': len(accessory_detections)
                            })
            
            # Method 2: Detect using template matching for common accessories
            # This is a simplified approach - in practice, you might have templates
            # for common accessories like stylus, small tools, etc.
            
            # Look for thin elongated objects (stylus, pens, etc.)
            edges = cv2.Canny(panel_roi, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                                   minLineLength=30, maxLineGap=5)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    if line_length > 40:  # Significant length
                        # Create bounding box around line
                        line_bbox = (min(x1, x2), min(y1, y2), 
                                   abs(x2-x1), abs(y2-y1))
                        
                        accessory_detections.append({
                            'bbox': (int(x + line_bbox[0]), int(y + line_bbox[1]), 
                                   int(line_bbox[2]), int(line_bbox[3])),
                            'confidence': 0.6,
                            'method': 'Accessory-Line',
                            'issue_type': 'misplaced_accessories_blocking_display',
                            'line_length': line_length,
                            'panel_id': len(accessory_detections)
                        })
        
        return accessory_detections

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Hybrid detection: YOLO + Smart Contours"""
        all_detections = []
        
        # Method 1: YOLO (preferred)
        yolo_detections = self.detect_with_yolo(image)
        all_detections.extend(yolo_detections)
        
        # Method 2: Contour with screen characteristics (if YOLO finds nothing)
        if len(yolo_detections) == 0:
            contour_detections = self.detect_with_contours(image)
            all_detections.extend(contour_detections)
        
        # Merge overlapping detections
        final_detections = self.merge_detections(all_detections)
        
        # Now detect display issues on the found panels
        display_issue_detections = []
        
        if final_detections:
            # Detect cables across screen
            cable_detections = self.detect_cables_across_screen(image, final_detections)
            display_issue_detections.extend(cable_detections)
            
            # Detect earbuds on display
            earbud_detections = self.detect_earbuds_on_display(image, final_detections)
            display_issue_detections.extend(earbud_detections)
            
            # Detect protective case overhang
            case_detections = self.detect_protective_case_overhang(image, final_detections)
            display_issue_detections.extend(case_detections)
            
            # Detect screen protector misalignment
            protector_detections = self.detect_screen_protector_misalignment(image, final_detections)
            display_issue_detections.extend(protector_detections)
            
            # Detect stickers on panel
            sticker_detections = self.detect_stickers_on_panel(image, final_detections)
            display_issue_detections.extend(sticker_detections)
            
            # Detect debris covering screen
            debris_detections = self.detect_debris_covering_screen(image, final_detections)
            display_issue_detections.extend(debris_detections)
            
            # Detect external mounts/holders
            mount_detections = self.detect_external_mounts_holders(image, final_detections)
            display_issue_detections.extend(mount_detections)
            
            # Detect misplaced accessories
            accessory_detections = self.detect_misplaced_accessories(image, final_detections)
            display_issue_detections.extend(accessory_detections)
        
        # Combine panel detections with display issue detections
        all_final_detections = final_detections + display_issue_detections
        
        return all_final_detections
