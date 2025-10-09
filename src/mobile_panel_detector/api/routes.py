"""
API Routes Module

This module contains all the Flask routes for the Mobile Panel Detection API.
"""

import base64
import os
import traceback
from flask import request, jsonify, send_file
import cv2
import numpy as np

from ..detector.panel_detector import AdvancedPanelDetector
from ..utils.image_utils import mark_image, save_processed_image, decode_image_from_request, validate_image
from ..utils.config import config


# Initialize detector
detector = AdvancedPanelDetector()


def _get_issue_description(issue_type):
    """Get human-readable description for issue types"""
    descriptions = {
        'cable_across_screen': 'Cable or wire crossing the display screen',
        'earbuds_on_display': 'Earbuds or small circular objects on the display',
        'protective_case_overhang': 'Protective case extending beyond screen boundaries',
        'screen_protector_misalignment': 'Screen protector misalignment or visible edges',
        'stickers_on_panel': 'Stickers or small rectangular objects on the display',
        'debris_covering_screen': 'Debris, dust, or particles covering the screen',
        'external_mounts_holders': 'External mounts or holders attached to the device',
        'misplaced_accessories_blocking_display': 'Misplaced accessories blocking the display'
    }
    return descriptions.get(issue_type, 'Unknown display issue')


def _convert_to_json_serializable(obj):
    """Convert NumPy types and other non-serializable objects to JSON-serializable types"""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def create_routes(app):
    """Create and register all routes with the Flask app"""
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "service": "Mobile Panel Detection API",
            "yolo_loaded": detector.model is not None,
            "version": "2.5 - Enhanced Display Issue Detection",
            "yolo_confidence": detector.min_yolo_confidence,
            "detection_mode": "mobile_panels_with_display_issues",
            "targets": "mobile devices (phones/tablets)",
            "excludes": "TVs, monitors, dark bars, large displays",
            "supports": "all orientations including sideways/rotated panels",
            "display_issues_detected": [
                "cables across screen",
                "earbuds on display", 
                "protective case overhang",
                "screen protector misalignment",
                "stickers on panel",
                "debris covering screen",
                "external mounts or holders",
                "misplaced accessories blocking display"
            ]
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
                image = decode_image_from_request(image_bytes)
                
            elif request.is_json and 'image' in request.json:
                base64_image = request.json['image']
                if ',' in base64_image:
                    base64_image = base64_image.split(',')[1]
                
                image_bytes = base64.b64decode(base64_image)
                image = decode_image_from_request(image_bytes)
            else:
                return jsonify({"error": "No image provided"}), 400
            
            if not validate_image(image):
                return jsonify({"error": "Invalid image format"}), 400
            
            # Detect panels and display issues
            detections, status_info = detector.detect(image)
            
            # Separate panel detections from display issue detections
            panel_detections = [d for d in detections if 'issue_type' not in d and 'status' not in d]
            display_issue_detections = [d for d in detections if 'issue_type' in d]
            
            # Group display issues by type
            issue_types = {}
            for issue in display_issue_detections:
                issue_type = issue['issue_type']
                if issue_type not in issue_types:
                    issue_types[issue_type] = []
                issue_types[issue_type].append(issue)
            
            # Mark the image
            marked_image = mark_image(image, detections)
            
            # Save processed image and get URL
            base_url = config[app.config.get('FLASK_CONFIG', 'default')].BASE_URL
            image_url, saved_filename = save_processed_image(
                marked_image, detections, original_filename, base_url
            )
            
            avg_confidence = sum(d['confidence'] for d in detections) / len(detections) if detections else 0.0
            
            # Determine overall result based on panels and issues
            has_panels = len(panel_detections) > 0
            has_issues = len(display_issue_detections) > 0
            
            if has_panels and has_issues:
                result = "NG - Panel with Display Issues Detected"
                message = f"Detected {len(panel_detections)} mobile panel(s) with {len(display_issue_detections)} display issue(s)"
            elif has_panels:
                result = "NG - Panel Detected"
                message = f"Detected {len(panel_detections)} mobile panel(s)"
            else:
                result = "OK - No Panel Detected"
                message = "No mobile panels detected"
            
            response = {
                "detected": has_panels,
                "has_display_issues": has_issues,
                "result": result,
                "panel_count": len(panel_detections),
                "display_issue_count": len(display_issue_detections),
                "confidence": round(avg_confidence, 3),
                "panels": [
                    {
                        "id": idx + 1,
                        "bbox": _convert_to_json_serializable(d['bbox']),
                        "confidence": round(d['confidence'], 3),
                        "method": d.get('method', 'Unknown'),
                        "angle": round(d.get('angle', 0), 1) if 'angle' in d else None,
                        "area": int(d['bbox'][2] * d['bbox'][3]),
                        "class": d.get('class', 'mobile_panel')
                    }
                    for idx, d in enumerate(panel_detections)
                ],
                "display_issues": [
                    {
                        "id": idx + 1,
                        "bbox": _convert_to_json_serializable(d['bbox']),
                        "confidence": round(d['confidence'], 3),
                        "method": d.get('method', 'Unknown'),
                        "issue_type": d['issue_type'],
                        "description": _get_issue_description(d['issue_type']),
                        "area": int(d['bbox'][2] * d['bbox'][3]),
                        "details": _convert_to_json_serializable({k: v for k, v in d.items() if k not in ['bbox', 'confidence', 'method', 'issue_type']})
                    }
                    for idx, d in enumerate(display_issue_detections)
                ],
                "issue_summary": {
                    issue_type: {
                        "count": len(issues),
                        "description": _get_issue_description(issue_type),
                        "avg_confidence": round(sum(i['confidence'] for i in issues) / len(issues), 3)
                    }
                    for issue_type, issues in issue_types.items()
                },
                "processed_image_url": image_url,
                "saved_filename": saved_filename,
                "message": message,
                "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
                "detection_mode": "mobile_panels_with_display_issues"
            }
            
            return jsonify(response), 200
            
        except Exception as e:
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
            image = decode_image_from_request(image_bytes)
            
            if not validate_image(image):
                return jsonify({"error": "Invalid image format"}), 400
            
            detections, status_info = detector.detect(image)
            marked_image = mark_image(image, detections)
            
            # Save processed image and get URL
            base_url = config[app.config.get('FLASK_CONFIG', 'default')].BASE_URL
            image_url, saved_filename = save_processed_image(
                marked_image, detections, original_filename, base_url
            )
            
            _, buffer = cv2.imencode('.jpg', marked_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            from io import BytesIO
            byte_io = BytesIO(buffer)
            byte_io.seek(0)
            
            # Use status_info from detection results
            status = status_info['status']
            panel_count = status_info['panel_count']
            
            # Return both the file and URL information
            response = send_file(
                byte_io,
                mimetype='image/jpeg',
                as_attachment=True,
                download_name=f'marked_{status}_{panel_count}panels.jpg'
            )
            
            # Add custom headers with URL information
            response.headers['X-Processed-Image-URL'] = image_url
            response.headers['X-Saved-Filename'] = saved_filename
            response.headers['X-Panel-Count'] = str(panel_count)
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
            base_url = config[app.config.get('FLASK_CONFIG', 'default')].BASE_URL
            
            for file in files:
                if file.filename:
                    original_filename = file.filename
                    image_bytes = file.read()
                    image = decode_image_from_request(image_bytes)
                    
                    if validate_image(image):
                        detections, status_info = detector.detect(image)
                        marked_image = mark_image(image, detections)
                        
                        # Save processed image and get URL
                        image_url, saved_filename = save_processed_image(
                            marked_image, detections, original_filename, base_url
                        )
                        
                        # Separate panel detections from display issue detections
                        panel_detections = [d for d in detections if 'issue_type' not in d and 'status' not in d]
                        display_issue_detections = [d for d in detections if 'issue_type' in d]
                        
                        avg_confidence = sum(d['confidence'] for d in detections) / len(detections) if detections else 0.0
                        
                        # Determine result
                        has_panels = len(panel_detections) > 0
                        has_issues = len(display_issue_detections) > 0
                        
                        if has_panels and has_issues:
                            result = "NG - Panel with Issues"
                        elif has_panels:
                            result = "NG - Panel Detected"
                        else:
                            result = "OK - No Panel"
                        
                        results.append({
                            "filename": file.filename,
                            "detected": has_panels,
                            "has_display_issues": has_issues,
                            "panel_count": len(panel_detections),
                            "display_issue_count": len(display_issue_detections),
                            "confidence": round(avg_confidence, 3),
                            "result": result,
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

    @app.route('/processed/<filename>')
    def serve_processed_image(filename):
        """Serve processed images"""
        processed_folder = app.config.get('PROCESSED_FOLDER', 'processed')
        return send_file(os.path.join(processed_folder, filename))
