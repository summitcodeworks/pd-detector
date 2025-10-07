# Mobile Panel Detection API Documentation

## Overview

The Mobile Panel Detection API is a Flask-based REST API for detecting mobile device panels in images using advanced computer vision techniques including YOLO, OpenCV, and hybrid detection methods.

## Features

- **Mobile-Specific Detection**: Only detects mobile devices (phones/tablets), excludes TVs and monitors
- **Dark Bar Filtering**: Recognizes that dark bars are not displays
- **Rotation Support**: Detects panels in any orientation (0°, 45°, 90°, 180°, 270°, etc.)
- **Multiple Input Formats**: Supports file uploads and base64 encoded images
- **Batch Processing**: Process multiple images at once
- **URL-Based Storage**: Returns full URLs for processed images instead of base64 data

## API Endpoints

### Health Check
```
GET /health
```
Returns API status and configuration information.

**Response:**
```json
{
  "status": "healthy",
  "service": "Mobile Panel Detection API",
  "yolo_loaded": true,
  "version": "2.4 - Mobile-Specific Detection",
  "yolo_confidence": 0.35,
  "detection_mode": "mobile_panels_only",
  "targets": "mobile devices (phones/tablets)",
  "excludes": "TVs, monitors, dark bars, large displays",
  "supports": "all orientations including sideways/rotated panels"
}
```

### Single Image Detection
```
POST /detect
```
Detect panels in a single image. Accepts:
- File upload via `file` parameter
- Base64 encoded image via JSON `image` parameter

**Request (File Upload):**
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/detect
```

**Request (Base64):**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"image":"data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."}' \
  http://localhost:5000/detect
```

**Response:**
```json
{
  "detected": true,
  "result": "NG - Panel Detected",
  "panel_count": 1,
  "confidence": 0.823,
  "detections": [
    {
      "id": 1,
      "bbox": [100, 150, 200, 300],
      "confidence": 0.823,
      "method": "YOLO-Mobile",
      "angle": 0.0,
      "area": 60000
    }
  ],
  "processed_image_url": "http://localhost:5000/processed/image_20241201_143022_a1b2c3d4.jpg",
  "saved_filename": "image_20241201_143022_a1b2c3d4.jpg",
  "message": "Detected 1 mobile panel(s)",
  "image_size": {"width": 1920, "height": 1080},
  "detection_mode": "mobile_panels_only"
}
```

### Image File Response
```
POST /detect/image
```
Returns the marked image directly as a JPEG file with additional URL information in headers.

**Headers:**
- `X-Processed-Image-URL`: URL to access the processed image
- `X-Saved-Filename`: Filename of the saved processed image
- `X-Panel-Count`: Number of panels detected
- `X-Status`: Detection status (OK/NG)

### Batch Processing
```
POST /detect/batch
```
Process multiple images at once via `files` parameter.

**Request:**
```bash
curl -X POST -F "files=@image1.jpg" -F "files=@image2.jpg" http://localhost:5000/detect/batch
```

**Response:**
```json
{
  "total_images": 2,
  "ng_count": 1,
  "ok_count": 1,
  "results": [
    {
      "filename": "image1.jpg",
      "detected": true,
      "panel_count": 1,
      "confidence": 0.823,
      "result": "NG",
      "processed_image_url": "http://localhost:5000/processed/image1_20241201_143022_a1b2c3d4.jpg",
      "saved_filename": "image1_20241201_143022_a1b2c3d4.jpg"
    },
    {
      "filename": "image2.jpg",
      "detected": false,
      "panel_count": 0,
      "confidence": 0.0,
      "result": "OK",
      "processed_image_url": "http://localhost:5000/processed/image2_20241201_143022_b2c3d4e5.jpg",
      "saved_filename": "image2_20241201_143022_b2c3d4e5.jpg"
    }
  ]
}
```

### Processed Image Access
```
GET /processed/<filename>
```
Access processed images directly via their URLs.

## Detection Methods

The API uses three complementary detection methods:

1. **YOLO Detection**: Uses YOLOv8 model to detect "cell phone" objects
2. **Contour Detection**: Finds rectangular objects with mobile screen characteristics
3. **Color-based Detection**: Detects mobile screen colors while excluding dark bars

Results from all methods are merged and overlapping detections are removed using Non-Maximum Suppression.

## Detection Specifications

### What It Detects:
- ✅ Mobile phones and tablets
- ✅ All orientations (0°, 45°, 90°, 180°, 270°)
- ✅ Active displays with content
- ✅ Rotated/sideways mobile devices

### What It Excludes:
- ❌ TVs and monitors (too large)
- ❌ Dark bars and black screens
- ❌ Very bright lights/reflections
- ❌ Large displays (> 20% of image)
- ❌ Uniform surfaces (walls, tables)

## Error Responses

All endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (for processed images)
- `500`: Internal Server Error

Error response format:
```json
{
  "error": "Error message description",
  "traceback": "Detailed error traceback (in debug mode)"
}
```

## Configuration

The API can be configured using environment variables:

```bash
# Server Configuration
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# Base URL for image URLs (important for production)
BASE_URL=http://your-domain.com

# File Storage
UPLOAD_FOLDER=uploads
PROCESSED_FOLDER=processed
MAX_CONTENT_LENGTH=16777216  # 16MB

# Detection Parameters
YOLO_CONFIDENCE_THRESHOLD=0.35
CONTOUR_CONFIDENCE_THRESHOLD=0.45
```

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider implementing rate limiting based on your requirements.

## Security Considerations

- File uploads are limited to 16MB by default
- Only image files are processed
- Processed images are stored locally and served via HTTP
- CORS is enabled by default (configure as needed for production)
