# Mobile Panel Detection API

A Flask-based REST API for detecting mobile panels in images using advanced computer vision techniques including YOLO, OpenCV, and hybrid detection methods.

## Features

- **Hybrid Detection**: Combines YOLO, contour detection, and color-based detection
- **Rotation Support**: Detects panels in any orientation (0°, 45°, 90°, 180°, 270°, etc.)
- **Multiple Input Formats**: Supports file uploads and base64 encoded images
- **Batch Processing**: Process multiple images at once
- **Real-time Results**: Returns marked images with detection overlays
- **Confidence Scoring**: Provides confidence levels for each detection

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd pd-backend

# Run setup script (creates venv and installs dependencies)
./setup.sh
```

### 2. Start the Server

**Development Mode (recommended for testing):**
```bash
python main.py
```

**Production Mode (with Gunicorn):**
```bash
python main.py --prod
```

**Custom Port:**
```bash
python main.py --port 8080
```

**All Options:**
```bash
python main.py --help
```

The API will be available at `http://localhost:6000`

### 3. Using Makefile (Alternative)

```bash
make dev          # Development server
make prod         # Production server
make restart      # Restart server
make health       # Check if server is running
make help         # Show all commands
```

## API Endpoints

### Health Check
```bash
GET /health
```
Returns API status and configuration information.

### Single Image Detection
```bash
POST /detect
```
Detect panels in a single image. Accepts:
- File upload via `file` parameter
- Base64 encoded image via JSON `image` parameter

### Image File Response
```bash
POST /detect/image
```
Returns the marked image directly as a JPEG file.

### Batch Processing
```bash
POST /detect/batch
```
Process multiple images at once via `files` parameter.

## Usage Examples

### Health Check
```bash
curl http://localhost:6000/health
```

### Single Image Detection
```bash
curl -X POST -F "file=@image.jpg" http://localhost:6000/detect
```

### Base64 Image Detection
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"image":"data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."}' \
  http://localhost:6000/detect
```

### Using Python
```python
import requests

# File upload
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:6000/detect', files={'file': f})
    result = response.json()
    print(f"Detected: {result['detected']}")
    print(f"Panel Count: {result['panel_count']}")
    print(f"Image URL: {result['processed_image_url']}")
```

## Response Format

### Successful Detection
```json
{
  "detected": true,
  "result": "NG - Panel Detected",
  "panel_count": 1,
  "confidence": 0.756,
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
  "processed_image_url": "http://localhost:6000/processed/image_20241008_143022_a1b2c3d4.jpg",
  "saved_filename": "image_20241008_143022_a1b2c3d4.jpg",
  "message": "Detected 1 mobile panel(s)",
  "image_size": {"width": 1920, "height": 1080},
  "detection_mode": "mobile_panels_only"
}
```

### No Detection
```json
{
  "detected": false,
  "result": "OK - No Panel Detected",
  "panel_count": 0,
  "confidence": 0.0,
  "detections": [],
  "processed_image_url": "http://localhost:6000/processed/image_20241008_143022_a1b2c3d4.jpg",
  "message": "No mobile panels detected"
}
```

## Configuration

You can configure the API using environment variables or by modifying `config.py`:

```bash
# Server Configuration
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_HOST=0.0.0.0
FLASK_PORT=6000

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

## Project Structure

```
pd-backend/
├── main.py                           # Single entry point (dev + prod)
├── config.py                         # Configuration settings
├── requirements.txt                  # Python dependencies
├── setup.sh                          # Setup script
├── restart_server.sh                 # Restart server script
├── Makefile                          # Common commands
├── src/                              # Source code
│   └── mobile_panel_detector/        # Main package
│       ├── app.py                    # Flask application factory
│       ├── api/                      # API routes
│       │   └── routes.py             # Flask routes
│       ├── detector/                 # Detection logic
│       │   └── panel_detector.py     # Core detection algorithms
│       └── utils/                    # Utilities
│           ├── config.py             # Configuration management
│           └── image_utils.py        # Image processing utilities
├── tests/                            # Test suite
│   ├── test_api.py                   # API endpoint tests
│   └── test_detector.py              # Detection algorithm tests
└── docs/                             # Documentation
    └── API.md                        # API documentation
```

## Docker Deployment

```bash
# Build the image
docker build -t panel-detection-api .

# Run the container
docker run -p 6000:6000 panel-detection-api

# Or use docker-compose
docker-compose up --build
```

## Development

### Running Tests
```bash
make test
# or
python -m pytest tests/ -v
```

### Code Quality
```bash
make lint          # Run linting
make format        # Format code
make clean         # Clean up temporary files
```

## Detection Methods

The API uses three complementary detection methods:

1. **YOLO Detection**: Uses YOLOv8 model to detect "cell phone" objects
2. **Contour Detection**: Finds rectangular objects with screen characteristics
3. **Color-based Detection**: Detects screen-like colors and patterns

Results from all methods are merged and overlapping detections are removed using Non-Maximum Suppression.

## Command Reference

### main.py Options
```bash
python main.py                        # Development mode (default)
python main.py --prod                 # Production mode with Gunicorn
python main.py --port 8080            # Custom port
python main.py --host 0.0.0.0         # Custom host
python main.py --workers 8            # Custom worker count (prod only)
python main.py --config production    # Use specific config
python main.py --help                 # Show all options
```

### Makefile Commands
```bash
make help          # Show all available commands
make install       # Install dependencies
make dev           # Start development server
make prod          # Start production server
make restart       # Restart the server
make health        # Check API health
make test          # Run tests
make lint          # Run linting
make format        # Format code
make clean         # Clean up temporary files
make docker-build  # Build Docker image
make docker-run    # Run Docker container
```

## Troubleshooting

### Common Issues

1. **YOLO Model Download**: The YOLO model will be downloaded automatically on first run. Ensure internet connectivity.

2. **Port Already in Use**: If port 6000 is already in use, specify a different port:
   ```bash
   python main.py --port 8080
   ```

3. **Memory Issues**: Large images may cause memory issues. Consider resizing images before processing.

4. **OpenCV Installation**: If OpenCV installation fails, try:
   ```bash
   pip install opencv-python-headless
   ```

5. **Restart Server**: If the server is stuck:
   ```bash
   ./restart_server.sh
   # or
   make restart
   ```

## License

[Add your license information here]

## Version

Current version: 2.4 - Mobile-Specific Detection with Simplified Entry Point
