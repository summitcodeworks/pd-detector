# Mobile Panel Detection API

A Flask-based REST API for detecting mobile panels in images using advanced computer vision techniques including YOLO, OpenCV, and hybrid detection methods.

## Project Structure

```
pd-backend/
├── src/                                    # Source code
│   └── mobile_panel_detector/             # Main package
│       ├── __init__.py                    # Package initialization
│       ├── app.py                         # Flask application factory
│       ├── cli.py                         # Command line interface
│       ├── api/                           # API routes
│       │   ├── __init__.py
│       │   └── routes.py                  # Flask routes
│       ├── detector/                      # Detection logic
│       │   ├── __init__.py
│       │   └── panel_detector.py          # Core detection algorithms
│       ├── utils/                         # Utilities
│       │   ├── __init__.py
│       │   ├── config.py                  # Configuration management
│       │   └── image_utils.py             # Image processing utilities
│       └── models/                        # Data models (future use)
│           └── __init__.py
├── tests/                                 # Test suite
│   ├── __init__.py
│   ├── test_api.py                        # API endpoint tests
│   └── test_detector.py                   # Detection algorithm tests
├── docs/                                  # Documentation
│   └── API.md                            # API documentation
├── main_new.py                           # New main entry point
├── setup.py                              # Package setup (legacy)
├── pyproject.toml                        # Modern Python packaging
├── requirements.txt                      # Dependencies
├── Makefile                              # Development commands
├── Dockerfile                            # Container configuration
├── docker-compose.yml                    # Container orchestration
└── README.md                             # This file
```

## Features

- **Hybrid Detection**: Combines YOLO, contour detection, and color-based detection
- **Rotation Support**: Detects panels in any orientation (0°, 45°, 90°, 180°, 270°, etc.)
- **Multiple Input Formats**: Supports file uploads and base64 encoded images
- **Batch Processing**: Process multiple images at once
- **Real-time Results**: Returns marked images with detection overlays
- **Confidence Scoring**: Provides confidence levels for each detection

## API Endpoints

### Health Check
```
GET /health
```
Returns API status and configuration information.

### Single Image Detection
```
POST /detect
```
Detect panels in a single image. Accepts:
- File upload via `file` parameter
- Base64 encoded image via JSON `image` parameter

### Image File Response
```
POST /detect/image
```
Returns the marked image directly as a JPEG file.

### Batch Processing
```
POST /detect/batch
```
Process multiple images at once via `files` parameter.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pd-backend
   ```

2. **Quick Setup (Recommended)**
   ```bash
   ./setup.sh
   ```

3. **Manual Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

4. **Run the application**

   **Option 1: New structured entry point**
   ```bash
   python main_new.py
   ```

   **Option 2: Using Makefile**
   ```bash
   make dev          # Development server
   make prod         # Production server
   make cli ARGS="--help"  # CLI with arguments
   ```

   **Option 3: Using CLI directly**
   ```bash
   python -m mobile_panel_detector.cli --help
   python -m mobile_panel_detector.cli --config development --debug
   ```

The API will be available at `http://localhost:5000`

## Usage Examples

### Using curl

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Single Image Detection:**
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/detect
```

**Base64 Image Detection:**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"image":"data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."}' \
  http://localhost:5000/detect
```

### Using Python requests

```python
import requests

# File upload
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/detect', files={'file': f})
    result = response.json()

# Base64 image
import base64
with open('image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
    
response = requests.post('http://localhost:5000/detect', 
                        json={'image': f'data:image/jpeg;base64,{image_data}'})
result = response.json()
```

## Response Format

### Successful Detection Response
```json
{
  "detected": true,
  "result": "NG - Panel Detected",
  "panel_count": 2,
  "confidence": 0.756,
  "detections": [
    {
      "id": 1,
      "bbox": [100, 150, 200, 300],
      "confidence": 0.823,
      "method": "YOLO",
      "angle": 0.0,
      "area": 60000
    }
  ],
  "processed_image_url": "http://localhost:5000/processed/image_20241201_143022_a1b2c3d4.jpg",
  "saved_filename": "image_20241201_143022_a1b2c3d4.jpg",
  "message": "Detected 2 mobile panel(s)",
  "image_size": {"width": 1920, "height": 1080},
  "detection_mode": "hybrid_rotation_support"
}
```

### No Detection Response
```json
{
  "detected": false,
  "result": "OK - No Panel Detected",
  "panel_count": 0,
  "confidence": 0.0,
  "detections": [],
  "processed_image_url": "http://localhost:5000/processed/image_20241201_143022_a1b2c3d4.jpg",
  "saved_filename": "image_20241201_143022_a1b2c3d4.jpg",
  "message": "No mobile panels detected",
  "image_size": {"width": 1920, "height": 1080},
  "detection_mode": "hybrid_rotation_support"
}
```

## Image Storage and URLs

The API now automatically saves all processed images and returns full URLs instead of base64 encoded data:

- **Processed images** are saved in the `processed/` directory
- **Unique filenames** are generated using timestamp and UUID
- **Full URLs** are returned in the response for easy access
- **Direct access** to processed images via `/processed/<filename>` endpoint

### Image URL Format
```
http://localhost:5000/processed/originalname_YYYYMMDD_HHMMSS_uuid.jpg
```

### Accessing Processed Images
You can directly access processed images using the returned URL:
```bash
curl http://localhost:5000/processed/image_20241201_143022_a1b2c3d4.jpg
```

## Detection Methods

The API uses three complementary detection methods:

1. **YOLO Detection**: Uses YOLOv8 model to detect "cell phone" objects
2. **Contour Detection**: Finds rectangular objects with screen characteristics
3. **Color-based Detection**: Detects screen-like colors (blue, white, cyan)

Results from all methods are merged and overlapping detections are removed using Non-Maximum Suppression.

## Configuration

The API can be configured using environment variables or by modifying `config.py`:

### Environment Variables
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

### Detection Parameters
- `min_yolo_confidence`: Minimum confidence for YOLO detections (default: 0.35)
- `min_contour_confidence`: Minimum confidence for contour detections (default: 0.45)
- Detection thresholds for screen characteristics
- Image processing parameters

## Docker Deployment

A Dockerfile is included for containerized deployment:

```bash
# Build the image
docker build -t panel-detection-api .

# Run the container
docker run -p 5000:5000 panel-detection-api
```

## Production Deployment

For production deployment, use Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 main:app
```

## Troubleshooting

### Common Issues

1. **YOLO Model Download**: The YOLO model will be downloaded automatically on first run. Ensure internet connectivity.

2. **Memory Issues**: Large images may cause memory issues. Consider resizing images before processing.

3. **OpenCV Installation**: If OpenCV installation fails, try:
   ```bash
   pip install opencv-python-headless
   ```

4. **CUDA Support**: For GPU acceleration, install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## API Version

Current version: 2.3 - Rotation Support

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/mobile_panel_detector

# Run specific test categories
pytest -m "not slow"          # Skip slow tests
pytest -m integration         # Run only integration tests
pytest -m unit               # Run only unit tests
```

### Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Development Commands
```bash
make help          # Show all available commands
make dev           # Start development server
make test          # Run tests
make lint          # Run linting
make format        # Format code
make clean         # Clean up temporary files
```

### Package Installation
```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Architecture

The project follows a clean architecture pattern:

- **API Layer** (`api/`): Flask routes and request handling
- **Business Logic** (`detector/`): Core detection algorithms
- **Utilities** (`utils/`): Configuration, image processing, and helper functions
- **Models** (`models/`): Data models (for future expansion)
- **Tests** (`tests/`): Comprehensive test suite
- **Documentation** (`docs/`): API and project documentation

### Key Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Dependency Injection**: Configuration and dependencies are injected
3. **Testability**: All components are designed to be easily testable
4. **Extensibility**: New detection methods can be easily added
5. **Configuration Management**: Environment-based configuration
6. **Error Handling**: Comprehensive error handling and logging

## License

[Add your license information here]
