"""
Configuration settings for Mobile Panel Detection API
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Server Configuration
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 6000))
    
    # YOLO Model Configuration
    YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'yolov8n.pt')
    YOLO_CONFIDENCE_THRESHOLD = float(os.getenv('YOLO_CONFIDENCE_THRESHOLD', 0.35))
    
    # Detection Configuration
    CONTOUR_CONFIDENCE_THRESHOLD = float(os.getenv('CONTOUR_CONFIDENCE_THRESHOLD', 0.45))
    MIN_PANEL_SIZE = int(os.getenv('MIN_PANEL_SIZE', 60))
    MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', 2048))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'app.log')
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', 'processed')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    
    # Base URL Configuration
    BASE_URL = os.getenv('BASE_URL', 'http://localhost:6000')

class DevelopmentConfig(Config):
    """Development configuration"""
    FLASK_DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    FLASK_DEBUG = False
    LOG_LEVEL = 'WARNING'

class TestingConfig(Config):
    """Testing configuration"""
    FLASK_DEBUG = True
    LOG_LEVEL = 'DEBUG'
    YOLO_CONFIDENCE_THRESHOLD = 0.3  # Lower threshold for testing

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
