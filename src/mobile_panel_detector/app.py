"""
Flask Application Factory

This module contains the Flask application factory and initialization logic.
"""

import os
from flask import Flask
from flask_cors import CORS

from .utils.config import config
from .api.routes import create_routes


def create_app(config_name=None):
    """
    Create and configure the Flask application.
    
    Args:
        config_name: Configuration name to use (development, production, testing)
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    if config_name is None:
        config_name = os.getenv('FLASK_CONFIG', 'default')
    
    app.config.from_object(config[config_name])
    
    # Configure upload folders
    app.config['UPLOAD_FOLDER'] = config[config_name].UPLOAD_FOLDER
    app.config['PROCESSED_FOLDER'] = config[config_name].PROCESSED_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = config[config_name].MAX_CONTENT_LENGTH
    
    # Create directories if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    
    # Initialize CORS
    CORS(app)
    
    # Register routes
    create_routes(app)
    
    return app
