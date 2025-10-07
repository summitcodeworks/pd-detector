#!/usr/bin/env python3
"""
Main Entry Point for Mobile Panel Detection API

This is the main entry point for the Mobile Panel Detection API.
It creates and runs the Flask application.
"""

import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mobile_panel_detector.app import create_app


def main():
    """Main entry point"""
    # Create Flask application
    app = create_app()
    
    # Print startup information
    print("=" * 70)
    print("Mobile Panel Detection API v2.4 - MOBILE-SPECIFIC DETECTION")
    print("=" * 70)
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
    
    # Run the application
    app.run(
        debug=app.config.get('FLASK_DEBUG', True),
        host=app.config.get('FLASK_HOST', '0.0.0.0'),
        port=app.config.get('FLASK_PORT', 5000)
    )


if __name__ == '__main__':
    main()
