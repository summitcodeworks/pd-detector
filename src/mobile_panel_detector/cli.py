"""
Command Line Interface for Mobile Panel Detection API
"""

import argparse
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from .app import create_app


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Mobile Panel Detection API - Mobile device panel detection with YOLO and OpenCV"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--config",
        choices=["development", "production", "testing"],
        default="development",
        help="Configuration to use (default: development)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['FLASK_CONFIG'] = args.config
    os.environ['FLASK_HOST'] = args.host
    os.environ['FLASK_PORT'] = str(args.port)
    os.environ['FLASK_DEBUG'] = str(args.debug).lower()
    
    # Create Flask application
    app = create_app(args.config)
    
    # Print startup information
    print("=" * 80)
    print("Mobile Panel Detection API v2.5 - ENHANCED DISPLAY ISSUE DETECTION")
    print("=" * 80)
    print(f"Configuration: {args.config}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print("Target: Mobile devices (phones/tablets) ONLY")
    print("Excludes: TVs, monitors, dark bars, large displays")
    print("Supports: ALL orientations (0°, 45°, 90°, 180°, 270°, etc.)")
    print("Size Limits: Max 20% of image dimensions")
    print("Brightness: Excludes dark bars (brightness > 30)")
    print("\nDisplay Issues Detected:")
    print("  ✓ Cables across screen")
    print("  ✓ Earbuds on display")
    print("  ✓ Protective case overhang")
    print("  ✓ Screen protector misalignment")
    print("  ✓ Stickers on panel")
    print("  ✓ Debris covering screen")
    print("  ✓ External mounts or holders")
    print("  ✓ Misplaced accessories blocking display")
    print("\nEndpoints:")
    print("  - POST /detect          (JSON with issue details)")
    print("  - POST /detect/image    (Image file)")
    print("  - POST /detect/batch    (Multiple)")
    print("  - GET  /health          (Status)")
    print(f"\nServer: http://{args.host}:{args.port}")
    print("=" * 80)
    
    # Run the application
    if args.workers > 1 and not args.debug:
        # Use Gunicorn for production
        try:
            import gunicorn.app.wsgiapp as wsgi
            sys.argv = [
                'gunicorn',
                '--bind', f"{args.host}:{args.port}",
                '--workers', str(args.workers),
                '--timeout', '120',
                '--keep-alive', '2',
                '--max-requests', '1000',
                '--max-requests-jitter', '100',
                'mobile_panel_detector.app:create_app()'
            ]
            wsgi.run()
        except ImportError:
            print("Warning: Gunicorn not available, running with Flask development server")
            app.run(debug=args.debug, host=args.host, port=args.port)
    else:
        # Use Flask development server
        app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
