#!/usr/bin/env python3
"""
Mobile Panel Detection API - Main Entry Point

This is the single entry point for the Mobile Panel Detection API.
Supports both development and production modes.

Usage:
    python main.py                    # Development mode (default)
    python main.py --prod             # Production mode with Gunicorn
    python main.py --port 5001        # Custom port
    python main.py --help             # Show help
"""

import os
import sys
import argparse

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mobile_panel_detector.app import create_app


def run_development(app, host='0.0.0.0', port=6000):
    """Run in development mode with Flask's built-in server"""
    print("=" * 70)
    print("Mobile Panel Detection API - DEVELOPMENT MODE")
    print("=" * 70)
    print("Target: Mobile devices (phones/tablets) ONLY")
    print("Excludes: TVs, monitors, dark bars, large displays")
    print("Supports: ALL orientations (0°, 45°, 90°, 180°, 270°, etc.)")
    print("\nEndpoints:")
    print("  - POST /detect          (JSON)")
    print("  - POST /detect/image    (Image file)")
    print("  - POST /detect/batch    (Multiple)")
    print("  - GET  /health          (Status)")
    print(f"\nServer: http://{host}:{port}")
    print("=" * 70)
    
    app.run(debug=True, host=host, port=port)


def run_production(app, host='0.0.0.0', port=6000, workers=4):
    """Run in production mode with Gunicorn"""
    print("=" * 70)
    print("Mobile Panel Detection API - PRODUCTION MODE")
    print("=" * 70)
    print(f"Starting with Gunicorn ({workers} workers)...")
    print(f"Server: http://{host}:{port}")
    print("=" * 70)
    
    # Import and run Gunicorn
    from gunicorn.app.base import BaseApplication
    
    class GunicornApp(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()
        
        def load_config(self):
            for key, value in self.options.items():
                if key in self.cfg.settings and value is not None:
                    self.cfg.set(key.lower(), value)
        
        def load(self):
            return self.application
    
    options = {
        'bind': f'{host}:{port}',
        'workers': workers,
        'timeout': 120,
        'keepalive': 2,
        'max_requests': 1000,
        'max_requests_jitter': 100,
    }
    
    GunicornApp(app, options).run()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Mobile Panel Detection API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Development mode (default)
  python main.py --prod             # Production mode
  python main.py --port 5001        # Custom port
  python main.py --prod --workers 8 # Production with 8 workers
        """
    )
    
    parser.add_argument(
        '--prod', '--production',
        action='store_true',
        help='Run in production mode with Gunicorn'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=6000,
        help='Port to run the server on (default: 6000)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to run the server on (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of Gunicorn workers (production only, default: 4)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='development',
        choices=['development', 'production', 'testing'],
        help='Configuration to use (default: development)'
    )
    
    args = parser.parse_args()
    
    # Override config if production mode is requested
    if args.prod:
        args.config = 'production'
    
    # Set environment variable for config
    os.environ['FLASK_CONFIG'] = args.config
    
    # Create Flask application
    app = create_app(args.config)
    
    # Run in appropriate mode
    if args.prod:
        run_production(app, host=args.host, port=args.port, workers=args.workers)
    else:
        run_development(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

