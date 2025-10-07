#!/usr/bin/env python3
"""
Production runner for Mobile Panel Detection API
"""
import os
import sys
from main import app
from config import config

def main():
    """Main entry point for production server"""
    # Get configuration
    config_name = os.getenv('FLASK_CONFIG', 'default')
    app.config.from_object(config[config_name])
    
    # Run the application
    if len(sys.argv) > 1 and sys.argv[1] == 'dev':
        # Development mode
        app.run(
            debug=True,
            host=app.config['FLASK_HOST'],
            port=app.config['FLASK_PORT']
        )
    else:
        # Production mode with Gunicorn
        import gunicorn.app.wsgiapp as wsgi
        sys.argv = [
            'gunicorn',
            '--bind', f"{app.config['FLASK_HOST']}:{app.config['FLASK_PORT']}",
            '--workers', '4',
            '--timeout', '120',
            '--keep-alive', '2',
            '--max-requests', '1000',
            '--max-requests-jitter', '100',
            'main:app'
        ]
        wsgi.run()

if __name__ == '__main__':
    main()
