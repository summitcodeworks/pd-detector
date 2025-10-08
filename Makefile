# Mobile Panel Detection API Makefile

.PHONY: help install dev prod test clean docker-build docker-run docker-stop lint format restart

# Default target
help:
	@echo "Mobile Panel Detection API - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install         Install dependencies in virtual environment"
	@echo "  dev             Run development server (default port 6000)"
	@echo "  prod            Run production server with Gunicorn"
	@echo ""
	@echo "Server Management:"
	@echo "  restart         Restart the server"
	@echo "  health          Check API health"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    Build Docker image"
	@echo "  docker-run      Run Docker container"
	@echo "  docker-stop     Stop Docker container"
	@echo "  docker-compose  Run with docker-compose"
	@echo ""
	@echo "Development:"
	@echo "  test            Run tests"
	@echo "  lint            Run linting"
	@echo "  format          Format code"
	@echo "  clean           Clean up temporary files"

# Setup commands
install:
	@echo "Setting up Mobile Panel Detection API..."
	@./setup.sh

dev:
	@echo "Starting development server..."
	@python main.py

prod:
	@echo "Starting production server..."
	@python main.py --prod

# Server management
restart:
	@echo "Restarting server..."
	@./restart_server.sh

health:
	@echo "Checking API health..."
	@curl -s http://localhost:6000/health | python -m json.tool || echo "Server is not running"

# Docker commands
docker-build:
	@echo "Building Docker image..."
	@docker build -t panel-detection-api .

docker-run:
	@echo "Running Docker container..."
	@docker run -p 6000:6000 --name panel-detection-api panel-detection-api

docker-stop:
	@echo "Stopping Docker container..."
	@docker stop panel-detection-api || true
	@docker rm panel-detection-api || true

docker-compose:
	@echo "Running with docker-compose..."
	@docker-compose up --build

# Development commands
test:
	@echo "Running tests..."
	@python -m pytest tests/ -v

lint:
	@echo "Running linting..."
	@flake8 main.py config.py src/ || true
	@mypy main.py config.py || true

format:
	@echo "Formatting code..."
	@black main.py config.py src/ || true
	@isort main.py config.py src/ || true

clean:
	@echo "Cleaning up..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
