# Mobile Panel Detection API Makefile

.PHONY: help install dev test clean docker-build docker-run docker-stop lint format

# Default target
help:
	@echo "Mobile Panel Detection API - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install     Install dependencies in virtual environment"
	@echo "  dev         Run development server"
	@echo "  prod        Run production server"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    Build Docker image"
	@echo "  docker-run      Run Docker container"
	@echo "  docker-stop     Stop Docker container"
	@echo "  docker-compose  Run with docker-compose"
	@echo ""
	@echo "Development:"
	@echo "  test        Run tests"
	@echo "  lint        Run linting"
	@echo "  format      Format code"
	@echo "  clean       Clean up temporary files"
	@echo ""
	@echo "Utilities:"
	@echo "  health      Check API health"
	@echo "  logs        Show application logs"

# Setup commands
install:
	@echo "Setting up Mobile Panel Detection API..."
	@./setup.sh

dev:
	@echo "Starting development server..."
	@python main_new.py

prod:
	@echo "Starting production server..."
	@python -m mobile_panel_detector.cli --config production --workers 4

cli:
	@echo "Starting with CLI..."
	@python -m mobile_panel_detector.cli $(ARGS)

# Docker commands
docker-build:
	@echo "Building Docker image..."
	@docker build -t panel-detection-api .

docker-run:
	@echo "Running Docker container..."
	@docker run -p 5000:5000 --name panel-detection-api panel-detection-api

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
	@flake8 main.py config.py run.py
	@mypy main.py config.py run.py

format:
	@echo "Formatting code..."
	@black main.py config.py run.py
	@isort main.py config.py run.py

clean:
	@echo "Cleaning up..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/

# Utility commands
health:
	@echo "Checking API health..."
	@curl -s http://localhost:5000/health | python -m json.tool

logs:
	@echo "Showing application logs..."
	@tail -f app.log
