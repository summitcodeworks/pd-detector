#!/bin/bash

# Mobile Panel Detection API Setup Script
echo "Setting up Mobile Panel Detection API..."
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Setup completed successfully!"
    echo ""
    echo "To start the API server:"
    echo "1. Activate the virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Run the application:"
    echo "   python main.py"
    echo ""
    echo "3. The API will be available at: http://localhost:5000"
    echo ""
    echo "For more information, see README.md"
else
    echo "ERROR: Setup failed. Please check the error messages above."
    exit 1
fi
