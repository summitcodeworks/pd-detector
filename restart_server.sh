#!/bin/bash

# Auto-restart script for Mobile Panel Detection API
echo "🔄 Restarting Mobile Panel Detection API..."

# Kill any existing server processes
echo "🛑 Stopping existing server processes..."
pkill -f "python -m src.mobile_panel_detector.cli" 2>/dev/null || true
pkill -f "mobile_panel_detector" 2>/dev/null || true

# Wait a moment for processes to terminate
sleep 2

# Check if port is still in use and kill any remaining processes
if lsof -ti:5001 > /dev/null 2>&1; then
    echo "🔧 Force killing processes on port 5001..."
    lsof -ti:5001 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Start the server
echo "🚀 Starting Mobile Panel Detection API on port 5001..."
python -m src.mobile_panel_detector.cli --port 5001

echo "✅ Server restart complete!"
