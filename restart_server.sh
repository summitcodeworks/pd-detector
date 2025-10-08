#!/bin/bash

# Auto-restart script for Mobile Panel Detection API
echo "🔄 Restarting Mobile Panel Detection API..."

# Kill any existing server processes
echo "🛑 Stopping existing server processes..."
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "gunicorn" 2>/dev/null || true

# Wait a moment for processes to terminate
sleep 2

# Check if port is still in use and kill any remaining processes
PORT=${1:-6000}
if lsof -ti:$PORT > /dev/null 2>&1; then
    echo "🔧 Force killing processes on port $PORT..."
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Start the server
echo "🚀 Starting Mobile Panel Detection API on port $PORT..."
python main.py --port $PORT

echo "✅ Server restart complete!"
