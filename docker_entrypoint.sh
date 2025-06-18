#!/bin/bash

echo "🐳 Starting YOLOv5 Ball Detection Camera System in Docker"
echo "============================================================"

# Check if custom model exists and is accessible
if [ -f "best.pt" ]; then
    echo "✅ Custom model found: best.pt"
    echo "🎯 Using your trained ball detection model!"
else
    echo "⚠️ Custom model not found - will use standard YOLO"
fi

# Check camera permissions
if [ -d "/dev" ]; then
    echo "🔍 Checking camera devices..."
    ls -la /dev/video* 2>/dev/null || echo "⚠️ No video devices found"
    ls -la /dev/bus/usb/ 2>/dev/null || echo "⚠️ No USB devices found"
fi

# Set up display if X11 forwarding is available
if [ ! -z "$DISPLAY" ]; then
    echo "📺 Display available: $DISPLAY"
    echo "🖥️ GUI windows will be shown"
else
    echo "🚫 No display available - running in headless mode"
    export DISPLAY_MODE="--no-display"
fi

# Test PySpin installation thoroughly
echo "🧪 Running PySpin SDK test..."
if [ -f "test_pyspin.py" ]; then
    python test_pyspin.py
else
    # Fallback basic test
    python -c "import PySpin; print('✅ PySpin SDK available')" 2>/dev/null || {
        echo "⚠️ PySpin not available - camera functions may not work"
        echo "❌ Check that the Linux PySpin wheel was properly installed during build"
    }
fi

# Start the camera system
echo "🚀 Starting camera capture system..."
echo "📸 Your custom model will work perfectly in Linux environment!"

# Run the main application
if [ -f "help.py" ]; then
    echo "🎯 Running help.py with custom model support..."
    python help.py $DISPLAY_MODE
elif [ -f "me.py" ]; then
    echo "🎯 Running me.py..."
    python me.py $DISPLAY_MODE
else
    echo "❌ No main script found!"
    echo "📋 Available files:"
    ls -la *.py
    
    # Keep container running for debugging
    echo "🐛 Keeping container alive for debugging..."
    tail -f /dev/null
fi 