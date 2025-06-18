# 🐳 Docker Setup for YOLOv5 Ball Detection System

## Why Docker?

✅ **Solves PosixPath Issues**: Your `best.pt` model will work perfectly in Linux container  
✅ **Clean Environment**: Isolated environment with all dependencies  
✅ **No Path Conversion**: Native Unix paths work seamlessly  
✅ **Consistent Results**: Same environment every time  

## Prerequisites

1. **Install Docker Desktop** on Windows
   - Download from: https://www.docker.com/products/docker-desktop/
   - Make sure it's running

2. **Install Docker Compose** (usually included with Docker Desktop)

3. **Get PySpin for Linux** (Important!)
   - Download the Linux version of PySpin SDK from FLIR website
   - Look for `spinnaker_python-*-linux_x86_64.whl`
   - Place it in your project directory

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# 1. Build and run the container
docker-compose up --build

# 2. To run in background
docker-compose up -d --build

# 3. To see logs
docker-compose logs -f

# 4. To stop
docker-compose down
```

### Option 2: Manual Docker Commands

```bash
# 1. Build the image
docker build -t yolo-ball-detection .

# 2. Run with GUI support (Windows with WSL2)
docker run -it --rm \
  --privileged \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ${PWD}:/app \
  --device /dev/bus/usb \
  -e DISPLAY=$DISPLAY \
  yolo-ball-detection

# 3. Run headless (no GUI)
docker run -it --rm \
  --privileged \
  -v ${PWD}:/app \
  --device /dev/bus/usb \
  yolo-ball-detection
```

## What Gets Fixed

🎯 **Your Custom Model**: `best.pt` will load perfectly in Linux environment  
🔧 **Path Issues**: All PosixPath objects work natively  
📦 **Dependencies**: Clean Python environment with all packages  
🎥 **Camera Access**: USB cameras accessible through device mounting  

## File Structure

```
Your Directory/
├── best.pt                    # Your custom model (works in Linux!)
├── help.py                    # Your main script
├── yolov5/                    # YOLOv5 directory
├── Dockerfile                 # Container definition
├── docker-compose.yml        # Easy container management
├── docker_entrypoint.sh      # Startup script
├── spinnaker_python*.whl     # PySpin for Linux (download separately)
└── synchronized_captures/    # Output directory
```

## Camera Setup in Docker

### USB Camera Access
```bash
# Check if cameras are visible
docker exec -it ball-detection-cameras ls -la /dev/bus/usb/
```

### Troubleshooting Camera Access
```bash
# Enter container for debugging
docker exec -it ball-detection-cameras bash

# Check USB devices
lsusb

# Check camera permissions
ls -la /dev/video*
```

## Display Setup (GUI Windows)

### For Windows with WSL2:
```bash
# Install X11 server (like VcXsrv)
# Configure to allow connections from Docker

# Set display
export DISPLAY=:0
```

### For Headless Mode:
```bash
# Run without display
docker run ... -e HEADLESS=true yolo-ball-detection
```

## Benefits of Docker Approach

| Issue | Windows Native | Docker Linux |
|-------|----------------|--------------|
| PosixPath errors | ❌ Requires conversion | ✅ Works natively |
| Model loading | ❌ Complex fixes needed | ✅ Loads directly |
| Dependencies | ❌ Environment conflicts | ✅ Clean isolation |
| Reproducibility | ❌ System dependent | ✅ Consistent |

## Performance Considerations

- **Slightly slower** than native due to containerization
- **USB bandwidth** may be limited through Docker
- **GPU access** requires additional setup (not covered here)
- **Memory usage** increased due to container overhead

## Commands You'll Use

```bash
# Start system
docker-compose up -d

# View live logs
docker-compose logs -f

# Stop system
docker-compose down

# Rebuild after changes
docker-compose up --build

# Debug inside container
docker exec -it ball-detection-cameras bash

# Check custom model
docker exec -it ball-detection-cameras python -c "import torch; print(torch.load('best.pt', map_location='cpu').keys())"
```

## Next Steps

1. **Download PySpin Linux wheel** and place in your directory
2. **Build the container**: `docker-compose up --build`
3. **Test your custom model**: It should load without any PosixPath errors!
4. **Enjoy seamless ball detection** with your trained model! 🎾

## Alternative: WSL2 (Simpler Option)

If Docker seems complex, you could also:
1. Install WSL2 (Windows Subsystem for Linux)
2. Install Python and dependencies in WSL2
3. Run your scripts directly in WSL2 environment

This would also solve the PosixPath issues with less complexity! 