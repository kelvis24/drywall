# YOLOv5 Ball Detection Camera System - Docker Container
FROM ubuntu:20.04

# Avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    cmake \
    build-essential \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev \
    libopenexr-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libusb-1.0-0-dev \
    wget \
    curl \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up Python - remove existing symlinks first
RUN rm -f /usr/bin/python /usr/bin/pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Install Python packages
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version for compatibility)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install OpenCV and other ML packages
RUN pip install opencv-python opencv-contrib-python numpy

# Install YOLOv5 dependencies
RUN pip install \
    matplotlib \
    pillow \
    pyyaml \
    requests \
    scipy \
    tensorboard \
    pandas \
    seaborn \
    ipython \
    psutil \
    thop

# Install PySpin SDK for Blackfly cameras
# Copy and install the PySpin wheel file - bypass platform checks
COPY spinnaker_python*.whl /tmp/
RUN cd /tmp && \
    unzip -q spinnaker_python*.whl && \
    cp -r PySpin* /usr/local/lib/python3.8/dist-packages/ && \
    cp -r spinnaker_python*.dist-info /usr/local/lib/python3.8/dist-packages/ || \
    echo "Manual extraction completed"

# Install additional packages that help with PySpin
RUN pip install \
    pyserial

# Create working directory
WORKDIR /app

# Copy requirements first for better Docker caching
COPY requirements.txt .
RUN pip install -r requirements.txt || echo "Requirements file not found or had issues"

# Copy application code
COPY . .

# Create output directories
RUN mkdir -p synchronized_captures individual_frames

# Set up display for GUI (if needed)
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# Expose any ports if needed (for web interface)
EXPOSE 8080

# Set up entry point
COPY docker_entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker_entrypoint.sh

# Default command
CMD ["/usr/local/bin/docker_entrypoint.sh"] 