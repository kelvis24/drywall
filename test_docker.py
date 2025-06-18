#!/usr/bin/env python3
"""
Test script to verify Docker container is working without PySpin
"""

import sys
import os
import time
import threading
from queue import Queue
from datetime import datetime
from pathlib import Path

def test_basic_imports():
    """Test basic Python imports"""
    print("🔍 Testing basic imports:")
    
    try:
        import cv2
        print("  ✅ OpenCV")
    except ImportError as e:
        print(f"  ❌ OpenCV: {e}")
    
    try:
        import numpy as np
        print("  ✅ NumPy")
    except ImportError as e:
        print(f"  ❌ NumPy: {e}")
    
    try:
        import torch
        print(f"  ✅ PyTorch (version: {torch.__version__})")
    except ImportError as e:
        print(f"  ❌ PyTorch: {e}")
    
    try:
        import matplotlib
        print("  ✅ Matplotlib")
    except ImportError as e:
        print(f"  ❌ Matplotlib: {e}")

def test_yolo_model():
    """Test YOLO model loading"""
    print("\n🎯 Testing YOLO model:")
    
    try:
        import torch
        
        # Check if model file exists
        model_paths = ['best.pt', 'best_fixed.pt', 'yolov5n.pt']
        model_found = None
        
        for path in model_paths:
            if os.path.exists(path):
                model_found = path
                print(f"  📁 Found model: {path}")
                break
        
        if model_found:
            try:
                # Try to load the model
                model = torch.load(model_found, map_location='cpu')
                print("  ✅ Model loaded successfully")
                
                # Check if it's a YOLO model
                if 'model' in model:
                    print("  ✅ Valid YOLO model structure")
                else:
                    print("  ⚠️ Model structure may not be YOLO format")
                    
            except Exception as e:
                print(f"  ❌ Failed to load model: {e}")
        else:
            print("  ⚠️ No model files found")
            
    except Exception as e:
        print(f"  ❌ YOLO test failed: {e}")

def test_file_system():
    """Test file system access"""
    print("\n💾 Testing file system:")
    
    try:
        # Create test directories
        test_dirs = ['synchronized_captures', 'individual_frames', 'test_output']
        
        for dir_name in test_dirs:
            os.makedirs(dir_name, exist_ok=True)
            print(f"  ✅ Created directory: {dir_name}")
        
        # Test file writing
        test_file = 'test_output/test.txt'
        with open(test_file, 'w') as f:
            f.write(f"Test file created at {datetime.now()}")
        print(f"  ✅ Created test file: {test_file}")
        
        # Test file reading
        with open(test_file, 'r') as f:
            content = f.read()
        print(f"  ✅ Read test file: {len(content)} characters")
        
    except Exception as e:
        print(f"  ❌ File system test failed: {e}")

def test_threading():
    """Test threading capabilities"""
    print("\n🧵 Testing threading:")
    
    try:
        def worker_function():
            time.sleep(0.1)
            return "Thread completed"
        
        # Test thread creation
        thread = threading.Thread(target=worker_function)
        thread.start()
        thread.join()
        print("  ✅ Thread creation and execution")
        
        # Test queue
        q = Queue()
        q.put("test")
        result = q.get()
        print(f"  ✅ Queue operations: {result}")
        
    except Exception as e:
        print(f"  ❌ Threading test failed: {e}")

def test_system_info():
    """Test system information"""
    print("\n💻 System information:")
    
    try:
        import platform
        print(f"  🖥️ OS: {platform.system()} {platform.release()}")
        print(f"  🐍 Python: {sys.version}")
        
        import torch
        print(f"  🔥 PyTorch: {torch.__version__}")
        print(f"  🖥️ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  🎮 CUDA version: {torch.version.cuda}")
            print(f"  🎮 GPU count: {torch.cuda.device_count()}")
        
    except Exception as e:
        print(f"  ❌ System info failed: {e}")

def main():
    print("🐳 Docker Container Test (No PySpin)")
    print("=" * 50)
    
    test_basic_imports()
    test_yolo_model()
    test_file_system()
    test_threading()
    test_system_info()
    
    print("\n" + "=" * 50)
    print("🎉 Docker container test completed!")
    print("📝 Note: PySpin is not available - camera functions will not work")
    print("🔧 To fix PySpin, you need a valid Linux PySpin wheel file")
    print("💡 The container is ready for development and testing!")

if __name__ == "__main__":
    main() 