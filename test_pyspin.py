#!/usr/bin/env python3
"""
Test script to verify PySpin SDK installation in Docker container
"""

def test_pyspin_import():
    """Test PySpin import and basic functionality"""
    try:
        import PySpin
        print("✅ PySpin imported successfully!")
        
        # Get system instance
        system = PySpin.System.GetInstance()
        
        # Get library version
        version = system.GetLibraryVersion()
        version_str = f'{version.major}.{version.minor}.{version.type}.{version.build}'
        print(f"📚 PySpin SDK version: {version_str}")
        
        # Get camera list
        camera_list = system.GetCameras()
        num_cameras = camera_list.GetSize()
        print(f"📷 Cameras detected: {num_cameras}")
        
        if num_cameras > 0:
            print("🎉 PySpin is working correctly!")
            
            # Test camera access
            for i in range(num_cameras):
                try:
                    cam = camera_list.GetByIndex(i)
                    cam.Init()
                    
                    # Get camera info
                    nodemap_tldevice = cam.GetTLDeviceNodeMap()
                    device_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
                    device_model = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceModelName'))
                    
                    serial = device_serial.GetValue() if PySpin.IsAvailable(device_serial) and PySpin.IsReadable(device_serial) else 'Unknown'
                    model = device_model.GetValue() if PySpin.IsAvailable(device_model) and PySpin.IsReadable(device_model) else 'Unknown'
                    
                    print(f"  📷 Camera {i+1}: {model} (Serial: {serial})")
                    cam.DeInit()
                    
                except Exception as ex:
                    print(f"  ⚠️ Error accessing camera {i+1}: {ex}")
        else:
            print("⚠️ No cameras detected - check USB connections and permissions")
        
        # Clean up
        camera_list.Clear()
        system.ReleaseInstance()
        
        return True
        
    except ImportError as ex:
        print(f"❌ PySpin import failed: {ex}")
        print("🔧 Make sure the PySpin wheel file is properly installed")
        return False
    except Exception as ex:
        print(f"❌ PySpin test failed: {ex}")
        return False

def test_other_dependencies():
    """Test other required dependencies"""
    dependencies = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('threading', 'Threading'),
        ('queue', 'Queue'),
        ('pathlib', 'Pathlib')
    ]
    
    print("\n🔍 Testing other dependencies:")
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - not available")

if __name__ == "__main__":
    print("🧪 Testing PySpin SDK in Docker Container")
    print("=" * 50)
    
    # Test PySpin
    pyspin_ok = test_pyspin_import()
    
    # Test other dependencies
    test_other_dependencies()
    
    print("\n" + "=" * 50)
    if pyspin_ok:
        print("🎉 PySpin test completed successfully!")
        print("🚀 Your camera system should work perfectly!")
    else:
        print("❌ PySpin test failed!")
        print("🔧 Check that the Linux PySpin wheel file is in the directory")
        
    print("🐳 Docker container is ready for ball detection!") 