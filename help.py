"""
Simplified Dual High-Speed Camera Capture System
Focus: Image capture, synchronization, and FPS reporting
WITH GPIO HARDWARE SYNCHRONIZATION FOR BLACKFLY S CAMERAS
"""

# Fix for OpenMP runtime conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
import PySpin
import cv2
import numpy as np
import os
import time
import threading
from queue import Queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import torch
import logging
from typing import List
from pathlib import Path, PosixPath, WindowsPath
import re
import traceback
import subprocess
import pickle
import platform

# Fix for PosixPath on Windows
class PathUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pathlib' and name == 'PosixPath':
            return WindowsPath if platform.system() == 'Windows' else PosixPath
        return super().find_class(module, name)

def fix_model_paths(model_path):
    """Permanently fix PosixPath issues in a model file by creating a Windows-compatible version"""
    if not os.path.exists(model_path):
        return None
        
    try:
        print(f"🔧 Attempting to fix PosixPath issues in {model_path}...")
        
        # Create backup
        backup_path = model_path + '.backup'
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(model_path, backup_path)
            print(f"📋 Created backup: {backup_path}")
        
        # Load with path fix
        with open(model_path, 'rb') as f:
            unpickler = PathUnpickler(f)
            checkpoint = unpickler.load()
        
        # Save the fixed version
        fixed_path = model_path.replace('.pt', '_fixed.pt')
        torch.save(checkpoint, fixed_path)
        print(f"✅ Fixed model saved as: {fixed_path}")
        
        return fixed_path
        
    except Exception as ex:
        print(f"❌ Could not fix model paths: {ex}")
        return None

def torch_load_with_path_fix(path, **kwargs):
    """Load torch model with PosixPath fix for Windows"""
    if platform.system() == 'Windows':
        # Method 1: Try to fix the model file directly
        try:
            # Load the raw pickle data and fix PosixPath objects
            import pickle
            with open(path, 'rb') as f:
                # Load with our custom unpickler
                unpickler = PathUnpickler(f)
                checkpoint = unpickler.load()
            return checkpoint
        except Exception as e1:
            # Method 2: Try loading with modified pickle protocol
            try:
                # Load with map_location to avoid device issues
                import torch
                # Temporarily disable the monkey patch to avoid recursion
                original_loads = pickle.loads
                def safe_loads(data):
                    return original_loads(data)
                pickle.loads = safe_loads
                
                result = torch.load(path, map_location='cpu', pickle_module=pickle)
                pickle.loads = original_loads
                return result
            except Exception as e2:
                print(f"Both path fix methods failed: {e1}, {e2}")
                pass
    
    # Fallback to regular torch.load
    return torch.load(path, **kwargs)

class YOLO_Wrapper:
    def __init__(self, model, names: List[str]):
        # used to find the correct indices
        self.names = {
            i: word
            for i, name in model.names.items()
            for word in names
            if re.search(rf"\b{re.escape(word)}\b", name)
        }
        self.indices = {v: k for k, v in self.names.items()}
        self.model = model

    def __call__(self, images):
        return self.model(images)

    @staticmethod
    def get_names(path_weight):
        path_weight = path_weight.replace("engine", "pt")
        # Load the checkpoint dictionary
        ckpt = torch_load_with_path_fix(path_weight, map_location="cpu")
        names = ckpt.get("model", {}).names
        del ckpt

        if isinstance(names, list):
            return {i:v for i, v in enumerate(names)}
        return names

class SimplifiedDualCapture:
    def __init__(self, enable_display=True, enable_gpio_sync=True):
        self.system = None
        self.cameras = []
        self.camera_list = None
        self.running = False
        
        # YOLO Model
        self.yolo_model = None
        
        # GPIO Synchronization
        self.enable_gpio_sync = enable_gpio_sync
        self.primary_camera_index = 0  # First camera will be primary
        
        self.output_dir = "synchronized_captures"
        self.individual_dir = "individual_frames"
        self.create_output_directories()
        
        # Display settings
        self.enable_display = enable_display
        self.display_queues = {}
        self.display_threads = []
        
        # Threading and synchronization
        self.save_executor = ThreadPoolExecutor(max_workers=8)
        self.capture_threads = []
        
        # Frame synchronization
        self.frame_sync_dict = {}  # Store frames waiting for sync
        self.sync_lock = threading.Lock()
        self.frame_counter = 0
        
        # Motion detection to save only interesting frames
        self.previous_frames = {}  # Store previous frame for each camera
        self.motion_threshold = 5000  # Increased threshold to ignore light reflections
        self.enable_motion_detection = True
        self.save_individual_frames = False  # Focus on synchronized saves only
        
        # Ball detection
        self.ball_detector = None
        self.setup_ball_detector()
        
        # Statistics
        self.stats = {
            'frames_captured': {},
            'frames_saved': 0,
            'frames_skipped': {},  # Frames skipped due to no motion
            'individual_frames_saved': {},
            'synchronized_frames_saved': 0,
            'balls_detected': 0,
            'start_time': time.time(),
            'last_fps_time': time.time(),
            'last_capture_counts': {},
        }

        print("🚀 SIMPLIFIED DUAL HIGH-SPEED CAPTURE SYSTEM")
        print("=" * 50)
        print("📸 Synchronized dual camera capture")
        print(f"🔗 GPIO Hardware Sync: {'ENABLED' if enable_gpio_sync else 'DISABLED'}")
        if enable_gpio_sync:
            print("   Primary camera will trigger secondary camera via GPIO")
        print("💾 SYNC images saved to: synchronized_captures/")
        if self.save_individual_frames:
            print("💾 Individual images saved to: individual_frames/")
        else:
            print("💾 Individual frames: DISABLED (sync only)")
        print("📊 FPS reporting enabled")
        print(f"📺 Live display: {'ENABLED' if enable_display else 'DISABLED'}")
        print(f"📹 Motion detection: {'ENABLED' if self.enable_motion_detection else 'DISABLED'}")
        if self.enable_motion_detection:
            print(f"   Enhanced detection - ignores light reflections")
            print(f"   Threshold: 200+ significantly changed pixels")

    def setup_ball_detector(self):
        """Initialize YOLOv5 ball detector with multiple fallback methods"""
        try:
            # Check if CUDA is available
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            print(f"🧠 Setting up ball detector on {device}")
            
            # Try multiple approaches for loading the model
            model_loaded = False
            
            # Method 1: Try to load custom model with path fix
            if os.path.exists('best.pt'):
                try:
                    print("🔄 Attempting to load custom model (best.pt)...")
                    
                    # Import YOLOv5 modules
                    import sys
                    sys.path.append('yolov5')
                    from models.common import DetectMultiBackend
                    from utils.general import check_img_size
                    
                    model_path = 'best.pt'
                    
                    # Check if we need to fix the model for Windows
                    if platform.system() == 'Windows':
                        # Try to create a fixed version first
                        fixed_path = fix_model_paths(model_path)
                        if fixed_path and os.path.exists(fixed_path):
                            print(f"🔧 Using fixed model: {fixed_path}")
                            model_path = fixed_path
                        elif os.path.exists('best_fixed.pt'):
                            print("🔧 Using existing fixed model: best_fixed.pt")  
                            model_path = 'best_fixed.pt'
                    
                    # Convert paths to strings to avoid PosixPath issues on Windows
                    weights = str(Path(model_path).absolute())
                    data = str(Path('yolov5/data/coco128.yaml').absolute())
                    
                    self.yolo_model = DetectMultiBackend(
                        weights,
                        device=device,
                        dnn=False,
                        data=data,
                        fp16=False
                    )
                    
                    # Configure model settings
                    self.yolo_model.conf = 0.25  # confidence threshold
                    self.yolo_model.iou = 0.45   # NMS IoU threshold
                    
                    # Get model stride and check image size
                    self.stride = int(self.yolo_model.stride)
                    self.img_size = check_img_size(640, s=self.stride)
                    
                    print("✅ Custom YOLOv5 model loaded successfully!")
                    model_loaded = True
                    
                except Exception as ex:
                    print(f"⚠️ Failed to load custom model: {ex}")
                    # If the main model failed, try the backup approach
                    if platform.system() == 'Windows':
                        try:
                            print("🔄 Trying alternative loading method...")
                            # Try with torch.load directly on fixed model
                            if os.path.exists('best_fixed.pt'):
                                checkpoint = torch.load('best_fixed.pt', map_location=device)
                                print("✅ Loaded fixed model directly with torch.load")
                                # Create a simple wrapper for the checkpoint
                                self.yolo_model = checkpoint
                                self.stride = 32
                                self.img_size = 640
                                model_loaded = True
                            else:
                                print("❌ No fixed model available")
                        except Exception as ex2:
                            print(f"⚠️ Alternative method also failed: {ex2}")
            
            # Method 2: Fallback to standard pretrained model
            if not model_loaded:
                try:
                    print("🔄 Falling back to standard YOLOv5n model...")
                    self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
                    self.yolo_model.conf = 0.25  # confidence threshold
                    self.yolo_model.iou = 0.45   # NMS IoU threshold
                    
                    # Set default values for standard model
                    self.stride = 32
                    self.img_size = 640
                    
                    print("✅ Standard YOLOv5n model loaded successfully!")
                    model_loaded = True
                    
                except Exception as ex:
                    print(f"⚠️ Failed to load standard model: {ex}")
            
            # Method 3: Last resort - disable ball detection
            if not model_loaded:
                print("❌ All model loading methods failed - disabling ball detection")
                self.yolo_model = None
                return
            
        except Exception as ex:
            print(f"❌ Error setting up ball detector: {ex}")
            traceback.print_exc()
            self.yolo_model = None

    def detect_ball(self, image):
        """Run YOLO ball detection on an image - works with both custom and standard models"""
        try:
            if self.yolo_model is None:
                print("❌ YOLO model not initialized!")
                return []
            
            # Ensure image is in BGR format for processing
            if len(image.shape) == 2:  # If grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Check if we're using a standard torch.hub model or custom DetectMultiBackend
            is_standard_model = hasattr(self.yolo_model, '__call__') and not hasattr(self.yolo_model, 'stride')
            
            if is_standard_model:
                # Standard YOLOv5 model from torch.hub
                # Convert BGR to RGB for the model
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Run inference
                results = self.yolo_model(image_rgb)
                
                # Get detections
                detections = results.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2, conf, cls)
                
                detected_balls = []
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    class_id = int(cls)
                    
                    # Filter for ball-related classes (sports ball, tennis ball, etc.)
                    if class_id in [32, 37, 38]:  # sports ball, tennis ball, baseball
                        class_name = results.names[class_id]
                        detected_balls.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': float(conf),
                            'type': class_name
                        })
                        print(f"🎾 Detected {class_name} with {conf:.2f} confidence at ({int(x1)}, {int(y1)})")
                
            else:
                # Custom model using DetectMultiBackend
                # Import YOLOv5 modules
                from utils.general import non_max_suppression, scale_boxes
                from utils.augmentations import letterbox
                
                # Preprocess image
                img = letterbox(image, self.img_size, stride=self.stride, auto=True)[0]
                img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.yolo_model.device)
                img = img.float()
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(img.shape) == 3:
                    img = img[None]  # expand for batch dim
                
                # Inference
                pred = self.yolo_model(img)
                
                # NMS
                pred = non_max_suppression(pred, self.yolo_model.conf, self.yolo_model.iou)
                
                detected_balls = []
                if len(pred) > 0 and len(pred[0]) > 0:
                    # Rescale boxes from img_size to image size
                    pred = pred[0]
                    pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], image.shape).round()
                    
                    # Process detections
                    for *xyxy, conf, cls in pred:
                        x1, y1, x2, y2 = map(int, xyxy)
                        class_id = int(cls)
                        class_name = self.yolo_model.names[class_id]
                        
                        detected_balls.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(conf),
                            'type': class_name
                        })
                        print(f"🎾 Detected {class_name} with {conf:.2f} confidence at ({x1}, {y1})")
            
            if detected_balls:
                print(f"✅ Found {len(detected_balls)} ball(s)!")
            else:
                print("🔍 No balls detected")
            
            return detected_balls
            
        except Exception as ex:
            print(f"❌ Error in ball detection: {ex}")
            traceback.print_exc()
            return []

    def detect_motion(self, camera_id, current_frame):
        """Detect significant motion and then check for ball if motion detected"""
        if not self.enable_motion_detection:
            return True
        
        try:
            # Convert to grayscale if needed
            if len(current_frame.shape) == 3:
                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            else:
                current_gray = current_frame.copy()
            
            # Get previous frame
            if self.previous_frames[camera_id] is None:
                # First frame - always save
                self.previous_frames[camera_id] = current_gray.copy()
                return True
            
            # Apply Gaussian blur to reduce noise and light flicker sensitivity
            current_blur = cv2.GaussianBlur(current_gray, (5, 5), 0)
            previous_blur = cv2.GaussianBlur(self.previous_frames[camera_id], (5, 5), 0)
            
            # Calculate difference between current and previous frame
            diff = cv2.absdiff(current_blur, previous_blur)
            
            # Apply threshold to ignore small changes (light reflections)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Count significant change pixels instead of total difference
            motion_pixels = np.sum(thresh > 0)
            
            # Update previous frame
            self.previous_frames[camera_id] = current_gray.copy()
            
            # Check if motion exceeds threshold (now based on number of changed pixels)
            pixel_threshold = 200  # Number of significantly changed pixels
            has_motion = motion_pixels > pixel_threshold
            
            if has_motion:
                print(f"📹 SIGNIFICANT Motion detected on Camera {camera_id}: {motion_pixels} pixels changed significantly")
                
                # If we have a ball detector, check for balls
                if hasattr(self, 'yolo_model'):
                    balls = self.detect_ball(current_frame)
                    if balls:
                        print(f"🎾 Found {len(balls)} ball(s) in motion!")
                        return True
                    else:
                        print("🔍 No balls detected in motion")
                        return False
                else:
                    return True  # If no ball detector, just use motion
            
            return has_motion
            
        except Exception as ex:
            print(f'❌ Error in motion detection for camera {camera_id}: {ex}')
            return True  # Save frame if detection fails

    def configure_gpio_sync(self, primary_cam, secondary_cam, primary_index, secondary_index):
        """Configure GPIO synchronization for Blackfly S cameras"""
        try:
            primary_nodemap = primary_cam.GetNodeMap()
            secondary_nodemap = secondary_cam.GetNodeMap()
            
            print(f"🔗 Configuring GPIO synchronization...")
            print(f"   Primary: Camera {primary_index + 1} (triggers secondary)")
            print(f"   Secondary: Camera {secondary_index + 1} (triggered by primary)")
            
            # Configure PRIMARY camera (strobe output)
            # Line1 = Pin 4 (white wire) = Opto-isolated output
            line_selector_primary = PySpin.CEnumerationPtr(primary_nodemap.GetNode("LineSelector"))
            if PySpin.IsAvailable(line_selector_primary) and PySpin.IsWritable(line_selector_primary):
                line1_entry = line_selector_primary.GetEntryByName("Line1")
                if PySpin.IsAvailable(line1_entry):
                    line_selector_primary.SetIntValue(line1_entry.GetValue())
                    
                    # Set Line1 to Output mode
                    line_mode = PySpin.CEnumerationPtr(primary_nodemap.GetNode("LineMode"))
                    if PySpin.IsAvailable(line_mode) and PySpin.IsWritable(line_mode):
                        output_mode = line_mode.GetEntryByName("Output")
                        if PySpin.IsAvailable(output_mode):
                            line_mode.SetIntValue(output_mode.GetValue())
                    
                    # Set Line1 source to ExposureActive (triggers when exposure starts)
                    line_source = PySpin.CEnumerationPtr(primary_nodemap.GetNode("LineSource"))
                    if PySpin.IsAvailable(line_source) and PySpin.IsWritable(line_source):
                        exposure_active = line_source.GetEntryByName("ExposureActive")
                        if PySpin.IsAvailable(exposure_active):
                            line_source.SetIntValue(exposure_active.GetValue())
            
            # Enable 3.3V output on Line2 (Pin 3 - red wire) for pull-up resistor
            line_selector_primary.SetIntValue(line_selector_primary.GetEntryByName("Line2").GetValue())
            v33_enable = PySpin.CBooleanPtr(primary_nodemap.GetNode("V3_3Enable"))
            if PySpin.IsAvailable(v33_enable) and PySpin.IsWritable(v33_enable):
                v33_enable.SetValue(True)
                print("   ✅ Primary camera: 3.3V output enabled (Pin 3)")
            
            # Configure SECONDARY camera (trigger input)
            # First, turn off trigger mode to configure it
            trigger_mode_secondary = PySpin.CEnumerationPtr(secondary_nodemap.GetNode("TriggerMode"))
            if PySpin.IsAvailable(trigger_mode_secondary) and PySpin.IsWritable(trigger_mode_secondary):
                trigger_off = trigger_mode_secondary.GetEntryByName("Off")
                if PySpin.IsAvailable(trigger_off):
                    trigger_mode_secondary.SetIntValue(trigger_off.GetValue())
            
            # Set trigger source to Line3 (Pin 1 - green wire = VAUX input)
            trigger_source = PySpin.CEnumerationPtr(secondary_nodemap.GetNode("TriggerSource"))
            if PySpin.IsAvailable(trigger_source) and PySpin.IsWritable(trigger_source):
                line3_entry = trigger_source.GetEntryByName("Line3")
                if PySpin.IsAvailable(line3_entry):
                    trigger_source.SetIntValue(line3_entry.GetValue())
                    print("   ✅ Secondary camera: Trigger source set to Line3 (Pin 1)")
            
            # Set trigger overlap to ReadOut for maximum frame rate
            trigger_overlap = PySpin.CEnumerationPtr(secondary_nodemap.GetNode("TriggerOverlap"))
            if PySpin.IsAvailable(trigger_overlap) and PySpin.IsWritable(trigger_overlap):
                readout_entry = trigger_overlap.GetEntryByName("ReadOut")
                if PySpin.IsAvailable(readout_entry):
                    trigger_overlap.SetIntValue(readout_entry.GetValue())
                    print("   ✅ Secondary camera: Trigger overlap set to ReadOut")
            
            # Enable trigger mode on secondary camera
            if PySpin.IsAvailable(trigger_mode_secondary) and PySpin.IsWritable(trigger_mode_secondary):
                trigger_on = trigger_mode_secondary.GetEntryByName("On")
                if PySpin.IsAvailable(trigger_on):
                    trigger_mode_secondary.SetIntValue(trigger_on.GetValue())
                    print("   ✅ Secondary camera: Trigger mode enabled")
            
            print("🔗 GPIO synchronization configured successfully!")
            print("   📋 Wiring check:")
            print("      Primary Pin 4 (white) → Secondary Pin 1 (green)")
            print("      Primary Pin 5 (blue) → Secondary Pin 6 (brown)") 
            print("      Primary Pin 6 (brown) → Secondary Pin 6 (brown)")
            print("      10kΩ resistor: Primary Pin 3 (red) → Primary Pin 4 (white)")
            
            # Verify trigger mode is still enabled
            trigger_mode_verify = PySpin.CEnumerationPtr(secondary_nodemap.GetNode("TriggerMode"))
            if PySpin.IsAvailable(trigger_mode_verify) and PySpin.IsReadable(trigger_mode_verify):
                current_mode = trigger_mode_verify.GetCurrentEntry().GetSymbolic()
                print(f"   🔍 Secondary camera trigger mode: {current_mode}")
                if current_mode != "On":
                    print("   ⚠️ WARNING: Trigger mode not active!")
                    
            return True
            
        except PySpin.SpinnakerException as ex:
            print(f"❌ Error configuring GPIO sync: {ex}")
            return False

    def create_output_directories(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.individual_dir):
            os.makedirs(self.individual_dir)

    def initialize_system(self):
        try:
            self.system = PySpin.System.GetInstance()
            version = self.system.GetLibraryVersion()
            version_str = f'{version.major}.{version.minor}.{version.type}.{version.build}'
            print(f'🎥 PySpin SDK version: {version_str}')
            return True
        except PySpin.SpinnakerException as ex:
            print(f'❌ Error initializing system: {ex}')
            return False

    def configure_camera(self, cam, camera_index):
        """Configure camera for high-speed capture"""
        nodemap = cam.GetNodeMap()

        # Set acquisition mode to continuous
        acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        if PySpin.IsAvailable(acquisition_mode) and PySpin.IsWritable(acquisition_mode):
            mode_continuous = acquisition_mode.GetEntryByName("Continuous")
            acquisition_mode.SetIntValue(mode_continuous.GetValue())

        # Set pixel format to Mono8 for speed
        pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
        if PySpin.IsAvailable(pixel_format) and PySpin.IsWritable(pixel_format):
            mono8 = pixel_format.GetEntryByName("Mono8")
            if PySpin.IsAvailable(mono8) and PySpin.IsReadable(mono8):
                pixel_format.SetIntValue(mono8.GetValue())

        # Get camera serial for specific configuration
        device_serial = PySpin.CStringPtr(cam.GetTLDeviceNodeMap().GetNode('DeviceSerialNumber'))
        serial_number = device_serial.GetValue() if PySpin.IsAvailable(device_serial) and PySpin.IsReadable(device_serial) else 'Unknown'
        
        # Set ROI dimensions - different for each camera
        roi_width = 720  # Same width for both cameras
        roi_height = 150  # Same height for both cameras
        
        # Adjust offsets based on camera
        if camera_index == 0:  # Camera 1 (first camera)
            roi_offset_x = 0
            roi_offset_y = 246  # Adjusted to focus more on the window area
            print(f"📷 Camera 1 (Serial: {serial_number}) - Configuring for side view at window")
        else:  # Camera 2 (second camera)
            roi_offset_x = 0
            roi_offset_y = 220  # Original offset for Camera 2
            print(f"📷 Camera 2 (Serial: {serial_number}) - Maintaining current view")

        # Get dimension nodes
        width_node = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
        height_node = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
        offset_x_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetX"))
        offset_y_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetY"))
        
        # Reset to max first, then set our values
        if PySpin.IsAvailable(width_node) and PySpin.IsWritable(width_node):
            width_node.SetValue(width_node.GetMax())
        if PySpin.IsAvailable(height_node) and PySpin.IsWritable(height_node):
            height_node.SetValue(height_node.GetMax())
        if PySpin.IsAvailable(offset_x_node) and PySpin.IsWritable(offset_x_node):
            offset_x_node.SetValue(0)
        if PySpin.IsAvailable(offset_y_node) and PySpin.IsWritable(offset_y_node):
            offset_y_node.SetValue(0)
        
        # Set our target dimensions
        if PySpin.IsAvailable(width_node) and PySpin.IsWritable(width_node):
            width_node.SetValue(min(roi_width, width_node.GetMax()))
        if PySpin.IsAvailable(height_node) and PySpin.IsWritable(height_node):
            height_node.SetValue(min(roi_height, height_node.GetMax()))
        if PySpin.IsAvailable(offset_x_node) and PySpin.IsWritable(offset_x_node):
            max_offset_x = width_node.GetMax() - width_node.GetValue()
            offset_x_node.SetValue(min(roi_offset_x, max_offset_x))
        if PySpin.IsAvailable(offset_y_node) and PySpin.IsWritable(offset_y_node):
            max_offset_y = height_node.GetMax() - height_node.GetValue()
            offset_y_node.SetValue(min(roi_offset_y, max_offset_y))

        print(f'🎯 Camera {camera_index+1} ROI: {width_node.GetValue()}x{height_node.GetValue()} at offset y={offset_y_node.GetValue()}')

        # Disable auto exposure and set manual values
        auto_exposure = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
        if PySpin.IsAvailable(auto_exposure) and PySpin.IsWritable(auto_exposure):
            exposure_off = auto_exposure.GetEntryByName("Off")
            if PySpin.IsAvailable(exposure_off):
                auto_exposure.SetIntValue(exposure_off.GetValue())

        exposure_time_node = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
        if PySpin.IsAvailable(exposure_time_node) and PySpin.IsWritable(exposure_time_node):
            exposure_time_node.SetValue(min(2000.0, exposure_time_node.GetMax()))

        # Disable auto gain and set manual gain
        auto_gain = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
        if PySpin.IsAvailable(auto_gain) and PySpin.IsWritable(auto_gain):
            gain_off = auto_gain.GetEntryByName("Off")
            if PySpin.IsAvailable(gain_off):
                auto_gain.SetIntValue(gain_off.GetValue())

        gain_node = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
        if PySpin.IsAvailable(gain_node) and PySpin.IsWritable(gain_node):
            gain_node.SetValue(min(20.0, gain_node.GetMax()))

        # Set frame rate to 350 FPS for better GPIO sync
        acq_frame_rate_enable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
        if PySpin.IsAvailable(acq_frame_rate_enable) and PySpin.IsWritable(acq_frame_rate_enable):
            acq_frame_rate_enable.SetValue(True)

        acq_frame_rate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
        if PySpin.IsAvailable(acq_frame_rate) and PySpin.IsWritable(acq_frame_rate):
            target_fps = 350.0  # Reduced from 500 for better GPIO sync
            max_fps = acq_frame_rate.GetMax()
            final_fps = min(target_fps, max_fps)
            acq_frame_rate.SetValue(final_fps)
            print(f'✅ Camera {camera_index+1}: FPS set to {final_fps:.1f}')

        # Optimize buffer handling
        stream_buffer_count = PySpin.CIntegerPtr(nodemap.GetNode("StreamBufferCountMode"))
        if PySpin.IsAvailable(stream_buffer_count) and PySpin.IsWritable(stream_buffer_count):
            manual_mode = stream_buffer_count.GetEntryByName("Manual")
            if PySpin.IsAvailable(manual_mode):
                stream_buffer_count.SetIntValue(manual_mode.GetValue())

        stream_buffer_count_manual = PySpin.CIntegerPtr(nodemap.GetNode("StreamBufferCountManual"))
        if PySpin.IsAvailable(stream_buffer_count_manual) and PySpin.IsWritable(stream_buffer_count_manual):
            stream_buffer_count_manual.SetValue(16)

        print(f'🛠️ Camera {camera_index+1} configured: {width_node.GetValue()}x{height_node.GetValue()}')

    def enumerate_cameras(self):
        try:
            self.camera_list = self.system.GetCameras()
            num_cameras = self.camera_list.GetSize()

            print(f'📷 Cameras detected: {num_cameras}')
            if num_cameras == 0:
                print('❌ No cameras detected!')
                return False

            for i in range(num_cameras):
                cam = self.camera_list.GetByIndex(i)
                cam.Init()

                nodemap_tldevice = cam.GetTLDeviceNodeMap()
                device_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
                device_model = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceModelName'))

                serial = device_serial.GetValue() if PySpin.IsAvailable(device_serial) and PySpin.IsReadable(device_serial) else 'Unknown'
                model = device_model.GetValue() if PySpin.IsAvailable(device_model) and PySpin.IsReadable(device_model) else 'Unknown'

                print(f'📷 Camera {i+1}: {model} (Serial: {serial})')
                self.configure_camera(cam, i)

                camera_info = {
                    'camera': cam,
                    'camera_id': i+1,
                    'serial': serial,
                    'model': model
                }

                self.cameras.append(camera_info)
                self.stats['frames_captured'][i+1] = 0
                self.stats['frames_skipped'][i+1] = 0
                self.stats['individual_frames_saved'][i+1] = 0
                self.stats['last_capture_counts'][i+1] = 0
                
                # Initialize motion detection
                self.previous_frames[i+1] = None
                
                # Initialize display queue if display is enabled
                if self.enable_display:
                    self.display_queues[i+1] = Queue(maxsize=2)

            # Configure GPIO synchronization if enabled and we have 2+ cameras
            if self.enable_gpio_sync and len(self.cameras) >= 2:
                primary_cam = self.cameras[self.primary_camera_index]['camera']
                secondary_index = 1 if self.primary_camera_index == 0 else 0
                secondary_cam = self.cameras[secondary_index]['camera']
                
                success = self.configure_gpio_sync(
                    primary_cam, secondary_cam, 
                    self.primary_camera_index, 
                    secondary_index
                )
                
                if not success:
                    print("⚠️ GPIO sync configuration failed - continuing with software sync")
                    self.enable_gpio_sync = False

            return len(self.cameras) > 0

        except PySpin.SpinnakerException as ex:
            print(f'❌ Error enumerating cameras: {ex}')
            return False

    def capture_thread(self, camera_info):
        """Capture frames from individual camera"""
        cam = camera_info['camera']
        camera_id = camera_info['camera_id']
        frame_count = 0

        try:
            cam.BeginAcquisition()
            print(f'🚀 Started capture for camera {camera_id} - targeting 350 FPS')

            while self.running:
                try:
                    image_result = cam.GetNextImage(50)  # 50ms timeout
                    if not image_result.IsIncomplete():
                        frame_count += 1
                        self.stats['frames_captured'][camera_id] += 1
                        
                        # Get image data
                        image_data = image_result.GetNDArray()
                        timestamp = datetime.now()
                        
                        # Check for motion before saving
                        has_motion = self.detect_motion(camera_id, image_data)
                        
                        if has_motion:
                            # Motion detected on this camera - try to synchronize with other camera
                            self.try_synchronize_frame(camera_id, image_data, frame_count, timestamp, has_motion=True)
                        else:
                            # No motion on this camera, but still check if other camera had recent motion
                            # This allows sync when only one camera detects motion but both cameras see the ball
                            self.try_synchronize_frame(camera_id, image_data, frame_count, timestamp, has_motion=False)
                            self.stats['frames_skipped'][camera_id] += 1
                        
                        # Send frame to display queue if display is enabled
                        if self.enable_display and camera_id in self.display_queues:
                            if not self.display_queues[camera_id].full():
                                # Convert to BGR for display
                                if len(image_data.shape) == 2:
                                    display_frame = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
                                else:
                                    display_frame = image_data.copy()
                                
                                # Add labels to display frame
                                height, width = display_frame.shape[:2]
                                cv2.putText(display_frame, f'Camera {camera_id} - LIVE', 
                                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                cv2.putText(display_frame, f'Frame: {frame_count}', 
                                           (width - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(display_frame, timestamp.strftime("%H:%M:%S.%f")[:-3], 
                                           (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                                
                                try:
                                    self.display_queues[camera_id].put_nowait(display_frame)
                                except:
                                    pass  # Skip if queue full

                    image_result.Release()

                except PySpin.SpinnakerException as ex:
                    if self.running and "timeout" not in str(ex).lower():
                        print(f'❌ Capture error for camera {camera_id}: {ex}')
                    continue

            cam.EndAcquisition()
            print(f'🛑 Stopped capture for camera {camera_id}')

        except PySpin.SpinnakerException as ex:
            print(f'❌ Error in capture thread for camera {camera_id}: {ex}')

    def try_synchronize_frame(self, camera_id, frame_data, frame_count, timestamp, has_motion=False):
        """Try to synchronize frames from multiple cameras using timestamp-based matching"""
        with self.sync_lock:
            # Use timestamp (rounded to milliseconds) as sync key for better matching
            timestamp_ms = int(timestamp.timestamp() * 1000)  # Convert to milliseconds
            
            # Store this frame with timestamp info
            frame_info = {
                'frame_data': frame_data.copy(),
                'timestamp': timestamp,
                'frame_count': frame_count,
                'timestamp_ms': timestamp_ms,
                'has_motion': has_motion
            }
            
            # Save individual frame only if enabled and to separate folder
            if self.save_individual_frames and has_motion:
                self.save_individual_frame(camera_id, frame_info)
            
            # Try to find a matching frame from other camera(s) within 50ms window (increased for better sync)
            sync_tolerance_ms = 10  # Increased tolerance for better synchronization
            
            # Look for frames from other cameras within the time window
            matching_frames = {camera_id: frame_info}
            
            # Check existing frames for potential matches
            for stored_timestamp_ms, stored_frames in list(self.frame_sync_dict.items()):
                time_diff = abs(timestamp_ms - stored_timestamp_ms)
                if time_diff <= sync_tolerance_ms:
                    # Found frames within sync window
                    for other_camera_id, other_frame_info in stored_frames.items():
                        if other_camera_id != camera_id:
                            matching_frames[other_camera_id] = other_frame_info
                    
                    # If we have frames from all cameras, check if any had motion
                    if len(matching_frames) == len(self.cameras):
                        # Check if any of the frames had motion detected
                        motion_detected = any(frame['has_motion'] for frame in matching_frames.values())
                        
                        if motion_detected:
                            self.save_synchronized_frames(matching_frames, f"sync_{timestamp_ms}")
                            print(f"✅ SYNC SAVED! Cameras matched within {time_diff}ms - motion detected")
                            # Remove the matched frames
                            if stored_timestamp_ms in self.frame_sync_dict:
                                del self.frame_sync_dict[stored_timestamp_ms]
                            return
            
            # Store this frame for future matching (always store, regardless of motion)
            if timestamp_ms not in self.frame_sync_dict:
                self.frame_sync_dict[timestamp_ms] = {}
            self.frame_sync_dict[timestamp_ms][camera_id] = frame_info
            
            # Clean up old frames (keep only recent 200ms worth - increased for better sync)
            current_time_ms = int(time.time() * 1000)
            keys_to_remove = [k for k in self.frame_sync_dict.keys() if current_time_ms - k > 200]
            for k in keys_to_remove:
                del self.frame_sync_dict[k]

    def save_individual_frame(self, camera_id, frame_info):
        """Save individual frame from one camera"""
        try:
            frame_data = frame_info['frame_data']
            timestamp = frame_info['timestamp']
            frame_count = frame_info['frame_count']
            
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Convert to BGR if grayscale
            if len(frame_data.shape) == 2:
                frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = frame_data.copy()
            
            # Add labels
            height, width = frame_bgr.shape[:2]
            cv2.putText(frame_bgr, f'Camera {camera_id}', 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f'Frame: {frame_count}', 
                       (width - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame_bgr, timestamp.strftime("%H:%M:%S.%f")[:-3], 
                       (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Save individual frame
            filename = f"cam{camera_id}_frame{frame_count:06d}_{timestamp_str}.jpg"
            filepath = os.path.join(self.individual_dir, filename)  # Use separate folder
            
            # Use thread pool for saving
            self.save_executor.submit(self._save_individual_image, filepath, frame_bgr, camera_id)
            
        except Exception as ex:
            print(f'❌ Error saving individual frame: {ex}')

    def save_synchronized_frames(self, frame_dict, sync_key):
        """Save synchronized frames from all cameras"""
        try:
            # Sort cameras by ID for consistent ordering
            sorted_cameras = sorted(frame_dict.keys())
            
            frames_to_save = []
            timestamp_str = None
            
            for camera_id in sorted_cameras:
                frame_info = frame_dict[camera_id]
                frame_data = frame_info['frame_data']
                timestamp = frame_info['timestamp']
                frame_count = frame_info['frame_count']
                
                if timestamp_str is None:
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                
                # Convert to BGR for display if needed
                if len(frame_data.shape) == 2:
                    frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGR)
                else:
                    frame_bgr = frame_data.copy()
                
                # Run detection on original frame
                if self.yolo_model is not None:
                    try:
                        # Stack mono to 3 channels for YOLO
                        frame_3ch = np.stack((frame_data,) * 3, axis=-1)
                        
                        # Run inference
                        results = self.yolo_model(frame_3ch, size=640)
                        
                        # Process detections
                        if len(results.pred[0]) > 0:  # If we have detections
                            boxes = results.pred[0]
                            
                            for box in boxes:
                                try:
                                    # Extract box information
                                    x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                                    
                                    # Get class name
                                    class_name = results.names[int(cls)]
                                    
                                    # Draw box
                                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    
                                    # Add label
                                    label = f"{class_name} {conf:.2f}"
                                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                    cv2.rectangle(frame_bgr, (x1, y1-25), (x1 + label_w, y1), (0, 255, 0), -1)
                                    cv2.putText(frame_bgr, label, (x1, y1-5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                                    
                                    print(f"🎯 Camera {camera_id}: Found {class_name} ({conf:.2f})")
                                    
                                except Exception as det_ex:
                                    print(f"⚠️ Error drawing detection: {det_ex}")
                                    continue
                            
                    except Exception as ex:
                        print(f"⚠️ Detection error on camera {camera_id}: {ex}")
                        traceback.print_exc()
                
                # Add camera labels
                height, width = frame_bgr.shape[:2]
                cv2.putText(frame_bgr, f'Camera {camera_id} [SYNC]', 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_bgr, f'Frame: {frame_count}', 
                           (width - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame_bgr, timestamp.strftime("%H:%M:%S.%f")[:-3], 
                           (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                frames_to_save.append(frame_bgr)
            
            # Create side-by-side image
            if len(frames_to_save) == 2:
                combined_image = np.hstack(frames_to_save)
            else:
                combined_image = np.hstack(frames_to_save)  # Works for any number
            
            # Save the synchronized image
            filename = f"SYNC_{sync_key}_{timestamp_str}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Use thread pool for saving to not block capture
            self.save_executor.submit(self._save_sync_image, filepath, combined_image, sync_key)
            
        except Exception as ex:
            print(f'❌ Error saving synchronized frames: {ex}')
            traceback.print_exc()

    def _save_individual_image(self, filepath, image, camera_id):
        """Save individual camera frame (runs in thread pool)"""
        try:
            cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.stats['individual_frames_saved'][camera_id] += 1
            self.stats['frames_saved'] += 1
            
            # Calculate current FPS for this camera
            current_time = time.time()
            runtime = current_time - self.stats['start_time']
            captured = self.stats['frames_captured'][camera_id]
            avg_fps = captured / runtime if runtime > 0 else 0
            
            if self.stats['individual_frames_saved'][camera_id] % 50 == 0:  # Print every 50 frames with FPS
                print(f"💾 Camera {camera_id}: {self.stats['individual_frames_saved'][camera_id]} saved | "
                      f"🚀 {avg_fps:.1f} FPS | {captured} total captured")
        except Exception as ex:
            print(f'❌ Error writing individual image for camera {camera_id}: {ex}')

    def _save_sync_image(self, filepath, image, sync_key):
        """Save synchronized frame (runs in thread pool)"""
        try:
            cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.stats['synchronized_frames_saved'] += 1
            
            # Show FPS for both cameras in sync message
            current_time = time.time()
            runtime = current_time - self.stats['start_time']
            fps_info = []
            for camera_id in self.stats['frames_captured']:
                captured = self.stats['frames_captured'][camera_id]
                avg_fps = captured / runtime if runtime > 0 else 0
                fps_info.append(f"Cam{camera_id}: {avg_fps:.1f}FPS")
            
            fps_str = " | ".join(fps_info)
            print(f"🔄 SYNC #{self.stats['synchronized_frames_saved']}: {os.path.basename(filepath)} | {fps_str}")
        except Exception as ex:
            print(f'❌ Error writing synchronized image: {ex}')

    def start_capture(self):
        if not self.cameras:
            print('❌ No cameras available!')
            return False

        self.running = True
        
        # Start capture threads
        for camera_info in self.cameras:
            thread = threading.Thread(target=self.capture_thread, args=(camera_info,))
            thread.daemon = True
            thread.start()
            self.capture_threads.append(thread)

        # Start display threads if display is enabled
        if self.enable_display:
            for camera_info in self.cameras:
                camera_id = camera_info['camera_id']
                thread = threading.Thread(target=self.display_thread, args=(camera_id,))
                thread.daemon = True
                thread.start()
                self.display_threads.append(thread)

        print('\n🚀 SYNCHRONIZED DUAL CAPTURE ACTIVE')
        if self.enable_display:
            print('🎮 Press "q" or ESC in any window to exit')
        else:
            print('🎮 Press Ctrl+C to exit')

        try:
            while self.running:
                time.sleep(1)  # Update every 1 second for more frequent FPS updates
                self.print_fps_stats()
        except KeyboardInterrupt:
            print('\n🛑 Stopping capture system...')
            self.stop()

        return True

    def print_fps_stats(self):
        """Print current FPS statistics - simplified and focused on FPS"""
        current_time = time.time()
        runtime = current_time - self.stats['start_time']
        time_interval = current_time - self.stats['last_fps_time']
        
        # Calculate FPS for each camera
        fps_summary = []
        for camera_id in self.stats['frames_captured']:
            captured = self.stats['frames_captured'][camera_id]
            last_count = self.stats['last_capture_counts'][camera_id]
            
            avg_fps = captured / runtime if runtime > 0 else 0
            recent_frames = captured - last_count
            realtime_fps = recent_frames / time_interval if time_interval > 0 else 0
            
            fps_summary.append(f"Cam{camera_id}: {realtime_fps:.1f}FPS (avg {avg_fps:.1f})")
            
            # Update last count
            self.stats['last_capture_counts'][camera_id] = captured
        
        # Show concise FPS info with motion detection stats
        fps_line = " | ".join(fps_summary)
        total_individual = sum(self.stats['individual_frames_saved'].values())
        total_skipped = sum(self.stats['frames_skipped'].values())
        sync_count = self.stats["synchronized_frames_saved"]
        
        print(f"🚀 FPS: {fps_line} | 💾 Saved: {total_individual} individual + {sync_count} sync | "
              f"⏭️ Skipped: {total_skipped} (no motion) | ⏱️ {runtime:.0f}s")
        
        self.stats['last_fps_time'] = current_time

    def display_thread(self, camera_id):
        """Display live video from camera"""
        if not self.enable_display:
            return
            
        window_name = f'Camera {camera_id} - Live View'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Calculate proper aspect ratio for 720x150 ROI
        roi_width, roi_height = 720, 150
        aspect_ratio = roi_width / roi_height  # 4.8:1 ratio
        
        # Set display window to maintain aspect ratio
        display_height = 200
        display_width = int(display_height * aspect_ratio)
        cv2.resizeWindow(window_name, display_width, display_height)
        
        print(f'📺 Started live display for camera {camera_id} - 350 FPS capture mode')

        while self.running:
            try:
                if not self.display_queues[camera_id].empty():
                    frame = self.display_queues[camera_id].get_nowait()
                    cv2.imshow(window_name, frame)
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:  # 'q' or ESC
                    print(f'👋 Exit key pressed in camera {camera_id} window')
                    self.stop()
                    break
                    
            except Exception as ex:
                if self.running:
                    print(f'❌ Display error for camera {camera_id}: {ex}')
                break

        cv2.destroyWindow(window_name)
        print(f'🛑 Stopped display for camera {camera_id}')

    def stop(self):
        self.running = False
        print('🛑 Shutting down...')
        time.sleep(0.5)
        self.save_executor.shutdown(wait=True)
        
        # Close all OpenCV windows
        if self.enable_display:
            cv2.destroyAllWindows()

    def pre_cleanup(self):
        """Prepare for cleanup by stopping all operations"""
        try:
            # Stop all operations
            self.running = False
            time.sleep(1.0)  # Give threads time to stop
            
            # Stop all camera operations first
            if hasattr(self, 'cameras') and self.cameras:
                for camera_info in self.cameras:
                    try:
                        cam = camera_info.get('camera')
                        if cam and cam.IsStreaming():
                            cam.EndAcquisition()
                    except:
                        pass
            
            # Clear all queues
            if hasattr(self, 'frame_queues'):
                for queue in self.frame_queues.values():
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                        except:
                            pass
            
            if hasattr(self, 'display_queues'):
                for queue in self.display_queues.values():
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                        except:
                            pass
            
            # Close display windows
            cv2.destroyAllWindows()
            time.sleep(0.5)  # Give time for windows to close
            
        except Exception as ex:
            print(f"⚠️ Warning during pre-cleanup: {ex}")

    def cleanup(self):
        """Clean up camera resources"""
        try:
            # Run pre-cleanup first
            self.pre_cleanup()
            
            # Clear all data structures that might hold references
            if hasattr(self, 'frame_queues'):
                self.frame_queues.clear()
                del self.frame_queues
            
            if hasattr(self, 'display_queues'):
                self.display_queues.clear()
                del self.display_queues
            
            if hasattr(self, 'previous_frames'):
                self.previous_frames.clear()
                del self.previous_frames
            
            if hasattr(self, 'frame_sync_dict'):
                self.frame_sync_dict.clear()
                del self.frame_sync_dict
            
            # Clean up cameras
            if hasattr(self, 'cameras') and self.cameras:
                for camera_info in self.cameras:
                    try:
                        cam = camera_info.get('camera')
                        if cam:
                            if cam.IsStreaming():
                                cam.EndAcquisition()
                            cam.DeInit()
                            camera_info['camera'] = None
                    except Exception as ex:
                        print(f"⚠️ Warning during camera cleanup: {ex}")
                self.cameras.clear()
                del self.cameras
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clean up camera list
            if hasattr(self, 'camera_list') and self.camera_list:
                try:
                    self.camera_list.Clear()
                    self.camera_list = None
                    del self.camera_list
                except Exception as ex:
                    print(f"⚠️ Warning during camera list cleanup: {ex}")
            
            # Final garbage collection before system release
            gc.collect()
            
            # Finally release the system instance
            if hasattr(self, 'system') and self.system:
                try:
                    self.system.ReleaseInstance()
                    self.system = None
                    del self.system
                except Exception as ex:
                    print(f"⚠️ Warning during system cleanup: {ex}")
            
            # Final cleanup
            gc.collect()
            cv2.destroyAllWindows()
            
        except Exception as ex:
            print(f"⚠️ Warning during final cleanup: {ex}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass

def main():
    import sys
    
    # Check for command line options
    enable_gpio_sync = True
    enable_display = True
    
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.lower() in ['--no-gpio', '--software-only']:
                enable_gpio_sync = False
                print("🔧 GPIO sync disabled via command line")
            elif arg.lower() in ['--no-display', '--headless']:
                enable_display = False
                print("🔧 Display disabled via command line")
            elif arg.lower() in ['--help', '-h']:
                print("📋 Usage: python simplified_dual_capture.py [options]")
                print("   --no-gpio        Disable GPIO hardware synchronization")
                print("   --no-display     Run without live display windows")
                print("   --help           Show this help message")
                return
    
    capture_system = SimplifiedDualCapture(
        enable_display=enable_display,
        enable_gpio_sync=enable_gpio_sync
    )
    
    try:
        if not capture_system.initialize_system(): 
            return
        if not capture_system.enumerate_cameras(): 
            return
        capture_system.start_capture()
    finally:
        capture_system.cleanup()

if __name__ == "__main__":
    main() 