import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

"""
Simplified Dual High-Speed Camera Capture System
Focus: Image capture, synchronization, and FPS reporting
WITH GPIO HARDWARE SYNCHRONIZATION FOR BLACKFLY S CAMERAS
"""

import PySpin
import cv2
import numpy as np
import os
import time
import threading
from queue import Queue, Full, Empty
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import torch

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
        
        # Frame queues for each camera (buffer between acquisition and processing)
        self.frame_queues = {}
        self.processing_threads = []
        self.MAX_QUEUE_SIZE = 100  # Maximum frames to buffer
        
        # Frame synchronization
        self.frame_sync_dict = {}  # Store frames waiting for sync
        self.sync_lock = threading.Lock()
        self.frame_counter = 0
        
        # Motion detection to save only interesting frames
        self.previous_frames = {}  # Store previous frame for each camera
        self.motion_threshold = 5000  # Increased threshold to ignore light reflections
        self.enable_motion_detection = True
        self.save_individual_frames = False  # Focus on synchronized saves only
        
        # Statistics
        self.stats = {
            'frames_captured': {},      # Raw frames from camera
            'frames_processed': {},     # Frames actually processed
            'frames_saved': 0,
            'frames_skipped': {},       # Frames skipped due to no motion
            'individual_frames_saved': {},
            'synchronized_frames_saved': 0,
            'start_time': time.time(),
            'last_fps_time': time.time(),
            'last_capture_counts': {},
            'last_process_counts': {},
            'camera_fps': {},           # Actual camera acquisition rate
            'processing_fps': {},       # Processing rate
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
            print("   Enhanced detection - ignores light reflections")
            print("   Threshold: 200+ significantly changed pixels")

    def detect_ball(self, image):
        """Run YOLO ball detection on an image"""
        try:
            if self.yolo_model is None:
                print("❌ YOLO model not initialized!")
                return []
            
            # Ensure image is in RGB format for YOLO
            if len(image.shape) == 2:  # If grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:  # If BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = self.yolo_model(image)
            
            # Get detections
            detections = results.xyxy[0].cpu().numpy()  # Get detection boxes in (x1, y1, x2, y2, conf, cls) format
            
            detected_balls = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if int(cls) in [32, 37, 38]:  # Sports ball, tennis ball, baseball classes
                    ball_type = "sports ball" if int(cls) == 32 else "tennis ball" if int(cls) == 37 else "baseball"
                    detected_balls.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(conf),
                        'type': ball_type
                    })
                    print(f"🎾 Detected {ball_type} with {conf:.2f} confidence at ({int(x1)}, {int(y1)})")
            
            if detected_balls:
                print(f"✅ Found {len(detected_balls)} ball(s)!")
            
            return detected_balls
        except Exception as ex:
            print(f"❌ Error in ball detection: {ex}")
            return []

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
            
            # Initialize YOLO model
            print("🔄 Loading YOLOv5n model...")
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
            self.yolo_model.classes = [32, 37, 38]  # Sports ball, tennis ball, baseball classes
            self.yolo_model.conf = 0.3  # Lower confidence threshold for better detection
            print("✅ YOLOv5n model loaded successfully!")
            
            return True
        except PySpin.SpinnakerException as ex:
            print(f'❌ Error initializing system: {ex}')
            return False

    def configure_camera(self, cam, camera_index):
        """Configure camera for high-speed capture"""
        try:
            # First stop acquisition if camera is streaming
            if cam.IsStreaming():
                cam.EndAcquisition()
                time.sleep(0.1)  # Give camera time to stop
            
            nodemap = cam.GetNodeMap()
            stream_node = cam.GetTLStreamNodeMap()
            device_node = cam.GetTLDeviceNodeMap()

            # Get camera serial for identification
            device_serial = PySpin.CStringPtr(cam.GetTLDeviceNodeMap().GetNode('DeviceSerialNumber'))
            serial_number = device_serial.GetValue() if PySpin.IsAvailable(device_serial) and PySpin.IsReadable(device_serial) else 'Unknown'

            # Special handling for Camera 2 - reduce initial load
            if camera_index == 1:  # Camera 2
                print(f"\n🔧 Configuring Camera 2 (Serial: {serial_number}) with reduced initial settings")
                
                # Set very conservative initial settings for Camera 2
                roi_width = 720  # Keep width
                roi_height = 220  # Keep height
                roi_offset_x = 0
                roi_offset_y = 272
                initial_fps = 100.0  # Start at lower FPS
                
                # Disable GPIO for initial startup
                if self.enable_gpio_sync:
                    print("⚠️ Temporarily disabling GPIO sync for Camera 2 startup")
                    trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerMode"))
                    if PySpin.IsAvailable(trigger_mode) and PySpin.IsWritable(trigger_mode):
                        trigger_off = trigger_mode.GetEntryByName("Off")
                        if PySpin.IsAvailable(trigger_off):
                            trigger_mode.SetIntValue(trigger_off.GetValue())
            else:  # Camera 1 - normal settings
                roi_width = 720
                roi_height = 220
                roi_offset_x = 0
                roi_offset_y = 294
                initial_fps = 350.0

            # Configure buffer handling mode
            handling_mode = PySpin.CEnumerationPtr(stream_node.GetNode('StreamBufferHandlingMode'))
            if PySpin.IsAvailable(handling_mode) and PySpin.IsWritable(handling_mode):
                handling_mode_newest = handling_mode.GetEntryByName('NewestOnly')
                handling_mode.SetIntValue(handling_mode_newest.GetValue())
                print(f"✅ Camera {camera_index+1}: Set to NewestOnly buffer mode")
            
            # Set buffer count mode to manual with larger buffers for Camera 2
            buffer_count_mode = PySpin.CEnumerationPtr(stream_node.GetNode('StreamBufferCountMode'))
            if PySpin.IsAvailable(buffer_count_mode) and PySpin.IsWritable(buffer_count_mode):
                buffer_count_manual = buffer_count_mode.GetEntryByName('Manual')
                buffer_count_mode.SetIntValue(buffer_count_manual.GetValue())
            
            buffer_count = PySpin.CIntegerPtr(stream_node.GetNode('StreamBufferCountManual'))
            if PySpin.IsAvailable(buffer_count) and PySpin.IsWritable(buffer_count):
                if camera_index == 1:  # Camera 2 gets more buffers
                    buffer_count.SetValue(64)  # Double the buffers for Camera 2
                    print(f"✅ Camera 2: Buffer count set to 64 for stability")
                else:
                    buffer_count.SetValue(32)
                    print(f"✅ Camera 1: Buffer count set to 32")

            # Configure packet size and delay
            packet_size = PySpin.CIntegerPtr(stream_node.GetNode('StreamPacketSize'))
            if PySpin.IsAvailable(packet_size) and PySpin.IsWritable(packet_size):
                if camera_index == 1:  # Camera 2 gets smaller packets
                    packet_size.SetValue(packet_size.GetMax() // 2)  # Half max packet size
                    print(f"✅ Camera 2: Reduced packet size for stability")
                else:
                    packet_size.SetValue(packet_size.GetMax())

            # Add delay between packets for Camera 2
            if camera_index == 1:
                packet_delay = PySpin.CIntegerPtr(stream_node.GetNode('StreamPacketDelay'))
                if PySpin.IsAvailable(packet_delay) and PySpin.IsWritable(packet_delay):
                    packet_delay.SetValue(250)  # 250µs delay between packets
                    print(f"✅ Camera 2: Added packet delay for stability")

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

            # Configure ROI dimensions
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

            # Set exposure and gain
            auto_exposure = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
            if PySpin.IsAvailable(auto_exposure) and PySpin.IsWritable(auto_exposure):
                exposure_off = auto_exposure.GetEntryByName("Off")
                if PySpin.IsAvailable(exposure_off):
                    auto_exposure.SetIntValue(exposure_off.GetValue())

            exposure_time_node = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
            if PySpin.IsAvailable(exposure_time_node) and PySpin.IsWritable(exposure_time_node):
                if camera_index == 1:  # Longer exposure for Camera 2 initially
                    exposure_time_node.SetValue(min(3000.0, exposure_time_node.GetMax()))
                else:
                    exposure_time_node.SetValue(min(2000.0, exposure_time_node.GetMax()))

            auto_gain = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
            if PySpin.IsAvailable(auto_gain) and PySpin.IsWritable(auto_gain):
                gain_off = auto_gain.GetEntryByName("Off")
                if PySpin.IsAvailable(gain_off):
                    auto_gain.SetIntValue(gain_off.GetValue())

            gain_node = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
            if PySpin.IsAvailable(gain_node) and PySpin.IsWritable(gain_node):
                gain_node.SetValue(min(20.0, gain_node.GetMax()))

            # Set frame rate
            acq_frame_rate_enable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
            if PySpin.IsAvailable(acq_frame_rate_enable) and PySpin.IsWritable(acq_frame_rate_enable):
                acq_frame_rate_enable.SetValue(True)

            acq_frame_rate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
            if PySpin.IsAvailable(acq_frame_rate) and PySpin.IsWritable(acq_frame_rate):
                max_fps = acq_frame_rate.GetMax()
                final_fps = min(initial_fps, max_fps)
                acq_frame_rate.SetValue(final_fps)
                print(f'✅ Camera {camera_index+1}: FPS set to {final_fps:.1f}')

            if camera_index == 1:
                print("\n📝 Camera 2 configured with conservative settings:")
                print(f"   • Initial FPS: {initial_fps} (will increase gradually)")
                print(f"   • Larger buffer count: 64")
                print(f"   • Reduced packet size")
                print(f"   • Added packet delay: 250µs")
                print(f"   • GPIO sync temporarily disabled")
                print(f"   • ROI: {width_node.GetValue()}x{height_node.GetValue()} at y={offset_y_node.GetValue()}")
                print("\nWill attempt to increase settings once stable...")

            return True
            
        except PySpin.SpinnakerException as ex:
            print(f"❌ Error configuring camera {camera_index+1}: {ex}")
            return False

    def enumerate_cameras(self):
        try:
            self.camera_list = self.system.GetCameras()
            num_cameras = self.camera_list.GetSize()

            print(f'📷 Cameras detected: {num_cameras}')
            if num_cameras == 0:
                print('❌ No cameras detected!')
                return False

            # First initialize all cameras
            for i in range(num_cameras):
                cam = self.camera_list.GetByIndex(i)
                cam.Init()
                time.sleep(0.5)  # Give each camera time to initialize

                nodemap_tldevice = cam.GetTLDeviceNodeMap()
                device_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
                device_model = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceModelName'))

                serial = device_serial.GetValue() if PySpin.IsAvailable(device_serial) and PySpin.IsReadable(device_serial) else 'Unknown'
                model = device_model.GetValue() if PySpin.IsAvailable(device_model) and PySpin.IsReadable(device_model) else 'Unknown'

                print(f'📷 Camera {i+1}: {model} (Serial: {serial})')
                
                camera_info = {
                    'camera': cam,
                    'camera_id': i+1,
                    'serial': serial,
                    'model': model
                }
                self.cameras.append(camera_info)

            # Then configure all cameras
            print("\n🔧 Configuring cameras for synchronized capture...")
            for i, camera_info in enumerate(self.cameras):
                print(f"\n⚙️ Configuring Camera {i+1}...")
                if not self.configure_camera(camera_info['camera'], i):
                    print(f"❌ Failed to configure Camera {i+1}")
                    return False
                time.sleep(1.0)  # Wait between camera configurations
                
            # Initialize statistics tracking
            for i in range(num_cameras):
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
                print("\n🔌 Setting up GPIO synchronization...")
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
                else:
                    print("✅ GPIO synchronization configured successfully")
                    time.sleep(1.0)  # Wait for GPIO setup to stabilize

            print("\n✅ All cameras configured successfully")
            return len(self.cameras) > 0

        except PySpin.SpinnakerException as ex:
            print(f'❌ Error enumerating cameras: {ex}')
            return False

    def capture_thread(self, camera_info):
        """Capture frames from individual camera - ONLY handles acquisition"""
        cam = camera_info['camera']
        camera_id = camera_info['camera_id']
        frame_count = 0
        error_count = 0
        MAX_ERRORS = 5  # Maximum consecutive errors before resetting camera
        RESET_DELAY = 2.0  # Increased from 1.0 to 2.0 seconds to wait after reset
        CAPTURE_TIMEOUT = 100  # Increased from 20 to 100 milliseconds to wait for frame

        # Camera 2 ramping variables
        is_camera_2 = camera_id == 2
        current_fps = 100.0 if is_camera_2 else 350.0
        target_fps = 350.0
        fps_step = 50.0  # Increase by 50 FPS at a time
        stable_frames_needed = 1000  # Need 1000 frames without errors to increase FPS
        stable_frame_count = 0
        last_fps_change_time = time.time()
        MIN_TIME_BETWEEN_CHANGES = 5.0  # Minimum seconds between FPS changes

        try:
            # Initialize frame queue for this camera
            self.frame_queues[camera_id] = Queue(maxsize=self.MAX_QUEUE_SIZE)
            self.stats['frames_captured'][camera_id] = 0
            self.stats['frames_processed'][camera_id] = 0
            self.stats['camera_fps'][camera_id] = 0
            self.stats['processing_fps'][camera_id] = 0
            
            # Start processing thread for this camera
            process_thread = threading.Thread(
                target=self.process_frames,
                args=(camera_id,),
                daemon=True
            )
            process_thread.start()
            self.processing_threads.append(process_thread)

            while self.running:
                try:
                    if not cam.IsStreaming():
                        # Configure event handling before starting acquisition
                        try:
                            # Get stream nodemap
                            stream_nodemap = cam.GetTLStreamNodeMap()
                            nodemap = cam.GetNodeMap()
                            
                            # Set buffer handling mode to NewestOnly
                            handling_mode = PySpin.CEnumerationPtr(stream_nodemap.GetNode('StreamBufferHandlingMode'))
                            if PySpin.IsAvailable(handling_mode) and PySpin.IsWritable(handling_mode):
                                handling_mode_newest = handling_mode.GetEntryByName('NewestOnly')
                                handling_mode.SetIntValue(handling_mode_newest.GetValue())
                            
                            # Set current FPS
                            if is_camera_2:
                                acq_frame_rate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
                                if PySpin.IsAvailable(acq_frame_rate) and PySpin.IsWritable(acq_frame_rate):
                                    acq_frame_rate.SetValue(current_fps)
                                    print(f"\n📈 Camera 2 current FPS: {current_fps}")
                            
                            # Start acquisition
                            cam.BeginAcquisition()
                            print(f'🚀 Started capture for camera {camera_id} - Camera set to {current_fps} FPS')
                            
                            # Reset error count after successful start
                            error_count = 0
                            stable_frame_count = 0
                            
                        except PySpin.SpinnakerException as ex:
                            print(f"⚠️ Error configuring camera {camera_id}: {ex}")
                            time.sleep(RESET_DELAY)
                            continue
                    
                    last_fps_time = time.time()
                    last_frame_count = 0
                    consecutive_timeouts = 0
                    MAX_TIMEOUTS = 3  # Maximum consecutive timeouts before resetting

                    while self.running and cam.IsStreaming():
                        try:
                            # Get next image with longer timeout
                            image_result = cam.GetNextImage(CAPTURE_TIMEOUT)
                            
                            if image_result.IsIncomplete():
                                print(f"⚠️ Incomplete frame from camera {camera_id}")
                                error_count += 1
                                stable_frame_count = 0  # Reset stable count
                                image_result.Release()
                            else:
                                frame_count += 1
                                self.stats['frames_captured'][camera_id] += 1
                                error_count = 0  # Reset error count on successful capture
                                consecutive_timeouts = 0  # Reset timeout counter
                                
                                # Increment stable frame count for Camera 2
                                if is_camera_2:
                                    stable_frame_count += 1
                                
                                # Get image data and timestamp
                                image_data = image_result.GetNDArray()
                                timestamp = datetime.now()
                                
                                # Release the image before processing
                                image_result.Release()
                                
                                # Try to add to processing queue, skip if full
                                try:
                                    self.frame_queues[camera_id].put_nowait({
                                        'frame': image_data,
                                        'timestamp': timestamp,
                                        'frame_count': frame_count
                                    })
                                except Full:
                                    pass  # Skip if queue full
                                    
                                # Calculate camera FPS every second
                                current_time = time.time()
                                if current_time - last_fps_time >= 1.0:
                                    camera_fps = (frame_count - last_frame_count) / (current_time - last_fps_time)
                                    self.stats['camera_fps'][camera_id] = camera_fps
                                    last_frame_count = frame_count
                                    last_fps_time = current_time

                                # Check if we should increase FPS for Camera 2
                                if (is_camera_2 and 
                                    stable_frame_count >= stable_frames_needed and 
                                    current_fps < target_fps and
                                    current_time - last_fps_change_time >= MIN_TIME_BETWEEN_CHANGES):
                                    
                                    # Increase FPS
                                    new_fps = min(current_fps + fps_step, target_fps)
                                    acq_frame_rate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
                                    if PySpin.IsAvailable(acq_frame_rate) and PySpin.IsWritable(acq_frame_rate):
                                        acq_frame_rate.SetValue(new_fps)
                                        current_fps = new_fps
                                        print(f"\n📈 Camera 2 FPS increased to {current_fps}")
                                        # Reset counters
                                        stable_frame_count = 0
                                        last_fps_change_time = current_time
                                        
                                        # If we've reached target FPS, re-enable GPIO sync
                                        if current_fps >= target_fps and self.enable_gpio_sync:
                                            print("\n🔌 Re-enabling GPIO sync for Camera 2...")
                                            trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerMode"))
                                            if PySpin.IsAvailable(trigger_mode) and PySpin.IsWritable(trigger_mode):
                                                trigger_on = trigger_mode.GetEntryByName("On")
                                                if PySpin.IsAvailable(trigger_on):
                                                    trigger_mode.SetIntValue(trigger_on.GetValue())
                                                    print("✅ GPIO sync re-enabled for Camera 2")

                            # Check if we need to reset due to errors
                            if error_count >= MAX_ERRORS:
                                print(f"⚠️ Too many errors on camera {camera_id} - resetting...")
                                raise Exception("Too many consecutive errors")

                        except PySpin.SpinnakerException as ex:
                            if "timeout" in str(ex).lower():
                                consecutive_timeouts += 1
                                if consecutive_timeouts >= MAX_TIMEOUTS:
                                    print(f"⚠️ Too many consecutive timeouts on camera {camera_id} - resetting...")
                                    raise Exception("Too many consecutive timeouts")
                            else:
                                print(f'⚠️ Capture error for camera {camera_id}: {ex}')
                                error_count += 1
                                stable_frame_count = 0  # Reset stable count
                                if error_count >= MAX_ERRORS:
                                    raise Exception("Too many consecutive errors")
                            continue

                except Exception as ex:
                    print(f"❌ Error in camera {camera_id} streaming loop: {ex}")
                    try:
                        # Stop acquisition and wait
                        if cam.IsStreaming():
                            cam.EndAcquisition()
                        print(f"🔄 Resetting camera {camera_id}...")
                        time.sleep(RESET_DELAY)  # Longer delay between retries
                        error_count = 0  # Reset error counter
                        stable_frame_count = 0  # Reset stable count
                        if is_camera_2:
                            # Reduce FPS after error
                            current_fps = max(100.0, current_fps - fps_step)
                            print(f"\n📉 Reducing Camera 2 FPS to {current_fps} after error")
                    except:
                        print(f"⚠️ Error while resetting camera {camera_id}")
                    continue

            # Clean shutdown
            try:
                if cam.IsStreaming():
                    cam.EndAcquisition()
                print(f'🛑 Stopped capture for camera {camera_id}')
            except:
                pass

        except Exception as ex:
            print(f'❌ Fatal error in capture thread for camera {camera_id}: {ex}')
            self.running = False  # Stop everything if we hit a fatal error

    def process_frames(self, camera_id):
        """Process frames from the camera's frame queue"""
        print(f'🔄 Started frame processing for camera {camera_id}')
        last_fps_time = time.time()
        processed_count = 0
        last_processed_count = 0
        
        while self.running:
            try:
                # Get frame from queue with timeout
                frame_data = self.frame_queues[camera_id].get(timeout=0.1)
                
                # Process the frame
                frame = frame_data['frame']
                timestamp = frame_data['timestamp']
                frame_count = frame_data['frame_count']
                
                # Detect motion
                has_motion = self.detect_motion(camera_id, frame)
                
                if has_motion:
                    # Try to synchronize with other camera
                    self.try_synchronize_frame(camera_id, frame, frame_count, timestamp, has_motion=True)
                else:
                    # No motion, but still check for sync
                    self.try_synchronize_frame(camera_id, frame, frame_count, timestamp, has_motion=False)
                    self.stats['frames_skipped'][camera_id] += 1
                
                # Update processing stats
                processed_count += 1
                self.stats['frames_processed'][camera_id] += 1
                
                # Calculate processing FPS every second
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    processing_fps = (processed_count - last_processed_count) / (current_time - last_fps_time)
                    self.stats['processing_fps'][camera_id] = processing_fps
                    last_processed_count = processed_count
                    last_fps_time = current_time
                
                # Handle display if enabled
                if self.enable_display and camera_id in self.display_queues:
                    if not self.display_queues[camera_id].full():
                        # Convert to BGR for display
                        if len(frame.shape) == 2:
                            display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        else:
                            display_frame = frame.copy()
                        
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
                            
            except Empty:
                continue  # No frames to process
            except Exception as ex:
                print(f'❌ Error processing frames for camera {camera_id}: {ex}')
                continue
        
        print(f'🛑 Stopped frame processing for camera {camera_id}')

    def detect_motion(self, camera_id, current_frame):
        """Detect significant motion by comparing with previous frame - robust against light changes"""
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
            
            return has_motion
            
        except Exception as ex:
            print(f'❌ Error in motion detection for camera {camera_id}: {ex}')
            return True  # Save frame if detection fails

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
                
                # Convert to BGR if grayscale
                if len(frame_data.shape) == 2:
                    frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGR)
                else:
                    frame_bgr = frame_data.copy()
                
                # Add simple labels with "SYNC" indicator
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
            # First save the original synchronized image
            cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Run ball detection
            print(f"🔍 Running ball detection on sync image {os.path.basename(filepath)}...")
            detected_balls = self.detect_ball(image)
            
            if detected_balls:
                # Create a copy for drawing
                annotated_image = image.copy()
                
                # Draw bounding boxes for each detected ball
                for i, ball in enumerate(detected_balls):
                    x1, y1, x2, y2 = ball['bbox']
                    conf = ball['confidence']
                    ball_type = ball['type']
                    
                    # Color based on ball type
                    if ball_type == "sports ball":
                        color = (0, 255, 0)  # Green
                    elif ball_type == "tennis ball":
                        color = (0, 255, 255)  # Yellow
                    else:  # baseball
                        color = (255, 0, 0)  # Blue
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add detailed label with ball type and confidence
                    label = f"{ball_type}: {conf:.2f}"
                    # Draw label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_image, (x1, y1-25), (x1 + label_w, y1), color, -1)
                    # Draw label text in black
                    cv2.putText(annotated_image, label, (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Save the annotated image with "_detected" suffix
                detected_filepath = filepath.replace(".jpg", "_detected.jpg")
                cv2.imwrite(detected_filepath, annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                
                # Print detailed detection results
                print(f"🎾 Found {len(detected_balls)} ball(s) in {os.path.basename(filepath)}!")
                for i, ball in enumerate(detected_balls):
                    x1, y1, x2, y2 = ball['bbox']
                    conf = ball['confidence']
                    ball_type = ball['type']
                    print(f"   Ball {i+1}: {ball_type} (conf={conf:.2f}) at ({x1},{y1})-({x2},{y2})")
                print(f"   💾 Saved annotated image as: {os.path.basename(detected_filepath)}")
            else:
                print(f"ℹ️ No balls detected in {os.path.basename(filepath)}")
            
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
            print(f'❌ Error in _save_sync_image: {ex}')

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
        """Print current FPS statistics - showing both camera and processing rates"""
        current_time = time.time()
        runtime = current_time - self.stats['start_time']
        
        # Show FPS info for each camera
        fps_summary = []
        for camera_id in self.stats['frames_captured']:
            camera_fps = self.stats['camera_fps'][camera_id]
            processing_fps = self.stats['processing_fps'][camera_id]
            
            fps_summary.append(
                f"Cam{camera_id}: {camera_fps:.1f}FPS (camera) | {processing_fps:.1f}FPS (processed)"
            )
        
        # Show concise FPS info with motion detection stats
        fps_line = " | ".join(fps_summary)
        total_individual = sum(self.stats['individual_frames_saved'].values())
        total_skipped = sum(self.stats['frames_skipped'].values())
        sync_count = self.stats["synchronized_frames_saved"]
        
        print(f"🎥 {fps_line}")
        print(f"💾 Saved: {total_individual} individual + {sync_count} sync | "
              f"⏭️ Skipped: {total_skipped} (no motion) | ⏱️ {runtime:.0f}s")

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

    def cleanup(self):
        try:
            for camera_info in self.cameras:
                try:
                    camera_info['camera'].DeInit()
                except:
                    pass

            if self.camera_list:
                self.camera_list.Clear()
            if self.system:
                self.system.ReleaseInstance()
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