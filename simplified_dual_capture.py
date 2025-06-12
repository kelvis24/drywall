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
from queue import Queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

class SimplifiedDualCapture:
    def __init__(self, enable_display=True, enable_gpio_sync=True):
        self.system = None
        self.cameras = []
        self.camera_list = None
        self.running = False
        
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
        
        # Statistics
        self.stats = {
            'frames_captured': {},
            'frames_saved': 0,
            'frames_skipped': {},  # Frames skipped due to no motion
            'individual_frames_saved': {},
            'synchronized_frames_saved': 0,
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

        # Set ROI dimensions
        roi_width, roi_height = 720, 150
        roi_offset_x, roi_offset_y = 0, 196
        
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