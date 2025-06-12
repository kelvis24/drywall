"""
Dual High-Speed Camera Capture System
Optimized for 500 FPS performance
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

class DualHighSpeedCapture:
    def __init__(self, enable_display=False, save_every_n_frames=1, enable_baseball_detection=True):
        self.system = None
        self.cameras = []
        self.camera_list = None
        self.running = False
        
        # Performance optimizations
        self.enable_display = enable_display
        self.save_every_n_frames = save_every_n_frames  # Save every N frames to reduce I/O
        self.enable_baseball_detection = enable_baseball_detection

        self.output_dir = "high_speed_captures"
        self.create_output_directories()
        
        # Simplified baseball detection parameters for single white baseball
        self.baseball_detection_params = {
            # Debug mode - enable to see processing steps
            'debug_mode': True,             # Show intermediate processing steps for tuning
            
            # Save only when ball detected
            'save_only_when_detected': False, # Save all frames, not just when baseball is found
            
            # Detection parameters - made MUCH stricter to reduce false positives
            'threshold_value': 180,         # Brightness threshold (was 160) - much higher = much stricter
            'min_area': 50,                 # Minimum area (was 20) - much higher = stricter
            'max_area': 1000,               # Maximum area (was 2000) - lower = stricter
            'min_radius': 4,                # Minimum radius (was 3) - higher = stricter
            'max_radius': 30,               # Maximum radius (was 50) - much lower = stricter
            'min_circularity': 0.5,         # Minimum circularity (was 0.3) - much higher = stricter
            'min_brightness': 150,          # Minimum brightness (was 120) - much higher = stricter
            'min_confidence': 0.6,          # Minimum confidence (was 0.4) - much higher = stricter
            
            # Expected number of balls for validation
            'expected_ball_count': 4,       # Expected number of baseballs in this side view
            'max_detections_per_camera': 6, # Maximum detections per camera (safety limit) - reduced
        }
        
        # Baseball tracking data
        self.baseball_tracking = {}     # Track detected baseballs across frames

        # Increased thread pool for better performance
        self.save_executor = ThreadPoolExecutor(max_workers=16)
        self.capture_threads = []
        self.display_threads = []
        self.display_queues = {}
        
        # Frame synchronization for side-by-side saving
        self.frame_sync_queue = Queue(maxsize=100)
        self.sync_thread = None

        self.stats = {
            'frames_captured': {},
            'frames_saved': {},
            'start_time': time.time(),
            # Add real-time FPS tracking
            'last_capture_count': {},
            'last_save_count': {},
            'last_fps_time': time.time()
        }

        print("🚀 DUAL HIGH-SPEED CAPTURE SYSTEM - 500 FPS OPTIMIZED")
        print("=" * 50)
        print(f"📺 Display: {'ENABLED' if enable_display else 'DISABLED (for max performance)'}")
        print(f"💾 Saving every {save_every_n_frames} frame(s)")
        print("📸 Side-by-side saving: ENABLED")
        print(f"⚾ Baseball detection: {'ENABLED' if enable_baseball_detection else 'DISABLED'}")

    def create_output_directories(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # No longer creating separate camera folders - all images saved together

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

    def configure_camera(self, cam, i):
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

        # Set custom ROI for specific camera positioning
        CUSTOM_WIDTH = 720   # Custom width in pixels
        CUSTOM_HEIGHT = 150  # Custom height in pixels
        CENTER_IMAGE = False  # Center the image region on sensor
        CUSTOM_OFFSET_X = 0    # Custom X offset (start position)
        CUSTOM_OFFSET_Y = 196  # Custom Y offset (start position)
        
        # Get dimension nodes
        width_node = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
        height_node = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
        offset_x_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetX"))
        offset_y_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetY"))
        
        # First, reset to maximum sensor size to get valid ranges
        if PySpin.IsAvailable(width_node) and PySpin.IsWritable(width_node):
            max_width = width_node.GetMax()
            width_node.SetValue(max_width)
        if PySpin.IsAvailable(height_node) and PySpin.IsWritable(height_node):
            max_height = height_node.GetMax()
            height_node.SetValue(max_height)
        if PySpin.IsAvailable(offset_x_node) and PySpin.IsWritable(offset_x_node):
            offset_x_node.SetValue(0)
        if PySpin.IsAvailable(offset_y_node) and PySpin.IsWritable(offset_y_node):
            offset_y_node.SetValue(0)
        
        # Calculate valid dimensions and offsets
        max_width = width_node.GetMax() if PySpin.IsAvailable(width_node) else 1920
        max_height = height_node.GetMax() if PySpin.IsAvailable(height_node) else 1200
        
        # Ensure our custom dimensions fit within sensor limits
        target_width = min(CUSTOM_WIDTH, max_width)
        target_height = min(CUSTOM_HEIGHT, max_height)
        
        # Calculate maximum valid offsets
        max_offset_x = max_width - target_width
        max_offset_y = max_height - target_height
        
        # Apply safe offset values
        safe_offset_x = min(CUSTOM_OFFSET_X, max_offset_x)
        safe_offset_y = min(CUSTOM_OFFSET_Y, max_offset_y)
        
        # Set dimensions first
        if PySpin.IsAvailable(width_node) and PySpin.IsWritable(width_node):
            width_node.SetValue(target_width)
        if PySpin.IsAvailable(height_node) and PySpin.IsWritable(height_node):
            height_node.SetValue(target_height)
        
        # Then set offsets (must be done after width/height)
        if PySpin.IsAvailable(offset_x_node) and PySpin.IsWritable(offset_x_node):
            offset_x_node.SetValue(safe_offset_x)
        if PySpin.IsAvailable(offset_y_node) and PySpin.IsWritable(offset_y_node):
            offset_y_node.SetValue(safe_offset_y)

        # Disable auto exposure
        auto_exposure = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
        if PySpin.IsAvailable(auto_exposure) and PySpin.IsWritable(auto_exposure):
            exposure_off = auto_exposure.GetEntryByName("Off")
            if PySpin.IsAvailable(exposure_off) and PySpin.IsReadable(exposure_off):
                auto_exposure.SetIntValue(exposure_off.GetValue())

        # Set exposure time for visible image (optimized from diagnostic test)
        exposure_time_node = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
        if PySpin.IsAvailable(exposure_time_node) and PySpin.IsWritable(exposure_time_node):
            target_exposure = 2000.0  # 2000µs - proven to work from brightness test
            exposure_time_node.SetValue(min(target_exposure, exposure_time_node.GetMax()))

        # Disable auto gain and set manual gain for visibility
        auto_gain = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
        if PySpin.IsAvailable(auto_gain) and PySpin.IsWritable(auto_gain):
            gain_off = auto_gain.GetEntryByName("Off")
            if PySpin.IsAvailable(gain_off) and PySpin.IsReadable(gain_off):
                auto_gain.SetIntValue(gain_off.GetValue())

        # Set optimal gain for visibility (from diagnostic test)
        gain_node = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
        if PySpin.IsAvailable(gain_node) and PySpin.IsWritable(gain_node):
            target_gain = 20.0  # 20dB - proven to work from brightness test
            gain_node.SetValue(min(target_gain, gain_node.GetMax()))

        # Maximize throughput
        throughput_node = PySpin.CIntegerPtr(nodemap.GetNode("DeviceLinkThroughputLimit"))
        if PySpin.IsAvailable(throughput_node) and PySpin.IsWritable(throughput_node):
            throughput_node.SetValue(throughput_node.GetMax())  # Use maximum throughput

        # Enable frame rate control
        acq_frame_rate_enable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
        if PySpin.IsAvailable(acq_frame_rate_enable) and PySpin.IsWritable(acq_frame_rate_enable):
            acq_frame_rate_enable.SetValue(True)

        # Set target frame rate to 500 FPS
        acq_frame_rate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
        if PySpin.IsAvailable(acq_frame_rate) and PySpin.IsWritable(acq_frame_rate):
            target_fps = 500.0
            max_fps = acq_frame_rate.GetMax()
            final_fps = min(target_fps, max_fps)
            acq_frame_rate.SetValue(final_fps)
            print(f'✅ Camera {i+1}: FPS set to {final_fps:.2f} (max available: {max_fps:.2f})')

        # Optimize buffer handling
        stream_buffer_count = PySpin.CIntegerPtr(nodemap.GetNode("StreamBufferCountMode"))
        if PySpin.IsAvailable(stream_buffer_count) and PySpin.IsWritable(stream_buffer_count):
            manual_mode = stream_buffer_count.GetEntryByName("Manual")
            if PySpin.IsAvailable(manual_mode) and PySpin.IsReadable(manual_mode):
                stream_buffer_count.SetIntValue(manual_mode.GetValue())

        stream_buffer_count_manual = PySpin.CIntegerPtr(nodemap.GetNode("StreamBufferCountManual"))
        if PySpin.IsAvailable(stream_buffer_count_manual) and PySpin.IsWritable(stream_buffer_count_manual):
            stream_buffer_count_manual.SetValue(16)  # Increase buffer count

        # Get actual applied values for verification
        actual_width = width_node.GetValue() if PySpin.IsAvailable(width_node) else target_width
        actual_height = height_node.GetValue() if PySpin.IsAvailable(height_node) else target_height
        actual_offset_x = offset_x_node.GetValue() if PySpin.IsAvailable(offset_x_node) else safe_offset_x
        actual_offset_y = offset_y_node.GetValue() if PySpin.IsAvailable(offset_y_node) else safe_offset_y
        
        print(f'🛠️ Camera {i+1} configured:')
        print(f'   📐 ROI: {actual_width}x{actual_height} at offset ({actual_offset_x}, {actual_offset_y})')
        print(f'   📏 Requested: {CUSTOM_WIDTH}x{CUSTOM_HEIGHT} at offset ({CUSTOM_OFFSET_X}, {CUSTOM_OFFSET_Y})')
        print(f'   📏 Applied: {actual_width}x{actual_height} at offset ({actual_offset_x}, {actual_offset_y})')
        print(f'   ⚡ Settings: {exposure_time_node.GetValue():.1f}µs exposure, {acq_frame_rate.GetValue():.2f} FPS')

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
                self.stats['frames_saved'][i+1] = 0
                if self.enable_display:
                    self.display_queues[i+1] = Queue(maxsize=2)  # Smaller queue

            return len(self.cameras) > 0

        except PySpin.SpinnakerException as ex:
            print(f'❌ Error enumerating cameras: {ex}')
            return False

    def stop(self):
        self.running = False
        print('🛑 Shutting down...')
        time.sleep(0.5)  # Shorter wait time
        self.save_executor.shutdown(wait=True)

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

    def detect_baseball(self, frame, camera_id):
        """
        Detect multiple WHITE baseballs against any background (grass, etc.)
        """
        if not self.enable_baseball_detection:
            return [], frame
            
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            
            annotated_frame = frame.copy()
            detected_baseballs = []
            
            # Debug mode setup
            debug_mode = self.baseball_detection_params.get('debug_mode', False)
            if debug_mode:
                print(f"\n{'='*60}")
                print(f"🔍 DEBUGGING CAMERA {camera_id} DETECTION")
                print(f"{'='*60}")
                print(f"Frame size: {gray.shape}")
                print(f"Frame brightness stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
            
            # Step 1: White object detection with debugging
            threshold_value = self.baseball_detection_params['threshold_value']
            _, white_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            
            if debug_mode:
                print(f"Threshold value: {threshold_value}")
                white_pixels = np.sum(white_mask > 0)
                total_pixels = white_mask.shape[0] * white_mask.shape[1]
                white_percentage = (white_pixels / total_pixels) * 100
                print(f"White pixels: {white_pixels}/{total_pixels} ({white_percentage:.1f}%)")
                
                # Save threshold image for visual inspection
                timestamp_str = time.strftime("%H%M%S")
                debug_filename = f"debug_cam{camera_id}_threshold_{timestamp_str}.jpg"
                debug_path = os.path.join(self.output_dir, debug_filename)
                cv2.imwrite(debug_path, white_mask)
                print(f"💾 Saved threshold debug image: {debug_filename}")
            
            # Minimal cleanup to preserve small balls
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            white_mask_cleaned = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            
            if debug_mode:
                # Save cleaned mask
                cleaned_filename = f"debug_cam{camera_id}_cleaned_{timestamp_str}.jpg"
                cleaned_path = os.path.join(self.output_dir, cleaned_filename)
                cv2.imwrite(cleaned_path, white_mask_cleaned)
                print(f"💾 Saved cleaned debug image: {cleaned_filename}")
            
            # Find contours
            contours, _ = cv2.findContours(white_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if debug_mode:
                print(f"Found {len(contours)} contours after morphological cleaning")
            
                # Create contour visualization
                contour_debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(contour_debug, contours, -1, (0, 255, 0), 1)
                contour_filename = f"debug_cam{camera_id}_contours_{timestamp_str}.jpg"
                contour_path = os.path.join(self.output_dir, contour_filename)
                cv2.imwrite(contour_path, contour_debug)
                print(f"💾 Saved contour debug image: {contour_filename}")
            
            # Step 2: Detailed analysis of each candidate
            rejection_stats = {
                'too_small_area': 0,
                'too_large_area': 0,
                'too_small_radius': 0,
                'too_large_radius': 0,
                'poor_circularity': 0,
                'too_dim': 0,
                'low_confidence': 0,
                'accepted': 0
            }
            
            candidate_details = []
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # Get bounding circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Basic measurements
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Brightness analysis
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, center, max(1, radius), 255, -1)
                mean_brightness = cv2.mean(gray, mask)[0]
                
                # Store candidate details
                candidate_info = {
                    'id': i+1,
                    'center': center,
                    'area': area,
                    'radius': radius,
                    'circularity': circularity,
                    'brightness': mean_brightness,
                    'rejected_reason': None
                }
                
                # Apply filters with detailed tracking
                if area < self.baseball_detection_params['min_area']:
                    candidate_info['rejected_reason'] = f"area too small ({area:.1f} < {self.baseball_detection_params['min_area']})"
                    rejection_stats['too_small_area'] += 1
                elif area > self.baseball_detection_params['max_area']:
                    candidate_info['rejected_reason'] = f"area too large ({area:.1f} > {self.baseball_detection_params['max_area']})"
                    rejection_stats['too_large_area'] += 1
                elif radius < self.baseball_detection_params['min_radius']:
                    candidate_info['rejected_reason'] = f"radius too small ({radius} < {self.baseball_detection_params['min_radius']})"
                    rejection_stats['too_small_radius'] += 1
                elif radius > self.baseball_detection_params['max_radius']:
                    candidate_info['rejected_reason'] = f"radius too large ({radius} > {self.baseball_detection_params['max_radius']})"
                    rejection_stats['too_large_radius'] += 1
                elif circularity < self.baseball_detection_params['min_circularity']:
                    candidate_info['rejected_reason'] = f"poor circularity ({circularity:.3f} < {self.baseball_detection_params['min_circularity']})"
                    rejection_stats['poor_circularity'] += 1
                elif mean_brightness < self.baseball_detection_params['min_brightness']:
                    candidate_info['rejected_reason'] = f"too dim ({mean_brightness:.1f} < {self.baseball_detection_params['min_brightness']})"
                    rejection_stats['too_dim'] += 1
                else:
                    # Calculate confidence
                    brightness_score = mean_brightness / 255.0
                    shape_score = min(1.0, circularity * 2.0)
                    size_score = min(1.0, area / 500.0)
                    confidence = (brightness_score * 0.4 + shape_score * 0.3 + size_score * 0.3)
                    
                    candidate_info['confidence'] = confidence
                    
                    if confidence < self.baseball_detection_params['min_confidence']:
                        candidate_info['rejected_reason'] = f"low confidence ({confidence:.3f} < {self.baseball_detection_params['min_confidence']})"
                        rejection_stats['low_confidence'] += 1
                    else:
                        # ACCEPTED!
                        rejection_stats['accepted'] += 1
                        candidate_info['rejected_reason'] = None
                        
                baseball_info = {
                    'center': center,
                    'radius': radius,
                    'area': area,
                    'circularity': circularity,
                            'brightness': mean_brightness,
                    'confidence': confidence,
                    'camera_id': camera_id
                }
                detected_baseballs.append(baseball_info)
                
                candidate_details.append(candidate_info)
            
            # Debug output - summary statistics
            if debug_mode:
                print(f"\n📊 DETECTION SUMMARY FOR CAMERA {camera_id}:")
                print(f"  Total candidates: {len(contours)}")
                print(f"  ✅ Accepted: {rejection_stats['accepted']}")
                print(f"  ❌ Rejected breakdown:")
                for reason, count in rejection_stats.items():
                    if reason != 'accepted' and count > 0:
                        print(f"    - {reason.replace('_', ' ').title()}: {count}")
                
                # Show details of accepted detections
                if detected_baseballs:
                    print(f"\n🎯 ACCEPTED DETECTIONS:")
                    for i, ball in enumerate(detected_baseballs):
                        print(f"  Ball {i+1}: center({ball['center'][0]},{ball['center'][1]}) "
                              f"r={ball['radius']} area={ball['area']:.1f} "
                              f"circ={ball['circularity']:.3f} bright={ball['brightness']:.1f} "
                              f"conf={ball['confidence']:.3f}")
                
                # Show some rejected candidates for analysis
                rejected_candidates = [c for c in candidate_details if c['rejected_reason']]
                if rejected_candidates:
                    print(f"\n🚫 NOTABLE REJECTED CANDIDATES (first 5):")
                    for candidate in rejected_candidates[:5]:
                        print(f"  #{candidate['id']}: {candidate['rejected_reason']} "
                              f"center({candidate['center'][0]},{candidate['center'][1]}) "
                              f"r={candidate['radius']} area={candidate['area']:.1f}")
                
                print(f"{'='*60}\n")
            
            # Sort by confidence and annotate frame
            detected_baseballs.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Limit maximum detections to prevent excessive false positives
            max_detections = self.baseball_detection_params['max_detections_per_camera']
            if len(detected_baseballs) > max_detections:
                if debug_mode:
                    print(f"⚠️  WARNING: Too many detections ({len(detected_baseballs)}) - limiting to top {max_detections}")
                detected_baseballs = detected_baseballs[:max_detections]
            
            # Final summary
            if debug_mode:
                expected_count = self.baseball_detection_params['expected_ball_count']
                actual_count = len(detected_baseballs)
                print(f"\n🎯 FINAL DETECTION RESULTS:")
                print(f"  Expected balls total: {expected_count}")
                print(f"  Detected in Camera {camera_id}: {actual_count}")
                if actual_count > expected_count:
                    print(f"  ⚠️  OVER-DETECTION: {actual_count - expected_count} extra detections")
                elif actual_count < expected_count // 2:  # Less than half expected per camera
                    print(f"  ⚠️  UNDER-DETECTION: Significantly fewer than expected")
                else:
                    print(f"  ✅ Detection count seems reasonable")
            
            # Draw all detected baseballs with better labeling
            for i, ball in enumerate(detected_baseballs):
                center = ball['center']
                radius = ball['radius']
                confidence = ball['confidence']
                
                # Color coding by confidence
                if confidence > 0.7:
                    color = (0, 255, 0)  # Green - high confidence
                elif confidence > 0.5:
                    color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    color = (0, 165, 255)  # Orange - low confidence
                
                # Draw detection circle
                cv2.circle(annotated_frame, center, radius, color, 2)
                
                # Draw center point
                cv2.circle(annotated_frame, center, 2, (0, 0, 255), -1)
                
                # Add detailed label
                label_y = max(center[1] - radius - 5, 15)
                cv2.putText(annotated_frame, f"#{i+1}", (center[0] - 10, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(annotated_frame, f"{confidence:.2f}", (center[0] - 15, label_y + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Add summary text to frame
            height, width = annotated_frame.shape[:2]
            summary_text = f"Detected: {len(detected_baseballs)} objects"
            cv2.putText(annotated_frame, summary_text, (10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return detected_baseballs, annotated_frame
            
        except Exception as ex:
            print(f'❌ Error in baseball detection for camera {camera_id}: {ex}')
            import traceback
            traceback.print_exc()
            return [], frame

    def _calculate_local_contrast(self, gray_frame, center, radius):
        """Calculate local contrast around detected region"""
        try:
            x, y = center
            # Create mask for the baseball region
            mask_inner = np.zeros(gray_frame.shape, dtype=np.uint8)
            cv2.circle(mask_inner, (x, y), radius, 255, -1)
            
            # Create mask for surrounding area
            mask_outer = np.zeros(gray_frame.shape, dtype=np.uint8)
            cv2.circle(mask_outer, (x, y), radius * 2, 255, -1)
            mask_surround = cv2.bitwise_xor(mask_outer, mask_inner)
            
            # Calculate mean brightness of baseball and surrounding area
            baseball_brightness = cv2.mean(gray_frame, mask_inner)[0]
            surround_brightness = cv2.mean(gray_frame, mask_surround)[0]
            
            # Return contrast ratio
            if surround_brightness > 0:
                return baseball_brightness / surround_brightness
            else:
                return 1.0
                
        except:
            return 1.0
    
    def _check_stripe_proximity(self, gray_frame, center, radius):
        """Check if detection is near red/black stripes (indicating it's on the board)"""
        try:
            x, y = center
            
            # Expand search area around the baseball
            search_radius = radius * 3
            
            # Define search region
            x1 = max(0, x - search_radius)
            x2 = min(gray_frame.shape[1], x + search_radius)
            y1 = max(0, y - search_radius)
            y2 = min(gray_frame.shape[0], y + search_radius)
            
            # Extract region around the detection
            region = gray_frame[y1:y2, x1:x2]
            
            if region.size == 0:
                return False
            
            # Look for stripe patterns (alternating dark and light bands)
            # Calculate horizontal intensity profile to detect stripes
            horizontal_profile = np.mean(region, axis=0)
            
            if len(horizontal_profile) < 10:
                return False
            
            # Look for variance in intensity (stripes create high variance)
            intensity_variance = np.var(horizontal_profile)
            
            # Also check vertical profile for horizontal stripes
            vertical_profile = np.mean(region, axis=1)
            vertical_variance = np.var(vertical_profile)
            
            # If there's significant variance in either direction, likely near stripes
            stripe_threshold = self.baseball_detection_params.get('stripe_threshold', 200)
            has_stripes = intensity_variance > stripe_threshold or vertical_variance > stripe_threshold
            
            if self.baseball_detection_params.get('debug_mode', False) and has_stripes:
                print(f"Stripe variance - H: {intensity_variance:.1f}, V: {vertical_variance:.1f}")
            
            return has_stripes
            
        except:
            return True  # Default to accepting if check fails
    
    def _calculate_baseball_confidence(self, area, circularity, solidity, contrast):
        """Calculate overall confidence score for baseball detection"""
        try:
            # Normalize each metric to 0-1 scale
            area_score = min(1.0, area / 500.0)  # Normalize area
            circularity_score = min(1.0, circularity)
            solidity_score = min(1.0, solidity)
            contrast_score = min(1.0, contrast / 2.0)  # Normalize contrast
            
            # Weighted combination
            confidence = (
                area_score * 0.2 +
                circularity_score * 0.4 +
                solidity_score * 0.3 +
                contrast_score * 0.1
            )
            
            return confidence
            
        except:
            return 0.5

    def _calculate_circle_confidence(self, gray_frame, x, y, radius):
        """Calculate confidence score for detected circle based on edge strength"""
        try:
            # Create mask for circle perimeter
            mask = np.zeros(gray_frame.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), radius, 255, 2)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray_frame, 50, 150)
            
            # Calculate edge pixels along circle perimeter
            circle_edges = cv2.bitwise_and(edges, mask)
            edge_count = np.sum(circle_edges > 0)
            perimeter = 2 * np.pi * radius
            
            # Confidence is ratio of edge pixels to total perimeter
            confidence = edge_count / perimeter if perimeter > 0 else 0
            return min(confidence, 1.0)
            
        except:
            return 0.5  # Default confidence
    
    def _estimate_velocity(self, camera_id, x, y):
        """Estimate baseball velocity based on position history"""
        try:
            current_time = time.time()
            
            # Initialize tracking for camera if not exists
            if camera_id not in self.baseball_tracking:
                self.baseball_tracking[camera_id] = []
            
            # Add current position
            self.baseball_tracking[camera_id].append({
                'position': (x, y),
                'timestamp': current_time
            })
            
            # Keep only recent positions (last 0.1 seconds)
            self.baseball_tracking[camera_id] = [
                p for p in self.baseball_tracking[camera_id] 
                if current_time - p['timestamp'] < 0.1
            ]
            
            # Calculate velocity if we have enough points
            positions = self.baseball_tracking[camera_id]
            if len(positions) >= 2:
                # Use first and last position for velocity calculation
                start_pos = positions[0]
                end_pos = positions[-1]
                
                dx = end_pos['position'][0] - start_pos['position'][0]
                dy = end_pos['position'][1] - start_pos['position'][1]
                dt = end_pos['timestamp'] - start_pos['timestamp']
                
                if dt > 0:
                    velocity_pixels_per_sec = np.sqrt(dx*dx + dy*dy) / dt
                    # Convert to approximate real-world velocity (assuming calibration)
                    # This is a rough estimate - you'd need proper calibration for accuracy
                    velocity_mph = velocity_pixels_per_sec * 0.1  # Rough conversion factor
                    return f"Vel: {velocity_mph:.1f} mph"
            
            return None
            
        except:
            return None

    def start_capture(self):
        if not self.cameras:
            print('❌ No cameras available!')
            return False

        self.running = True
        
        # Start frame synchronization thread for side-by-side saving
        self.sync_thread = threading.Thread(target=self.frame_sync_thread)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        
        # Start capture threads
        for camera_info in self.cameras:
            thread = threading.Thread(target=self.capture_thread, args=(camera_info,))
            thread.daemon = True
            thread.start()
            self.capture_threads.append(thread)

        # Start display threads only if enabled
        if self.enable_display:
            for camera_info in self.cameras:
                camera_id = camera_info['camera_id']
                thread = threading.Thread(target=self.display_thread, args=(camera_id,))
                thread.daemon = True
                thread.start()
                self.display_threads.append(thread)

        print('\n🚀 500 FPS HIGH-SPEED CAPTURE ACTIVE')
        if self.enable_display:
            print('🎮 Press "q" or ESC in any window to exit')
        else:
            print('🎮 Press Ctrl+C to exit')

        try:
            while self.running:
                time.sleep(2)  # More frequent updates
                current_time = time.time()
                runtime = current_time - self.stats['start_time']
                time_interval = current_time - self.stats['last_fps_time']
                
                # Clear previous lines and print FPS prominently
                print('\n' + '='*60)
                print('📊 CAMERA FPS PERFORMANCE:')
                print('='*60)
                
                total_combined_saves = 0
                for camera_id in self.stats['frames_captured']:
                    # Current totals
                    captured = self.stats['frames_captured'][camera_id]
                    saved = self.stats['frames_saved'][camera_id]
                    total_combined_saves = max(total_combined_saves, saved)  # Combined saves are the same for all cameras
                    
                    # Calculate average FPS (like your original)
                    avg_capture_fps = captured / runtime if runtime > 0 else 0
                    avg_save_fps = saved / runtime if runtime > 0 else 0
                    
                    # Calculate real-time FPS (like SpinView)
                    if camera_id in self.stats['last_capture_count']:
                        recent_captures = captured - self.stats['last_capture_count'][camera_id]
                        recent_saves = saved - self.stats['last_save_count'][camera_id]
                        realtime_capture_fps = recent_captures / time_interval if time_interval > 0 else 0
                        realtime_save_fps = recent_saves / time_interval if time_interval > 0 else 0
                    else:
                        realtime_capture_fps = avg_capture_fps
                        realtime_save_fps = avg_save_fps
                    
                    print(f'📷 Camera {camera_id}:')
                    print(f'   🚀 Capture: {realtime_capture_fps:.1f} FPS (real-time) | {avg_capture_fps:.1f} FPS (avg)')
                    print(f'   📋 Total: {captured} frames captured')
                    
                    # Update last counts for next iteration
                    self.stats['last_capture_count'][camera_id] = captured
                    self.stats['last_save_count'][camera_id] = saved
                
                # Show combined save statistics
                print(f'🔄 Combined Images: {total_combined_saves} side-by-side files saved')
                print('='*60)
                
                self.stats['last_fps_time'] = current_time
        except KeyboardInterrupt:
            print('\n🛑 Stopping capture system...')
            self.stop()

        return True

    def capture_thread(self, camera_info):
        cam = camera_info['camera']
        camera_id = camera_info['camera_id']
        frame_count = 0

        try:
            cam.BeginAcquisition()
            print(f'🚀 Started high-speed capture for camera {camera_id}')

            while self.running:
                try:
                    # Reduced timeout for faster operation
                    image_result = cam.GetNextImage(50)  # 50ms timeout
                    if not image_result.IsIncomplete():
                        frame_count += 1
                        self.stats['frames_captured'][camera_id] += 1
                        
                        # Save every single frame
                        if True:  # Save every frame regardless of count
                            # Get image data without unnecessary copying
                            image_data = image_result.GetNDArray()
                            timestamp = datetime.now()
                            
                            # Convert to BGR for detection processing
                            if len(image_data.shape) == 2:
                                bgr_frame = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
                            else:
                                bgr_frame = image_data.copy()
                            
                            # Apply baseball detection to get both original and annotated versions
                            detected_baseballs, annotated_frame = self.detect_baseball(bgr_frame.copy(), camera_id)
                            
                            # Send frame to sync queue for side-by-side saving (include both versions)
                            frame_info = {
                                'camera_id': camera_id,
                                'frame_data_original': image_data.copy(),  # Original grayscale/color
                                'frame_data_annotated': annotated_frame.copy(),  # Annotated BGR
                                'detections': detected_baseballs,
                                'frame_count': frame_count,
                                'timestamp': timestamp
                            }
                            try:
                                self.frame_sync_queue.put_nowait(frame_info)
                            except:
                                pass  # Skip if queue is full

                        # Handle display only if enabled and queue not full
                        if self.enable_display and camera_id in self.display_queues:
                            if not self.display_queues[camera_id].full():
                                display_frame = image_result.GetNDArray().copy()
                                if len(display_frame.shape) == 2:
                                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
                                
                                # Apply baseball detection to display frame
                                detected_baseballs, display_frame = self.detect_baseball(display_frame, camera_id)
                                
                                # Add enhanced labels for display
                                height, width = display_frame.shape[:2]
                                timestamp_display = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                
                                # Camera label
                                cv2.putText(display_frame, f'Camera {camera_id}', 
                                          (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                # Frame count
                                cv2.putText(display_frame, f'Frame: {frame_count}', 
                                          (width - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                # Timestamp
                                cv2.putText(display_frame, timestamp_display, 
                                          (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                                
                                # Baseball detection count
                                if detected_baseballs:
                                    detection_text = f'Baseballs: {len(detected_baseballs)}'
                                    cv2.putText(display_frame, detection_text, 
                                              (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                                
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

    def frame_sync_thread(self):
        """Thread to save every frame individually - no synchronization required"""
        while self.running:
            try:
                # Get frame from queue
                frame_info = self.frame_sync_queue.get(timeout=1.0)
                camera_id = frame_info['camera_id']
                
                # Save every frame immediately - no synchronization
                should_save = True
                
                # If save_only_when_detected is enabled, check for ball detection
                if self.baseball_detection_params.get('save_only_when_detected', False):
                    detections = frame_info.get('detections', [])
                    should_save = len(detections) > 0
                    if should_save:
                        print(f"🎯 Detection triggered save: Camera {camera_id} found {len(detections)} object(s)")
                
                if should_save:
                    # Save individual frame (both original and annotated)
                    self.save_individual_frame(frame_info)
                
            except:
                continue

    def save_individual_frame(self, frame_info):
        """Save individual frame from one camera with both original and annotated versions"""
        try:
            camera_id = frame_info['camera_id']
            frame_data_original = frame_info['frame_data_original']
            frame_data_annotated = frame_info['frame_data_annotated']
            frame_count = frame_info['frame_count']
            timestamp = frame_info['timestamp']
            detections = frame_info.get('detections', [])
            
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Convert original to BGR if grayscale
            if len(frame_data_original.shape) == 2:
                original_bgr = cv2.cvtColor(frame_data_original, cv2.COLOR_GRAY2BGR)
            else:
                original_bgr = frame_data_original.copy()
            
            # Add labels to original image
            height, width = original_bgr.shape[:2]
            cv2.putText(original_bgr, f'Camera {camera_id} - Original', 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(original_bgr, f'Frame: {frame_count}', 
                       (width - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            time_text = timestamp.strftime("%H:%M:%S.%f")[:-3]
            cv2.putText(original_bgr, time_text, 
                       (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Add labels to annotated image
            annotated_copy = frame_data_annotated.copy()
            cv2.putText(annotated_copy, f'Camera {camera_id} - Detected', 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_copy, f'Frame: {frame_count}', 
                       (width - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Detection count
            if detections:
                detection_text = f'Objects: {len(detections)}'
                cv2.putText(annotated_copy, detection_text, 
                           (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.putText(annotated_copy, time_text, 
                       (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Create side-by-side image
            side_by_side = np.hstack([original_bgr, annotated_copy])
            
            # Save image
            detection_count = len(detections)
            filename = f"cam{camera_id}_frame{frame_count:06d}_{detection_count}objs_{timestamp_str}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, side_by_side, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            print(f"💾 Saved: {filename} ({detection_count} detections)")
            
            # Update stats
            self.stats['frames_saved'][camera_id] += 1
                
        except Exception as ex:
            print(f'❌ Error saving individual frame: {ex}')

    def save_combined_frame_4_panel(self, frame_dict):
        """Save frames from multiple cameras in 4-panel layout: Original1, Annotated1, Original2, Annotated2"""
        try:
            # Sort frames by camera_id to ensure consistent order
            sorted_frames = sorted(frame_dict.items(), key=lambda x: x[0])
            
            all_panels = []
            timestamp_str = None
            total_detections = 0
            
            for camera_id, frame_info in sorted_frames:
                frame_data_original = frame_info['frame_data_original']
                frame_data_annotated = frame_info['frame_data_annotated']
                frame_count = frame_info['frame_count']
                timestamp = frame_info['timestamp']
                detections = frame_info.get('detections', [])
                total_detections += len(detections)
                
                if timestamp_str is None:
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                
                # Convert original to BGR if grayscale
                if len(frame_data_original.shape) == 2:
                    original_bgr = cv2.cvtColor(frame_data_original, cv2.COLOR_GRAY2BGR)
                else:
                    original_bgr = frame_data_original.copy()
                
                # Add labels to original image
                height, width = original_bgr.shape[:2]
                cv2.putText(original_bgr, f'Camera {camera_id} - Original', 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(original_bgr, f'Frame: {frame_count}', 
                           (width - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                time_text = timestamp.strftime("%H:%M:%S.%f")[:-3]
                cv2.putText(original_bgr, time_text, 
                           (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Add labels to annotated image
                annotated_copy = frame_data_annotated.copy()
                cv2.putText(annotated_copy, f'Camera {camera_id} - Detected', 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated_copy, f'Frame: {frame_count}', 
                           (width - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Detection count
                if detections:
                    detection_text = f'Objects: {len(detections)}'
                    cv2.putText(annotated_copy, detection_text, 
                               (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                cv2.putText(annotated_copy, time_text, 
                           (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Add both panels for this camera
                all_panels.extend([original_bgr, annotated_copy])
            
            # Create 2x2 grid layout if we have 2 cameras (4 panels total)
            if len(all_panels) == 4:
                # Top row: Camera 1 original, Camera 1 annotated
                top_row = np.hstack([all_panels[0], all_panels[1]])
                # Bottom row: Camera 2 original, Camera 2 annotated  
                bottom_row = np.hstack([all_panels[2], all_panels[3]])
                # Combine rows
                combined_image = np.vstack([top_row, bottom_row])
            elif len(all_panels) == 2:
                # Single camera: side by side
                combined_image = np.hstack(all_panels)
            else:
                # Fallback: horizontal layout
                combined_image = np.hstack(all_panels)
            
            # Save combined image
            filename = f"4panel_detection_{total_detections}objs_{timestamp_str}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, combined_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            print(f"💾 Saved 4-panel image: {filename} ({total_detections} total detections)")
            
            # Update stats for all cameras
            for camera_id, _ in sorted_frames:
                self.stats['frames_saved'][camera_id] += 1
                
        except Exception as ex:
            print(f'❌ Error saving 4-panel combined frame: {ex}')

    def save_combined_frame(self, frame_dict):
        """Legacy save method - redirects to 4-panel version"""
        self.save_combined_frame_4_panel(frame_dict)

    def save_frame_async(self, frame_data, camera_id, frame_number, timestamp):
        """Legacy method - now handled by frame_sync_thread"""
        pass

    def display_thread(self, camera_id):
        if not self.enable_display:
            return
            
        window_name = f'Camera {camera_id} - 500 FPS'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Calculate proper aspect ratio for 720x150 ROI
        # Scale to fit screen while maintaining aspect ratio
        roi_width, roi_height = 720, 150
        aspect_ratio = roi_width / roi_height  # 4.8:1 ratio
        
        # Set display window to maintain aspect ratio (make it wider, not square)
        display_height = 200  # Reasonable height for display
        display_width = int(display_height * aspect_ratio)  # Calculate width to maintain ratio
        
        cv2.resizeWindow(window_name, display_width, display_height)
        print(f'📺 Started display for camera {camera_id} - {display_width}x{display_height} (maintaining {aspect_ratio:.1f}:1 aspect ratio)')

        while self.running:
            try:
                if not self.display_queues[camera_id].empty():
                    frame = self.display_queues[camera_id].get_nowait()
                    cv2.imshow(window_name, frame)
                
                # Check for exit key with minimal delay
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:  # 'q' or ESC
                    self.stop()
                    break
            except:
                pass

        cv2.destroyWindow(window_name)

def main():
    # For maximum performance, disable display and save every frame
    # capture_system = DualHighSpeedCapture(enable_display=False, save_every_n_frames=1, enable_baseball_detection=True)
    
    # For testing with display and baseball detection - save every frame
    capture_system = DualHighSpeedCapture(enable_display=True, save_every_n_frames=1, enable_baseball_detection=True)
    
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
