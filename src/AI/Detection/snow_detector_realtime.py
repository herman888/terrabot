"""
Terrabot Real-Time Snow Detection with Mainframe Integration
Adaptive CV-based snow detection with socket communication to C++ mainframe
"""

import cv2
import numpy as np
import socket
import json
import time
import threading
from dataclasses import dataclass, asdict
from typing import List, Tuple


@dataclass
class SnowDetection:
    """Snow detection data structure matching C++ mainframe"""
    x: float              # Center X coordinate (normalized 0-1)
    y: float              # Center Y coordinate (normalized 0-1)
    width: float          # Bounding box width (normalized 0-1)
    height: float         # Bounding box height (normalized 0-1)
    confidence: float     # Detection confidence (0-1)
    timestamp: float      # Unix timestamp
    
    def to_dict(self):
        return asdict(self)


class AdaptiveSnowDetector:
    """
    Adaptive computer vision-based snow detector
    Uses dynamic thresholding and sky exclusion
    """
    
    def __init__(self, sky_exclusion_ratio: float = 0.3, min_area_ratio: float = 0.001):
        """
        Initialize detector
        
        Args:
            sky_exclusion_ratio: Fraction of top image to exclude (sky)
            min_area_ratio: Minimum contour area as fraction of total image
        """
        self.sky_exclusion_ratio = sky_exclusion_ratio
        self.min_area_ratio = min_area_ratio
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
    def detect_snow(self, frame: np.ndarray) -> List[SnowDetection]:
        """
        Detect snow regions in frame
        
        Args:
            frame: Input BGR image
            
        Returns:
            List of snow detections
        """
        height, width = frame.shape[:2]
        timestamp = time.time()
        
        # Define ROI - exclude sky
        sky_threshold = int(height * self.sky_exclusion_ratio)
        roi = frame[sky_threshold:, :]
        
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_roi)
        
        # Calculate adaptive thresholds
        v_mean = np.mean(v)
        v_std = np.std(v)
        s_mean = np.mean(s)
        s_std = np.std(s)
        
        # Dynamic threshold calculation
        min_value = max(200, int(v_mean + 0.8 * v_std))
        max_saturation = min(50, int(s_mean + 0.5 * s_std))
        
        # Create snow mask
        lower_snow = np.array([0, 0, min_value])
        upper_snow = np.array([180, max_saturation, 255])
        snow_mask_roi = cv2.inRange(hsv_roi, lower_snow, upper_snow)
        
        # Morphological operations
        snow_mask_roi = cv2.morphologyEx(snow_mask_roi, cv2.MORPH_OPEN, self.kernel, iterations=1)
        snow_mask_roi = cv2.morphologyEx(snow_mask_roi, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(snow_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and convert to detections
        min_contour_area = (width * height) * self.min_area_ratio
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Adjust Y coordinate for sky exclusion
            y += sky_threshold
            
            # Calculate normalized center and dimensions
            center_x = (x + w / 2) / width
            center_y = (y + h / 2) / height
            norm_width = w / width
            norm_height = h / height
            
            # Calculate confidence based on area and brightness
            roi_crop = v[max(0, y-sky_threshold):min(v.shape[0], y+h-sky_threshold), 
                        max(0, x):min(v.shape[1], x+w)]
            if roi_crop.size > 0:
                avg_brightness = np.mean(roi_crop)
                confidence = min(1.0, avg_brightness / 255.0)
            else:
                confidence = 0.5
            
            detection = SnowDetection(
                x=float(center_x),
                y=float(center_y),
                width=float(norm_width),
                height=float(norm_height),
                confidence=float(confidence),
                timestamp=timestamp
            )
            detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[SnowDetection]) -> np.ndarray:
        """Draw detections on frame for visualization"""
        output = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw sky exclusion line
        sky_y = int(height * self.sky_exclusion_ratio)
        cv2.line(output, (0, sky_y), (width, sky_y), (255, 0, 0), 2)
        cv2.putText(output, "Sky Region (Ignored)", (10, sky_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw detections
        for det in detections:
            # Convert normalized to pixel coordinates
            x1 = int((det.x - det.width / 2) * width)
            y1 = int((det.y - det.height / 2) * height)
            x2 = int((det.x + det.width / 2) * width)
            y2 = int((det.y + det.height / 2) * height)
            
            # Color based on confidence
            color = (0, 255, 0) if det.confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"Snow {det.confidence:.2f}"
            cv2.putText(output, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Info text
        info = f"Detections: {len(detections)}"
        cv2.putText(output, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return output


class MainframeClient:
    """
    Socket client to communicate with C++ mainframe
    """
    
    def __init__(self, host: str = "localhost", port: int = 5555):
        """
        Initialize mainframe client
        
        Args:
            host: Mainframe server host
            port: Mainframe server port
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self, timeout: int = 5) -> bool:
        """
        Connect to mainframe server
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if connected successfully
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"[MainframeClient] Connected to mainframe at {self.host}:{self.port}")
            return True
        except (socket.timeout, ConnectionRefusedError) as e:
            print(f"[MainframeClient] Connection failed: {e}")
            self.connected = False
            return False
    
    def send_detections(self, detections: List[SnowDetection]) -> bool:
        """
        Send detections to mainframe
        
        Args:
            detections: List of snow detections
            
        Returns:
            True if sent successfully
        """
        if not self.connected or not self.socket:
            return False
        
        try:
            # Serialize detections to JSON
            data = json.dumps([det.to_dict() for det in detections])
            message = data.encode('utf-8') + b'\n'
            
            self.socket.sendall(message)
            return True
        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"[MainframeClient] Send failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from mainframe"""
        if self.socket:
            self.socket.close()
            self.connected = False
            print("[MainframeClient] Disconnected from mainframe")


class RealTimeSnowDetectionSystem:
    """
    Main system integrating camera, detector, and mainframe communication
    """
    
    def __init__(self, 
                 camera_id: int = 0,
                 enable_display: bool = True,
                 enable_mainframe: bool = True,
                 mainframe_host: str = "localhost",
                 mainframe_port: int = 5555):
        """
        Initialize real-time detection system
        
        Args:
            camera_id: Camera device ID
            enable_display: Show live video with detections
            enable_mainframe: Enable mainframe communication
            mainframe_host: Mainframe server host
            mainframe_port: Mainframe server port
        """
        self.camera_id = camera_id
        self.enable_display = enable_display
        self.enable_mainframe = enable_mainframe
        
        self.detector = AdaptiveSnowDetector()
        self.camera = None
        self.mainframe_client = MainframeClient(mainframe_host, mainframe_port) if enable_mainframe else None
        
        self.running = False
        self.fps = 0
        self.frame_times = []
        
    def initialize(self) -> bool:
        """Initialize camera and mainframe connection"""
        print("[DetectionSystem] Initializing...")
        
        # Initialize camera
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            print(f"[DetectionSystem] ERROR: Cannot open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        print("[DetectionSystem] Camera initialized")
        
        # Connect to mainframe
        if self.enable_mainframe and self.mainframe_client:
            if not self.mainframe_client.connect():
                print("[DetectionSystem] WARNING: Mainframe connection failed, continuing without it")
        
        return True
    
    def run(self):
        """Main detection loop"""
        if not self.initialize():
            return
        
        self.running = True
        print("[DetectionSystem] Starting detection loop...")
        print("Press 'q' to quit, 'r' to reconnect to mainframe")
        
        try:
            while self.running:
                start_time = time.time()
                
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("[DetectionSystem] Failed to capture frame")
                    break
                
                # Run detection
                detections = self.detector.detect_snow(frame)
                
                # Send to mainframe
                if self.enable_mainframe and self.mainframe_client and detections:
                    if not self.mainframe_client.send_detections(detections):
                        print("[DetectionSystem] Failed to send to mainframe")
                
                # Update FPS
                frame_time = time.time() - start_time
                self.frame_times.append(frame_time)
                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)
                self.fps = len(self.frame_times) / sum(self.frame_times)
                
                # Display
                if self.enable_display:
                    height, width = frame.shape[:2]
                    annotated = self.detector.draw_detections(frame, detections)
                    
                    # Add FPS and connection status
                    fps_text = f"FPS: {self.fps:.1f}"
                    cv2.putText(annotated, fps_text, (width - 120, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if self.mainframe_client:
                        status_color = (0, 255, 0) if self.mainframe_client.connected else (0, 0, 255)
                        status_text = "Mainframe: Connected" if self.mainframe_client.connected else "Mainframe: Disconnected"
                        cv2.putText(annotated, status_text, (10, height - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
                    
                    cv2.imshow("Terrabot Snow Detection", annotated)
                    
                    # Handle keyboard
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[DetectionSystem] Quit signal received")
                        break
                    elif key == ord('r') and self.mainframe_client:
                        print("[DetectionSystem] Reconnecting to mainframe...")
                        self.mainframe_client.connect()
                
        except KeyboardInterrupt:
            print("[DetectionSystem] Interrupted by user")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean up resources"""
        print("[DetectionSystem] Shutting down...")
        self.running = False
        
        if self.camera:
            self.camera.release()
        
        if self.enable_display:
            cv2.destroyAllWindows()
        
        if self.mainframe_client:
            self.mainframe_client.disconnect()
        
        print("[DetectionSystem] Shutdown complete")


def main():
    """Entry point"""
    print("=" * 50)
    print("  Terrabot Real-Time Snow Detection")
    print("  With Mainframe Integration")
    print("=" * 50)
    
    # Configuration
    CAMERA_ID = 0
    ENABLE_DISPLAY = True
    ENABLE_MAINFRAME = True
    MAINFRAME_HOST = "localhost"
    MAINFRAME_PORT = 5555
    
    # Create and run system
    system = RealTimeSnowDetectionSystem(
        camera_id=CAMERA_ID,
        enable_display=ENABLE_DISPLAY,
        enable_mainframe=ENABLE_MAINFRAME,
        mainframe_host=MAINFRAME_HOST,
        mainframe_port=MAINFRAME_PORT
    )
    
    system.run()


if __name__ == "__main__":
    main()
