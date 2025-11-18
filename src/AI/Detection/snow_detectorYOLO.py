"""
Terrabot Snow Detection Module
Real-time snow detection using YOLO (You Only Look Once)
Processes camera feed and identifies snow patches for removal
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import json
import socket
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import threading
from queue import Queue


@dataclass
class SnowDetection:
    """Data structure for snow detection results"""
    x: float                # Center X coordinate (normalized 0-1)
    y: float                # Center Y coordinate (normalized 0-1)
    width: float            # Bounding box width (normalized 0-1)
    height: float           # Bounding box height (normalized 0-1)
    confidence: float       # Detection confidence score (0-1)
    distance: float         # Estimated distance to snow patch in meters
    timestamp: float        # Unix timestamp of detection
    
    def to_dict(self):
        """Convert detection to dictionary for serialization"""
        return asdict(self)
    
    def to_json(self):
        """Convert detection to JSON string"""
        return json.dumps(self.to_dict())


class DistanceEstimator:
    """
    Estimates distance to detected objects using camera parameters
    and bounding box dimensions
    """
    
    def __init__(self, focal_length: float = 500.0, known_width: float = 0.5):
        """
        Initialize distance estimator
        
        Args:
            focal_length: Camera focal length in pixels (calibrate with known objects)
            known_width: Approximate real-world width of typical snow patch in meters
        """
        self.focal_length = focal_length
        self.known_width = known_width
        
    def estimate_distance(self, bbox_width_pixels: float, image_width: int) -> float:
        """
        Estimate distance using pinhole camera model
        
        Distance = (Known_Width * Focal_Length) / Perceived_Width
        
        Args:
            bbox_width_pixels: Width of bounding box in pixels
            image_width: Total image width in pixels
            
        Returns:
            Estimated distance in meters
        """
        if bbox_width_pixels == 0:
            return float('inf')
        
        # Calculate distance using similar triangles
        distance = (self.known_width * self.focal_length) / bbox_width_pixels
        
        return distance
    
    def calibrate_focal_length(self, known_distance: float, 
                               measured_width_pixels: float) -> None:
        """
        Calibrate focal length using object at known distance
        
        Args:
            known_distance: Actual distance to object in meters
            measured_width_pixels: Measured width in pixels
        """
        self.focal_length = (measured_width_pixels * known_distance) / self.known_width
        print(f"[DistanceEstimator] Focal length calibrated to: {self.focal_length:.2f} pixels")


class YOLOSnowDetector:
    """
    Main snow detection class using YOLO for real-time processing
    """
    
    def __init__(self, 
                 model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 device: str = "cuda"):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights (use pre-trained or custom trained)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for non-maximum suppression
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Initialize YOLO model
        print(f"[YOLODetector] Loading YOLO model from {model_path}...")
        self.model = None  # TODO: Uncomment when YOLO is installed
        # self.model = YOLO(model_path)
        # self.model.to(device)
        
        # Initialize distance estimator
        self.distance_estimator = DistanceEstimator()
        
        # Performance tracking
        self.fps = 0
        self.frame_times = []
        
        print(f"[YOLODetector] Initialized with confidence={confidence_threshold}, device={device}")
    
    def detect_snow(self, frame: np.ndarray) -> List[SnowDetection]:
        """
        Detect snow patches in a single frame
        
        Args:
            frame: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            List of SnowDetection objects
        """
        start_time = time.time()
        detections = []
        
        # TODO: Uncomment when YOLO is installed
        # Run YOLO inference
        # results = self.model(frame, 
        #                     conf=self.confidence_threshold,
        #                     iou=self.iou_threshold,
        #                     verbose=False)
        
        # Placeholder detection for testing (remove when YOLO is active)
        # Simulate detection in center of frame
        h, w = frame.shape[:2]
        dummy_detection = SnowDetection(
            x=0.5,
            y=0.5,
            width=0.2,
            height=0.2,
            confidence=0.85,
            distance=2.5,
            timestamp=time.time()
        )
        detections.append(dummy_detection)
        
        # TODO: Process actual YOLO results
        # for result in results:
        #     boxes = result.boxes
        #     for box in boxes:
        #         # Extract bounding box coordinates
        #         x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        #         confidence = float(box.conf[0])
        #         
        #         # Calculate normalized center and dimensions
        #         center_x = ((x1 + x2) / 2) / w
        #         center_y = ((y1 + y2) / 2) / h
        #         bbox_width = (x2 - x1) / w
        #         bbox_height = (y2 - y1) / h
        #         
        #         # Estimate distance
        #         distance = self.distance_estimator.estimate_distance(x2 - x1, w)
        #         
        #         detection = SnowDetection(
        #             x=float(center_x),
        #             y=float(center_y),
        #             width=float(bbox_width),
        #             height=float(bbox_height),
        #             confidence=confidence,
        #             distance=distance,
        #             timestamp=time.time()
        #         )
        #         detections.append(detection)
        
        # Update FPS
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        self.fps = len(self.frame_times) / sum(self.frame_times)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[SnowDetection]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detections to draw
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        for det in detections:
            # Convert normalized coordinates to pixel coordinates
            x1 = int((det.x - det.width / 2) * w)
            y1 = int((det.y - det.height / 2) * h)
            x2 = int((det.x + det.width / 2) * w)
            y2 = int((det.y + det.height / 2) * h)
            
            # Draw bounding box
            color = (0, 255, 0) if det.confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence and distance
            label = f"Snow {det.confidence:.2f} | {det.distance:.2f}m"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(annotated, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated


class DetectionStreamServer:
    """
    Streams detection results to C++ mainframe via socket communication
    """
    
    def __init__(self, host: str = "localhost", port: int = 5555):
        """
        Initialize detection stream server
        
        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.socket = None
        self.client_socket = None
        self.running = False
        self.detection_queue = Queue(maxsize=10)
        
    def start(self):
        """Start the server and listen for connections"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        self.running = True
        
        print(f"[DetectionServer] Listening on {self.host}:{self.port}")
        
        # Start sender thread
        threading.Thread(target=self._sender_thread, daemon=True).start()
    
    def _sender_thread(self):
        """Thread to accept connections and send detections"""
        while self.running:
            try:
                print("[DetectionServer] Waiting for C++ mainframe connection...")
                self.client_socket, addr = self.socket.accept()
                print(f"[DetectionServer] Connected to {addr}")
                
                while self.running:
                    # Get detection from queue
                    detections = self.detection_queue.get()
                    
                    # Serialize and send
                    data = json.dumps([det.to_dict() for det in detections])
                    message = data.encode('utf-8') + b'\n'
                    
                    try:
                        self.client_socket.sendall(message)
                    except (BrokenPipeError, ConnectionResetError):
                        print("[DetectionServer] Connection lost")
                        break
                        
            except Exception as e:
                print(f"[DetectionServer] Error: {e}")
                time.sleep(1)
    
    def send_detections(self, detections: List[SnowDetection]):
        """Add detections to send queue"""
        if not self.detection_queue.full():
            self.detection_queue.put(detections)
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.socket:
            self.socket.close()


class RealTimeDetectionSystem:
    """
    Main system that integrates camera, detector, and communication
    """
    
    def __init__(self, 
                 camera_id: int = 0,
                 model_path: str = "yolov8n.pt",
                 enable_display: bool = True,
                 enable_server: bool = True):
        """
        Initialize real-time detection system
        
        Args:
            camera_id: Camera device ID (0 for default camera)
            model_path: Path to YOLO model
            enable_display: Show live video feed with detections
            enable_server: Enable socket server for C++ communication
        """
        self.camera_id = camera_id
        self.enable_display = enable_display
        self.enable_server = enable_server
        
        # Initialize components
        self.detector = YOLOSnowDetector(model_path=model_path)
        self.camera = None
        self.server = DetectionStreamServer() if enable_server else None
        
        self.running = False
        
    def initialize(self) -> bool:
        """Initialize camera and server"""
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
        
        # Initialize server
        if self.enable_server and self.server:
            self.server.start()
        
        return True
    
    def run(self):
        """Main detection loop"""
        if not self.initialize():
            return
        
        self.running = True
        print("[DetectionSystem] Starting detection loop...")
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("[DetectionSystem] Failed to capture frame")
                    break
                
                # Run detection
                detections = self.detector.detect_snow(frame)
                
                # Send to mainframe
                if self.enable_server and self.server and detections:
                    self.server.send_detections(detections)
                
                # Display results
                if self.enable_display:
                    annotated = self.detector.draw_detections(frame, detections)
                    cv2.imshow("Terrabot Snow Detection", annotated)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[DetectionSystem] Quit signal received")
                        break
                    elif key == ord('c'):
                        # Calibration mode
                        print("[DetectionSystem] Calibration mode - not implemented")
                
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
        
        if self.server:
            self.server.stop()
        
        print("[DetectionSystem] Shutdown complete")


def main():
    """Entry point for snow detection system"""
    print("=" * 50)
    print("  Terrabot Snow Detection System")
    print("  YOLO Real-Time Processing")
    print("=" * 50)
    
    # Configuration
    CAMERA_ID = 0
    MODEL_PATH = "yolov8n.pt"  # TODO: Replace with custom trained model
    ENABLE_DISPLAY = True
    ENABLE_SERVER = True
    
    # Create and run detection system
    system = RealTimeDetectionSystem(
        camera_id=CAMERA_ID,
        model_path=MODEL_PATH,
        enable_display=ENABLE_DISPLAY,
        enable_server=ENABLE_SERVER
    )
    
    system.run()


if __name__ == "__main__":
    main()
