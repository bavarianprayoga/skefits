import cv2
import logging
import time
from typing import Optional, Dict, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self):
        self.cap = None
        self.frame_dims: Optional[Tuple[int, int]] = None
        self.last_frame_time = 0
        self.frame_delay = 0.033  # ~30 FPS
        self.camera_index = 0

    def init_camera(self) -> bool:
        """Initialize camera with basic configuration"""
        try:
            logger.info(f"Trying camera index {self.camera_index}")
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.warning(f"Failed to open camera {self.camera_index}")
                return False
                
            # Set basic camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Validate camera works
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                self.frame_dims = (frame.shape[1], frame.shape[0])
                logger.info(f"Successfully initialized camera {self.camera_index}")
                return True
                
            logger.warning(f"Camera {self.camera_index} failed validation")
            return False
            
        except Exception as e:
            logger.error(f"Error initializing camera {self.camera_index}: {str(e)}")
            return False
            
    def get_frame(self) -> Optional[Dict]:
        """Get current frame if available"""
        if not self.cap or not self.cap.isOpened():
            return None
            
        try:
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_delay:
                return None
                
            # Get new frame
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                logger.warning("Failed to read frame")
                return None
                
            self.last_frame_time = current_time
            
            return {
                'frame': frame,
                'timestamp': current_time,
                'dims': self.frame_dims
            }
            
        except Exception as e:
            logger.error(f"Failed to get frame: {str(e)}")
            return None
            
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
            self.frame_dims = None
            logger.info("Camera released")