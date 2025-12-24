"""
Ball Tracker Fusion Module
Combines TrackNet and CourtSide (YOLO) detections using Kalman Filter
for robust ball tracking with interpolation.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Optional, Tuple, List

from BallDetection import BallDetector


class CourtSideDetector:
    """
    Ball detector using CourtSide YOLOv11 model
    """
    def __init__(self, model_path: str = 'weights/courtside_yolo.pt', conf_threshold: float = 0.25, device: str = None):
        from ultralytics import YOLO
        import torch
        
        # Auto-detect GPU if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.ball_class_id = 1  # tennis_ball class
        
        print(f"CourtSide YOLO using device: {self.device}")
        
    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """
        Detect tennis ball in frame
        Returns: (x, y, confidence) or None if not detected
        """
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False, device=self.device)
        
        best_detection = None
        best_conf = 0
        
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls == self.ball_class_id:
                conf = float(box.conf[0])
                if conf > best_conf:
                    # Get center of bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    best_detection = (cx, cy, conf)
                    best_conf = conf
                    
        return best_detection


class BallKalmanFilter:
    """
    Kalman Filter for ball position tracking and prediction
    State: [x, y, vx, vy] (position and velocity)
    """
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1],  # vy = vy
        ])
        
        # Measurement matrix (we only observe x, y)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        
        # Measurement noise
        self.kf.R = np.eye(2) * 10
        
        # Process noise
        self.kf.Q = np.eye(4) * 0.1
        self.kf.Q[2, 2] = 1  # Higher uncertainty for velocity
        self.kf.Q[3, 3] = 1
        
        # Initial covariance
        self.kf.P *= 100
        
        self.initialized = False
        self.frames_without_detection = 0
        self.max_frames_without_detection = 10
        
    def initialize(self, x: float, y: float):
        """Initialize filter with first detection"""
        self.kf.x = np.array([[x], [y], [0], [0]])
        self.initialized = True
        self.frames_without_detection = 0
        
    def predict(self) -> Tuple[float, float]:
        """Predict next position"""
        self.kf.predict()
        return float(self.kf.x[0, 0]), float(self.kf.x[1, 0])
    
    def update(self, x: float, y: float):
        """Update with measurement"""
        self.kf.update(np.array([[x], [y]]))
        self.frames_without_detection = 0
        
    def get_position(self) -> Tuple[float, float]:
        """Get current estimated position"""
        return float(self.kf.x[0, 0]), float(self.kf.x[1, 0])
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current estimated velocity"""
        return float(self.kf.x[2, 0]), float(self.kf.x[3, 0])
    
    def increment_no_detection(self):
        """Called when no detection in current frame"""
        self.frames_without_detection += 1
        
    def is_tracking_valid(self) -> bool:
        """Check if tracking is still valid"""
        return self.frames_without_detection < self.max_frames_without_detection


class BallTrackerFusion:
    """
    Fuses TrackNet and CourtSide detections using Kalman Filter
    """
    def __init__(
        self, 
        tracknet_weights: str = 'weights/tracknet.pth',
        courtside_weights: str = 'weights/courtside_yolo.pt',
        tracknet_weight: float = 0.6,
        courtside_weight: float = 0.4,
        max_detection_distance: float = 100,
        min_courtside_conf: float = 0.15,
        kalman_gate_base_px: float = 120.0,
        kalman_gate_vel_mult: float = 2.0,
        kalman_gate_max_px: float = 350.0,
        device: str = None,
    ):
        """
        Args:
            tracknet_weights: Path to TrackNet weights
            courtside_weights: Path to CourtSide YOLO weights
            tracknet_weight: Weight for TrackNet detections in fusion (0-1)
            courtside_weight: Weight for CourtSide detections in fusion (0-1)
            max_detection_distance: Max distance between detections to consider them same ball
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Initialize detectors
        self.tracknet = BallDetector(tracknet_weights, out_channels=2, device=device)
        self.courtside = CourtSideDetector(courtside_weights, device=device)
        
        # Fusion weights
        self.tracknet_weight = tracknet_weight
        self.courtside_weight = courtside_weight
        self.max_detection_distance = max_detection_distance
        self.min_courtside_conf = min_courtside_conf
        
        # Outlier gating vs Kalman prediction (prevents "teleport" jumps)
        self.kalman_gate_base_px = float(kalman_gate_base_px)
        self.kalman_gate_vel_mult = float(kalman_gate_vel_mult)
        self.kalman_gate_max_px = float(kalman_gate_max_px)
        
        # Kalman Filter for tracking
        self.kalman = BallKalmanFilter()
        
        # Storage for trajectory
        self.trajectory: List[Tuple[Optional[int], Optional[int], str]] = []
        self.frame_count = 0
        
    def detect_ball(self, frame: np.ndarray) -> Tuple[Optional[int], Optional[int], str]:
        """
        Detect ball using both detectors and fuse results
        
        Returns:
            (x, y, source) where source is 'tracknet', 'courtside', 'fused', 'kalman', or 'none'
        """
        self.frame_count += 1
        
        # Get TrackNet detection
        self.tracknet.detect_ball(frame)
        tracknet_pos = None
        if self.tracknet.xy_coordinates[-1][0] is not None:
            tracknet_pos = (
                int(self.tracknet.xy_coordinates[-1][0]),
                int(self.tracknet.xy_coordinates[-1][1])
            )
        
        # Get CourtSide detection
        courtside_result = self.courtside.detect(frame)
        courtside_pos = None
        courtside_conf = 0
        if courtside_result is not None:
            courtside_pos = (courtside_result[0], courtside_result[1])
            courtside_conf = courtside_result[2]
            # Drop low-confidence YOLO detections (common source of far-away outliers)
            if courtside_conf < self.min_courtside_conf:
                courtside_pos = None
                courtside_conf = 0
        
        # Fuse detections
        final_pos, source = self._fuse_detections(tracknet_pos, courtside_pos, courtside_conf)
        
        # Update Kalman filter with outlier gating
        if not self.kalman.initialized:
            if final_pos is not None:
                self.kalman.initialize(final_pos[0], final_pos[1])
        else:
            # Predict first (advance state)
            self.kalman.predict()
            kx, ky = self.kalman.get_position()
            
            if final_pos is not None:
                # Gate measurement vs predicted position.
                # Allowed distance grows with current estimated velocity.
                vx, vy = self.kalman.get_velocity()
                vel_mag = float(np.sqrt(vx * vx + vy * vy))
                gate = min(self.kalman_gate_max_px, self.kalman_gate_base_px + self.kalman_gate_vel_mult * vel_mag)
                dist_meas = float(np.sqrt((final_pos[0] - kx) ** 2 + (final_pos[1] - ky) ** 2))
                
                if dist_meas <= gate:
                    self.kalman.update(final_pos[0], final_pos[1])
                else:
                    # Reject outlier measurement; keep smooth Kalman prediction
                    self.kalman.increment_no_detection()
                    
                    # AUTO-RESET: if we've lost tracking for too long, reset initialization
                    # so we can snap back to a valid detection far away.
                    if not self.kalman.is_tracking_valid():
                        self.kalman.initialized = False
                        # Try to re-initialize immediately with this measurement if it looks plausible
                        # (e.g. if we have a high-confidence TrackNet/Fusion hit)
                        self.kalman.initialize(final_pos[0], final_pos[1])
                        source = source  # keep original source
                    else:
                        final_pos = (int(kx), int(ky))
                        source = 'kalman'
            else:
                self.kalman.increment_no_detection()
                if self.kalman.is_tracking_valid():
                    final_pos = (int(kx), int(ky))
                    source = 'kalman'
                else:
                    # LOST: reset initialization so we can snap to next detection
                    self.kalman.initialized = False
        
        # Store in trajectory
        if final_pos is not None:
            self.trajectory.append((final_pos[0], final_pos[1], source))
        else:
            self.trajectory.append((None, None, 'none'))
            
        return final_pos[0] if final_pos else None, final_pos[1] if final_pos else None, source
    
    def _fuse_detections(
        self, 
        tracknet_pos: Optional[Tuple[int, int]], 
        courtside_pos: Optional[Tuple[int, int]],
        courtside_conf: float
    ) -> Tuple[Optional[Tuple[int, int]], str]:
        """
        Fuse detections from both models
        """
        # Both detected
        if tracknet_pos is not None and courtside_pos is not None:
            # Check if detections are close enough to be the same ball
            dist = np.sqrt(
                (tracknet_pos[0] - courtside_pos[0])**2 + 
                (tracknet_pos[1] - courtside_pos[1])**2
            )
            
            if dist <= self.max_detection_distance:
                # Weighted average of positions
                # Adjust weights based on CourtSide confidence
                adjusted_courtside_weight = self.courtside_weight * courtside_conf
                total_weight = self.tracknet_weight + adjusted_courtside_weight
                
                fused_x = int(
                    (tracknet_pos[0] * self.tracknet_weight + 
                     courtside_pos[0] * adjusted_courtside_weight) / total_weight
                )
                fused_y = int(
                    (tracknet_pos[1] * self.tracknet_weight + 
                     courtside_pos[1] * adjusted_courtside_weight) / total_weight
                )
                return (fused_x, fused_y), 'fused'
            else:
                # Detections too far apart - use the one closer to Kalman prediction
                if self.kalman.initialized and self.kalman.is_tracking_valid():
                    kx, ky = self.kalman.get_position()
                    dist_tracknet = np.sqrt((tracknet_pos[0] - kx)**2 + (tracknet_pos[1] - ky)**2)
                    dist_courtside = np.sqrt((courtside_pos[0] - kx)**2 + (courtside_pos[1] - ky)**2)
                    
                    if dist_tracknet < dist_courtside:
                        return tracknet_pos, 'tracknet'
                    else:
                        return courtside_pos, 'courtside'
                else:
                    # No Kalman prediction, prefer TrackNet (trained specifically for tennis)
                    return tracknet_pos, 'tracknet'
        
        # Only TrackNet detected
        elif tracknet_pos is not None:
            return tracknet_pos, 'tracknet'
        
        # Only CourtSide detected
        elif courtside_pos is not None:
            return courtside_pos, 'courtside'
        
        # Neither detected
        return None, 'none'
    
    def get_trajectory(self) -> List[Tuple[Optional[int], Optional[int], str]]:
        """Get full trajectory"""
        return self.trajectory
    
    def get_interpolated_trajectory(self) -> List[Tuple[int, int]]:
        """
        Get trajectory with gaps filled by interpolation
        """
        if not self.trajectory:
            return []
        
        # Find valid points
        valid_points = []
        for i, (x, y, source) in enumerate(self.trajectory):
            if x is not None and y is not None:
                valid_points.append((i, x, y))
        
        if len(valid_points) < 2:
            return [(x, y) for x, y, _ in self.trajectory if x is not None]
        
        # Interpolate gaps
        result = []
        for i in range(len(self.trajectory)):
            x, y, _ = self.trajectory[i]
            if x is not None and y is not None:
                result.append((x, y))
            else:
                # Find surrounding valid points
                prev_point = None
                next_point = None
                
                for idx, px, py in valid_points:
                    if idx < i:
                        prev_point = (idx, px, py)
                    elif idx > i and next_point is None:
                        next_point = (idx, px, py)
                        break
                
                if prev_point and next_point:
                    # Linear interpolation
                    t = (i - prev_point[0]) / (next_point[0] - prev_point[0])
                    interp_x = int(prev_point[1] + t * (next_point[1] - prev_point[1]))
                    interp_y = int(prev_point[2] + t * (next_point[2] - prev_point[2]))
                    result.append((interp_x, interp_y))
                elif prev_point:
                    result.append((prev_point[1], prev_point[2]))
                elif next_point:
                    result.append((next_point[1], next_point[2]))
        
        return result
    
    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """Get current estimated velocity from Kalman filter"""
        if self.kalman.initialized:
            return self.kalman.get_velocity()
        return None
    
    def reset(self):
        """Reset tracker state"""
        self.kalman = BallKalmanFilter()
        self.trajectory = []
        self.frame_count = 0


# Example usage and testing
if __name__ == '__main__':
    import cv2
    
    # Initialize fusion tracker
    tracker = BallTrackerFusion()
    
    # Test on video
    cap = cv2.VideoCapture('video/rally1.mp4')
    
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        x, y, source = tracker.detect_ball(frame)
        
        if x is not None:
            print(f"Frame {frame_num}: Ball at ({x}, {y}) - Source: {source}")
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, source, (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Ball Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_num += 1
        if frame_num > 100:  # Limit for testing
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Get interpolated trajectory
    trajectory = tracker.get_interpolated_trajectory()
    print(f"\nTotal trajectory points: {len(trajectory)}")

