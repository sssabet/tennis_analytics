"""
Tennis Court Keypoint Detector
Based on: https://github.com/MuhammadMoinFaisal/tennis_analysis
Uses fine-tuned ResNet-50 to detect 14 tennis court keypoints
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import os
from typing import Optional, List, Tuple, Dict


def create_resnet50_keypoint_model(num_keypoints=14):
    """
    Create a ResNet-50 model for keypoint detection
    Based on: https://github.com/MuhammadMoinFaisal/tennis_analysis
    Outputs 14 keypoints (28 values: x,y for each)
    """
    # Use ResNet-50 as backbone
    model = models.resnet50(weights=None)
    
    # Replace final FC layer to output 28 values (14 keypoints * 2 coordinates)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_keypoints * 2)
    
    return model


class CourtKeypointDetector:
    """
    Detects tennis court keypoints using ResNet-50 based deep learning model
    Based on: https://github.com/MuhammadMoinFaisal/tennis_analysis
    Falls back to classical CV if model not available
    """
    def __init__(self, model_path: Optional[str] = None, device: str = None):
        """
        Args:
            model_path: Path to pretrained ResNet-50 keypoints model (keypoints_model_50.pth)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        self.model = None
        self.use_model = False
        
        # Standard ImageNet transforms for ResNet
        self.transforms = None
        
        if model_path and os.path.exists(model_path):
            try:
                print(f"Loading ResNet-50 court keypoint model from {model_path}...")
                self.model = create_resnet50_keypoint_model(num_keypoints=14)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
                self.model.eval().to(self.device)
                self.use_model = True
                print("ResNet-50 court keypoint model loaded successfully!")
                
                # Set up transforms
                try:
                    from torchvision import transforms
                    self.transforms = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                    ])
                except Exception as e:
                    print(f"Warning: Could not set up transforms: {e}")
                    
            except Exception as e:
                print(f"Warning: Could not load court model: {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to classical CV approach")
        
        self.last_keypoints = None
        
    def detect(self, frame: np.ndarray) -> Tuple[Optional[List[Tuple]], str]:
        """
        Detect court keypoints
        
        Returns:
            (corners, source) where corners is list of 4 corner points
            and source is 'model', 'cv', or 'kalman'
        """
        h, w = frame.shape[:2]
        
        if self.use_model and self.model is not None:
            # Use deep learning model
            keypoints = self._detect_with_model(frame)
            if keypoints:
                # Extract 4 corners from keypoints
                corners = self._keypoints_to_corners(keypoints, w, h)
                if corners:
                    return corners, 'model'
        
        # Fallback to classical CV (Hough lines)
        corners = self._detect_with_cv(frame)
        if corners:
            return corners, 'cv'
        
        # Use Kalman prediction if available
        if self.last_keypoints:
            return self.last_keypoints, 'kalman'
        
        return None, 'none'
    
    def _detect_with_model(self, frame: np.ndarray) -> Optional[List[Tuple]]:
        """
        Detect keypoints using ResNet-50 model
        Model outputs 28 values (x,y for 14 keypoints) in 224x224 image coordinates
        """
        try:
            h, w = frame.shape[:2]
            input_size = 224  # Model input size
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            if self.transforms:
                input_tensor = self.transforms(frame_rgb).unsqueeze(0).to(self.device)
            else:
                # Fallback: manual preprocessing
                frame_resized = cv2.resize(frame_rgb, (input_size, input_size))
                frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
                # Normalize with ImageNet stats
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                input_tensor = ((frame_tensor - mean) / std).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Output shape: (1, 28) - 14 keypoints * 2 coordinates in 224x224 space
            output = output[0].cpu().numpy()
            
            # Parse keypoints - scale from 224x224 to original frame size
            keypoints = []
            for i in range(14):
                x_224 = output[i * 2]
                y_224 = output[i * 2 + 1]
                # Scale to original frame dimensions
                x = int(x_224 * w / input_size)
                y = int(y_224 * h / input_size)
                keypoints.append((x, y))
            
            return keypoints
            
        except Exception as e:
            print(f"ResNet model detection error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _detect_with_cv(self, frame: np.ndarray) -> Optional[List[Tuple]]:
        """Fallback: Detect court corners using classical CV with improved method"""
        try:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive threshold for better line detection
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            
            # Detect edges
            canny = cv2.Canny(bw, 50, 150)
            
            # Find lines using HoughLinesP
            lines = cv2.HoughLinesP(canny, 1, np.pi/180, threshold=100, 
                                   minLineLength=int(min(w, h) * 0.1), 
                                   maxLineGap=20)
            
            if lines is None or len(lines) < 4:
                # Try with lower threshold
                lines = cv2.HoughLinesP(canny, 1, np.pi/180, threshold=50, 
                                       minLineLength=int(min(w, h) * 0.05), 
                                       maxLineGap=30)
            
            if lines is None or len(lines) < 2:
                return None
            
            # Find all line intersections
            intersections = []
            for i, line1 in enumerate(lines[:min(20, len(lines))]):  # Limit to avoid too many
                for line2 in lines[i+1:min(20, len(lines))]:
                    x1, y1, x2, y2 = line1[0]
                    x3, y3, x4, y4 = line2[0]
                    
                    # Calculate intersection
                    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                    if abs(denom) > 1e-6:
                        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
                        u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
                        
                        if 0 <= t <= 1 and 0 <= u <= 1:
                            px = int(x1 + t*(x2-x1))
                            py = int(y1 + t*(y2-y1))
                            # Check if within frame bounds
                            if 10 <= px < w-10 and 10 <= py < h-10:
                                intersections.append((px, py))
            
            if len(intersections) < 4:
                # Not enough intersections - use edge-based approach
                # Find corners from image edges
                corners = self._find_corners_from_edges(frame, canny)
                return corners
            
            # Cluster intersections to find 4 main corners
            intersections = np.array(intersections)
            
            # Use k-means or simple clustering to find 4 corner regions
            # For simplicity, find extreme points
            if len(intersections) >= 4:
                # Find corners based on position
                # Top-left: min(x+y)
                # Top-right: min(-x+y)  
                # Bottom-left: min(x-y)
                # Bottom-right: max(x+y)
                corners = [
                    tuple(intersections[np.argmin(intersections[:, 0] + intersections[:, 1])]),  # top-left
                    tuple(intersections[np.argmin(-intersections[:, 0] + intersections[:, 1])]),  # top-right
                    tuple(intersections[np.argmin(intersections[:, 0] - intersections[:, 1])]),  # bottom-left
                    tuple(intersections[np.argmax(intersections[:, 0] + intersections[:, 1])]),  # bottom-right
                ]
                
                # Validate corners make sense (top < bottom, left < right)
                if (corners[0][1] < corners[2][1] and corners[1][1] < corners[3][1] and
                    corners[0][0] < corners[1][0] and corners[2][0] < corners[3][0]):
                    return corners
            
            return None
            
        except Exception as e:
            return None
    
    def _find_corners_from_edges(self, frame: np.ndarray, edges: np.ndarray) -> Optional[List[Tuple]]:
        """Find corners using edge detection and contour analysis"""
        try:
            h, w = frame.shape[:2]
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find largest contour (likely the court)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) >= 4:
                # Get bounding rectangle
                x, y, w_rect, h_rect = cv2.boundingRect(approx)
                
                # Use bounding box corners
                corners = [
                    (x, y),  # top-left
                    (x + w_rect, y),  # top-right
                    (x, y + h_rect),  # bottom-left
                    (x + w_rect, y + h_rect)  # bottom-right
                ]
                return corners
            
            return None
            
        except Exception as e:
            return None
    
    def _keypoints_to_corners(self, keypoints: List[Tuple], w: int, h: int) -> Optional[List[Tuple]]:
        """
        Convert 14 keypoints to 4 corner points
        
        TennisCourtDetector keypoint layout (14 points):
        0: top-left corner
        1: top-right corner  
        2: bottom-left corner
        3: bottom-right corner
        4-13: Additional court points (service lines, etc.)
        
        We use keypoints 0-3 directly as corners.
        """
        try:
            if keypoints is None or len(keypoints) < 4:
                return None
            
            # Use first 4 keypoints as corners
            corners = [
                tuple(keypoints[0]),  # top-left
                tuple(keypoints[1]),  # top-right
                tuple(keypoints[2]),  # bottom-left  
                tuple(keypoints[3]),  # bottom-right
            ]
            
            # Validate corners are within frame bounds
            valid = True
            for corner in corners:
                if corner[0] < 0 or corner[0] >= w or corner[1] < 0 or corner[1] >= h:
                    valid = False
                    break
            
            if not valid:
                return None
            
            # Validate corners make geometric sense
            # Top corners should be above bottom corners
            if not (corners[0][1] < corners[2][1] and corners[1][1] < corners[3][1]):
                # Try to reorder corners based on Y coordinate
                all_pts = list(corners)
                all_pts.sort(key=lambda p: p[1])  # Sort by Y
                top_pts = sorted(all_pts[:2], key=lambda p: p[0])  # Top 2, sorted by X
                bottom_pts = sorted(all_pts[2:], key=lambda p: p[0])  # Bottom 2, sorted by X
                corners = [top_pts[0], top_pts[1], bottom_pts[0], bottom_pts[1]]
            
            self.last_keypoints = corners
            return corners
            
        except Exception as e:
            return None


# Simplified version that works without model weights
def detect_court_keypoints_cv(frame: np.ndarray) -> Optional[List[Tuple]]:
    """
    Detect court corners using classical computer vision
    This is a fallback when the deep learning model is not available
    """
    detector = CourtKeypointDetector()
    corners, source = detector.detect(frame)
    return corners


if __name__ == '__main__':
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python TennisCourtKeypointDetector.py <video_path> [model_path]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    detector = CourtKeypointDetector(model_path=model_path)
    
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        corners, source = detector.detect(frame)
        
        if corners:
            print(f"Detected corners ({source}): {corners}")
            for i, corner in enumerate(corners):
                cv2.circle(frame, corner, 10, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (corner[0]+15, corner[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw court outline
            cv2.line(frame, corners[0], corners[1], (0, 0, 255), 2)
            cv2.line(frame, corners[2], corners[3], (0, 0, 255), 2)
            cv2.line(frame, corners[0], corners[2], (0, 0, 255), 2)
            cv2.line(frame, corners[1], corners[3], (0, 0, 255), 2)
        
        cv2.imshow('Court Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

