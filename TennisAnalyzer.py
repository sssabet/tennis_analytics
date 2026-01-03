"""
Tennis Analyzer - Unified video processing with all detections and annotations
Combines: Court detection, Ball tracking (fusion), Player tracking (YOLOv8-pose), Homography
         Shot Classification, Mini Court Visualization, Pose Skeleton Overlay
Outputs: Single annotated video with optional picture-in-picture court view
"""

import cv2
import numpy as np
from numpy import pi, ones, zeros, uint8, where, cos, sin, float32
import mediapipe as mp
from typing import Optional, Tuple, List, Dict
import os
from pathlib import Path
from ultralytics import YOLO

from TraceHeader import findIntersection, calculatePixels
from BallTrackerFusion import BallTrackerFusion
from BallMapping import euclideanDistance, withinCircle
from CourtMapping import (
    courtMap,
    showLines,
    showPoint,
    heightP,
    widthP,
    givePoint,
    yOffset,
    courtHeight,
    courtWidth,
)
from TennisCourtKeypointDetector import CourtKeypointDetector
from AudioAnalyzer import AudioAnalyzer, draw_audio_overlay, draw_audio_waveform

# YOLOv8 Pose keypoint indices
YOLO_POSE_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    # Face
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    # Upper body
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    # Torso
    (5, 11),
    (6, 12),
    (11, 12),
    # Lower body
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


class TennisAnalyzer:
    """
    Unified tennis video analyzer with all detection capabilities
    """

    def __init__(
        self,
        tracknet_weights: str = "weights/tracknet.pth",
        courtside_weights: str = "weights/courtside_yolo.pt",
        device: str = None,
        show_court_overlay: bool = True,
        show_ball: bool = True,
        show_players: bool = True,
        show_trajectory: bool = True,
        show_pose_skeleton: bool = True,
        show_court_keypoints: bool = True,
        pip_size: float = 0.3,  # Picture-in-picture size ratio
    ):
        """
        Args:
            tracknet_weights: Path to TrackNet weights
            courtside_weights: Path to CourtSide YOLO weights
            device: Device to use ('cuda', 'cpu', or None for auto)
            show_court_overlay: Show court lines overlay
            show_ball: Show ball detection
            show_players: Show player tracking
            show_trajectory: Show ball trajectory trail
            show_pose_skeleton: Show YOLOv8 pose skeleton overlay
            show_court_keypoints: Show detected court keypoints
            pip_size: Size of picture-in-picture court view (0-1)
        """
        self.device = device
        self.show_court_overlay = show_court_overlay
        self.show_ball = show_ball
        self.show_players = show_players
        self.show_trajectory = show_trajectory
        self.show_pose_skeleton = show_pose_skeleton
        self.show_court_keypoints = show_court_keypoints
        self.pip_size = pip_size

        # Initialize ball tracker
        print("Initializing Ball Tracker (Fusion: TrackNet + CourtSide + Kalman)...")
        self.ball_tracker = BallTrackerFusion(
            tracknet_weights=tracknet_weights,
            courtside_weights=courtside_weights,
            device=device,
        )

        # Initialize court detector - use keypoint-based approach (TennisCourtDetector style)
        print("Initializing Court Detector (Keypoint-based + CV fallback)...")
        # Use the TennisCourtDetector model
        base_dir = Path(__file__).resolve().parent
        court_model_path = str(base_dir / "weights/court_keypoints.pth")
        self.court_detector = CourtKeypointDetector(
            model_path=court_model_path, device=device
        )

        # Initialize YOLOv8 pose estimator
        print("Initializing YOLOv8 Pose Estimator...")
        pose_device = device if device else ("cuda" if self._check_cuda() else "cpu")
        self.pose_model = YOLO(
            str(base_dir / "weights/yolov8_pose.pt")
        )  # Nano pose model for speed
        self.pose_device = pose_device

        # Keep MediaPipe as fallback
        self.mp_pose = mp.solutions.pose
        self.pose1 = self.mp_pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.25,
            min_tracking_confidence=0.25,
        )
        self.pose2 = self.mp_pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.25,
            min_tracking_confidence=0.25,
        )

        # Skeleton colors
        self.skeleton_color_p1 = (0, 255, 0)  # Green for player 1
        self.skeleton_color_p2 = (0, 165, 255)  # Orange for player 2
        self.keypoint_color = (0, 0, 255)  # Red for keypoints
        self.court_kp_color = (255, 0, 255)  # Magenta for court keypoints

        # Audio analyzer will be initialized per video
        self.audio_analyzer = None
        self.show_audio_overlay = True  # Always show audio overlay

        # State variables
        self.frame_count = 0
        self.ball_trajectory = []
        self.court_corners = None
        self.homography_matrix = None

        # Ball position history for shot detection
        self.ball_positions_history = []
        self.player_positions_history = {}
        self.ball_shot_frames = []
        self.shot_classifications = {}

        # Player smoothing
        self.player1_pos = None
        self.player2_pos = None
        self.smoothing_factor = 3

        # Colors
        self.colors = {
            "ball": (0, 255, 0),  # Green
            "ball_fused": (0, 255, 0),  # Green
            "ball_tracknet": (255, 0, 0),  # Blue
            "ball_courtside": (0, 165, 255),  # Orange
            "ball_kalman": (255, 255, 0),  # Cyan
            "trajectory": (0, 255, 255),  # Yellow
            "player1": (255, 0, 255),  # Magenta
            "player2": (255, 128, 0),  # Orange
            "court": (0, 0, 255),  # Red
            "hand_radius": (255, 0, 0),  # Blue
        }

    def _check_cuda(self):
        """Check if CUDA is available"""
        import torch

        return torch.cuda.is_available()

    def process_video(
        self,
        input_path: str,
        output_path: str,
        show_display: bool = False,
        max_frames: int = None,
    ) -> Dict:
        """
        Process video with full analysis

        Args:
            input_path: Input video path
            output_path: Output video path
            show_display: Show live preview
            max_frames: Maximum frames to process (None for all)

        Returns:
            Dictionary with processing statistics
        """
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        # Calculate output dimensions with sidebar
        sidebar_width = 300  # Width of left sidebar for overlays
        right_sidebar_width = 260  # Right sidebar for shot history / speeds
        # Sidebar panels:
        # - Court View: top-down court homography view (taller aspect)
        # - Field Homography: original camera view with court lines projected via inverse homography
        court_pip_scale = 0.75  # make the court view smaller (was effectively 1.0)
        pip_width = max(1, int((sidebar_width - 20) * court_pip_scale))
        pip_height = int(pip_width * heightP / widthP)  # maintain aspect ratio
        output_width = width + sidebar_width + right_sidebar_width
        output_height = height

        # Setup video writer
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )
        # Use mp4v for OpenCV compatibility, will re-encode to H.264 later
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        temp_output = output_path.replace('.mp4', '_temp.mp4')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (output_width, output_height))

        # Crop regions for player detection
        crop1 = self._create_crop_config(0.5, 0, 1, 0.33, 0, 0, width, height)
        crop2 = self._create_crop_config(0.83, 0, 1, 0.60, 0.40, 0, width, height)

        # Processing stats
        stats = {
            "total_frames": total_frames,
            "ball_detections": 0,
            "detection_sources": {},
            "player_detections": 0,
            "court_detections": 0,
        }

        print(f"Processing {total_frames} frames...")
        print(f"Output: {output_path}")
        print(f"Frame size: {width}x{height}")
        print(f"PiP size: {int(widthP * self.pip_size)}x{int(heightP * self.pip_size)}")

        # Reset position histories
        self.ball_positions_history = []
        self.player_positions_history = {}
        self.ball_shot_frames = []
        self.shot_classifications = {}

        # Initialize audio analyzer for shot detection
        print("Extracting and analyzing audio...")
        self.audio_analyzer = AudioAnalyzer(input_path, fps=fps)
        audio_ok = self.audio_analyzer.extract_audio()
        if audio_ok:
            self.audio_analyzer.analyze_intensity()
            audio_shot_frames = self.audio_analyzer.detect_shots_from_audio()
            print(
                f"  Audio shots detected at frames: {audio_shot_frames[:10]}..."
                if len(audio_shot_frames) > 10
                else f"  Audio shots detected at frames: {audio_shot_frames}"
            )
        else:
            print("  Audio analysis unavailable - continuing without audio")

        # Extra length for line detection
        extraLen = width / 3

        # Previous court corners for stability
        prev_corners = None
        self.homography_matrix = None

        # Shot detection helpers (merge audio + ball net-crossing)
        net_y = yOffset + int(
            courtHeight * 0.5
        )  # net line in homography/court-map coordinates
        prev_ball_court_y = None
        last_ball_detect_frame = -10_000
        last_net_cross_frame = -10_000
        min_shot_gap_frames = 12  # avoid double-counting; tuned for ~30fps
        # Shot gating: require an audio peak near the net-crossing.
        # We use the raw intensity curve (not just find_peaks) to avoid missing peaks.
        audio_ball_merge_window_frames = (
            10  # +/- frames window to match audio peak to net-crossing
        )
        audio_intensity_gate = 0.35  # normalized intensity threshold (0..1)

        def _audio_peak_near(frame_idx: int) -> bool:
            if (
                self.audio_analyzer is None
                or not self.audio_analyzer.intensity_per_frame
            ):
                return False
            intensities = self.audio_analyzer.intensity_per_frame
            n = len(intensities)
            # Frame indices in this code are 1-based; intensity array is 0-based.
            idx = max(0, min(n - 1, int(frame_idx) - 1))
            lo = max(0, idx - audio_ball_merge_window_frames)
            hi = min(n, idx + audio_ball_merge_window_frames + 1)
            if hi <= lo:
                return False
            return float(max(intensities[lo:hi])) >= float(audio_intensity_gate)

        # Ball speed/velocity (computed in court-map coordinates, then converted to m/s, km/h)
        # CourtMapping ratio is based on doubles court width (10.97m) and length (23.77m).
        meters_per_px_x = 10.97 / max(1, courtWidth)
        meters_per_px_y = 23.77 / max(1, courtHeight)
        prev_ball_court_xy = None
        prev_ball_frame = None
        current_ball_speed_kmh = 0.0
        current_ball_vxy_mps = (0.0, 0.0)

        # Cap speeds to a realistic tennis upper bound (fastest recorded serves ~263.4 km/h)
        MAX_BALL_SPEED_KMH = 263.4
        SPEED_SMOOTH_ALPHA = 0.25  # 0..1, higher = more reactive

        # Per-shot max speed tracking (max over a small window around the shot event)
        speed_samples = []  # list[(frame_idx, speed_kmh)]
        SPEED_SAMPLE_KEEP = 90
        SHOT_SPEED_PRE_FRAMES = 6
        SHOT_SPEED_POST_FRAMES = 8
        active_shot = (
            None  # dict with max_speed_kmh updated for a few frames after event
        )

        # Track players on court-map each frame to assign hitter (top vs bottom)
        # We map "top_player" = smaller y in court-map; "bottom_player" = larger y.
        top_player_id = None
        bottom_player_id = None

        # Shot stats
        # Each entry: {frame, time_s, max_speed_kmh, lr, hitter, reason}
        shot_events = []
        player_shot_stats = {
            "top": {"count": 0, "lr": {"L2R": 0, "R2L": 0}, "last_speed_kmh": 0.0},
            "bottom": {"count": 0, "lr": {"L2R": 0, "R2L": 0}, "last_speed_kmh": 0.0},
        }

        self.frame_count = 0
        debug_printed = False

        # Pre-render a canonical 2D court template (top-down) once
        court_template = np.zeros((heightP, widthP, 3), dtype=np.uint8)
        cv2.rectangle(court_template, (0, 0), (widthP, heightP), (188, 145, 103), -1)
        court_template = showLines(court_template)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and self.frame_count >= max_frames:
                break

            self.frame_count += 1
            annotated_frame = frame.copy()
            court_view = None
            court_map_view = court_template.copy()

            # === COURT DETECTION (FUSION) ===
            corners, court_source = self.court_detector.detect(frame)

            # Debug output on first frame
            if not debug_printed and self.frame_count == 1:
                print(
                    f"First frame - Court detection: source={court_source}, corners={corners}"
                )
                debug_printed = True

            if "court_sources" not in stats:
                stats["court_sources"] = {}
            if court_source not in stats["court_sources"]:
                stats["court_sources"][court_source] = 0
            stats["court_sources"][court_source] += 1

            # Use detected corners or fall back to previous or default
            if corners and all(c is not None for c in corners):
                prev_corners = corners
                stats["court_detections"] += 1
            elif prev_corners is not None:
                # Use previous corners if current detection failed
                corners = prev_corners
                court_source = "kalman"  # Indicate we're using previous detection
            else:
                # No detection and no previous - use default corners based on frame size
                # Default: assume court takes up center 60% of frame
                margin_x = int(width * 0.2)
                margin_y = int(height * 0.15)
                corners = [
                    (margin_x, margin_y),  # top-left
                    (width - margin_x, margin_y),  # top-right
                    (margin_x, height - margin_y),  # bottom-left
                    (width - margin_x, height - margin_y),  # bottom-right
                ]
                court_source = "default"
                if prev_corners is None:
                    prev_corners = corners

            # Create homography view (warped perspective of court)
            court_view = None
            try:
                if corners and all(c is not None for c in corners):
                    court_view, self.homography_matrix = courtMap(
                        frame, corners[0], corners[1], corners[2], corners[3]
                    )
                    # Just the warped video - no lines overlay
            except Exception as e:
                # If homography fails, create a default court view
                print(f"Warning: Court homography failed: {e}")
                court_view = None

            # Create default court view if homography failed
            if court_view is None:
                court_view = np.zeros((heightP, widthP, 3), dtype=np.uint8)
                cv2.rectangle(
                    court_view, (0, 0), (widthP, heightP), (188, 145, 103), -1
                )
                court_view = showLines(court_view)

            # === PLAYER DETECTION (YOLOv8-Pose) ===
            player_positions_frame = {}
            player_court_positions = {}
            shoulders1, shoulders2 = None, None
            keypoints1, keypoints2 = None, None
            if self.show_players:
                player_data = self._detect_players(frame, crop1, crop2)
                if player_data:
                    (
                        feet1,
                        feet2,
                        hands1,
                        hands2,
                        nose1,
                        nose2,
                        shoulders1,
                        shoulders2,
                        keypoints1,
                        keypoints2,
                    ) = player_data
                    stats["player_detections"] += 1

                    # Draw pose skeleton overlay
                    if self.show_pose_skeleton:
                        if keypoints1 is not None:
                            annotated_frame = self._draw_skeleton(
                                annotated_frame, keypoints1, self.skeleton_color_p1, 1
                            )
                        if keypoints2 is not None:
                            annotated_frame = self._draw_skeleton(
                                annotated_frame, keypoints2, self.skeleton_color_p2, 2
                            )

                    # Draw player positions
                    if feet1 and feet1[0] and feet1[1]:
                        pos1 = (
                            int((feet1[0][0] + feet1[1][0]) / 2),
                            int(max(feet1[0][1], feet1[1][1])),
                        )
                        player_positions_frame[1] = pos1

                        # Draw on court view
                        if (
                            court_view is not None
                            and self.homography_matrix is not None
                        ):
                            court_view = showPoint(
                                court_view, self.homography_matrix, pos1
                            )
                            try:
                                p1_map = givePoint(self.homography_matrix, pos1)
                                cv2.circle(
                                    court_map_view,
                                    (int(p1_map[0]), int(p1_map[1])),
                                    10,
                                    self.colors["player1"],
                                    -1,
                                )
                                player_court_positions[1] = (
                                    int(p1_map[0]),
                                    int(p1_map[1]),
                                )
                            except Exception:
                                pass

                    if feet2 and feet2[0] and feet2[1]:
                        pos2 = (
                            int((feet2[0][0] + feet2[1][0]) / 2),
                            int(max(feet2[0][1], feet2[1][1])),
                        )
                        player_positions_frame[2] = pos2

                        if (
                            court_view is not None
                            and self.homography_matrix is not None
                        ):
                            court_view = showPoint(
                                court_view, self.homography_matrix, pos2
                            )
                            try:
                                p2_map = givePoint(self.homography_matrix, pos2)
                                cv2.circle(
                                    court_map_view,
                                    (int(p2_map[0]), int(p2_map[1])),
                                    10,
                                    self.colors["player2"],
                                    -1,
                                )
                                player_court_positions[2] = (
                                    int(p2_map[0]),
                                    int(p2_map[1]),
                                )
                            except Exception:
                                pass

                    # Draw hand radius circles
                    if hands1 and hands1[1] and nose1:
                        try:
                            radius1 = int(
                                0.65
                                * euclideanDistance(nose1, pos1 if feet1 else hands1[1])
                            )
                            cv2.circle(
                                annotated_frame,
                                hands1[1],
                                radius1,
                                self.colors["hand_radius"],
                                1,
                            )
                        except:
                            pass
                    if hands2 and hands2[1] and nose2:
                        try:
                            radius2 = int(
                                0.6
                                * euclideanDistance(nose2, pos2 if feet2 else hands2[1])
                            )
                            cv2.circle(
                                annotated_frame,
                                hands2[1],
                                radius2,
                                self.colors["hand_radius"],
                                1,
                            )
                        except:
                            pass

            # === COURT KEYPOINTS OVERLAY ===
            if self.show_court_keypoints and corners:
                # Get all 14 keypoints if available (from model)
                all_keypoints = self.court_detector.get_all_keypoints()
                annotated_frame = self._draw_court_keypoints(
                    annotated_frame, corners, all_keypoints
                )

            # Store player positions for shot classification
            self.player_positions_history[self.frame_count] = player_positions_frame

            # Update top/bottom player IDs (based on court-map Y)
            if player_court_positions:
                # Sort by y ascending: first is top, last is bottom
                sorted_players = sorted(
                    player_court_positions.items(), key=lambda kv: kv[1][1]
                )
                top_player_id = sorted_players[0][0]
                bottom_player_id = sorted_players[-1][0]

            # === BALL DETECTION (FUSION) ===
            ball_position_frame = {}
            is_ball_net_crossing = False
            if self.show_ball:
                x, y, source = self.ball_tracker.detect_ball(frame)

                if source not in stats["detection_sources"]:
                    stats["detection_sources"][source] = 0
                stats["detection_sources"][source] += 1

                if x is not None and y is not None:
                    last_ball_detect_frame = self.frame_count
                    stats["ball_detections"] += 1
                    self.ball_trajectory.append((x, y, self.frame_count, source))
                    ball_position_frame[1] = (x, y)

                    # Draw ball
                    color = self.colors.get(f"ball_{source}", self.colors["ball"])
                    cv2.circle(annotated_frame, (x, y), 10, color, -1)
                    cv2.circle(annotated_frame, (x, y), 12, (255, 255, 255), 2)
                    cv2.putText(
                        annotated_frame,
                        source[:3].upper(),
                        (x + 15, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )

                    # Draw on court view
                    if court_view is not None and self.homography_matrix is not None:
                        try:
                            ball_court = givePoint(self.homography_matrix, (x, y))
                            cv2.circle(
                                court_view,
                                (int(ball_court[0]), int(ball_court[1])),
                                10,
                                (0, 255, 255),
                                -1,
                            )
                            cv2.circle(
                                court_map_view,
                                (int(ball_court[0]), int(ball_court[1])),
                                10,
                                (0, 255, 255),
                                -1,
                            )

                            # Velocity + speed (use previous mapped point)
                            if (
                                prev_ball_court_xy is not None
                                and prev_ball_frame is not None
                            ):
                                dt = max(
                                    1e-6,
                                    (self.frame_count - prev_ball_frame)
                                    / max(1e-6, fps),
                                )
                                dx_px = ball_court[0] - prev_ball_court_xy[0]
                                dy_px = ball_court[1] - prev_ball_court_xy[1]
                                vx_mps = (dx_px * meters_per_px_x) / dt
                                vy_mps = (dy_px * meters_per_px_y) / dt
                                current_ball_vxy_mps = (vx_mps, vy_mps)
                                speed_mps = float(
                                    np.sqrt(vx_mps * vx_mps + vy_mps * vy_mps)
                                )
                                raw_speed_kmh = float(speed_mps * 3.6)
                                raw_speed_kmh = max(
                                    0.0, min(raw_speed_kmh, MAX_BALL_SPEED_KMH)
                                )
                                current_ball_speed_kmh = (
                                    1.0 - SPEED_SMOOTH_ALPHA
                                ) * float(
                                    current_ball_speed_kmh
                                ) + SPEED_SMOOTH_ALPHA * raw_speed_kmh
                            prev_ball_court_xy = (
                                int(ball_court[0]),
                                int(ball_court[1]),
                            )
                            prev_ball_frame = self.frame_count

                            # Speed samples buffer for per-shot max estimation
                            speed_samples.append(
                                (self.frame_count, float(current_ball_speed_kmh))
                            )
                            if len(speed_samples) > SPEED_SAMPLE_KEEP:
                                speed_samples = speed_samples[-SPEED_SAMPLE_KEEP:]

                            # Optional: draw a small velocity arrow on the 2D map
                            if prev_ball_court_xy is not None:
                                vx_mps, vy_mps = current_ball_vxy_mps
                                # Scale arrow length for display
                                arrow_scale = 12.0  # px per (m/s), purely visual
                                ax = int(prev_ball_court_xy[0] + vx_mps * arrow_scale)
                                ay = int(prev_ball_court_xy[1] + vy_mps * arrow_scale)
                                cv2.arrowedLine(
                                    court_map_view,
                                    (
                                        int(prev_ball_court_xy[0]),
                                        int(prev_ball_court_xy[1]),
                                    ),
                                    (ax, ay),
                                    (0, 200, 255),
                                    2,
                                    tipLength=0.25,
                                )

                            # Detect net-crossing (side change) in court-map coordinates
                            if prev_ball_court_y is not None:
                                prev_side = prev_ball_court_y - net_y
                                cur_side = ball_court[1] - net_y
                                crossed = (
                                    (prev_side == 0)
                                    or (cur_side == 0)
                                    or (prev_side * cur_side < 0)
                                )
                                moved_enough = (
                                    abs(ball_court[1] - prev_ball_court_y) > 10
                                )
                                if (
                                    crossed
                                    and moved_enough
                                    and (self.frame_count - last_net_cross_frame) > 6
                                ):
                                    is_ball_net_crossing = True
                                    last_net_cross_frame = self.frame_count
                            prev_ball_court_y = ball_court[1]
                        except:
                            pass

            # Store ball position for shot detection (as list format for detector)
            self.ball_positions_history.append(ball_position_frame)

            # === TRAJECTORY TRAIL ===
            if self.show_trajectory and len(self.ball_trajectory) > 1:
                trail = self.ball_trajectory[-50:]  # Last 50 points
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    thickness = max(1, int(4 * alpha))
                    pt1 = (trail[i - 1][0], trail[i - 1][1])
                    pt2 = (trail[i][0], trail[i][1])
                    cv2.line(
                        annotated_frame, pt1, pt2, self.colors["trajectory"], thickness
                    )

            # === CREATE SIDEBAR ===
            sidebar = np.zeros((height, sidebar_width, 3), dtype=np.uint8)
            sidebar[:] = (30, 30, 30)  # Dark gray background

            # Current Y position for drawing elements in sidebar
            sidebar_y = 10

            # --- 2D COURT MAP (schematic) ---
            map_pip_height = max(
                1, int(pip_height * 0.55)
            )  # smaller than the homography warp view
            if pip_width > 0 and map_pip_height > 0 and court_map_view is not None:
                map_pip = cv2.resize(court_map_view, (pip_width, map_pip_height))
                cv2.rectangle(
                    map_pip,
                    (0, 0),
                    (pip_width - 1, map_pip_height - 1),
                    (100, 100, 100),
                    1,
                )
                cv2.putText(
                    map_pip,
                    "2D Map",
                    (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )
                x_offset = (sidebar_width - pip_width) // 2
                sidebar[
                    sidebar_y : sidebar_y + map_pip_height,
                    x_offset : x_offset + pip_width,
                ] = map_pip
                sidebar_y += map_pip_height + 12

            # --- COURT HOMOGRAPHY VIEW (warped video) ---
            if court_view is None:
                court_view = court_template.copy()
            if pip_width > 0 and pip_height > 0 and court_view is not None:
                pip = cv2.resize(court_view, (pip_width, pip_height))
                cv2.rectangle(
                    pip, (0, 0), (pip_width - 1, pip_height - 1), (100, 100, 100), 1
                )
                cv2.putText(
                    pip,
                    "Homography",
                    (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )
                x_offset = (sidebar_width - pip_width) // 2
                y2 = sidebar_y
                y3 = min(height, sidebar_y + pip_height)
                if y3 > y2:
                    sidebar[y2:y3, x_offset : x_offset + pip_width] = pip[
                        : (y3 - y2), :
                    ]
                    sidebar_y += (y3 - y2) + 15

            # --- AUDIO SECTION ---
            cv2.putText(
                sidebar,
                "AUDIO",
                (10, sidebar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            sidebar_y += 20

            is_audio_shot = False
            is_new_shot = False
            shot_reason = None

            if self.audio_analyzer is not None:
                intensity = self.audio_analyzer.get_intensity_at_frame(self.frame_count)
                # Allow small tolerance because audio peak timing can be off by a frame or two
                is_audio_shot = self.audio_analyzer.is_shot_frame(
                    self.frame_count, tolerance=2
                )

                # Draw audio waveform in sidebar
                if self.audio_analyzer.intensity_per_frame:
                    sidebar = draw_audio_waveform(
                        sidebar,
                        self.audio_analyzer.intensity_per_frame,
                        self.frame_count,
                        x=10,
                        y=sidebar_y,
                        width=sidebar_width - 20,
                        height=45,
                    )
                    sidebar_y += 55

                # Draw audio intensity bar in sidebar
                sidebar = draw_audio_overlay(
                    sidebar,
                    intensity,
                    is_audio_shot,
                    x=10,
                    y=sidebar_y,
                    width=sidebar_width - 70,
                    height=18,
                )
                sidebar_y += 30

                # Track shot detection (MERGED, robust):
                # Count a shot ONLY when ball crosses the net AND there is a nearby audio peak
                # in a +/- window (using raw intensity, not only peak detector).
                last_shot_frame = getattr(self, "_last_shot_frame", -100)
                is_shot_event = bool(
                    is_ball_net_crossing and _audio_peak_near(self.frame_count)
                )

                if (
                    is_shot_event
                    and (self.frame_count - last_shot_frame) > min_shot_gap_frames
                ):
                    self._last_shot_frame = self.frame_count
                    self._shot_count = getattr(self, "_shot_count", 0) + 1
                    is_new_shot = True
                    shot_reason = "ball+audio"
                    self._last_shot_reason = shot_reason

                    # Estimate shot speed and assign hitter + L2R/R2L direction (based on ball motion in court-map)
                    hitter_side = None
                    lr_dir = None

                    # Determine hitter side based on which side of net the ball came from (only reliable for net-crossing)
                    if is_ball_net_crossing and prev_ball_court_y is not None:
                        # If ball was below net (larger y) and moved upward across net => bottom hit
                        # If ball was above net and moved downward across net => top hit
                        if prev_ball_court_y > net_y:
                            hitter_side = "bottom"
                        else:
                            hitter_side = "top"
                    else:
                        # Fallback: infer from current vertical velocity sign
                        vx_mps, vy_mps = current_ball_vxy_mps
                        hitter_side = "bottom" if vy_mps < 0 else "top"

                    # L2R / R2L based on horizontal velocity sign in court-map
                    vx_mps, vy_mps = current_ball_vxy_mps
                    lr_dir = "L2R" if vx_mps >= 0 else "R2L"

                    # Start an active shot; finalize max speed after a short post window
                    pre_start = self.frame_count - SHOT_SPEED_PRE_FRAMES
                    pre_window = [
                        s
                        for (f, s) in speed_samples
                        if pre_start <= f <= self.frame_count
                    ]
                    initial_max = (
                        max(pre_window) if pre_window else float(current_ball_speed_kmh)
                    )
                    active_shot = {
                        "frame": self.frame_count,
                        "time_s": float(self.frame_count) / max(1e-6, float(fps)),
                        "max_speed_kmh": float(initial_max),
                        "lr": lr_dir,
                        "hitter": hitter_side,
                        "reason": shot_reason,
                        "post_left": int(SHOT_SPEED_POST_FRAMES),
                    }
                    if hitter_side in player_shot_stats:
                        player_shot_stats[hitter_side]["count"] += 1
                        player_shot_stats[hitter_side]["lr"][lr_dir] += 1

            # Update in-progress shot max speed (post window) and finalize when done
            if active_shot is not None:
                active_shot["max_speed_kmh"] = max(
                    float(active_shot["max_speed_kmh"]), float(current_ball_speed_kmh)
                )
                active_shot["post_left"] -= 1
                if active_shot["post_left"] <= 0:
                    active_shot["max_speed_kmh"] = max(
                        0.0,
                        min(float(active_shot["max_speed_kmh"]), MAX_BALL_SPEED_KMH),
                    )
                    shot_events.append(active_shot)
                    hitter_side = active_shot.get("hitter")
                    if hitter_side in player_shot_stats:
                        player_shot_stats[hitter_side]["last_speed_kmh"] = float(
                            active_shot["max_speed_kmh"]
                        )
                    active_shot = None

            # --- SHOT DETECTED SECTION ---
            cv2.line(
                sidebar,
                (10, sidebar_y),
                (sidebar_width - 10, sidebar_y),
                (60, 60, 60),
                1,
            )
            sidebar_y += 8

            # Show shot indicator (persists for ~1 second after detection)
            last_shot_frame = getattr(self, "_last_shot_frame", -100)
            shot_count = getattr(self, "_shot_count", 0)
            last_shot_reason = getattr(self, "_last_shot_reason", "")
            frames_since_shot = self.frame_count - last_shot_frame

            if frames_since_shot < 30:  # Show for 30 frames (~1 sec)
                # Flash effect: brighter when more recent
                brightness = int(255 * (1 - frames_since_shot / 30))
                shot_color = (0, brightness, 255)  # Red-orange

                cv2.putText(
                    sidebar,
                    "SHOT DETECTED!",
                    (10, sidebar_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    shot_color,
                    2,
                )
                if last_shot_reason:
                    cv2.putText(
                        sidebar,
                        f"({last_shot_reason})",
                        (10, sidebar_y + 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (180, 180, 180),
                        1,
                    )
                sidebar_y += 32

                # Also flash on main video
                if frames_since_shot < 15:
                    cv2.putText(
                        annotated_frame,
                        "SHOT!",
                        (width // 2 - 50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3,
                    )
            else:
                cv2.putText(
                    sidebar,
                    "Waiting for shot...",
                    (10, sidebar_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (80, 80, 80),
                    1,
                )
                sidebar_y += 32

            # Show shot count
            cv2.putText(
                sidebar,
                f"Total shots: {shot_count}",
                (10, sidebar_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (150, 150, 150),
                1,
            )
            sidebar_y += 20

            # --- DETECTION INFO ---
            cv2.line(
                sidebar,
                (10, sidebar_y + 5),
                (sidebar_width - 10, sidebar_y + 5),
                (60, 60, 60),
                1,
            )
            sidebar_y += 13
            cv2.putText(
                sidebar,
                "DETECTION",
                (10, sidebar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            sidebar_y += 25

            ball_src = source if self.show_ball else "N/A"
            cv2.putText(
                sidebar,
                f"Ball: {ball_src}",
                (15, sidebar_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (130, 130, 130),
                1,
            )
            sidebar_y += 16
            cv2.putText(
                sidebar,
                f"Court: {court_source}",
                (15, sidebar_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (130, 130, 130),
                1,
            )
            sidebar_y += 16
            cv2.putText(
                sidebar,
                f"Frame: {self.frame_count}/{total_frames}",
                (15, sidebar_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (130, 130, 130),
                1,
            )

            # === COMBINE SIDEBAR + MAIN FRAME ===
            rightbar = np.zeros((height, right_sidebar_width, 3), dtype=np.uint8)
            rightbar[:] = (30, 30, 30)

            # --- RIGHT SIDEBAR: SHOT HISTORY + MAX SPEED ---
            ry = 10
            cv2.putText(
                rightbar,
                "SHOT HISTORY",
                (10, ry + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                1,
            )
            ry += 26
            cv2.line(
                rightbar, (10, ry), (right_sidebar_width - 10, ry), (60, 60, 60), 1
            )
            ry += 12

            cv2.putText(
                rightbar,
                f"Cap: {MAX_BALL_SPEED_KMH:.1f} km/h",
                (10, ry + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (150, 150, 150),
                1,
            )
            ry += 18
            cv2.putText(
                rightbar,
                f"Top: {player_shot_stats['top']['count']}  Bot: {player_shot_stats['bottom']['count']}",
                (10, ry + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (150, 150, 150),
                1,
            )
            ry += 22

            recent = shot_events[-8:]
            if active_shot is not None:
                recent = (shot_events + [active_shot])[-8:]

            for evt in reversed(recent):
                if ry + 44 >= height:
                    break
                speed = float(evt.get("max_speed_kmh", 0.0))
                hitter = evt.get("hitter", "n/a")
                lr = evt.get("lr", "")
                tsec = float(evt.get("time_s", 0.0))
                reason = evt.get("reason", "")

                bar_w = right_sidebar_width - 20
                bar_h = 8
                filled = int(bar_w * min(max(speed, 0.0) / MAX_BALL_SPEED_KMH, 1.0))
                cv2.rectangle(
                    rightbar, (10, ry), (10 + bar_w, ry + bar_h), (50, 50, 50), -1
                )
                cv2.rectangle(
                    rightbar, (10, ry), (10 + filled, ry + bar_h), (0, 165, 255), -1
                )
                cv2.rectangle(
                    rightbar, (10, ry), (10 + bar_w, ry + bar_h), (80, 80, 80), 1
                )
                ry += bar_h + 12

                cv2.putText(
                    rightbar,
                    f"{speed:5.1f} km/h  {hitter} {lr}",
                    (10, ry),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (230, 230, 230),
                    1,
                )
                ry += 16
                cv2.putText(
                    rightbar,
                    f"t={tsec:5.1f}s  {reason}",
                    (10, ry),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (140, 140, 140),
                    1,
                )
                ry += 18
                cv2.line(
                    rightbar, (10, ry), (right_sidebar_width - 10, ry), (50, 50, 50), 1
                )
                ry += 10

            final_frame = np.zeros((height, output_width, 3), dtype=np.uint8)
            final_frame[:, :sidebar_width] = sidebar
            final_frame[:, sidebar_width : sidebar_width + width] = annotated_frame
            final_frame[:, sidebar_width + width :] = rightbar

            # Draw separator line between sidebar and main video
            cv2.line(
                final_frame,
                (sidebar_width, 0),
                (sidebar_width, height),
                (80, 80, 80),
                2,
            )
            cv2.line(
                final_frame,
                (sidebar_width + width, 0),
                (sidebar_width + width, height),
                (80, 80, 80),
                2,
            )

            # Write frame
            out.write(final_frame)

            # Display if requested
            if show_display:
                cv2.imshow("Tennis Analysis", final_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nStopped by user")
                    break

            # Progress
            if self.frame_count % 50 == 0:
                pct = 100 * self.frame_count / total_frames
                print(f"  Frame {self.frame_count}/{total_frames} ({pct:.1f}%)")

        # Cleanup
        cap.release()
        out.release()
        if show_display:
            cv2.destroyAllWindows()

        # Final stats
        stats["processed_frames"] = self.frame_count
        stats["ball_detection_rate"] = (
            100 * stats["ball_detections"] / self.frame_count
            if self.frame_count > 0
            else 0
        )
        
        # Re-encode video with H.264 codec for browser compatibility
        if os.path.exists(temp_output):
            print(f"\nRe-encoding video with H.264 codec for browser compatibility...")
            import subprocess
            try:
                # Use ffmpeg to re-encode with H.264
                cmd = [
                    'ffmpeg', '-y', '-i', temp_output,
                    '-c:v', 'libx264',  # H.264 video codec
                    '-preset', 'medium',  # Encoding speed/quality tradeoff
                    '-crf', '23',  # Quality (lower = better, 18-28 is reasonable)
                    '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                    '-movflags', '+faststart',  # Enable streaming
                    output_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                # Remove temp file
                os.remove(temp_output)
                print(f"Video re-encoded successfully!")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to re-encode video: {e}")
                print(f"stderr: {e.stderr}")
                # Fallback: just rename the temp file
                if os.path.exists(temp_output):
                    os.rename(temp_output, output_path)
                print(f"Using original encoding (may not play in browser)")
            except Exception as e:
                print(f"Warning: Error during re-encoding: {e}")
                if os.path.exists(temp_output):
                    os.rename(temp_output, output_path)
        
        print(f"Output saved to: {output_path}")

        return stats

    def _create_crop_config(
        self, x_ratio, x_offset, x_center, y_ratio, y_offset, y_center, width, height
    ):
        """Create crop configuration for player detection"""

        class Crop:
            pass

        crop = Crop()
        crop.x = int(width * x_ratio)
        crop.y = int(height * y_ratio)
        if x_center:
            crop.xoffset = int((width - crop.x) / 2)
        else:
            crop.xoffset = int(width * x_offset)
        if y_center:
            crop.yoffset = int((height - crop.y) / 2)
        else:
            crop.yoffset = int(height * y_offset)
        return crop

    def _detect_court(self, frame, width, height, extraLen):
        """Detect court corners using Hough lines"""
        try:
            # Preprocessing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 156, 255, cv2.THRESH_BINARY)
            canny = cv2.Canny(bw, 100, 200)

            # Find lines with most intersections
            hPLines = cv2.HoughLinesP(
                canny, 1, pi / 180, threshold=150, minLineLength=100, maxLineGap=10
            )
            if hPLines is None:
                return None

            # Find intersections and mask court area
            dilation = cv2.dilate(bw, ones((5, 5), uint8), iterations=1)
            nonRectArea = dilation.copy()

            intersectNum = zeros((len(hPLines), 2))
            for i, hPLine1 in enumerate(hPLines):
                Line1 = [[hPLine1[0][0], hPLine1[0][1]], [hPLine1[0][2], hPLine1[0][3]]]
                for hPLine2 in hPLines:
                    Line2 = [
                        [hPLine2[0][0], hPLine2[0][1]],
                        [hPLine2[0][2], hPLine2[0][3]],
                    ]
                    if Line1 != Line2:
                        x1, y1 = min(Line1[0][0], Line1[1][0]), min(
                            Line1[0][1], Line1[1][1]
                        )
                        x2, y2 = max(Line1[0][0], Line1[1][0]), max(
                            Line1[0][1], Line1[1][1]
                        )
                        intersect = findIntersection(
                            Line1, Line2, x1 - 200, y1 - 200, x2 + 200, y2 + 200
                        )
                        if intersect:
                            intersectNum[i][0] += 1
                intersectNum[i][1] = i

            # Flood fill from top lines
            intersectNum = intersectNum[(-intersectNum)[:, 0].argsort()]
            for i, hPLine in enumerate(hPLines):
                x1, y1, x2, y2 = hPLine[0]
                for p in range(min(8, len(intersectNum))):
                    if i == intersectNum[p][1] and intersectNum[i][0] > 0:
                        cv2.floodFill(
                            nonRectArea,
                            zeros((height + 2, width + 2), uint8),
                            (x1, y1),
                            1,
                        )
                        cv2.floodFill(
                            nonRectArea,
                            zeros((height + 2, width + 2), uint8),
                            (x2, y2),
                            1,
                        )

            dilation[where(nonRectArea == 255)] = 0
            dilation[where(nonRectArea == 1)] = 255
            eroded = cv2.erode(dilation, ones((5, 5), uint8))
            cannyMain = cv2.Canny(eroded, 90, 100)

            # Find extreme lines
            hLines = cv2.HoughLines(cannyMain, 2, pi / 180, 300)
            if hLines is None:
                return None

            # Axis definitions
            axis_top = [[-extraLen, 0], [width + extraLen, 0]]
            axis_right = [[width + extraLen, 0], [width + extraLen, height]]
            axis_bottom = [[-extraLen, height], [width + extraLen, height]]
            axis_left = [[-extraLen, 0], [-extraLen, height]]

            # Find extreme lines
            lines = {
                "xOLeft": None,
                "xORight": None,
                "yOTop": None,
                "yOBottom": None,
                "xFLeft": None,
                "xFRight": None,
                "yFTop": None,
                "yFBottom": None,
            }
            vals = {
                "xOLeft": width + extraLen,
                "xORight": -extraLen,
                "yOTop": height,
                "yOBottom": 0,
                "xFLeft": width + extraLen,
                "xFRight": -extraLen,
                "yFTop": height,
                "yFBottom": 0,
            }

            for hLine in hLines:
                for rho, theta in hLine:
                    a, b = cos(theta), sin(theta)
                    x0, y0 = a * rho, b * rho
                    x1 = int(x0 + width * (-b))
                    y1 = int(y0 + width * a)
                    x2 = int(x0 - width * (-b))
                    y2 = int(y0 - width * a)
                    line = [[x1, y1], [x2, y2]]

                    for axis_name, axis in [
                        ("xO", axis_top),
                        ("yO", axis_left),
                        ("xF", axis_bottom),
                        ("yF", axis_right),
                    ]:
                        intersect = findIntersection(
                            axis, line, -extraLen, 0, width + extraLen, height
                        )
                        if intersect:
                            if axis_name in ["xO", "xF"]:
                                if intersect[0] < vals[f"{axis_name}Left"]:
                                    vals[f"{axis_name}Left"] = intersect[0]
                                    lines[f"{axis_name}Left"] = line
                                if intersect[0] > vals[f"{axis_name}Right"]:
                                    vals[f"{axis_name}Right"] = intersect[0]
                                    lines[f"{axis_name}Right"] = line
                            else:
                                if intersect[1] < vals[f"{axis_name}Top"]:
                                    vals[f"{axis_name}Top"] = intersect[1]
                                    lines[f"{axis_name}Top"] = line
                                if intersect[1] > vals[f"{axis_name}Bottom"]:
                                    vals[f"{axis_name}Bottom"] = intersect[1]
                                    lines[f"{axis_name}Bottom"] = line

            # Check if we found all lines
            required = [
                "xOLeft",
                "xORight",
                "yOTop",
                "yOBottom",
                "xFLeft",
                "xFRight",
                "yFTop",
                "yFBottom",
            ]
            if any(lines[k] is None for k in required):
                return None

            # Adjust top lines
            lines["yOTop"][0][1] += 4
            lines["yOTop"][1][1] += 4
            lines["yFTop"][0][1] += 4
            lines["yFTop"][1][1] += 4

            # Find corners
            topLeft = findIntersection(
                lines["xOLeft"], lines["yOTop"], -extraLen, 0, width + extraLen, height
            )
            topRight = findIntersection(
                lines["xORight"], lines["yFTop"], -extraLen, 0, width + extraLen, height
            )
            bottomLeft = findIntersection(
                lines["xFLeft"],
                lines["yOBottom"],
                -extraLen,
                0,
                width + extraLen,
                height,
            )
            bottomRight = findIntersection(
                lines["xFRight"],
                lines["yFBottom"],
                -extraLen,
                0,
                width + extraLen,
                height,
            )

            return [topLeft, topRight, bottomLeft, bottomRight]

        except Exception as e:
            return None

    def _detect_players_yolo_on_crop(self, frame, crop):
        """Run YOLOv8 pose detection on a specific crop region and return pose data"""
        try:
            # Extract crop region - crop.x is width, crop.y is height, xoffset/yoffset are the start positions
            crop_x = crop.xoffset
            crop_y = crop.yoffset
            crop_w = crop.x
            crop_h = crop.y

            # Ensure valid crop bounds
            h_frame, w_frame = frame.shape[:2]
            crop_x = max(0, min(crop_x, w_frame - 1))
            crop_y = max(0, min(crop_y, h_frame - 1))
            crop_w = min(crop_w, w_frame - crop_x)
            crop_h = min(crop_h, h_frame - crop_y)

            cropped = frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

            if cropped.size == 0:
                return None, None

            # Run YOLOv8 pose detection on crop
            results = self.pose_model(
                cropped, device=self.pose_device, verbose=False, conf=0.25
            )

            if results and len(results) > 0 and results[0].keypoints is not None:
                keypoints_data = results[0].keypoints
                boxes = results[0].boxes

                if boxes is not None and len(boxes) > 0:
                    # Find largest person in the crop (most likely the player)
                    best_idx = 0
                    best_area = 0
                    for i, box in enumerate(boxes.xyxy):
                        x1, y1, x2, y2 = box.cpu().numpy()
                        area = (x2 - x1) * (y2 - y1)
                        if area > best_area:
                            best_area = area
                            best_idx = i

                    if best_idx < len(keypoints_data.xy):
                        kps = keypoints_data.xy[best_idx].cpu().numpy()
                        conf = (
                            keypoints_data.conf[best_idx].cpu().numpy()
                            if keypoints_data.conf is not None
                            else None
                        )

                        # Offset keypoints back to full frame coordinates
                        kps_global = kps.copy()
                        kps_global[:, 0] += crop_x  # Add crop x offset
                        kps_global[:, 1] += crop_y  # Add crop y offset

                        # Filter out low-confidence keypoints by setting them to 0
                        # Higher threshold for head keypoints (0-4) since they tend to be noisy
                        min_conf_body = 0.5  # Body keypoints
                        min_conf_head = 0.7  # Head keypoints (nose, eyes, ears)
                        head_indices = {
                            0,
                            1,
                            2,
                            3,
                            4,
                        }  # nose, left_eye, right_eye, left_ear, right_ear

                        if conf is not None:
                            for i in range(len(kps_global)):
                                if i < len(conf):
                                    threshold = (
                                        min_conf_head
                                        if i in head_indices
                                        else min_conf_body
                                    )
                                    if conf[i] < threshold:
                                        kps_global[i] = [0, 0]  # Mark as invalid

                        # Calculate body center from shoulders/hips for outlier detection
                        body_keypoints = []
                        for idx in [5, 6, 11, 12]:  # shoulders and hips
                            if idx < len(kps_global):
                                px, py = kps_global[idx]
                                if px > 0 and py > 0:
                                    body_keypoints.append((px, py))

                        if body_keypoints:
                            body_center_x = np.mean([p[0] for p in body_keypoints])
                            body_center_y = np.mean([p[1] for p in body_keypoints])

                            # Calculate typical body size from shoulder width or hip width
                            body_size = 150  # Default
                            if len(body_keypoints) >= 2:
                                widths = []
                                for idx1, idx2 in [
                                    (5, 6),
                                    (11, 12),
                                ]:  # shoulder pair, hip pair
                                    if idx1 < len(kps_global) and idx2 < len(
                                        kps_global
                                    ):
                                        p1 = kps_global[idx1]
                                        p2 = kps_global[idx2]
                                        if p1[0] > 0 and p2[0] > 0:
                                            widths.append(abs(p1[0] - p2[0]))
                                if widths:
                                    body_size = (
                                        max(widths) * 4
                                    )  # Allow 4x body width as max distance

                            # Filter keypoints too far from body center
                            for i in range(len(kps_global)):
                                x, y = kps_global[i]
                                if x > 0 and y > 0:
                                    dist = np.sqrt(
                                        (x - body_center_x) ** 2
                                        + (y - body_center_y) ** 2
                                    )
                                    if dist > body_size:
                                        kps_global[i] = [
                                            0,
                                            0,
                                        ]  # Outlier - too far from body

                        # Filter keypoints outside frame bounds
                        for i in range(len(kps_global)):
                            x, y = kps_global[i]
                            if x < 0 or x > w_frame or y < 0 or y > h_frame:
                                kps_global[i] = [0, 0]  # Mark as invalid

                        pose_data = self._extract_pose_data(kps_global, conf)
                        return pose_data, kps_global

            return None, None

        except Exception as e:
            return None, None

    def _extract_pose_data(self, keypoints, confidence=None):
        """Extract pose data from YOLOv8 keypoints"""
        try:
            min_conf = 0.3

            def get_kp(idx):
                if idx >= len(keypoints):
                    return None
                x, y = keypoints[idx]
                if x <= 0 or y <= 0:
                    return None
                if confidence is not None and idx < len(confidence):
                    if confidence[idx] < min_conf:
                        return None
                return (int(x), int(y))

            # Extract key body parts
            nose = get_kp(YOLO_POSE_KEYPOINTS["nose"])
            l_shoulder = get_kp(YOLO_POSE_KEYPOINTS["left_shoulder"])
            r_shoulder = get_kp(YOLO_POSE_KEYPOINTS["right_shoulder"])
            l_wrist = get_kp(YOLO_POSE_KEYPOINTS["left_wrist"])
            r_wrist = get_kp(YOLO_POSE_KEYPOINTS["right_wrist"])
            l_ankle = get_kp(YOLO_POSE_KEYPOINTS["left_ankle"])
            r_ankle = get_kp(YOLO_POSE_KEYPOINTS["right_ankle"])

            # Build return data
            feet = [l_ankle, r_ankle] if l_ankle or r_ankle else None
            hands = [l_wrist, r_wrist] if l_wrist or r_wrist else None
            shoulders = [l_shoulder, r_shoulder] if l_shoulder or r_shoulder else None

            if feet or hands:
                return {
                    "feet": feet,
                    "hands": hands,
                    "nose": nose,
                    "shoulders": shoulders,
                }
            return None

        except Exception:
            return None

    def _draw_skeleton(self, frame, keypoints, color, player_id):
        """Draw pose skeleton on frame"""
        if keypoints is None or len(keypoints) == 0:
            return frame

        # Draw skeleton connections
        for connection in SKELETON_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_pt = keypoints[start_idx]
                end_pt = keypoints[end_idx]

                # Check if points are valid
                if (
                    start_pt[0] > 0
                    and start_pt[1] > 0
                    and end_pt[0] > 0
                    and end_pt[1] > 0
                ):
                    pt1 = (int(start_pt[0]), int(start_pt[1]))
                    pt2 = (int(end_pt[0]), int(end_pt[1]))
                    cv2.line(frame, pt1, pt2, color, 2)

        # Draw keypoints
        for i, kp in enumerate(keypoints):
            x, y = kp
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 4, self.keypoint_color, -1)
                cv2.circle(frame, (int(x), int(y)), 4, color, 1)

        return frame

    def _draw_court_keypoints(self, frame, corners, all_keypoints=None):
        """
        Draw court keypoints on frame
        If all_keypoints (14 keypoints) is available, draw all of them.
        Otherwise, just draw the 4 corners.
        """
        # Keypoint labels for all 14 points (based on typical tennis court layout)
        all_labels = [
            "TL",  # 0: Top-left corner
            "TR",  # 1: Top-right corner
            "BL",  # 2: Bottom-left corner
            "BR",  # 3: Bottom-right corner
            "TSL",  # 4: Top service line left
            "TSR",  # 5: Top service line right
            "BSL",  # 6: Bottom service line left
            "BSR",  # 7: Bottom service line right
            "TC",  # 8: Top center (net)
            "BC",  # 9: Bottom center (net)
            "TML",  # 10: Top middle left
            "TMR",  # 11: Top middle right
            "BML",  # 12: Bottom middle left
            "BMR",  # 13: Bottom middle right
        ]

        # Draw all 14 keypoints if available
        if all_keypoints and len(all_keypoints) >= 14:
            for i, kp in enumerate(all_keypoints):
                if kp and kp[0] >= 0 and kp[1] >= 0:
                    # Different colors for corners vs other keypoints
                    if i < 4:
                        # Corners: larger, more prominent
                        color = self.court_kp_color  # Magenta
                        radius = 8
                        thickness = -1
                    else:
                        # Other keypoints: smaller, different color
                        color = (0, 255, 255)  # Cyan
                        radius = 5
                        thickness = -1

                    # Draw keypoint circle
                    cv2.circle(frame, kp, radius, color, thickness)
                    cv2.circle(frame, kp, radius + 2, (255, 255, 255), 1)

                    # Draw label
                    label = all_labels[i] if i < len(all_labels) else str(i)
                    cv2.putText(
                        frame,
                        label,
                        (kp[0] + 10, kp[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )

        # If we only have corners (CV fallback), draw just those
        elif corners:
            labels = ["TL", "TR", "BL", "BR"]
            for i, corner in enumerate(corners):
                if corner:
                    # Draw keypoint circle
                    cv2.circle(frame, corner, 8, self.court_kp_color, -1)
                    cv2.circle(frame, corner, 10, (255, 255, 255), 2)
                    # Draw label
                    cv2.putText(
                        frame,
                        labels[i],
                        (corner[0] + 12, corner[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.court_kp_color,
                        2,
                    )

        # Draw court lines connecting corners
        if corners and len(corners) >= 4 and all(c is not None for c in corners):
            cv2.line(frame, corners[0], corners[1], self.court_kp_color, 2)  # Top
            cv2.line(frame, corners[2], corners[3], self.court_kp_color, 2)  # Bottom
            cv2.line(frame, corners[0], corners[2], self.court_kp_color, 2)  # Left
            cv2.line(frame, corners[1], corners[3], self.court_kp_color, 2)  # Right

        # If we have all keypoints, draw additional court lines
        if all_keypoints and len(all_keypoints) >= 14:
            # Draw service lines if keypoints are available
            if len(all_keypoints) > 7 and all_keypoints[4] and all_keypoints[5]:
                # Top service line
                cv2.line(frame, all_keypoints[4], all_keypoints[5], (0, 255, 255), 1)
            if len(all_keypoints) > 7 and all_keypoints[6] and all_keypoints[7]:
                # Bottom service line
                cv2.line(frame, all_keypoints[6], all_keypoints[7], (0, 255, 255), 1)
            # Draw center/net line if available
            if len(all_keypoints) > 9 and all_keypoints[8] and all_keypoints[9]:
                cv2.line(frame, all_keypoints[8], all_keypoints[9], (0, 255, 255), 1)

        return frame

    def _detect_players(self, frame, crop1, crop2):
        """Detect player positions using YOLOv8-pose on crop regions"""
        try:
            # Run YOLOv8 pose detection on each player's crop region
            player1_data, keypoints1 = self._detect_players_yolo_on_crop(frame, crop1)
            player2_data, keypoints2 = self._detect_players_yolo_on_crop(frame, crop2)

            # Extract data for each player
            feet1, hands1, nose1, shoulders1 = None, None, None, None
            feet2, hands2, nose2, shoulders2 = None, None, None, None

            if player1_data:
                feet1 = player1_data.get("feet")
                hands1 = player1_data.get("hands")
                nose1 = player1_data.get("nose")
                shoulders1 = player1_data.get("shoulders")

            if player2_data:
                feet2 = player2_data.get("feet")
                hands2 = player2_data.get("hands")
                nose2 = player2_data.get("nose")
                shoulders2 = player2_data.get("shoulders")

            if feet1 or feet2 or hands1 or hands2:
                return (
                    feet1,
                    feet2,
                    hands1,
                    hands2,
                    nose1,
                    nose2,
                    shoulders1,
                    shoulders2,
                    keypoints1,
                    keypoints2,
                )
            return None

        except Exception as e:
            print(f"Player detection error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _draw_info_overlay(self, frame, ball_source, court_source="N/A"):
        """Draw information overlay on frame"""
        h, w = frame.shape[:2]

        # Background for text
        cv2.rectangle(frame, (10, 10), (250, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, 120), (255, 255, 255), 1)

        # Frame info
        cv2.putText(
            frame,
            f"Frame: {self.frame_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Ball: {ball_source}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            frame,
            f"Court: {court_source}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Trajectory: {len(self.ball_trajectory)} pts",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (128, 128, 128),
            1,
        )


def run_full_analysis(args):
    """
    Run full analysis with all detections
    """
    import torch

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("TRACE - Full Tennis Analysis")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Initialize analyzer with all features enabled
    analyzer = TennisAnalyzer(
        tracknet_weights=args.weights,
        courtside_weights=args.courtside_weights,
        device=device,
        show_court_overlay=True,
        show_ball=True,
        show_players=True,
        show_trajectory=True,
        show_pose_skeleton=True,
        show_court_keypoints=True,
    )

    # Process video
    stats = analyzer.process_video(
        input_path=args.input,
        output_path=args.output,
        show_display=not args.no_display,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Frames processed: {stats['processed_frames']}")
    print(f"Ball detection rate: {stats['ball_detection_rate']:.1f}%")

    print(f"\nBall Detection Sources:")
    for source, count in sorted(
        stats["detection_sources"].items(), key=lambda x: -x[1]
    ):
        pct = 100 * count / stats["processed_frames"]
        print(f"  {source:12s}: {count:4d} ({pct:.1f}%)")

    if "court_sources" in stats:
        print(f"\nCourt Detection Sources:")
        for source, count in sorted(
            stats["court_sources"].items(), key=lambda x: -x[1]
        ):
            pct = 100 * count / stats["processed_frames"]
            print(f"  {source:12s}: {count:4d} ({pct:.1f}%)")
    print(f"\nOutput saved to: {args.output}")

    return stats


# Test script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tennis Analyzer - Full video analysis"
    )
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument(
        "--output", "-o", default="output/analyzed.mp4", help="Output video path"
    )
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--weights", default="weights/tracknet.pth")
    parser.add_argument("--courtside-weights", default="weights/courtside_yolo.pt")
    parser.add_argument("--no-display", action="store_true")

    args = parser.parse_args()
    run_full_analysis(args)
