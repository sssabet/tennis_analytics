# Tennis Analytics

Tennis video analysis tool with AI-powered ball tracking, player pose estimation, and court detection.

<img width="1913" height="750" alt="readme" src="https://github.com/user-attachments/assets/92c79b55-042d-440e-afe4-b116a8d0a8b8" />

## Installation

```bash
pip install -r requirements.txt
```

Ensure model weights are in the `weights/` folder, with git lfs:
- `tracknet.pth` - TrackNet ball tracking
- `courtside_yolo.pt` - YOLO ball detection
- `court_keypoints.pth` - Court keypoint detection
- `yolov8_pose.pt` - Player pose estimation

## Usage

### Basic Usage

```bash
python main.py --input video.mp4
```

### Options

| Option | Description |
|--------|-------------|
| `--input`, `-i` | Input video path (required) |
| `--output`, `-o` | Output video path (default: `input_traced.mp4`) |
| `--ball-only` | Ball tracking only (faster) |
| `--device` | `cuda`, `cpu`, or `auto` (default: `auto`) |
| `--no-display` | Run without display window |
| `--detector` | `fusion`, `tracknet`, or `courtside` (default: `fusion`) |

### Examples

```bash
# Full analysis
python main.py --input video.mp4

# Ball tracking only
python main.py --input video.mp4 --ball-only

# Force GPU usage
python main.py --input video.mp4 --device cuda

# Custom output, no display
python main.py --input video.mp4 --output result.mp4 --no-display
```

## Features

- **Ball Tracking**: Fusion of TrackNet + CourtSide YOLO + Kalman Filter
- **Player Detection**: YOLOv8 pose estimation with skeleton overlay
- **Court Detection**: ResNet-50 keypoint detection
- **Audio Analysis**: Shot detection from audio peaks

## How it Works

### Ball Detection
The system uses a robust fusion approach for ball tracking:
- **TrackNet**: A specialized deep learning model trained for tennis ball tracking in broadcast videos.
- **CourtSide YOLO**: A YOLO-based detector used as a secondary source to improve detection in various conditions.
- **Kalman Filter**: Fuses detections from both models and provides smooth predictions even when the ball is temporarily obscured or missed by the detectors.
- **Interpolation**: Fills in gaps in the trajectory to provide a continuous ball path.

### Court Detection
Court detection is handled through a multi-stage process:
- **ResNet-50 Keypoint Detector**: A fine-tuned ResNet-50 model identifies 14 critical court keypoints (corners, service lines, etc.).
- **Classical CV Fallback**: If the deep learning model fails, the system falls back to classical computer vision techniques (Hough lines, Canny edges, and contour analysis) to identify the court boundaries.
- **Homography**: Uses the detected keypoints to map the court to a top-down view for tactical analysis.

### Shot Detection
The system analyzes the audio stream from the video to detect when a ball is hit:
- **Audio Extraction**: Extracts the audio track from the video using FFmpeg.
- **Intensity Analysis**: Calculates the RMS (Root Mean Square) intensity for each frame.
- **Peak Detection**: Identifies sharp spikes in audio intensity that correspond to the sound of a tennis shot, using prominence-based peak detection from `scipy`.
- **Visual Feedback**: Provides a real-time waveform and shot indicator in the output video.

## Project Structure

```
TRACE/
├── main.py                      # CLI entry point
├── TennisAnalyzer.py           # Main analysis orchestrator
├── AudioAnalyzer.py            # Audio-based shot detection
├── BallTrackerFusion.py        # Ball detection fusion
├── BallDetection.py            # TrackNet ball detector
├── BallTrackNet.py             # TrackNet model architecture
├── TennisCourtKeypointDetector.py  # Court keypoint detection
├── CourtMapping.py             # Court homography utilities
├── BallMapping.py              # Ball position utilities
├── TraceHeader.py              # Shared utilities
├── requirements.txt            # Dependencies
└── weights/                    # Model weights

## Credits & References

This project utilizes several open-source models and research:

- **TrackNet**: Based on [TrackNet: Tennis Ball Tracking from Broadcast Video by Deep Learning Networks](https://github.com/yuchuanhuang/TrackNet) by Yu-Chuan Huang.
- **Court Detection**: ResNet-50 keypoint detection inspired by [Muhammad Moin Faisal's Tennis Analysis](https://github.com/MuhammadMoinFaisal/tennis_analysis).
- **CourtSide YOLO**: Ball detection using weights from the [CourtSide](https://github.com/viren-m-mehta/CourtSide) project.
- **Player Pose**: Human pose estimation powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
```
