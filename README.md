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
# Custom output, no display
python main.py --input video.mp4 --output result.mp4 --no-display
```

## Features

- **Ball Tracking**: Fusion of TrackNet + YOLO (from CourtSide) + Kalman Filter
- **Player Detection**: YOLOv8 pose estimation with skeleton overlay
- **Court Detection**: ResNet-50 keypoint detection
- **Audio Analysis**: Shot detection from audio peaks

## How it Works

### Ball Detection
The system uses a fusion approach for ball tracking:
- **TrackNet**: A deep learning model trained for tennis ball tracking in broadcast videos.
- **CourtSide YOLO**: Fine-tuned YOLOv11n model used as a secondary source to improve detection in various conditions.  https://huggingface.co/Davidsv/CourtSide-Computer-Vision-v1
- **Kalman Filter**: Fuses detections from both models, smooths the trajectory, and predicts ball position during short occlusions.

### Court Detection
Court detection is handled through a multi-stage process:
- **ResNet-50 Keypoint Detector**: A ResNet-50 model (23.6M parameters) identifies 14 critical court keypoints (corners, service lines, etc.).
- **Homography**: Uses the detected keypoints (from either method) to map the court to a top-down view for tactical analysis.

### Shot Detection
The system uses a **fusion approach** combining audio analysis and ball movement to detect shots:
- **Audio Extraction**: Extracts the audio track from the video using FFmpeg.
- **Intensity Analysis**: Calculates the RMS (Root Mean Square) intensity for each frames which provides a measure of audio energy/volume per frame. Then identifies sharp spikes in audio intensity that correspond to the sound of a tennis shot, using prominence-based peak detection from `scipy`.
- **Ball Net-Crossing Detection**: Tracks ball position in court-map coordinates and detects when the ball crosses the net (changes from one side to the other). This uses the ball's Y-coordinate relative to the net line: `crossed = (prev_side * cur_side < 0)` where sides are calculated as `ball_y - net_y`.
- **Fusion Logic**: A shot is only counted when **both conditions are met**: (1) the ball crosses the net, AND (2) there is an audio peak within ±10 frames of the net-crossing event. This dual-requirement reduces false positives from crowd noise or ball bounces that don't result in shots.

## Credits & References

This project utilizes several open-source models and research:

- **TrackNet**: Based on [TrackNet: Tennis Ball Tracking from Broadcast Video by Deep Learning Networks](https://github.com/yuchuanhuang/TrackNet) by Yu-Chuan Huang.
- **Court Detection**: ResNet-50 keypoint detection architecture based on [Muhammad Moin Faisal's Tennis Analysis](https://github.com/MuhammadMoinFaisal/tennis_analysis).
- **CourtSide YOLO**: Ball detection using weights from the [CourtSide](https://github.com/viren-m-mehta/CourtSide) project.
- **Player Pose**: Human pose estimation powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
```
