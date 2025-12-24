# TRACE

**Tennis video analysis with AI**

Detects court lines, ball, players with pose estimation, and audio-based shot detection.

![Video](https://github.com/hgupt3/TRACE/assets/112455192/627e8ca6-86c1-4409-938d-2b45e875bbfa)

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the model weights in the `weights/` folder:
   - `weights/tracknet.pth` - Ball tracking
   - `weights/courtside_yolo.pt` - YOLO ball detection
   - `weights/court_keypoints.pth` - Court keypoint detection
   - `weights/yolov8_pose.pt` - Player pose estimation

## Usage

### Basic Usage

```bash
python main.py --input video.mp4
```

### Options

| Option | Description |
|--------|-------------|
| `--input`, `-i` | Path to input video (required) |
| `--output`, `-o` | Path to output video (default: `input_traced.mp4`) |
| `--no-display` | Run without display window |
| `--device` | `cuda`, `cpu`, or `auto` (default: `auto`) |
| `--ball-only` | Only run ball tracking (faster) |

### Examples

```bash
# Full analysis with GPU
python main.py --input video.mp4 --device cuda

# Ball tracking only (faster)
python main.py --input video.mp4 --ball-only

# Custom output, no display
python main.py --input video.mp4 --output result.mp4 --no-display
```

## Features

### Video Output Layout

The output video has a **sidebar on the left** containing:
- **Court View** - Top-down homography view
- **Audio Waveform** - Real-time audio intensity
- **Shot Detection** - Visual indicator when shots are detected
- **Detection Info** - Ball/court detection sources

The **main video** on the right shows:
- Player pose skeletons (YOLOv8)
- Ball tracking with trajectory
- Court keypoints

### Detection Methods

| Component | Method |
|-----------|--------|
| **Ball** | TrackNet + CourtSide YOLO + Kalman Filter fusion |
| **Players** | YOLOv8-pose with skeleton overlay |
| **Court** | ResNet-50 keypoint detection |
| **Shots** | Audio peak detection |

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
└── weights/
    ├── tracknet.pth            # TrackNet ball tracking
    ├── courtside_yolo.pt       # YOLO ball detection
    ├── court_keypoints.pth     # Court keypoint detection
    └── yolov8_pose.pt          # YOLOv8 pose estimation
```

## Required Libraries

- opencv-python
- torch
- numpy
- mediapipe
- ultralytics
- scipy
- filterpy

Install all:
```bash
pip install -r requirements.txt
```

## Credits

Ball tracking based on:
> Yu-Chuan Huang, "TrackNet: Tennis Ball Tracking from Broadcast Video by Deep Learning Networks," Master Thesis, National Chiao Tung University, Taiwan, 2018.
