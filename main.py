#!/usr/bin/env python3
"""
TRACE - Tennis video analysis CLI
Processes tennis videos to detect court lines, ball, and player landmarks.
"""

import argparse
import sys
import os
from pathlib import Path

def _resolve_path(path_str: str, base_dir: Path) -> str:
    """
    Resolve a path in a user-friendly way:
    - If given path exists as-is (absolute or relative to CWD), keep it.
    - Otherwise, try resolving relative to base_dir (repo/script directory).
    - Fall back to the original string.
    """
    if path_str is None:
        return None
    p = Path(path_str)
    if p.exists():
        return str(p)
    alt = base_dir / path_str
    if alt.exists():
        return str(alt)
    return str(p)

def main():
    parser = argparse.ArgumentParser(
        description='TRACE: Tennis video analysis tool for detecting court lines, ball, and player landmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input video.mp4                     # Full analysis (default)
  python main.py --input video.mp4 --ball-only         # Ball tracking only
  python main.py --input video.mp4 --device cuda       # Force GPU
  python main.py --input video.mp4 --output result.mp4 --no-display
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to output video file (default: input_filename_traced.mp4)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run without displaying video frames (useful for headless servers)'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default='weights/tracknet.pth',
        help='Path to TrackNet weights file (default: weights/tracknet.pth)'
    )
    
    parser.add_argument(
        '--courtside-weights',
        type=str,
        default='weights/courtside_yolo.pt',
        help='Path to CourtSide YOLO weights file (default: weights/courtside_yolo.pt)'
    )
    
    parser.add_argument(
        '--ball-only',
        action='store_true',
        help='Only run ball tracking with fusion (TrackNet + CourtSide + Kalman)'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='(Default) Full analysis: ball + players + court + audio detection'
    )
    
    parser.add_argument(
        '--detector',
        type=str,
        choices=['fusion', 'tracknet', 'courtside'],
        default='fusion',
        help='Ball detector to use: fusion (both+Kalman), tracknet, or courtside (default: fusion)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use for inference: cuda, cpu, or auto (default: auto)'
    )
    
    args = parser.parse_args()

    # Resolve default/resource paths relative to this script when needed
    base_dir = Path(__file__).resolve().parent
    args.weights = _resolve_path(args.weights, base_dir)
    args.courtside_weights = _resolve_path(args.courtside_weights, base_dir)
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Set output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_traced{input_path.suffix}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Ball-only mode with fusion tracker
    if args.ball_only:
        run_ball_tracking_only(args)
        return
    
    # Default: Full analysis mode (ball + players + court + audio)
    from TennisAnalyzer import run_full_analysis
    run_full_analysis(args)


def run_ball_tracking_only(args):
    """
    Run ball tracking only with fusion detector (TrackNet + CourtSide + Kalman Filter)
    """
    import cv2
    import torch
    from collections import Counter
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Ball Tracking Mode: {args.detector}")
    print(f"Device: {device}")
    print(f"Processing video: {args.input}")
    print(f"Output will be saved to: {args.output}")
    
    # Initialize detector based on mode
    if args.detector == 'fusion':
        from BallTrackerFusion import BallTrackerFusion
        tracker = BallTrackerFusion(
            tracknet_weights=args.weights,
            courtside_weights=args.courtside_weights,
            device=device
        )
        detect_func = lambda frame: tracker.detect_ball(frame)
    elif args.detector == 'tracknet':
        from BallDetection import BallDetector
        detector = BallDetector(args.weights, out_channels=2, device=device)
        def detect_func(frame):
            detector.detect_ball(frame)
            pos = detector.xy_coordinates[-1]
            if pos[0] is not None:
                return int(pos[0]), int(pos[1]), 'tracknet'
            return None, None, 'none'
    else:  # courtside
        from BallTrackerFusion import CourtSideDetector
        detector = CourtSideDetector(args.courtside_weights, device=device)
        def detect_func(frame):
            result = detector.detect(frame)
            if result:
                return result[0], result[1], 'courtside'
            return None, None, 'none'
    
    # Open video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{args.input}'", file=sys.stderr)
        sys.exit(1)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Process frames
    sources = Counter()
    frame_num = 0
    trajectory = []
    
    print(f"Processing {total_frames} frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        x, y, source = detect_func(frame)
        sources[source] += 1
        
        # Draw ball position
        if x is not None and y is not None:
            trajectory.append((x, y, frame_num))
            
            # Color based on source
            colors = {
                'fused': (0, 255, 0),      # Green
                'tracknet': (255, 0, 0),    # Blue
                'courtside': (0, 165, 255), # Orange
                'kalman': (255, 255, 0),    # Cyan
            }
            color = colors.get(source, (255, 255, 255))
            
            cv2.circle(frame, (x, y), 8, color, -1)
            cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(frame, source, (x + 15, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw trajectory trail (last 30 points)
        trail_points = [(t[0], t[1]) for t in trajectory[-30:]]
        for i in range(1, len(trail_points)):
            alpha = i / len(trail_points)
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, trail_points[i-1], trail_points[i], (0, 255, 255), thickness)
        
        # Write frame
        out.write(frame)
        
        # Display if not headless
        if not args.no_display:
            cv2.imshow('Ball Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break
        
        frame_num += 1
        if frame_num % 50 == 0:
            print(f"  Processed {frame_num}/{total_frames} frames ({100*frame_num/total_frames:.1f}%)")
    
    # Cleanup
    cap.release()
    out.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    
    # Print statistics
    detections = sum(1 for s in sources.keys() if s != 'none') 
    detected_frames = total_frames - sources.get('none', 0)
    
    print(f"\n{'='*50}")
    print(f"Ball Tracking Complete!")
    print(f"{'='*50}")
    print(f"Total frames: {total_frames}")
    print(f"Detected in: {detected_frames} frames ({100*detected_frames/total_frames:.1f}%)")
    print(f"\nDetection sources:")
    for source, count in sources.most_common():
        print(f"  {source:12s}: {count:4d} frames ({100*count/total_frames:.1f}%)")
    print(f"\nOutput saved to: {args.output}")


if __name__ == '__main__':
    main()

