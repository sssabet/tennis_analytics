"""
Audio Analyzer for Tennis Shot Detection
Analyzes audio intensity and pitch to detect ball hits
"""

import numpy as np
import subprocess
import os
import tempfile
from scipy import signal
from scipy.io import wavfile


class AudioAnalyzer:
    """Analyzes audio from video to detect tennis shot sounds"""
    
    def __init__(self, video_path, fps=30, peak_threshold: float = 0.45, min_shot_interval: int = 30):
        self.video_path = video_path
        self.fps = fps
        self.audio_data = None
        self.sample_rate = None
        self.intensity_per_frame = None
        self.shot_frames = []
        # Normalized threshold for peak detection (0-1). Lower => more sensitive.
        self.peak_threshold = float(peak_threshold)
        # Minimum frames between detected peaks (~1 sec at 30fps).
        self.min_shot_interval = int(min_shot_interval)
        
    def extract_audio(self):
        """Extract audio from video using ffmpeg"""
        try:
            # Create temp file for audio
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav.close()
            
            # Extract audio using ffmpeg
            cmd = [
                'ffmpeg', '-y', '-i', self.video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1',
                temp_wav.name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if os.path.exists(temp_wav.name) and os.path.getsize(temp_wav.name) > 0:
                self.sample_rate, self.audio_data = wavfile.read(temp_wav.name)
                os.unlink(temp_wav.name)
                print(f"  Audio extracted: {len(self.audio_data)} samples at {self.sample_rate}Hz")
                return True
            else:
                print("  Warning: Could not extract audio from video")
                if os.path.exists(temp_wav.name):
                    os.unlink(temp_wav.name)
                return False
                
        except Exception as e:
            print(f"  Audio extraction error: {e}")
            return False
    
    def analyze_intensity(self):
        """Calculate audio intensity for each video frame"""
        if self.audio_data is None:
            return None
        
        # Samples per frame
        samples_per_frame = int(self.sample_rate / self.fps)
        num_frames = len(self.audio_data) // samples_per_frame
        
        # Calculate RMS intensity for each frame
        self.intensity_per_frame = []
        for i in range(num_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            frame_audio = self.audio_data[start:end].astype(float)
            
            # RMS (Root Mean Square) for intensity
            rms = np.sqrt(np.mean(frame_audio ** 2))
            self.intensity_per_frame.append(rms)
        
        # Normalize to 0-1 range
        if self.intensity_per_frame:
            max_intensity = max(self.intensity_per_frame)
            if max_intensity > 0:
                self.intensity_per_frame = [i / max_intensity for i in self.intensity_per_frame]
        
        return self.intensity_per_frame
    
    def detect_shots_from_audio(self):
        """Detect shot frames based on audio peaks"""
        if self.intensity_per_frame is None:
            return []
        
        intensity = np.array(self.intensity_per_frame)
        
        # Find peaks in audio intensity
        # Use scipy's find_peaks with prominence for better detection
        peaks, properties = signal.find_peaks(
            intensity,
            height=self.peak_threshold,
            distance=self.min_shot_interval,
            prominence=0.2
        )
        
        self.shot_frames = peaks.tolist()
        print(f"  Detected {len(self.shot_frames)} potential shots from audio")
        
        return self.shot_frames
    
    def get_intensity_at_frame(self, frame_idx):
        """Get audio intensity for a specific frame"""
        if self.intensity_per_frame is None or frame_idx >= len(self.intensity_per_frame):
            return 0.0
        return self.intensity_per_frame[frame_idx]
    
    def is_shot_frame(self, frame_idx, tolerance=3):
        """Check if a frame is near a detected audio shot"""
        for shot_frame in self.shot_frames:
            if abs(frame_idx - shot_frame) <= tolerance:
                return True
        return False
    
    def get_frequency_at_frame(self, frame_idx, window_size=2048):
        """Get dominant frequency for a frame (for pitch detection)"""
        if self.audio_data is None:
            return 0.0
        
        samples_per_frame = int(self.sample_rate / self.fps)
        start = frame_idx * samples_per_frame
        end = min(start + window_size, len(self.audio_data))
        
        if end - start < 256:
            return 0.0
        
        frame_audio = self.audio_data[start:end].astype(float)
        
        # Apply window function
        windowed = frame_audio * np.hanning(len(frame_audio))
        
        # FFT
        fft = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(windowed), 1.0 / self.sample_rate)
        
        # Find dominant frequency (ignore very low frequencies)
        valid_mask = freqs > 100  # Ignore below 100Hz
        if np.any(valid_mask):
            valid_fft = fft[valid_mask]
            valid_freqs = freqs[valid_mask]
            dominant_idx = np.argmax(valid_fft)
            return valid_freqs[dominant_idx]
        
        return 0.0


def draw_audio_overlay(frame, intensity, is_shot=False, x=20, y=None, width=200, height=30):
    """Draw audio intensity bar on frame"""
    import cv2
    
    if y is None:
        y = frame.shape[0] - 60  # Bottom of frame
    
    # Background bar
    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 1)
    
    # Intensity fill
    fill_width = int(width * min(intensity, 1.0))
    if fill_width > 0:
        # Color based on intensity (green -> yellow -> red)
        if intensity < 0.5:
            color = (0, int(255 * intensity * 2), 0)  # Green to yellow
        else:
            color = (0, 255, int(255 * (1 - intensity) * 2))  # Yellow to red
        
        if is_shot:
            color = (0, 0, 255)  # Red for detected shots
        
        cv2.rectangle(frame, (x + 2, y + 2), (x + fill_width - 2, y + height - 2), color, -1)
    
    # Label
    label = "SHOT!" if is_shot else "Audio"
    label_color = (0, 0, 255) if is_shot else (255, 255, 255)
    cv2.putText(frame, label, (x + width + 10, y + height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
    
    return frame


def draw_audio_waveform(frame, intensities, current_frame, x=20, y=None, width=300, height=50):
    """Draw recent audio waveform on frame"""
    import cv2
    
    if y is None:
        y = frame.shape[0] - 120  # Above the intensity bar
    
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 1)
    
    # Draw waveform (last N frames)
    window = 100  # Show last 100 frames
    start_frame = max(0, current_frame - window)
    end_frame = min(len(intensities), current_frame + 1)
    
    if end_frame > start_frame:
        wave_data = intensities[start_frame:end_frame]
        num_points = len(wave_data)
        
        if num_points > 1:
            points = []
            for i, intensity in enumerate(wave_data):
                px = x + int((i / window) * width)
                py = y + height - int(intensity * height * 0.9) - 2
                points.append((px, py))
            
            # Draw waveform line
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (0, 200, 0), 1)
            
            # Mark current position
            if points:
                cv2.circle(frame, points[-1], 3, (0, 255, 255), -1)
    
    # Center line
    center_y = y + height // 2
    cv2.line(frame, (x, center_y), (x + width, center_y), (60, 60, 60), 1)
    
    # Threshold line (visual guide only; should roughly match default AudioAnalyzer.peak_threshold)
    threshold_y = y + height - int(0.45 * height * 0.9) - 2
    cv2.line(frame, (x, threshold_y), (x + width, threshold_y), (0, 0, 150), 1, cv2.LINE_AA)
    
    return frame

