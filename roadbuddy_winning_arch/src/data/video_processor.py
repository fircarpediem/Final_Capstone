"""
Video Processing Module
Handles dual-stream video processing: global context + high-res keyframes
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple
from pathlib import Path
from loguru import logger


class VideoProcessor:
    """
    Dual-stream video processor for RoadBuddy
    
    Stream A: Global context (low-res, many frames)
    Stream B: Local detail (high-res, key frames)
    """
    
    def __init__(self, config):
        self.config = config
        self.target_fps = config.data.target_fps
        self.max_frames_global = config.data.max_frames_global
        self.num_keyframes = config.data.num_keyframes
        self.global_resolution = tuple(config.data.global_resolution)
        self.native_resolution = tuple(config.data.native_resolution)
        
        logger.info(f"VideoProcessor initialized:")
        logger.info(f"  Global: {self.max_frames_global} frames @ {self.global_resolution}")
        logger.info(f"  Keyframes: {self.num_keyframes} frames @ native resolution")
    
    def process(self, video_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Main processing pipeline
        
        Args:
            video_path: Path to video file
            
        Returns:
            global_frames: List of low-res frames for context
            keyframes_highres: List of high-res keyframes for detail
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.debug(f"Processing video: {video_path.name}")
        
        # Load all frames
        all_frames = self._load_video(video_path)
        
        if len(all_frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")
        
        logger.debug(f"Loaded {len(all_frames)} frames")
        
        # Stream A: Global context
        global_frames = self._sample_global_frames(all_frames)
        global_frames = [self._resize(f, self.global_resolution) for f in global_frames]
        
        # Stream B: Keyframes
        keyframe_indices = self._select_keyframes(all_frames)
        keyframes_highres = [all_frames[i] for i in keyframe_indices]
        
        logger.debug(f"Sampled {len(global_frames)} global frames, {len(keyframes_highres)} keyframes")
        
        return global_frames, keyframes_highres
    
    def _load_video(self, video_path: Path) -> List[np.ndarray]:
        """Load all frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def _sample_global_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Sample frames for global context
        Uses uniform sampling
        """
        num_frames = len(frames)
        target_count = min(self.max_frames_global, num_frames)
        
        indices = np.linspace(0, num_frames - 1, target_count, dtype=int)
        sampled = [frames[i] for i in indices]
        
        return sampled
    
    def _select_keyframes(self, frames: List[np.ndarray]) -> List[int]:
        """
        Intelligent keyframe selection
        
        Strategy:
        1. Baseline: Start, middle, end
        2. TODO: Scene change detection
        3. TODO: Quality-based selection
        """
        num_frames = len(frames)
        
        if self.num_keyframes == 3:
            # Simple baseline: evenly distributed
            return [
                0,                          # First frame
                num_frames // 2,            # Middle frame
                num_frames - 1              # Last frame
            ]
        else:
            # General case: uniform distribution
            indices = np.linspace(0, num_frames - 1, self.num_keyframes, dtype=int)
            return indices.tolist()
    
    def _resize(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize frame to target size
        
        Args:
            frame: Input frame (H, W, C)
            target_size: (width, height)
        """
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    
    def extract_frame_at_timestamp(self, video_path: str, timestamp: float) -> np.ndarray:
        """Extract single frame at specific timestamp (in seconds)"""
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        cap.release()
        
        if not ret:
            raise ValueError(f"Cannot extract frame at {timestamp}s")
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class SceneChangeDetector:
    """
    Advanced scene change detection for intelligent keyframe selection
    TODO: Implement for future improvement
    """
    
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold
    
    def detect(self, frames: List[np.ndarray]) -> List[int]:
        """
        Detect scene changes in video
        
        Returns:
            List of frame indices where scene changes occur
        """
        changes = []
        
        for i in range(1, len(frames)):
            # Compute frame difference
            diff = self._compute_difference(frames[i-1], frames[i])
            
            if diff > self.threshold:
                changes.append(i)
        
        return changes
    
    def _compute_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute difference between two frames"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Compute mean absolute difference
        diff = np.mean(np.abs(gray1.astype(float) - gray2.astype(float)))
        
        return diff


class FrameQualityScorer:
    """
    Score frame quality (blur, brightness, etc.)
    TODO: Implement for future improvement
    """
    
    def __init__(self):
        pass
    
    def score(self, frame: np.ndarray) -> float:
        """
        Score frame quality (0-1, higher is better)
        
        Factors:
        - Blur (Laplacian variance)
        - Brightness (mean intensity)
        - Contrast (std intensity)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Blur score (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(laplacian_var / 1000, 1.0)  # Normalize
        
        # Brightness score (prefer not too dark/bright)
        mean_intensity = gray.mean() / 255.0
        brightness_score = 1.0 - abs(mean_intensity - 0.5) * 2
        
        # Contrast score
        std_intensity = gray.std() / 128.0
        contrast_score = min(std_intensity, 1.0)
        
        # Combined score
        quality = (blur_score * 0.5 + brightness_score * 0.3 + contrast_score * 0.2)
        
        return quality
