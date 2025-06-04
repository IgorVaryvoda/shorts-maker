"""
Scene detection for FPV drone videos.
Optimized for high-motion aerial footage without audio.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
import logging
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

try:
    from ..utils.config import Config
except ImportError:
    from utils.config import Config

logger = logging.getLogger(__name__)


class SceneSegment:
    """Represents a detected scene segment."""

    def __init__(self, start_time: float, end_time: float, score: float,
                 features: Dict[str, Any]):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.score = score
        self.features = features

    def __repr__(self):
        return f"SceneSegment({self.start_time:.1f}s-{self.end_time:.1f}s, score={self.score:.3f})"


class FPVSceneDetector:
    """Scene detector optimized for FPV drone footage."""

    def __init__(self, config: Config):
        """Initialize FPV scene detector."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Detection parameters
        self.scene_threshold = config.get('segmentation.scene_threshold', 0.4)
        self.motion_threshold = config.get('segmentation.motion_threshold', 0.3)
        self.frame_sample_rate = config.get('segmentation.frame_sample_rate', 0.5)
        self.min_duration = config.get('segmentation.min_duration', 15)
        self.max_duration = config.get('segmentation.max_duration', 60)

        # Scoring weights
        self.visual_weight = config.get('segmentation.scoring.visual_interest', 0.6)
        self.motion_weight = config.get('segmentation.scoring.motion_activity', 0.4)

        # Initialize optical flow detector
        self.optical_flow = cv2.calcOpticalFlowPyrLK
        self.feature_detector = cv2.goodFeaturesToTrack

        logger.info("Initialized FPV Scene Detector")

    def detect_scenes(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect scenes in FPV drone footage.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback for progress updates (frame_num, total_frames, message)

        Returns:
            List of scene dictionaries with timing and scoring information
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        self.logger.info(f"Analyzing video: {total_frames} frames, {duration:.1f}s, {fps:.1f}fps")

        # Calculate sampling interval
        sample_interval = max(1, int(fps / self.frame_sample_rate))

        scenes = []
        frame_data = []

        # Initialize tracking variables
        prev_frame = None
        prev_gray = None
        frame_num = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Update progress for every frame
                if progress_callback:
                    progress_callback(frame_num, total_frames, f"Analyzing frame {frame_num + 1}/{total_frames}")

                # Sample frames at specified rate
                if frame_num % sample_interval == 0:
                    # Convert to grayscale for analysis
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Calculate frame metrics
                    metrics = self._calculate_frame_metrics(frame, gray, prev_gray)

                    frame_data.append({
                        'frame_num': frame_num,
                        'timestamp': frame_num / fps,
                        'metrics': metrics
                    })

                    prev_gray = gray.copy()

                prev_frame = frame.copy()
                frame_num += 1

        finally:
            cap.release()

        # Final progress update
        if progress_callback:
            progress_callback(total_frames, total_frames, f"Completed analysis of {total_frames} frames")

        # Detect scene boundaries
        scene_boundaries = self._detect_scene_boundaries(frame_data)

        # Create scene segments
        scenes = self._create_scene_segments(scene_boundaries, frame_data, fps)

        # Score scenes
        for scene in scenes:
            scene['score'] = self._score_scene(scene)

        self.logger.info(f"Detected {len(scenes)} scenes")

        return scenes

    def _calculate_frame_metrics(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        prev_gray: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate metrics for a single frame."""
        metrics = {}

        # Visual interest (edge density, contrast)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Contrast (standard deviation of pixel intensities)
        contrast = np.std(gray.astype(np.float32))

        # Brightness variance
        brightness_var = np.var(gray.astype(np.float32))

        # Combine visual interest metrics
        visual_interest = (edge_density * 0.4 +
                          (contrast / 255.0) * 0.4 +
                          (brightness_var / 10000.0) * 0.2)

        metrics['visual_interest'] = min(1.0, visual_interest)
        metrics['edge_density'] = edge_density
        metrics['contrast'] = contrast / 255.0
        metrics['brightness_var'] = brightness_var / 10000.0

        # Motion analysis (if previous frame available)
        if prev_gray is not None:
            try:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray,
                    np.array([[x, y] for x in range(0, gray.shape[1], 20)
                             for y in range(0, gray.shape[0], 20)], dtype=np.float32),
                    None
                )

                # Handle the return values properly
                if len(flow) >= 2:
                    p1, status = flow[0], flow[1]

                    if status is not None and len(status) > 0:
                        # Calculate motion magnitude
                        good_points = p1[status.flatten() == 1]
                        if len(good_points) > 0:
                            motion_vectors = good_points - np.array([[x, y] for x in range(0, gray.shape[1], 20)
                                                                   for y in range(0, gray.shape[0], 20)], dtype=np.float32)[status.flatten() == 1]
                            motion_magnitude = np.mean(np.linalg.norm(motion_vectors, axis=1))
                            metrics['motion_magnitude'] = min(1.0, motion_magnitude / 50.0)
                        else:
                            metrics['motion_magnitude'] = 0.0
                    else:
                        metrics['motion_magnitude'] = 0.0
                else:
                    metrics['motion_magnitude'] = 0.0

                # Frame difference
                frame_diff = cv2.absdiff(prev_gray, gray)
                diff_score = np.mean(frame_diff) / 255.0
                metrics['frame_difference'] = diff_score

                # SSIM for scene change detection
                try:
                    ssim_score = ssim(prev_gray, gray)
                    metrics['ssim'] = ssim_score
                    metrics['scene_change'] = 1.0 - ssim_score
                except Exception as e:
                    self.logger.debug(f"SSIM calculation failed: {e}")
                    metrics['ssim'] = 1.0
                    metrics['scene_change'] = 0.0

            except Exception as e:
                self.logger.debug(f"Motion analysis failed: {e}")
                metrics['motion_magnitude'] = 0.0
                metrics['frame_difference'] = 0.0
                metrics['ssim'] = 1.0
                metrics['scene_change'] = 0.0
        else:
            metrics['motion_magnitude'] = 0.0
            metrics['frame_difference'] = 0.0
            metrics['ssim'] = 1.0
            metrics['scene_change'] = 0.0

        return metrics

    def _detect_scene_boundaries(self, frame_data: List[Dict[str, Any]]) -> List[int]:
        """Detect scene boundaries based on frame metrics."""
        if len(frame_data) < 2:
            return [0, len(frame_data) - 1]

        boundaries = [0]  # Start with first frame

        # Look for significant changes in scene_change metric
        for i in range(1, len(frame_data) - 1):
            scene_change = frame_data[i]['metrics'].get('scene_change', 0.0)

            # Add safety check for division by zero
            if scene_change > self.scene_threshold:
                # Ensure minimum distance between boundaries
                if len(boundaries) == 0 or i - boundaries[-1] > 10:
                    boundaries.append(i)

        # Add final frame
        if boundaries[-1] != len(frame_data) - 1:
            boundaries.append(len(frame_data) - 1)

        return boundaries

    def _create_scene_segments(
        self,
        boundaries: List[int],
        frame_data: List[Dict[str, Any]],
        fps: float
    ) -> List[Dict[str, Any]]:
        """Create scene segments from boundaries."""
        scenes = []

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            start_time = frame_data[start_idx]['timestamp']
            end_time = frame_data[end_idx]['timestamp']
            duration = end_time - start_time

            # Filter by duration
            if duration < self.min_duration or duration > self.max_duration:
                continue

            # Calculate average metrics for the scene
            scene_frames = frame_data[start_idx:end_idx + 1]
            avg_metrics = self._average_metrics(scene_frames)

            scene = {
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'start_frame': frame_data[start_idx]['frame_num'],
                'end_frame': frame_data[end_idx]['frame_num'],
                'metrics': avg_metrics
            }

            scenes.append(scene)

        return scenes

    def _average_metrics(self, scene_frames: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average metrics for a scene."""
        if not scene_frames:
            return {}

        # Get all metric keys
        metric_keys = set()
        for frame in scene_frames:
            metric_keys.update(frame['metrics'].keys())

        # Calculate averages
        avg_metrics = {}
        for key in metric_keys:
            values = [frame['metrics'].get(key, 0.0) for frame in scene_frames]
            avg_metrics[key] = np.mean(values)

        return avg_metrics

    def _score_scene(self, scene: Dict[str, Any]) -> float:
        """Score a scene based on its metrics."""
        metrics = scene.get('metrics', {})

        # Visual interest score
        visual_score = (
            metrics.get('visual_interest', 0.0) * 0.4 +
            metrics.get('edge_density', 0.0) * 0.3 +
            metrics.get('contrast', 0.0) * 0.3
        )

        # Motion score
        motion_score = (
            metrics.get('motion_magnitude', 0.0) * 0.7 +
            metrics.get('frame_difference', 0.0) * 0.3
        )

        # Duration bonus (prefer scenes closer to target duration)
        target_duration = self.config.get('segmentation.target_duration', 30)
        duration_score = 1.0 - abs(scene['duration'] - target_duration) / target_duration
        duration_score = max(0.0, duration_score)

        # Combine scores
        total_score = (
            visual_score * self.visual_weight +
            motion_score * self.motion_weight +
            duration_score * 0.2
        )

        return min(1.0, total_score)


def detect_scenes(video_path: str, config: Dict[str, Any]) -> List[SceneSegment]:
    """
    Detect scenes in FPV drone video.

    Args:
        video_path: Path to video file
        config: Configuration dictionary

    Returns:
        List of detected scene segments
    """
    detector = FPVSceneDetector(config)
    return detector.detect_scenes(video_path)