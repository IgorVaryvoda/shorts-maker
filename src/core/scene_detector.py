"""
Scene detection for FPV drone videos.
Optimized for high-motion aerial footage without audio.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

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


class FastFPVSceneDetector:
    """Ultra-fast scene detector optimized for FPV drone footage."""

    def __init__(self, config: Config):
        """Initialize fast FPV scene detector."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Detection parameters - optimized for speed
        self.scene_threshold = config.get('segmentation.scene_threshold', 0.2)
        self.motion_threshold = config.get('segmentation.motion_threshold', 0.1)
        self.frame_sample_rate = config.get('segmentation.frame_sample_rate', 0.2)
        self.min_duration = config.get('segmentation.min_duration', 10)
        self.max_duration = config.get('segmentation.max_duration', 90)

        # Scoring weights
        self.visual_weight = config.get('segmentation.scoring.visual_interest', 0.6)
        self.motion_weight = config.get('segmentation.scoring.motion_activity', 0.4)

        # Performance optimizations
        self.use_threading = config.get('performance.num_threads', 4) > 1
        self.downsample_factor = 4  # Process at 1/4 resolution for speed
        self.adaptive_sampling = True  # Sample more frames in high-motion areas

        # Content filtering settings
        self.avoid_low_motion = config.get('segmentation.content_filtering.avoid_low_motion', True)
        self.min_motion_threshold = config.get('segmentation.content_filtering.min_motion_threshold', 0.05)
        self.avoid_static_scenes = config.get('segmentation.content_filtering.avoid_static_scenes', True)
        self.min_visual_interest = config.get('segmentation.content_filtering.min_visual_interest', 0.3)

        logger.info("Initialized Fast FPV Scene Detector with optimizations and content filtering")

    def detect_scenes(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ultra-fast scene detection using optimized algorithms.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps

        # Calculate downsampled dimensions for speed
        small_width = width // self.downsample_factor
        small_height = height // self.downsample_factor

        self.logger.info(f"Analyzing video: {total_frames} frames, {duration:.1f}s, {fps:.1f}fps")
        self.logger.info(f"Processing at {small_width}x{small_height} for speed")

        # Use adaptive sampling for better quality
        frame_indices = self._calculate_adaptive_sampling(total_frames, fps)

        self.logger.info(f"Sampling {len(frame_indices)} frames (adaptive sampling)")

        scenes = []
        frame_data = []

        # Process frames in batches for better performance
        batch_size = 50
        prev_small_gray = None

        try:
            for batch_start in range(0, len(frame_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(frame_indices))
                batch_indices = frame_indices[batch_start:batch_end]

                # Process batch
                batch_data = self._process_frame_batch(
                    cap, batch_indices, fps, small_width, small_height, prev_small_gray
                )

                frame_data.extend(batch_data)

                # Update progress
                if progress_callback:
                    progress = batch_end
                    total = len(frame_indices)
                    progress_callback(progress, total, f"Analyzing frame {progress}/{total} (fast mode)")

                # Update previous frame for next batch
                if batch_data:
                    prev_small_gray = batch_data[-1].get('small_gray')

        finally:
            cap.release()

        # Final progress update
        if progress_callback:
            progress_callback(len(frame_indices), len(frame_indices), f"Completed fast analysis of {len(frame_indices)} frames")

        # Detect scene boundaries using optimized algorithm
        scene_boundaries = self._detect_scene_boundaries_fast(frame_data)

        # Create scene segments
        scenes = self._create_scene_segments(scene_boundaries, frame_data, fps)

        # Score scenes with improved algorithm
        for scene in scenes:
            scene['score'] = self._score_scene_fast(scene)

        self.logger.info(f"Detected {len(scenes)} scenes using fast algorithm")

        return scenes

    def _calculate_adaptive_sampling(self, total_frames: int, fps: float) -> List[int]:
        """Calculate adaptive frame sampling for better quality."""
        base_interval = max(1, int(fps / self.frame_sample_rate))

        if not self.adaptive_sampling:
            return list(range(0, total_frames, base_interval))

        # Adaptive sampling: more frames at the beginning and end, fewer in middle
        indices = []

        # Dense sampling for first 30 seconds (scene changes more likely)
        dense_frames = min(int(30 * fps), total_frames // 3)
        dense_interval = max(1, int(fps / 1.0))  # 1 fps for first part
        indices.extend(range(0, dense_frames, dense_interval))

        # Sparse sampling for middle section
        middle_start = dense_frames
        middle_end = total_frames - dense_frames
        if middle_end > middle_start:
            sparse_interval = max(base_interval * 2, int(fps / 0.1))  # 0.1 fps for middle
            indices.extend(range(middle_start, middle_end, sparse_interval))

        # Dense sampling for last 30 seconds
        if total_frames > dense_frames:
            final_start = max(middle_end, total_frames - dense_frames)
            indices.extend(range(final_start, total_frames, dense_interval))

        # Remove duplicates and sort
        indices = sorted(list(set(indices)))

        return indices

    def _process_frame_batch(
        self,
        cap: cv2.VideoCapture,
        frame_indices: List[int],
        fps: float,
        small_width: int,
        small_height: int,
        prev_small_gray: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Process a batch of frames efficiently."""
        batch_data = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # Downsample for speed
            small_frame = cv2.resize(frame, (small_width, small_height))
            small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # Fast metrics calculation
            metrics = self._calculate_fast_metrics(small_gray, prev_small_gray)

            frame_data = {
                'frame_num': frame_idx,
                'timestamp': frame_idx / fps,
                'metrics': metrics,
                'small_gray': small_gray.copy()  # Keep for next iteration
            }

            batch_data.append(frame_data)
            prev_small_gray = small_gray

        return batch_data

    def _calculate_fast_metrics(
        self,
        small_gray: np.ndarray,
        prev_small_gray: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate metrics optimized for speed."""
        metrics = {}

        # Fast visual interest using histogram
        hist = cv2.calcHist([small_gray], [0], None, [32], [0, 256])  # Reduced bins for speed
        hist_norm = hist.flatten() / hist.sum()
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        visual_interest = entropy / 5.0  # Normalize to 0-1

        metrics['visual_interest'] = min(1.0, visual_interest)
        metrics['entropy'] = entropy

        # Fast contrast using standard deviation
        contrast = np.std(small_gray.astype(np.float32)) / 255.0
        metrics['contrast'] = contrast

        # Motion analysis (if previous frame available)
        if prev_small_gray is not None:
            # Simple frame difference for speed
            diff = cv2.absdiff(prev_small_gray, small_gray)
            motion_score = np.mean(diff) / 255.0

            metrics['motion_magnitude'] = motion_score
            metrics['frame_difference'] = motion_score
            metrics['scene_change'] = min(1.0, motion_score * 3.0)  # Amplify for detection
        else:
            metrics['motion_magnitude'] = 0.0
            metrics['frame_difference'] = 0.0
            metrics['scene_change'] = 0.0

        return metrics

    def _detect_scene_boundaries_fast(self, frame_data: List[Dict[str, Any]]) -> List[int]:
        """Fast scene boundary detection optimized for short segments."""
        if len(frame_data) < 2:
            return [0, len(frame_data) - 1]

        boundaries = [0]  # Start with first frame

        # Use smaller window for shorter segments
        window_size = 3
        scene_changes = []

        # Calculate scene change scores
        for i in range(len(frame_data)):
            scene_change = frame_data[i]['metrics'].get('scene_change', 0.0)
            scene_changes.append(scene_change)

        # Find peaks in scene changes with lower threshold for more segments
        for i in range(window_size, len(scene_changes) - window_size):
            current_score = scene_changes[i]

            # Check if this is a local maximum above threshold
            if current_score > self.scene_threshold:
                # Check if it's higher than surrounding frames
                window_scores = scene_changes[i-window_size:i+window_size+1]
                if current_score == max(window_scores):
                    # Ensure minimum distance between boundaries (shorter for more segments)
                    if len(boundaries) == 0 or i - boundaries[-1] > 10:
                        boundaries.append(i)

        # Force more segments if we don't have enough
        if len(boundaries) <= 2:  # Less than 2 segments
            self.logger.warning("Forcing more segments for better shorts")

            total_duration = frame_data[-1]['timestamp'] - frame_data[0]['timestamp']
            target_segment_duration = self.config.get('segmentation.target_duration', 10)

            # Create segments every target_duration seconds
            boundaries = [0]
            current_time = target_segment_duration

            for i, frame in enumerate(frame_data):
                if frame['timestamp'] >= current_time:
                    boundaries.append(i)
                    current_time += target_segment_duration

                    # Create more segments for shorts
                    if len(boundaries) >= 12:  # Max 11 segments
                        break

        # Add final frame
        if boundaries[-1] != len(frame_data) - 1:
            boundaries.append(len(frame_data) - 1)

        self.logger.info(f"Created {len(boundaries) - 1} scene boundaries using fast algorithm")
        return boundaries

    def _create_scene_segments(
        self,
        boundaries: List[int],
        frame_data: List[Dict[str, Any]],
        fps: float
    ) -> List[Dict[str, Any]]:
        """Create scene segments from boundaries with content filtering."""
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

            # Content filtering - skip boring segments
            if self._is_boring_segment(avg_metrics):
                self.logger.debug(f"Skipping boring segment: {start_time:.1f}s-{end_time:.1f}s")
                continue

            scene = {
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'start_frame': frame_data[start_idx]['frame_num'],
                'end_frame': frame_data[end_idx]['frame_num'],
                'metrics': avg_metrics
            }

            scenes.append(scene)

        self.logger.info(f"Created {len(scenes)} valid scenes after content filtering")
        return scenes

    def _is_boring_segment(self, metrics: Dict[str, float]) -> bool:
        """Check if a segment is boring (e.g., landing, static scene)."""

        # Skip segments with very low motion (landings, hovering)
        if self.avoid_low_motion:
            motion = metrics.get('motion_magnitude', 0.0)
            if motion < self.min_motion_threshold:
                return True

        # Skip segments with low visual interest (static scenes)
        if self.avoid_static_scenes:
            visual_interest = metrics.get('visual_interest', 0.0)
            if visual_interest < self.min_visual_interest:
                return True

        # Skip segments with very low contrast (boring/flat scenes)
        contrast = metrics.get('contrast', 0.0)
        if contrast < 0.1:  # Very low contrast threshold
            return True

        return False

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

    def _score_scene_fast(self, scene: Dict[str, Any]) -> float:
        """Fast scene scoring algorithm."""
        metrics = scene.get('metrics', {})

        # Visual interest score (using entropy for better quality assessment)
        visual_score = (
            metrics.get('visual_interest', 0.0) * 0.5 +
            metrics.get('entropy', 0.0) / 5.0 * 0.3 +
            metrics.get('contrast', 0.0) * 0.2
        )

        # Motion score
        motion_score = metrics.get('motion_magnitude', 0.0)

        # Duration bonus (prefer scenes closer to target duration)
        target_duration = self.config.get('segmentation.target_duration', 30)
        duration_score = 1.0 - abs(scene['duration'] - target_duration) / target_duration
        duration_score = max(0.0, duration_score)

        # Combine scores with improved weighting
        total_score = (
            visual_score * self.visual_weight +
            motion_score * self.motion_weight +
            duration_score * 0.2
        )

        return min(1.0, total_score)


# Keep the old class for compatibility but use the new one
FPVSceneDetector = FastFPVSceneDetector


def detect_scenes(video_path: str, config: Dict[str, Any]) -> List[SceneSegment]:
    """
    Detect scenes in FPV drone video using fast algorithm.
    """
    detector = FastFPVSceneDetector(config)
    return detector.detect_scenes(video_path)