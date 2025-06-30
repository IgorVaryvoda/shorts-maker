"""
Scene detection for FPV drone videos.
Optimized for high-motion aerial footage without audio.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

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


class UltraFastFPVSceneDetector:
    """Ultra-fast, reliable scene detector optimized for FPV drone footage."""

    def __init__(self, config: Config):
        """Initialize ultra-fast FPV scene detector."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Detection parameters - optimized for reliability and speed
        self.scene_threshold = config.get('segmentation.scene_threshold', 0.15)
        self.motion_threshold = config.get('segmentation.motion_threshold', 0.05)
        self.frame_sample_rate = config.get('segmentation.frame_sample_rate', 0.3)
        self.min_duration = config.get('segmentation.min_duration', 8)
        self.max_duration = config.get('segmentation.max_duration', 25)

        # Scoring weights
        self.visual_weight = config.get('segmentation.scoring.visual_interest', 0.6)
        self.motion_weight = config.get('segmentation.scoring.motion_activity', 0.4)

        # Performance optimizations - read from config
        self.downsample_factor = config.get('performance.downsampling_factor', 6)
        self.batch_size = 200       # Much larger batches for efficiency

        # Content filtering
        min_motion_key = 'segmentation.content_filtering.min_motion_threshold'
        self.min_motion_threshold = config.get(min_motion_key, 0.03)
        min_visual_key = 'segmentation.content_filtering.min_visual_interest'
        self.min_visual_interest = config.get(min_visual_key, 0.2)

        self.logger.info(
            "Initialized Ultra-Fast FPV Scene Detector - "
            "Optimized for speed and reliability"
        )

    def detect_scenes(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ultra-fast scene detection with simplified, reliable algorithm.
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

        # Calculate aggressive downsampling for maximum speed
        small_width = max(160, width // self.downsample_factor)  # Min 160px wide
        small_height = max(90, height // self.downsample_factor)  # Min 90px tall

        self.logger.info(f"Ultra-fast analysis: {total_frames} frames, {duration:.1f}s")
        self.logger.info(
            f"Processing at {small_width}x{small_height} ({self.downsample_factor}x downsampled)"
        )

        # Smart sampling - fewer frames, better distribution
        frame_indices = self._smart_sampling(total_frames, fps, duration)
        self.logger.info(
            f"Analyzing {len(frame_indices)} key frames (smart sampling)"
        )

        # Process frames in large batches
        frame_data = []
        prev_small_gray = None

        try:
            for batch_start in range(0, len(frame_indices), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(frame_indices))
                batch_indices = frame_indices[batch_start:batch_end]

                batch_data = self._process_frame_batch_ultra_fast(
                    cap, batch_indices, fps, small_width, small_height, prev_small_gray
                )
                frame_data.extend(batch_data)

                # Progress update
                if progress_callback:
                    progress = batch_end
                    msg = f"Ultra-fast analysis {progress}/{len(frame_indices)}"
                    progress_callback(progress, len(frame_indices), msg)

                if batch_data:
                    prev_small_gray = batch_data[-1].get('small_gray')

        finally:
            cap.release()

        # Simplified boundary detection
        boundaries = self._detect_boundaries_smart(frame_data, duration)

        # Create segments
        scenes = self._create_segments_fast(boundaries, frame_data)

        # Score scenes
        for scene in scenes:
            scene['score'] = self._score_scene_simple(scene)

        msg = f"Detected {len(scenes)} high-quality scenes in {len(frame_indices)} frame analysis"
        self.logger.info(msg)
        return scenes

    def _smart_sampling(self, total_frames: int, fps: float, duration: float) -> List[int]:
        """Smart sampling strategy using config frame sample rate."""
        # Use frame sample rate from config (frames per second to analyze)
        sample_interval = max(1, int(fps / self.frame_sample_rate))

        # Generate base samples using config sample rate
        indices = list(range(0, total_frames, sample_interval))

        # Add strategic samples at video thirds for better coverage
        strategic_points = [
            total_frames // 4,      # 25%
            total_frames // 2,      # 50%
            (3 * total_frames) // 4 # 75%
        ]

        for point in strategic_points:
            # Add 3 samples around each strategic point
            for offset in [-int(fps), 0, int(fps)]:
                sample = point + offset
                if 0 <= sample < total_frames and sample not in indices:
                    indices.append(sample)

        return sorted(list(set(indices)))

    def _process_frame_batch_ultra_fast(
        self,
        cap: cv2.VideoCapture,
        frame_indices: List[int],
        fps: float,
        small_width: int,
        small_height: int,
        prev_small_gray: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Ultra-fast batch processing with minimal computations."""
        batch_data = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Ultra-aggressive downsampling
            small_frame = cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
            small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # Minimal metrics for speed
            metrics = self._calculate_minimal_metrics(small_gray, prev_small_gray)

            batch_data.append({
                'frame_num': frame_idx,
                'timestamp': frame_idx / fps,
                'metrics': metrics,
                'small_gray': small_gray
            })
            prev_small_gray = small_gray

        return batch_data

    def _calculate_minimal_metrics(
        self,
        small_gray: np.ndarray,
        prev_small_gray: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Minimal metrics calculation for maximum speed."""
        # Fast visual interest using standard deviation (proxy for detail)
        visual_interest = np.std(small_gray.astype(np.float32)) / 255.0

        # Fast contrast
        contrast = visual_interest  # Reuse std calculation

        # Motion (if previous frame available)
        if prev_small_gray is not None:
            # Simple absolute difference for motion
            motion = np.mean(cv2.absdiff(prev_small_gray, small_gray)) / 255.0
        else:
            motion = 0.0

        return {
            'visual_interest': min(1.0, visual_interest * 2.0),  # Amplify for better detection
            'contrast': contrast,
            'motion_magnitude': motion,
            'scene_change': min(1.0, motion * 2.0)  # Simple scene change metric
        }

    def _detect_boundaries_smart(self, frame_data: List[Dict[str, Any]], duration: float) -> List[int]:
        """Smart boundary detection with safety nets - always finds some scenes."""
        if len(frame_data) < 3:
            return [0, len(frame_data) - 1]

        boundaries = [0]

        # Collect metrics
        motion_scores = [frame['metrics']['motion_magnitude'] for frame in frame_data]
        visual_scores = [frame['metrics']['visual_interest'] for frame in frame_data]

        # Use more permissive thresholds - lower percentiles
        motion_threshold = max(0.02, np.percentile(motion_scores, 60))  # Lower from 70
        visual_threshold = max(0.15, np.percentile(visual_scores, 60))  # Lower from 75

        self.logger.info(f"Detection thresholds: motion={motion_threshold:.3f}, visual={visual_threshold:.3f}")

        # Minimum distance for segments (allow more segments)
        min_segment_frames = max(2, len(frame_data) // 12)  # Allow up to 12 segments (was 8)

        # Main detection loop with lower significance threshold
        significant_boundaries = []
        for i in range(min_segment_frames, len(frame_data) - min_segment_frames):
            motion = motion_scores[i]
            visual = visual_scores[i]

            # Combined significance score with lower threshold
            significance = (motion / motion_threshold) * 0.6 + (visual / visual_threshold) * 0.4

            # Lower significance threshold for better detection
            if significance > 0.8 and (i - boundaries[-1]) >= min_segment_frames:  # Lower from 1.2
                significant_boundaries.append((significance, i))

        # Sort by significance and take the best ones
        significant_boundaries.sort(reverse=True)

        # Add boundaries, but not too many
        max_boundaries = min(10, len(significant_boundaries))  # Allow up to 10 boundaries
        for _, idx in significant_boundaries[:max_boundaries]:
            boundaries.append(idx)

        # Safety net: If we don't have enough boundaries, force some based on time
        if len(boundaries) < 3 and duration > 30:  # Need at least 2 segments for 30s+ videos
            self.logger.warning("Not enough boundaries detected, adding time-based boundaries")

            # Add boundaries every 20 seconds as fallback
            for time_point in [20, 40, 60, 80, 100]:
                if time_point < duration:
                    # Find closest frame to this time
                    target_time = time_point
                    closest_idx = min(range(len(frame_data)),
                                     key=lambda i: abs(frame_data[i]['timestamp'] - target_time))

                    # Make sure it's not too close to existing boundaries
                    if all(abs(closest_idx - b) >= min_segment_frames for b in boundaries):
                        boundaries.append(closest_idx)

        # Sort and ensure we have final boundary
        boundaries = sorted(list(set(boundaries)))
        if boundaries[-1] != len(frame_data) - 1:
            boundaries.append(len(frame_data) - 1)

        self.logger.info(f"Smart detection: {len(boundaries) - 1} segments from {len(frame_data)} frames")

        # Debug info
        if len(boundaries) <= 2:
            self.logger.warning(f"Very few boundaries detected! Motion range: {min(motion_scores):.3f}-{max(motion_scores):.3f}")
            self.logger.warning(f"Visual range: {min(visual_scores):.3f}-{max(visual_scores):.3f}")

        return boundaries

    def _create_segments_fast(self, boundaries: List[int], frame_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create segments with fast filtering."""
        scenes = []

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            start_time = frame_data[start_idx]['timestamp']
            end_time = frame_data[end_idx]['timestamp']
            duration = end_time - start_time

            # Duration filtering
            if duration < self.min_duration or duration > self.max_duration:
                continue

            # Calculate average metrics
            segment_frames = frame_data[start_idx:end_idx + 1]
            avg_motion = np.mean([f['metrics']['motion_magnitude'] for f in segment_frames])
            avg_visual = np.mean([f['metrics']['visual_interest'] for f in segment_frames])

            # Quality filtering
            if avg_motion < self.min_motion_threshold or avg_visual < self.min_visual_interest:
                continue

            scenes.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'start_frame': frame_data[start_idx]['frame_num'],
                'end_frame': frame_data[end_idx]['frame_num'],
                'metrics': {
                    'motion_magnitude': avg_motion,
                    'visual_interest': avg_visual,
                    'contrast': np.mean([f['metrics']['contrast'] for f in segment_frames])
                }
            })

        return scenes

    def _score_scene_simple(self, scene: Dict[str, Any]) -> float:
        """Simple, reliable scene scoring."""
        metrics = scene['metrics']

        # Weighted score
        motion_score = metrics['motion_magnitude']
        visual_score = metrics['visual_interest']

        # Duration bonus (prefer 15-20 second segments)
        target_duration = 15
        duration_score = 1.0 - abs(scene['duration'] - target_duration) / target_duration
        duration_score = max(0.3, duration_score)  # Don't penalize too much

        # Combined score
        total_score = (
            visual_score * self.visual_weight +
            motion_score * self.motion_weight +
            duration_score * 0.2
        )

        return min(1.0, total_score)


# Use the new ultra-fast detector
FastFPVSceneDetector = UltraFastFPVSceneDetector
FPVSceneDetector = UltraFastFPVSceneDetector


def detect_scenes(video_path: str, config: Dict[str, Any]) -> List[SceneSegment]:
    """
    Detect scenes in FPV drone video using fast algorithm.
    """
    detector = FastFPVSceneDetector(config)
    return detector.detect_scenes(video_path)
