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

        # Content filtering settings (more lenient)
        self.avoid_low_motion = config.get('segmentation.content_filtering.avoid_low_motion', True)
        self.min_motion_threshold = config.get('segmentation.content_filtering.min_motion_threshold', 0.02)
        self.avoid_static_scenes = config.get('segmentation.content_filtering.avoid_static_scenes', True)
        self.min_visual_interest = config.get('segmentation.content_filtering.min_visual_interest', 0.15)

        # Dynamic duration settings
        self.dynamic_duration = config.get('segmentation.dynamic_duration', True)
        self.quality_extension_threshold = config.get('segmentation.quality_extension_threshold', 0.6)
        self.max_extension_duration = config.get('segmentation.max_extension_duration', 10)

        logger.info("Initialized Fast FPV Scene Detector with aggressive scene detection and dynamic duration")

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

        # Validate video properties
        if fps <= 0:
            cap.release()
            raise ValueError(f"Invalid FPS value: {fps}. Video file may be corrupted or unsupported.")
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Invalid frame count: {total_frames}. Video file may be corrupted or unsupported.")
        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError(f"Invalid video dimensions: {width}x{height}. Video file may be corrupted or unsupported.")

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
        """Calculate uniform sampling to ensure we don't miss any good content."""
        base_interval = max(1, int(fps / self.frame_sample_rate))

        # Use uniform sampling across the entire video to catch all content
        indices = list(range(0, total_frames, base_interval))

        # Add some extra samples at key intervals to catch transitions
        # Sample every 5 seconds regardless of base rate
        five_second_interval = int(fps * 5)
        for i in range(0, total_frames, five_second_interval):
            if i not in indices:
                indices.append(i)

        # Add samples at 1/4, 1/2, 3/4 points to ensure middle coverage
        quarter_points = [
            total_frames // 4,
            total_frames // 2,
            (3 * total_frames) // 4
        ]

        for point in quarter_points:
            # Add several samples around each quarter point
            for offset in range(-int(fps * 2), int(fps * 2), int(fps * 0.5)):
                sample_frame = point + offset
                if 0 <= sample_frame < total_frames and sample_frame not in indices:
                    indices.append(sample_frame)

        # Remove duplicates and sort
        indices = sorted(list(set(indices)))

        self.logger.info(f"Uniform sampling: {len(indices)} frames across entire video")
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
                'timestamp': frame_idx / max(fps, 1.0),  # Prevent division by zero
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
        """Scene boundary detection optimized for 10-20 second segments."""
        if len(frame_data) < 2:
            return [0, len(frame_data) - 1]

        boundaries = [0]  # Start with first frame

        # Collect all metrics for comprehensive analysis
        scene_changes = []
        motion_scores = []
        visual_scores = []
        contrast_scores = []

        for i in range(len(frame_data)):
            metrics = frame_data[i]['metrics']
            scene_changes.append(metrics.get('scene_change', 0.0))
            motion_scores.append(metrics.get('motion_magnitude', 0.0))
            visual_scores.append(metrics.get('visual_interest', 0.0))
            contrast_scores.append(metrics.get('contrast', 0.0))

        # Minimum distance between boundaries for 10+ second segments
        min_boundary_distance = max(8, int(len(frame_data) * 0.05))  # At least 8 frames or 5% of video

        # Multiple detection strategies with longer segment focus

        # 1. Traditional scene change detection (significant changes only)
        for i in range(5, len(scene_changes) - 5):
            if scene_changes[i] > self.scene_threshold * 1.5:  # Higher threshold for longer segments
                if len(boundaries) == 0 or i - boundaries[-1] > min_boundary_distance:
                    boundaries.append(i)

        # 2. Major motion changes (significant action sequences)
        motion_threshold = max(0.04, np.percentile(motion_scores, 80))  # Higher threshold
        for i in range(10, len(motion_scores) - 10):
            current_motion = motion_scores[i]
            if current_motion > motion_threshold:
                # Check if this is a major motion increase
                prev_avg = np.mean(motion_scores[i-10:i])
                if current_motion > prev_avg * 2.0:  # More significant change required
                    if len(boundaries) == 0 or i - boundaries[-1] > min_boundary_distance:
                        boundaries.append(i)

        # 3. Strong visual interest peaks only
        visual_threshold = max(0.5, np.percentile(visual_scores, 75))  # Higher threshold
        for i in range(5, len(visual_scores) - 5):
            current_visual = visual_scores[i]
            if current_visual > visual_threshold:
                # Check if this is a strong local peak
                window = visual_scores[i-5:i+6]
                if current_visual >= max(window) * 0.95:  # Must be clear peak
                    if len(boundaries) == 0 or i - boundaries[-1] > min_boundary_distance:
                        boundaries.append(i)

        # 4. Major contrast changes only
        for i in range(5, len(contrast_scores) - 5):
            current_contrast = contrast_scores[i]
            if i > 10:
                prev_avg = np.mean(contrast_scores[i-10:i])
                if abs(current_contrast - prev_avg) > 0.08:  # More significant contrast change
                    if len(boundaries) == 0 or i - boundaries[-1] > min_boundary_distance:
                        boundaries.append(i)

        # 5. Force boundaries for very long videos (every 25-30 seconds)
        total_duration = frame_data[-1]['timestamp'] - frame_data[0]['timestamp']
        if total_duration > 60:  # Only for longer videos
            interval_duration = 25  # Force boundary every 25 seconds (was 15)
            current_time = interval_duration

            for i, frame in enumerate(frame_data):
                if frame['timestamp'] >= current_time:
                    if len(boundaries) == 0 or i - boundaries[-1] > min_boundary_distance:
                        boundaries.append(i)
                    current_time += interval_duration

        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))

        # Ensure we don't have too many short segments
        if len(boundaries) > 12 and total_duration > 120:  # Max 12 segments for 2+ minute videos
            self.logger.warning("Too many boundaries detected, keeping only strongest ones")
            # Keep only the strongest boundaries based on scene change scores
            boundary_scores = []
            for b in boundaries[1:-1]:  # Exclude first and last
                if b < len(scene_changes):
                    boundary_scores.append((scene_changes[b], b))

            # Sort by score and keep top boundaries
            boundary_scores.sort(reverse=True)
            top_boundaries = [0] + [b for _, b in boundary_scores[:10]] + [boundaries[-1]]
            boundaries = sorted(list(set(top_boundaries)))

        # Add final frame
        if boundaries[-1] != len(frame_data) - 1:
            boundaries.append(len(frame_data) - 1)

        self.logger.info(f"Created {len(boundaries) - 1} scene boundaries optimized for 10-20s segments")
        return boundaries

    def _create_scene_segments(
        self,
        boundaries: List[int],
        frame_data: List[Dict[str, Any]],
        fps: float
    ) -> List[Dict[str, Any]]:
        """Create scene segments with dynamic duration based on content quality."""
        scenes = []

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            start_time = frame_data[start_idx]['timestamp']
            end_time = frame_data[end_idx]['timestamp']
            duration = end_time - start_time

            # Calculate metrics first to determine if we should extend
            scene_frames = frame_data[start_idx:end_idx + 1]
            avg_metrics = self._average_metrics(scene_frames)

            # Dynamic duration adjustment based on content quality
            if self.dynamic_duration and duration < self.max_duration:
                extended_duration = self._calculate_dynamic_duration(
                    avg_metrics, duration, boundaries, i, frame_data
                )

                if extended_duration > duration:
                    # Extend the segment
                    new_end_time = start_time + extended_duration

                    # Find the frame index for the new end time
                    for j in range(end_idx, len(frame_data)):
                        if frame_data[j]['timestamp'] >= new_end_time:
                            end_idx = j
                            break

                    end_time = frame_data[end_idx]['timestamp']
                    duration = end_time - start_time

                    # Recalculate metrics with extended segment
                    scene_frames = frame_data[start_idx:end_idx + 1]
                    avg_metrics = self._average_metrics(scene_frames)

                    self.logger.debug(f"Extended high-quality segment to {duration:.1f}s")

            # Filter by duration constraints
            if duration < self.min_duration or duration > self.max_duration:
                continue

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

        self.logger.info(f"Created {len(scenes)} dynamic-duration scenes after content filtering")
        return scenes

    def _calculate_dynamic_duration(
        self,
        metrics: Dict[str, float],
        base_duration: float,
        boundaries: List[int],
        current_boundary_idx: int,
        frame_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate dynamic duration based on content quality."""

        # Calculate quality score
        visual_interest = metrics.get('visual_interest', 0.0)
        motion_activity = metrics.get('motion_magnitude', 0.0)
        contrast = metrics.get('contrast', 0.0)

        quality_score = (
            visual_interest * 0.4 +
            motion_activity * 0.4 +
            contrast * 0.2
        )

        # If quality is high, extend the duration
        if quality_score > self.quality_extension_threshold:
            extension_factor = min(2.0, quality_score / self.quality_extension_threshold)
            max_extension = self.max_extension_duration * (extension_factor - 1.0)

            # Don't extend beyond the next boundary (if it exists)
            max_possible_duration = base_duration
            if current_boundary_idx + 2 < len(boundaries):
                next_boundary_time = frame_data[boundaries[current_boundary_idx + 2]]['timestamp']
                current_start_time = frame_data[boundaries[current_boundary_idx]]['timestamp']
                max_possible_duration = min(
                    self.max_duration,
                    next_boundary_time - current_start_time
                )

            extended_duration = min(
                base_duration + max_extension,
                max_possible_duration
            )

            return extended_duration

        return base_duration

    def _is_boring_segment(self, metrics: Dict[str, float]) -> bool:
        """Check if a segment is boring (e.g., landing, static scene) - less strict filtering."""

        # Skip segments with very low motion (landings, hovering) - more lenient
        if self.avoid_low_motion:
            motion = metrics.get('motion_magnitude', 0.0)
            if motion < self.min_motion_threshold:
                return True

        # Skip segments with very low visual interest (static scenes) - more lenient
        if self.avoid_static_scenes:
            visual_interest = metrics.get('visual_interest', 0.0)
            if visual_interest < self.min_visual_interest:
                return True

        # Skip segments with extremely low contrast (completely flat/boring scenes)
        contrast = metrics.get('contrast', 0.0)
        if contrast < 0.05:  # Very low contrast threshold (was 0.1)
            return True

        # Additional check: skip if BOTH motion and visual interest are very low
        motion = metrics.get('motion_magnitude', 0.0)
        visual_interest = metrics.get('visual_interest', 0.0)
        if motion < 0.01 and visual_interest < 0.1:  # Both extremely low
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