"""
Main video processor for FPV drone footage.
Orchestrates scene detection, color grading, and video output.
"""

import cv2
import numpy as np
import ffmpeg
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import datetime
import json
import os

# Handle imports for both CLI and module usage
try:
    from .scene_detector import FPVSceneDetector, SceneSegment
    from .color_grader import ColorGrader
    from ..utils.config import Config
    from ..utils.progress import VideoProcessingProgress, ProgressBar
except ImportError:
    # Fallback for CLI usage
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from core.scene_detector import FPVSceneDetector, SceneSegment
    from core.color_grader import ColorGrader
    from utils.config import Config
    from utils.progress import VideoProcessingProgress, ProgressBar

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Main video processor for FPV drone footage."""

    def __init__(self, config_path_or_config: Optional[str] = None):
        """Initialize video processor."""
        if isinstance(config_path_or_config, str) or config_path_or_config is None:
            self.config = Config(config_path_or_config)
        else:
            self.config = config_path_or_config

        self.scene_detector = FPVSceneDetector(self.config)
        self.color_grader = ColorGrader(self.config)
        self.logger = logging.getLogger(__name__)

        # Video settings
        self.output_resolution = tuple(self.config.get('video.output_resolution', [1080, 1920]))
        self.fps = self.config.get('video.fps', 30)
        self.quality = self.config.get('video.quality', 'high')
        self.codec = self.config.get('video.codec', 'h264')
        self.bitrate = self.config.get('video.bitrate', '5M')

        # Processing settings
        self.use_gpu = self.config.get('performance.use_gpu', True)
        self.num_threads = self.config.get('performance.num_threads', 4)

        self.logger.info(f"Initialized Video Processor - {self.output_resolution[0]}x{self.output_resolution[1]} @ {self.fps}fps")

    def process_video(
        self,
        input_path: str,
        output_dir: str = "output",
        lut_path: Optional[str] = None,
        max_segments: int = 3
    ) -> List[str]:
        """
        Process a video to create shorts.

        Args:
            input_path: Path to input video
            output_dir: Directory for output files
            lut_path: Optional LUT file path
            max_segments: Maximum number of segments to create

        Returns:
            List of output file paths
        """
        # Validate input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print(f"🎬 Processing Video: {os.path.basename(input_path)}")
        print("=" * 60)

        # Get video info for progress tracking
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        cap.release()

        print(f"📊 Video Info: {width}x{height}, {duration:.1f}s, {fps:.1f}fps, {total_frames:,} frames")
        print(f"🎯 Target: {max_segments} shorts, LUT: {os.path.basename(lut_path) if lut_path else 'None'}")
        print("=" * 60)

        try:
            # Stage 1: Scene Detection
            print("\n🎬 Stage 1: Scene Detection")
            print("-" * 40)

            def scene_progress_callback(frame_num: int, total_frames: int, message: str = ""):
                if frame_num % 200 == 0 or frame_num == total_frames:
                    percentage = (frame_num / total_frames) * 100
                    bar_width = 30
                    filled = int(bar_width * percentage / 100)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    print(f"\r{bar} {percentage:5.1f}% {message}", end="", flush=True)

            scenes = self.scene_detector.detect_scenes(
                input_path,
                progress_callback=scene_progress_callback
            )

            print(f"\n✅ Found {len(scenes)} scenes")

            # Stage 2: Select Best Segments
            print("\n🎯 Stage 2: Selecting Best Segments")
            print("-" * 40)

            segments = self._select_best_segments(scenes, max_segments)
            print(f"✅ Selected {len(segments)} segments for processing")

            if not segments:
                print("❌ No suitable segments found")
                return []

            # Display selected segments
            for i, segment in enumerate(segments, 1):
                print(f"  {i}. {segment['start_time']:.1f}s - {segment['end_time']:.1f}s "
                      f"({segment['duration']:.1f}s, score: {segment['score']:.3f})")

            output_paths = []

            # Stage 3: Process Each Segment
            print(f"\n🎥 Stage 3: Processing {len(segments)} Segments")
            print("-" * 40)

            for i, segment in enumerate(segments):
                print(f"\n📹 Processing Segment {i+1}/{len(segments)}")
                print(f"   Time: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s ({segment['duration']:.1f}s)")

                output_path = self._process_segment_with_progress(
                    input_path,
                    segment,
                    output_dir,
                    i,
                    lut_path
                )

                if output_path:
                    output_paths.append(output_path)
                    file_size = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"   ✅ Created: {os.path.basename(output_path)} ({file_size:.1f}MB)")
                else:
                    print(f"   ❌ Failed to process segment {i+1}")

            # Final Summary
            print("\n" + "=" * 60)
            print("🎉 Processing Complete!")
            print("=" * 60)

            if output_paths:
                total_size = sum(os.path.getsize(f) for f in output_paths) / (1024 * 1024)
                print(f"✅ Created {len(output_paths)} shorts ({total_size:.1f}MB total)")
                for i, path in enumerate(output_paths, 1):
                    print(f"  {i}. {os.path.basename(path)}")
            else:
                print("❌ No shorts were created")

            return output_paths

        except Exception as e:
            print(f"\n❌ Processing failed: {e}")
            raise

    def _select_best_segments(self, scenes: List[Dict[str, Any]], max_segments: int) -> List[Dict[str, Any]]:
        """Select best segments from detected scenes."""
        # Sort scenes by score (highest first)
        sorted_scenes = sorted(scenes, key=lambda x: x.get('score', 0), reverse=True)

        # Filter scenes that meet duration requirements
        min_duration = self.config.get('segmentation.min_duration', 15)
        max_duration = self.config.get('segmentation.max_duration', 60)

        valid_scenes = [
            scene for scene in sorted_scenes
            if min_duration <= scene['duration'] <= max_duration
        ]

        # Select top segments
        selected = valid_scenes[:max_segments]

        self.logger.info(f"Selected {len(selected)} segments from {len(scenes)} detected scenes")

        return selected

    def _process_segment_with_progress(
        self,
        input_path: str,
        segment: Dict[str, Any],
        output_dir: str,
        segment_index: int,
        lut_path: Optional[str]
    ) -> Optional[str]:
        """Process a single segment with detailed progress tracking."""
        try:
            # Calculate segment info
            start_time = segment['start_time']
            end_time = segment['end_time']
            duration = segment['duration']

            # Open input video
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate frame range
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            total_segment_frames = end_frame - start_frame

            # Generate output filename
            timestamp = int(start_time)
            output_filename = f"short_{segment_index + 1:02d}_{timestamp}s.mp4"
            output_path = os.path.join(output_dir, output_filename)

            # Calculate output dimensions (9:16 aspect ratio)
            target_width = 1080
            target_height = 1920

            # Determine crop area (center crop)
            if width / height > target_width / target_height:
                # Video is wider, crop horizontally
                crop_height = height
                crop_width = int(height * target_width / target_height)
                crop_x = (width - crop_width) // 2
                crop_y = 0
            else:
                # Video is taller, crop vertically
                crop_width = width
                crop_height = int(width * target_height / target_width)
                crop_x = 0
                crop_y = (height - crop_height) // 2

            print(f"   📐 Crop: {crop_width}x{crop_height} → {target_width}x{target_height}")

            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Load LUT if provided
            lut = None
            if lut_path and os.path.exists(lut_path):
                try:
                    from ..utils.lut_loader import LUTLoader
                    lut = LUTLoader.load_cube(lut_path)
                    print(f"   🎨 Loaded LUT: {lut.title}")
                except Exception as e:
                    print(f"   ⚠️  LUT loading failed: {e}")

            frame_count = 0

            # Process frames with progress
            print(f"   🎞️  Processing {total_segment_frames} frames...")

            while frame_count < total_segment_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Show progress every 30 frames
                if frame_count % 30 == 0 or frame_count == total_segment_frames - 1:
                    percentage = (frame_count / total_segment_frames) * 100
                    bar_width = 20
                    filled = int(bar_width * percentage / 100)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    print(f"\r   {bar} {percentage:5.1f}% Frame {frame_count + 1}/{total_segment_frames}", end="", flush=True)

                # Apply color grading if LUT is available
                if lut:
                    try:
                        # Convert to float [0,1]
                        frame_float = frame.astype(np.float32) / 255.0
                        # Apply LUT
                        graded_frame = lut.apply(frame_float, intensity=0.8)
                        # Convert back to uint8
                        frame = (graded_frame * 255).astype(np.uint8)
                    except Exception as e:
                        # If LUT fails, continue without it
                        pass

                # Crop frame
                cropped = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

                # Resize to target resolution
                resized = cv2.resize(cropped, (target_width, target_height))

                # Write frame
                out.write(resized)
                frame_count += 1

            print()  # New line after progress bar

            # Cleanup
            cap.release()
            out.release()

            # Verify output file
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
            else:
                return None

        except Exception as e:
            print(f"\n   ❌ Error processing segment: {e}")
            return None

    def analyze_video(self, input_path: str) -> Dict[str, Any]:
        """
        Analyze video without processing.

        Args:
            input_path: Path to input video

        Returns:
            Analysis results
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        # Get video info for progress tracking
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Create simple progress bar for analysis
        print(f"🔍 Analyzing video: {os.path.basename(input_path)}")
        print(f"📊 Total frames: {total_frames:,} ({total_frames/fps:.1f}s @ {fps:.1f}fps)")
        print("=" * 60)

        def analysis_progress_callback(frame_num: int, total_frames: int, message: str = ""):
            # Show progress every 100 frames to avoid spam
            if frame_num % 100 == 0 or frame_num == total_frames:
                percentage = (frame_num / total_frames) * 100
                bar_width = 40
                filled = int(bar_width * percentage / 100)
                bar = "█" * filled + "░" * (bar_width - filled)

                print(f"\r{bar} {percentage:5.1f}% {message}", end="", flush=True)

        # Detect scenes with progress
        scenes = self.scene_detector.detect_scenes(
            input_path,
            progress_callback=analysis_progress_callback
        )

        # Complete progress bar
        print("\n" + "=" * 60)
        print("✅ Analysis complete!")

        # Compile analysis results
        cap = cv2.VideoCapture(input_path)
        video_info = {
            'file_path': input_path,
            'file_size_mb': os.path.getsize(input_path) / (1024 * 1024),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'resolution': (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        cap.release()

        # Filter scenes by duration
        min_duration = self.config.get('segmentation.min_duration', 15)
        max_duration = self.config.get('segmentation.max_duration', 60)

        valid_scenes = [
            scene for scene in scenes
            if min_duration <= scene['duration'] <= max_duration
        ]

        # Sort by score
        valid_scenes.sort(key=lambda x: x.get('score', 0), reverse=True)

        analysis = {
            'video_info': video_info,
            'total_scenes': len(scenes),
            'valid_scenes': len(valid_scenes),
            'best_scenes': valid_scenes[:5],  # Top 5 scenes
            'scene_details': scenes
        }

        return analysis

    def _select_scenes_for_shorts(self, scenes: List[SceneSegment],
                                target_duration: int, num_shorts: int) -> List[SceneSegment]:
        """Select best scenes for creating shorts."""
        # Filter scenes by duration (allow some flexibility)
        min_duration = max(15, target_duration - 10)
        max_duration = min(60, target_duration + 15)

        suitable_scenes = [
            s for s in scenes
            if min_duration <= s.duration <= max_duration
        ]

        if not suitable_scenes:
            # Fallback: use all scenes and we'll trim them
            suitable_scenes = scenes

        # Sort by score and take the best ones
        suitable_scenes.sort(key=lambda s: s.score, reverse=True)

        # Select non-overlapping scenes
        selected = []
        for scene in suitable_scenes:
            if len(selected) >= num_shorts:
                break

            # Check for overlap with already selected scenes
            overlap = False
            for selected_scene in selected:
                if self._scenes_overlap(scene, selected_scene):
                    overlap = True
                    break

            if not overlap:
                selected.append(scene)

        return selected

    def _scenes_overlap(self, scene1: SceneSegment, scene2: SceneSegment) -> bool:
        """Check if two scenes overlap significantly."""
        overlap_threshold = self.config.get('segmentation.overlap_threshold', 0.3)

        # Calculate overlap
        start = max(scene1.start_time, scene2.start_time)
        end = min(scene1.end_time, scene2.end_time)

        if start >= end:
            return False  # No overlap

        overlap_duration = end - start
        min_duration = min(scene1.duration, scene2.duration)

        return (overlap_duration / min_duration) > overlap_threshold

    def _process_scene(self, input_path: str, scene: SceneSegment,
                      output_path: Path, lut_path: Optional[str],
                      target_duration: int, platform: str, index: int) -> Optional[str]:
        """Process a single scene into a short video."""
        try:
            # Calculate timing
            start_time = scene.start_time
            duration = min(scene.duration, target_duration)

            # If scene is longer than target, take the best part
            if scene.duration > target_duration:
                # Take from the middle for better content
                excess = scene.duration - target_duration
                start_time += excess / 2

            # Generate output filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            input_name = Path(input_path).stem
            output_filename = f"{input_name}_short_{index}_{timestamp}.mp4"
            output_file = output_path / output_filename

            # Extract and process video segment
            temp_file = output_path / f"temp_{index}_{timestamp}.mp4"

            # Step 1: Extract segment with FFmpeg
            self.logger.info(f"Extracting segment: {start_time:.1f}s - {start_time + duration:.1f}s")

            (
                ffmpeg
                .input(input_path, ss=start_time, t=duration)
                .output(
                    str(temp_file),
                    vcodec='libx264',
                    acodec='aac' if self._has_audio(input_path) else None,
                    r=self.fps,
                    **self._get_ffmpeg_quality_params()
                )
                .overwrite_output()
                .run(quiet=True)
            )

            # Step 2: Apply color grading and cropping
            self.logger.info("Applying color grading and cropping...")
            self._apply_color_grading_and_crop(
                str(temp_file), str(output_file), lut_path, platform
            )

            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()

            return str(output_file)

        except Exception as e:
            self.logger.error(f"Failed to process scene {index}: {e}")
            return None

    def _apply_color_grading_and_crop(self, input_file: str, output_file: str,
                                    lut_path: Optional[str], platform: str) -> None:
        """Apply color grading and crop video to vertical format."""
        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_file}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate crop parameters for 9:16 aspect ratio
        target_width, target_height = self.output_resolution
        crop_params = self._calculate_crop_params(width, height, target_width, target_height)

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output = output_file.replace('.mp4', '_temp.mp4')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (target_width, target_height))

        frame_count = 0
        self.logger.info(f"Processing {total_frames} frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply color grading
            if lut_path or self.color_grader.default_lut:
                graded_frame = self.color_grader.grade_image(frame, lut_path)
            else:
                graded_frame = self.color_grader.apply_auto_corrections(frame)

            # Crop to vertical format
            cropped_frame = self._crop_frame(graded_frame, crop_params)

            # Resize to target resolution
            if cropped_frame.shape[:2] != (target_height, target_width):
                cropped_frame = cv2.resize(cropped_frame, (target_width, target_height))

            out.write(cropped_frame)
            frame_count += 1

            # Progress logging
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                self.logger.debug(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")

        cap.release()
        out.release()

        # Re-encode with FFmpeg for better compression and audio
        self.logger.info("Final encoding...")

        # Check if original has audio
        has_audio = self._has_audio(input_file)

        input_stream = ffmpeg.input(temp_output)
        output_stream = input_stream.video

        if has_audio:
            audio_stream = ffmpeg.input(input_file).audio
            output_stream = ffmpeg.output(
                output_stream, audio_stream, output_file,
                vcodec='libx264', acodec='aac',
                **self._get_ffmpeg_quality_params()
            )
        else:
            output_stream = ffmpeg.output(
                output_stream, output_file,
                vcodec='libx264',
                **self._get_ffmpeg_quality_params()
            )

        ffmpeg.run(output_stream, overwrite_output=True, quiet=True)

        # Clean up temp file
        Path(temp_output).unlink()

    def _calculate_crop_params(self, width: int, height: int,
                             target_width: int, target_height: int) -> Dict[str, int]:
        """Calculate crop parameters for vertical format."""
        # Target aspect ratio (9:16)
        target_aspect = target_width / target_height
        current_aspect = width / height

        if current_aspect > target_aspect:
            # Video is wider, crop horizontally
            new_width = int(height * target_aspect)
            new_height = height
            x_offset = (width - new_width) // 2
            y_offset = 0
        else:
            # Video is taller, crop vertically
            new_width = width
            new_height = int(width / target_aspect)
            x_offset = 0
            y_offset = (height - new_height) // 2

        return {
            'x': x_offset,
            'y': y_offset,
            'width': new_width,
            'height': new_height
        }

    def _crop_frame(self, frame: np.ndarray, crop_params: Dict[str, int]) -> np.ndarray:
        """Crop frame according to parameters."""
        x = crop_params['x']
        y = crop_params['y']
        w = crop_params['width']
        h = crop_params['height']

        return frame[y:y+h, x:x+w]

    def _has_audio(self, video_path: str) -> bool:
        """Check if video has audio track."""
        try:
            probe = ffmpeg.probe(video_path)
            audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
            return len(audio_streams) > 0
        except:
            return False

    def _get_ffmpeg_quality_params(self) -> Dict[str, Any]:
        """Get FFmpeg quality parameters based on settings."""
        quality_presets = {
            'low': {'crf': 28, 'preset': 'fast'},
            'medium': {'crf': 23, 'preset': 'medium'},
            'high': {'crf': 18, 'preset': 'slow'},
            'ultra': {'crf': 15, 'preset': 'slower'}
        }

        params = quality_presets.get(self.quality, quality_presets['high'])

        # Add bitrate if specified
        if self.bitrate:
            params['b:v'] = self.bitrate

        return params

    def _generate_recommendations(self, scenes: List[SceneSegment],
                                color_analysis: Dict[str, float]) -> Dict[str, Any]:
        """Generate processing recommendations."""
        recommendations = {
            'scene_selection': {
                'total_scenes': len(scenes),
                'recommended_scenes': min(len(scenes), 5),
                'best_duration_range': [20, 45] if scenes else [30, 30]
            },
            'color_grading': {
                'needs_exposure_correction': abs(color_analysis.get('exposure_bias', 0)) > 0.2,
                'needs_contrast_boost': color_analysis.get('contrast', 0.5) < 0.3,
                'needs_saturation_boost': color_analysis.get('saturation', 0.5) < 0.4,
                'recommended_lut_intensity': 0.8 if color_analysis.get('dynamic_range', 0.5) > 0.6 else 0.6
            },
            'processing': {
                'estimated_time_per_short': '2-5 minutes',
                'recommended_gpu_usage': self.use_gpu,
                'optimal_batch_size': min(3, len(scenes))
            }
        }

        return recommendations


def create_video_processor(config_path_or_config: Optional[str] = None) -> VideoProcessor:
    """Create a video processor instance."""
    return VideoProcessor(config_path_or_config)