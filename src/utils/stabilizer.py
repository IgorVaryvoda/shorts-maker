"""
Video stabilization utility using FFmpeg's vidstab filters.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import shutil

logger = logging.getLogger(__name__)


class VideoStabilizer:
    """Video stabilization using FFmpeg's vidstab filters."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the stabilizer with configuration."""
        self.config = config
        self.stabilization_config = config.get('stabilization', {})

        # FFmpeg settings
        self.ffmpeg_path = self.stabilization_config.get('ffmpeg_path', 'ffmpeg')
        self.temp_dir = self.stabilization_config.get('temp_dir', '.cache/stabilization')
        self.keep_temp_files = self.stabilization_config.get('keep_temp_files', False)
        self.preserve_original = self.stabilization_config.get('preserve_original', True)
        self.output_suffix = self.stabilization_config.get('output_suffix', '_stabilized')

        # Stabilization parameters
        self.smoothness = self.stabilization_config.get('smoothness', 10)  # FFmpeg vidstab smoothness
        self.shakiness = self.stabilization_config.get('shakiness', 5)     # Shakiness detection
        self.accuracy = self.stabilization_config.get('accuracy', 15)      # Accuracy of detection

        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)

        logger.info(f"Initialized VideoStabilizer with FFmpeg vidstab filters")

    def is_ffmpeg_available(self) -> bool:
        """Check if FFmpeg with vidstab is available."""
        try:
            # Check FFmpeg availability
            result = subprocess.run(
                [self.ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                logger.warning("FFmpeg not found")
                return False

            # Check if vidstab filters are available
            result = subprocess.run(
                [self.ffmpeg_path, '-filters'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if 'vidstabdetect' in result.stdout and 'vidstabtransform' in result.stdout:
                logger.info("FFmpeg with vidstab filters is available")
                return True
            else:
                logger.warning("FFmpeg found but vidstab filters not available")
                return False

        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            logger.warning(f"FFmpeg not found or not accessible: {e}")
            return False

    def stabilize_video(self, input_path: str, output_path: Optional[str] = None,
                       progress_callback: Optional[callable] = None) -> Optional[str]:
        """
        Stabilize a video using FFmpeg's vidstab filters.

        Args:
            input_path: Path to input video file
            output_path: Optional output path. If None, generates based on input path
            progress_callback: Optional callback for progress updates

        Returns:
            Path to stabilized video file, or None if stabilization failed
        """
        if not self.is_ffmpeg_available():
            logger.error("FFmpeg with vidstab filters is not available.")
            logger.info("Install FFmpeg with libvidstab support for video stabilization.")
            return None

        input_path = Path(input_path)
        if not input_path.exists():
            logger.error(f"Input video file not found: {input_path}")
            return None

        # Generate output path if not provided
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}{self.output_suffix}{input_path.suffix}"
        else:
            output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary transform file
        transform_file = Path(self.temp_dir) / f"{input_path.stem}_transforms.trf"

        logger.info(f"Stabilizing video: {input_path} -> {output_path}")

        try:
            # Step 1: Detect camera motion
            if progress_callback:
                progress_callback(10, "Analyzing camera motion...")

            detect_cmd = [
                self.ffmpeg_path,
                '-i', str(input_path),
                '-vf', f'vidstabdetect=shakiness={self.shakiness}:accuracy={self.accuracy}:result={transform_file}',
                '-f', 'null',
                '-'
            ]

            logger.debug(f"Running motion detection: {' '.join(detect_cmd)}")

            detect_process = subprocess.run(
                detect_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if detect_process.returncode != 0:
                logger.error(f"Motion detection failed: {detect_process.stderr}")
                return None

            if not transform_file.exists():
                logger.error("Transform file was not created")
                return None

            # Step 2: Apply stabilization
            if progress_callback:
                progress_callback(50, "Applying stabilization...")

            stabilize_cmd = [
                self.ffmpeg_path,
                '-i', str(input_path),
                '-vf', f'vidstabtransform=input={transform_file}:smoothing={self.smoothness}:crop=black',
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-y',  # Overwrite output file
                str(output_path)
            ]

            logger.debug(f"Running stabilization: {' '.join(stabilize_cmd)}")

            stabilize_process = subprocess.run(
                stabilize_cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if stabilize_process.returncode != 0:
                logger.error(f"Stabilization failed: {stabilize_process.stderr}")
                return None

            if progress_callback:
                progress_callback(90, "Finalizing...")

            # Clean up transform file
            if transform_file.exists():
                transform_file.unlink()

            if output_path.exists():
                logger.info(f"Successfully stabilized video: {output_path}")

                if progress_callback:
                    progress_callback(100, "Stabilization complete!")

                return str(output_path)
            else:
                logger.error("Stabilized video was not created")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Stabilization process timed out")
            return None
        except Exception as e:
            logger.error(f"Error during stabilization: {e}")
            return None
        finally:
            # Clean up transform file if it still exists
            if transform_file.exists():
                try:
                    transform_file.unlink()
                except:
                    pass

    def stabilize_video_simple(self, input_path: str) -> Optional[str]:
        """
        Simple stabilization with default settings.

        Args:
            input_path: Path to input video file

        Returns:
            Path to stabilized video file, or None if stabilization failed
        """
        return self.stabilize_video(input_path)

    def cleanup_temp_files(self):
        """Clean up temporary files if not configured to keep them."""
        if not self.keep_temp_files and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary stabilization files: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp files: {e}")


def create_stabilizer(config: Dict[str, Any]) -> VideoStabilizer:
    """Factory function to create a VideoStabilizer instance."""
    return VideoStabilizer(config)