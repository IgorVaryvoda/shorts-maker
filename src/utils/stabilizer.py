"""
Video stabilization utility using Gyroflow.
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
    """Video stabilization using Gyroflow."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the stabilizer with configuration."""
        self.config = config
        self.stabilization_config = config.get('stabilization', {})

        # Gyroflow settings
        self.gyroflow_path = self.stabilization_config.get('gyroflow_path', 'gyroflow')
        self.temp_dir = self.stabilization_config.get('temp_dir', '.cache/stabilization')
        self.keep_temp_files = self.stabilization_config.get('keep_temp_files', False)
        self.preserve_original = self.stabilization_config.get('preserve_original', True)
        self.output_suffix = self.stabilization_config.get('output_suffix', '_stabilized')

        # Gyroflow parameters
        self.smoothness = self.stabilization_config.get('smoothness', 0.5)
        self.lens_correction = self.stabilization_config.get('lens_correction', True)
        self.horizon_lock = self.stabilization_config.get('horizon_lock', False)

        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)

        logger.info(f"Initialized VideoStabilizer with gyroflow path: {self.gyroflow_path}")

    def is_gyroflow_available(self) -> bool:
        """Check if gyroflow is available in the system."""
        try:
            result = subprocess.run(
                [self.gyroflow_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"Gyroflow found: {result.stdout.strip()}")
                return True
            else:
                logger.warning(f"Gyroflow command failed: {result.stderr}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            logger.warning(f"Gyroflow not found or not accessible: {e}")
            return False

    def stabilize_video(self, input_path: str, output_path: Optional[str] = None,
                       progress_callback: Optional[callable] = None) -> Optional[str]:
        """
        Stabilize a video using gyroflow.

        Args:
            input_path: Path to input video file
            output_path: Optional output path. If None, generates based on input path
            progress_callback: Optional callback for progress updates

        Returns:
            Path to stabilized video file, or None if stabilization failed
        """
        if not self.is_gyroflow_available():
            logger.error("Gyroflow is not available. Please install gyroflow and ensure it's in your PATH.")
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

        logger.info(f"Stabilizing video: {input_path} -> {output_path}")

        if progress_callback:
            progress_callback(0, "Starting gyroflow stabilization...")

        try:
            # Build gyroflow command
            cmd = [
                self.gyroflow_path,
                str(input_path),
                '--output', str(output_path),
                '--smoothness', str(self.smoothness)
            ]

            # Add optional parameters
            if self.lens_correction:
                cmd.append('--lens-correction')

            if self.horizon_lock:
                cmd.append('--horizon-lock')

            logger.debug(f"Running gyroflow command: {' '.join(cmd)}")

            if progress_callback:
                progress_callback(10, "Running gyroflow...")

            # Run gyroflow
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )

            # Monitor progress
            stdout_lines = []
            stderr_lines = []

            while True:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()

                if stdout_line:
                    stdout_lines.append(stdout_line.strip())
                    logger.debug(f"Gyroflow stdout: {stdout_line.strip()}")

                    # Try to extract progress information
                    if progress_callback and ('progress' in stdout_line.lower() or '%' in stdout_line):
                        try:
                            # Look for percentage in the output
                            import re
                            match = re.search(r'(\d+)%', stdout_line)
                            if match:
                                percentage = int(match.group(1))
                                progress_callback(10 + (percentage * 0.8), f"Stabilizing... {percentage}%")
                        except:
                            pass

                if stderr_line:
                    stderr_lines.append(stderr_line.strip())
                    logger.debug(f"Gyroflow stderr: {stderr_line.strip()}")

                if process.poll() is not None:
                    break

            # Get remaining output
            remaining_stdout, remaining_stderr = process.communicate()
            if remaining_stdout:
                stdout_lines.extend(remaining_stdout.strip().split('\n'))
            if remaining_stderr:
                stderr_lines.extend(remaining_stderr.strip().split('\n'))

            if progress_callback:
                progress_callback(90, "Finalizing stabilization...")

            # Check if gyroflow succeeded
            if process.returncode == 0:
                if output_path.exists():
                    logger.info(f"Successfully stabilized video: {output_path}")

                    if progress_callback:
                        progress_callback(100, "Stabilization complete!")

                    return str(output_path)
                else:
                    logger.error(f"Gyroflow completed but output file not found: {output_path}")
                    return None
            else:
                logger.error(f"Gyroflow failed with return code {process.returncode}")
                logger.error(f"Stderr: {' '.join(stderr_lines)}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Gyroflow process timed out")
            return None
        except Exception as e:
            logger.error(f"Error running gyroflow: {e}")
            return None

    def stabilize_video_simple(self, input_path: str) -> Optional[str]:
        """
        Simple stabilization with basic gyroflow command.
        Falls back to this if the advanced method fails.

        Args:
            input_path: Path to input video file

        Returns:
            Path to stabilized video file, or None if stabilization failed
        """
        if not self.is_gyroflow_available():
            return None

        input_path = Path(input_path)
        output_path = input_path.parent / f"{input_path.stem}{self.output_suffix}{input_path.suffix}"

        try:
            # Simple gyroflow command
            cmd = [self.gyroflow_path, str(input_path)]

            logger.info(f"Running simple gyroflow stabilization: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                # Gyroflow typically outputs to the same directory with a suffix
                # Try to find the output file
                possible_outputs = [
                    input_path.parent / f"{input_path.stem}_stabilized{input_path.suffix}",
                    input_path.parent / f"{input_path.stem}_gyroflow{input_path.suffix}",
                    output_path
                ]

                for possible_output in possible_outputs:
                    if possible_output.exists():
                        logger.info(f"Found stabilized video: {possible_output}")
                        return str(possible_output)

                logger.warning("Gyroflow completed but couldn't find output file")
                return None
            else:
                logger.error(f"Simple gyroflow failed: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"Error in simple gyroflow stabilization: {e}")
            return None

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