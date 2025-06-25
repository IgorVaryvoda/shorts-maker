"""Progress tracking utilities for video processing."""

import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import sys


class ProgressStatus(Enum):
    """Progress status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressInfo:
    """Progress information container."""
    current: int = 0
    total: int = 100
    status: ProgressStatus = ProgressStatus.PENDING
    message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def percentage(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100.0)

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def eta(self) -> Optional[float]:
        """Estimate time remaining in seconds."""
        if self.current == 0 or self.start_time is None:
            return None

        elapsed = self.elapsed_time
        rate = self.current / elapsed
        remaining = self.total - self.current

        if rate > 0:
            return remaining / rate
        return None


class ProgressBar:
    """Simple progress bar for terminal output."""

    def __init__(self, total: int, width: int = 50, show_eta: bool = True):
        self.total = total
        self.width = width
        self.show_eta = show_eta
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0

    def update(self, current: int, message: str = ""):
        """Update progress bar."""
        self.current = current

        # Throttle updates to avoid spam
        now = time.time()
        if now - self.last_update < 0.1 and current < self.total:
            return
        self.last_update = now

        percentage = (current / self.total) * 100 if self.total > 0 else 0
        filled = int(self.width * current / self.total) if self.total > 0 else 0
        bar = "█" * filled + "░" * (self.width - filled)

        # Calculate ETA
        eta_str = ""
        if self.show_eta and current > 0:
            elapsed = now - self.start_time
            rate = current / elapsed
            if rate > 0:
                remaining = (self.total - current) / rate
                eta_str = f" ETA: {self._format_time(remaining)}"

        # Format message
        msg_str = f" {message}" if message else ""

        # Print progress bar
        sys.stdout.write(f"\r{bar} {percentage:5.1f}%{eta_str}{msg_str}")
        sys.stdout.flush()

        if current >= self.total:
            elapsed = now - self.start_time
            print(f"\nCompleted in {self._format_time(elapsed)}")

    def _format_time(self, seconds: float) -> str:
        """Format time duration."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


class ProgressTracker:
    """Advanced progress tracker with multiple stages."""

    def __init__(self, stages: Dict[str, int]):
        """
        Initialize progress tracker.

        Args:
            stages: Dictionary mapping stage names to their total steps
        """
        self.stages = stages
        self.current_stage = None
        self.stage_progress = {}
        self.callbacks = []
        self.start_time = time.time()
        self.stage_start_time = None
        self._lock = threading.Lock()

        # Initialize stage progress
        for stage_name in stages:
            self.stage_progress[stage_name] = ProgressInfo(total=stages[stage_name])

    def add_callback(self, callback: Callable[[str, ProgressInfo], None]):
        """Add progress callback function."""
        self.callbacks.append(callback)

    def start_stage(self, stage_name: str, message: str = ""):
        """Start a new processing stage."""
        with self._lock:
            if stage_name not in self.stages:
                raise ValueError(f"Unknown stage: {stage_name}")

            self.current_stage = stage_name
            self.stage_start_time = time.time()

            progress = self.stage_progress[stage_name]
            progress.status = ProgressStatus.RUNNING
            progress.message = message or f"Starting {stage_name}"
            progress.start_time = self.stage_start_time
            progress.current = 0

            self._notify_callbacks(stage_name, progress)
            print(f"\n🎬 {stage_name.replace('_', ' ').title()}: {progress.message}")

    def update_stage(self, current: int, message: str = ""):
        """Update current stage progress."""
        with self._lock:
            if self.current_stage is None:
                return

            progress = self.stage_progress[self.current_stage]
            progress.current = min(current, progress.total)
            if message:
                progress.message = message

            self._notify_callbacks(self.current_stage, progress)

            # Update progress bar
            percentage = progress.percentage
            bar_width = 40
            filled = int(bar_width * percentage / 100)
            bar = "█" * filled + "░" * (bar_width - filled)

            eta_str = ""
            if progress.eta:
                eta_str = f" ETA: {self._format_time(progress.eta)}"

            sys.stdout.write(f"\r  {bar} {percentage:5.1f}%{eta_str} {message}")
            sys.stdout.flush()

    def complete_stage(self, message: str = ""):
        """Complete current stage."""
        with self._lock:
            if self.current_stage is None:
                return

            progress = self.stage_progress[self.current_stage]
            progress.current = progress.total
            progress.status = ProgressStatus.COMPLETED
            progress.end_time = time.time()
            if message:
                progress.message = message

            self._notify_callbacks(self.current_stage, progress)

            elapsed = progress.elapsed_time
            print(f"\n  ✅ Completed in {self._format_time(elapsed)}")

    def fail_stage(self, error_message: str):
        """Mark current stage as failed."""
        with self._lock:
            if self.current_stage is None:
                return

            progress = self.stage_progress[self.current_stage]
            progress.status = ProgressStatus.FAILED
            progress.message = error_message
            progress.end_time = time.time()

            self._notify_callbacks(self.current_stage, progress)
            print(f"\n  ❌ Failed: {error_message}")

    def get_overall_progress(self) -> ProgressInfo:
        """Get overall progress across all stages."""
        total_steps = sum(self.stages.values())
        completed_steps = sum(
            progress.current for progress in self.stage_progress.values()
        )

        # Determine overall status
        statuses = [progress.status for progress in self.stage_progress.values()]
        if ProgressStatus.FAILED in statuses:
            status = ProgressStatus.FAILED
        elif ProgressStatus.RUNNING in statuses:
            status = ProgressStatus.RUNNING
        elif all(s == ProgressStatus.COMPLETED for s in statuses):
            status = ProgressStatus.COMPLETED
        else:
            status = ProgressStatus.PENDING

        return ProgressInfo(
            current=completed_steps,
            total=total_steps,
            status=status,
            start_time=self.start_time
        )

    def print_summary(self):
        """Print final progress summary."""
        overall = self.get_overall_progress()
        total_time = overall.elapsed_time

        print(f"\n{'='*60}")
        print(f"📊 Processing Summary")
        print(f"{'='*60}")
        print(f"Total time: {self._format_time(total_time)}")
        print(f"Overall status: {overall.status.value.upper()}")
        print()

        for stage_name, progress in self.stage_progress.items():
            status_icon = {
                ProgressStatus.COMPLETED: "✅",
                ProgressStatus.FAILED: "❌",
                ProgressStatus.RUNNING: "🔄",
                ProgressStatus.PENDING: "⏳"
            }.get(progress.status, "❓")

            stage_time = self._format_time(progress.elapsed_time)
            print(f"{status_icon} {stage_name.replace('_', ' ').title()}: {stage_time}")

        print(f"{'='*60}")

    def _notify_callbacks(self, stage_name: str, progress: ProgressInfo):
        """Notify all registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(stage_name, progress)
            except Exception as e:
                print(f"Warning: Progress callback failed: {e}")

    def _format_time(self, seconds: float) -> str:
        """Format time duration."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


class VideoProcessingProgress:
    """Specialized progress tracker for video processing."""

    def __init__(self, total_frames: int, fps: float = 30.0):
        self.total_frames = total_frames
        self.fps = max(fps, 1.0)  # Prevent division by zero
        self.total_duration = total_frames / self.fps

        # Define processing stages
        stages = {
            "loading": 10,
            "scene_detection": total_frames // 10,  # Sample frames
            "color_grading": total_frames,
            "cropping": total_frames,
            "encoding": total_frames,
            "finalizing": 5
        }

        self.tracker = ProgressTracker(stages)

    def start_loading(self):
        """Start video loading stage."""
        self.tracker.start_stage("loading", "Loading video file...")

    def update_loading(self, current: int, message: str = ""):
        """Update loading progress."""
        self.tracker.update_stage(current, message)

    def complete_loading(self):
        """Complete loading stage."""
        self.tracker.complete_stage("Video loaded successfully")

    def start_scene_detection(self):
        """Start scene detection stage."""
        self.tracker.start_stage("scene_detection", "Analyzing video content...")

    def update_scene_detection(self, frame_num: int):
        """Update scene detection progress."""
        sample_frame = frame_num // 10  # We sample every 10th frame
        message = f"Analyzing frame {frame_num}/{self.total_frames}"
        self.tracker.update_stage(sample_frame, message)

    def complete_scene_detection(self, num_scenes: int):
        """Complete scene detection stage."""
        self.tracker.complete_stage(f"Found {num_scenes} potential scenes")

    def start_color_grading(self):
        """Start color grading stage."""
        self.tracker.start_stage("color_grading", "Applying color grading...")

    def update_color_grading(self, frame_num: int):
        """Update color grading progress."""
        message = f"Processing frame {frame_num}/{self.total_frames}"
        self.tracker.update_stage(frame_num, message)

    def complete_color_grading(self):
        """Complete color grading stage."""
        self.tracker.complete_stage("Color grading applied")

    def start_cropping(self):
        """Start cropping stage."""
        self.tracker.start_stage("cropping", "Smart cropping to 9:16...")

    def update_cropping(self, frame_num: int):
        """Update cropping progress."""
        message = f"Cropping frame {frame_num}/{self.total_frames}"
        self.tracker.update_stage(frame_num, message)

    def complete_cropping(self):
        """Complete cropping stage."""
        self.tracker.complete_stage("Video cropped to vertical format")

    def start_encoding(self):
        """Start encoding stage."""
        self.tracker.start_stage("encoding", "Encoding final video...")

    def update_encoding(self, frame_num: int):
        """Update encoding progress."""
        message = f"Encoding frame {frame_num}/{self.total_frames}"
        self.tracker.update_stage(frame_num, message)

    def complete_encoding(self):
        """Complete encoding stage."""
        self.tracker.complete_stage("Video encoded successfully")

    def start_finalizing(self):
        """Start finalizing stage."""
        self.tracker.start_stage("finalizing", "Finalizing output...")

    def update_finalizing(self, current: int, message: str = ""):
        """Update finalizing progress."""
        self.tracker.update_stage(current, message)

    def complete_finalizing(self, output_path: str):
        """Complete finalizing stage."""
        self.tracker.complete_stage(f"Saved to {output_path}")

    def fail_stage(self, error_message: str):
        """Mark current stage as failed."""
        self.tracker.fail_stage(error_message)

    def print_summary(self):
        """Print processing summary."""
        self.tracker.print_summary()