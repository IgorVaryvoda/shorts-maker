"""
Configuration management for Shorts Creator.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for Shorts Creator."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            self._config = self._get_default_config()
            return

        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Loaded config from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self._config = self._get_default_config()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'video.fps')."""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to YAML file."""
        save_path = Path(path) if path else self.config_path

        try:
            with open(save_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            logger.info(f"Saved config to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    @property
    def data(self) -> Dict[str, Any]:
        """Get the raw configuration dictionary."""
        return self._config.copy()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for FPV drone videos."""
        return {
            "video": {
                "output_resolution": [1080, 1920],
                "fps": 30,
                "quality": "high",
                "codec": "h264",
                "bitrate": "5M",
                "supported_formats": ["mp4", "mov", "avi", "mkv", "webm"],
                "max_input_resolution": [3840, 2160]
            },
            "segmentation": {
                "min_duration": 15,
                "max_duration": 60,
                "target_duration": 30,
                "scene_threshold": 0.4,
                "motion_threshold": 0.3,
                "overlap_threshold": 0.3,
                "frame_sample_rate": 1,
                "histogram_bins": 256,
                "optical_flow_enabled": True,
                "scoring": {
                    "visual_interest": 0.6,  # Higher for FPV
                    "motion_activity": 0.4,  # Important for drone footage
                    "face_presence": 0.0     # Not relevant for FPV
                }
            },
            "color_grading": {
                "default_lut": "luts/cinematic.cube",
                "lut_intensity": 0.8,
                "auto_exposure": True,
                "auto_contrast": True,
                "auto_saturation": False,
                "exposure_offset": 0.0,
                "contrast_boost": 1.0,
                "saturation_boost": 1.0,
                "highlights": 0.0,
                "shadows": 0.0,
                "input_colorspace": "rec709",
                "working_colorspace": "rec709",
                "output_colorspace": "rec709"
            },
            "cropping": {
                "face_detection": False,  # Not needed for FPV
                "object_detection": True,
                "motion_tracking": True,
                "crop_padding": 0.1,
                "smooth_tracking": True,
                "tracking_smoothness": 0.8,
                "top_safe_area": 0.15,
                "bottom_safe_area": 0.20,
                "side_safe_area": 0.05
            },
            "performance": {
                "use_gpu": True,
                "gpu_device": 0,
                "num_threads": 4,
                "chunk_size": 1000,
                "streaming_mode": False,
                "max_memory_usage": "8GB",
                "temp_dir": "/tmp/shorts_creator",
                "enable_cache": True,
                "cache_dir": ".cache",
                "cache_duration": 7
            },
            "output": {
                "naming_pattern": "{input_name}_short_{index}_{timestamp}",
                "timestamp_format": "%Y%m%d_%H%M%S",
                "create_subdirs": True,
                "subdir_pattern": "{date}/{input_name}",
                "include_metadata": True,
                "creator_tag": "ShortsCreator"
            },
            "platforms": {
                "youtube_shorts": {
                    "max_duration": 60,
                    "recommended_duration": 30,
                    "aspect_ratio": [9, 16],
                    "max_file_size": "100MB"
                },
                "tiktok": {
                    "max_duration": 60,
                    "recommended_duration": 15,
                    "aspect_ratio": [9, 16],
                    "max_file_size": "72MB"
                },
                "instagram_reels": {
                    "max_duration": 90,
                    "recommended_duration": 30,
                    "aspect_ratio": [9, 16],
                    "max_file_size": "100MB"
                }
            }
        }


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from file."""
    return Config(config_path)