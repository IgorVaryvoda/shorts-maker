"""
Color grading system with LUT support for FPV drone videos.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

# Handle imports for both CLI and module usage
try:
    from ..utils.lut_loader import LUT3D, load_lut, LUTLoader
except ImportError:
    # Fallback for CLI usage
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.lut_loader import LUT3D, load_lut, LUTLoader

logger = logging.getLogger(__name__)


class ColorGrader:
    """Color grading system with LUT support."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lut_intensity = config.get('lut_intensity', 0.8)
        self.auto_exposure = config.get('auto_exposure', True)
        self.auto_contrast = config.get('auto_contrast', True)
        self.auto_saturation = config.get('auto_saturation', False)

        # Manual adjustments
        self.exposure_offset = config.get('exposure_offset', 0.0)
        self.contrast_boost = config.get('contrast_boost', 1.0)
        self.saturation_boost = config.get('saturation_boost', 1.0)
        self.highlights = config.get('highlights', 0.0)
        self.shadows = config.get('shadows', 0.0)

        # Load default LUT if specified
        self.default_lut = None
        default_lut_path = config.get('default_lut')
        if default_lut_path and Path(default_lut_path).exists():
            try:
                self.default_lut = load_lut(default_lut_path)
                logger.info(f"Loaded default LUT: {default_lut_path}")
            except Exception as e:
                logger.warning(f"Failed to load default LUT: {e}")

        logger.info("Initialized Color Grader")

    def apply_lut(self, image: np.ndarray, lut_path: Optional[str] = None,
                  intensity: Optional[float] = None) -> np.ndarray:
        """
        Apply LUT to image.

        Args:
            image: Input image (BGR format)
            lut_path: Path to LUT file (optional, uses default if None)
            intensity: LUT intensity (optional, uses config default if None)

        Returns:
            Color-graded image
        """
        if intensity is None:
            intensity = self.lut_intensity

        # Load LUT
        lut = None
        if lut_path:
            try:
                lut = load_lut(lut_path)
            except Exception as e:
                logger.warning(f"Failed to load LUT {lut_path}: {e}")

        if lut is None:
            lut = self.default_lut

        if lut is None:
            logger.warning("No LUT available, returning original image")
            return image.copy()

        # Convert BGR to RGB for LUT processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        rgb_float = rgb_image.astype(np.float32) / 255.0

        # Apply LUT
        graded_rgb = lut.apply(rgb_float, intensity)

        # Convert back to BGR and uint8
        graded_rgb = np.clip(graded_rgb * 255.0, 0, 255).astype(np.uint8)
        graded_bgr = cv2.cvtColor(graded_rgb, cv2.COLOR_RGB2BGR)

        return graded_bgr

    def apply_auto_corrections(self, image: np.ndarray) -> np.ndarray:
        """Apply automatic color corrections."""
        corrected = image.copy()

        if self.auto_exposure:
            corrected = self._auto_exposure_correction(corrected)

        if self.auto_contrast:
            corrected = self._auto_contrast_correction(corrected)

        if self.auto_saturation:
            corrected = self._auto_saturation_correction(corrected)

        return corrected

    def apply_manual_adjustments(self, image: np.ndarray) -> np.ndarray:
        """Apply manual color adjustments."""
        adjusted = image.copy()

        # Convert to float for processing
        img_float = adjusted.astype(np.float32) / 255.0

        # Exposure adjustment
        if self.exposure_offset != 0.0:
            img_float = self._adjust_exposure(img_float, self.exposure_offset)

        # Contrast adjustment
        if self.contrast_boost != 1.0:
            img_float = self._adjust_contrast(img_float, self.contrast_boost)

        # Saturation adjustment
        if self.saturation_boost != 1.0:
            img_float = self._adjust_saturation(img_float, self.saturation_boost)

        # Highlights and shadows
        if self.highlights != 0.0 or self.shadows != 0.0:
            img_float = self._adjust_highlights_shadows(img_float, self.highlights, self.shadows)

        # Convert back to uint8
        adjusted = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)

        return adjusted

    def grade_image(self, image: np.ndarray, lut_path: Optional[str] = None,
                   intensity: Optional[float] = None) -> np.ndarray:
        """
        Apply complete color grading pipeline.

        Args:
            image: Input image (BGR format)
            lut_path: Path to LUT file (optional)
            intensity: LUT intensity (optional)

        Returns:
            Fully color-graded image
        """
        # Step 1: Auto corrections
        corrected = self.apply_auto_corrections(image)

        # Step 2: Apply LUT
        graded = self.apply_lut(corrected, lut_path, intensity)

        # Step 3: Manual adjustments
        final = self.apply_manual_adjustments(graded)

        return final

    def _auto_exposure_correction(self, image: np.ndarray) -> np.ndarray:
        """Automatic exposure correction for FPV footage."""
        # Convert to LAB color space for better luminance control
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)

        # Calculate target exposure based on histogram
        hist = cv2.calcHist([l_channel.astype(np.uint8)], [0], None, [256], [0, 256])

        # Find the 50th percentile (median)
        total_pixels = l_channel.size
        cumsum = np.cumsum(hist)
        median_idx = np.where(cumsum >= total_pixels * 0.5)[0][0]

        # Target median around 128 (middle gray)
        target_median = 128
        exposure_adjustment = target_median / (median_idx + 1e-7)

        # Limit adjustment to reasonable range
        exposure_adjustment = np.clip(exposure_adjustment, 0.5, 2.0)

        # Apply adjustment
        l_channel = np.clip(l_channel * exposure_adjustment, 0, 255)
        lab[:, :, 0] = l_channel.astype(np.uint8)

        # Convert back to BGR
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return corrected

    def _auto_contrast_correction(self, image: np.ndarray) -> np.ndarray:
        """Automatic contrast correction."""
        # Convert to LAB for luminance processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # Calculate current contrast (standard deviation)
        current_std = np.std(l_channel)

        # Target contrast for FPV footage (higher for dramatic effect)
        target_std = 45.0

        if current_std > 5:  # Avoid division by very small numbers
            contrast_factor = target_std / current_std
            contrast_factor = np.clip(contrast_factor, 0.8, 1.5)  # Limit adjustment

            # Apply contrast adjustment
            mean_l = np.mean(l_channel)
            l_adjusted = (l_channel - mean_l) * contrast_factor + mean_l
            l_adjusted = np.clip(l_adjusted, 0, 255)

            lab[:, :, 0] = l_adjusted.astype(np.uint8)

        # Convert back to BGR
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return corrected

    def _auto_saturation_correction(self, image: np.ndarray) -> np.ndarray:
        """Automatic saturation correction."""
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1].astype(np.float32)

        # Calculate current saturation
        current_sat = np.mean(s_channel)

        # Target saturation for FPV footage
        target_sat = 120.0

        if current_sat > 10:  # Avoid very low saturation images
            sat_factor = target_sat / current_sat
            sat_factor = np.clip(sat_factor, 0.8, 1.3)  # Limit adjustment

            # Apply saturation adjustment
            s_adjusted = np.clip(s_channel * sat_factor, 0, 255)
            hsv[:, :, 1] = s_adjusted.astype(np.uint8)

        # Convert back to BGR
        corrected = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return corrected

    def _adjust_exposure(self, image: np.ndarray, exposure: float) -> np.ndarray:
        """Adjust exposure (brightness)."""
        # Exposure adjustment in linear space
        adjusted = image * (2.0 ** exposure)
        return np.clip(adjusted, 0.0, 1.0)

    def _adjust_contrast(self, image: np.ndarray, contrast: float) -> np.ndarray:
        """Adjust contrast."""
        # Contrast adjustment around middle gray
        adjusted = (image - 0.5) * contrast + 0.5
        return np.clip(adjusted, 0.0, 1.0)

    def _adjust_saturation(self, image: np.ndarray, saturation: float) -> np.ndarray:
        """Adjust saturation."""
        # Convert to HSV
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv_float = hsv.astype(np.float32)

        # Adjust saturation channel
        hsv_float[:, :, 1] = np.clip(hsv_float[:, :, 1] * saturation, 0, 255)

        # Convert back to BGR
        hsv_adjusted = hsv_float.astype(np.uint8)
        bgr_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

        return bgr_adjusted.astype(np.float32) / 255.0

    def _adjust_highlights_shadows(self, image: np.ndarray, highlights: float,
                                 shadows: float) -> np.ndarray:
        """Adjust highlights and shadows separately."""
        # Create luminance mask
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray_norm = gray.astype(np.float32) / 255.0

        # Create masks for highlights and shadows
        highlight_mask = np.power(gray_norm, 2)  # Emphasize bright areas
        shadow_mask = 1.0 - np.power(gray_norm, 0.5)  # Emphasize dark areas

        # Apply adjustments
        adjusted = image.copy()

        if highlights != 0.0:
            highlight_adj = 1.0 + highlights
            for c in range(3):
                adjusted[:, :, c] = adjusted[:, :, c] * (
                    1.0 + highlight_mask * (highlight_adj - 1.0)
                )

        if shadows != 0.0:
            shadow_adj = 1.0 + shadows
            for c in range(3):
                adjusted[:, :, c] = adjusted[:, :, c] * (
                    1.0 + shadow_mask * (shadow_adj - 1.0)
                )

        return np.clip(adjusted, 0.0, 1.0)

    def analyze_image(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze image characteristics for grading recommendations."""
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        analysis = {
            'brightness': np.mean(gray) / 255.0,
            'contrast': np.std(gray) / 255.0,
            'saturation': np.mean(hsv[:, :, 1]) / 255.0,
            'exposure_bias': self._calculate_exposure_bias(gray),
            'color_temperature': self._estimate_color_temperature(image),
            'dynamic_range': (np.max(gray) - np.min(gray)) / 255.0
        }

        return analysis

    def _calculate_exposure_bias(self, gray: np.ndarray) -> float:
        """Calculate exposure bias recommendation."""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # Find 50th percentile
        total_pixels = gray.size
        cumsum = np.cumsum(hist)
        median_idx = np.where(cumsum >= total_pixels * 0.5)[0][0]

        # Calculate bias (negative = underexposed, positive = overexposed)
        target_median = 128
        bias = (median_idx - target_median) / 128.0

        return bias

    def _estimate_color_temperature(self, image: np.ndarray) -> float:
        """Estimate color temperature (simplified)."""
        # Calculate average color channels
        b_avg = np.mean(image[:, :, 0])
        g_avg = np.mean(image[:, :, 1])
        r_avg = np.mean(image[:, :, 2])

        # Simple color temperature estimation
        if r_avg > 0 and b_avg > 0:
            ratio = r_avg / b_avg
            # Rough mapping to Kelvin (simplified)
            if ratio > 1.2:
                temp = 3000  # Warm
            elif ratio < 0.8:
                temp = 7000  # Cool
            else:
                temp = 5500  # Neutral
        else:
            temp = 5500  # Default

        return temp


def create_color_grader(config: Dict[str, Any]) -> ColorGrader:
    """Create a color grader instance."""
    return ColorGrader(config)