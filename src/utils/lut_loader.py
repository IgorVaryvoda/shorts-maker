"""
LUT (Look-Up Table) loader and processor for color grading.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
import re

logger = logging.getLogger(__name__)


class LUT3D:
    """3D LUT for color grading."""

    def __init__(self, size: int, data: np.ndarray, title: str = ""):
        self.size = size
        self.data = data  # Shape: (size, size, size, 3)
        self.title = title

        if data.shape != (size, size, size, 3):
            raise ValueError(f"LUT data shape {data.shape} doesn't match size {size}")

    def apply(self, image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """
        Apply LUT to an image.

        Args:
            image: Input image (H, W, 3) with values in [0, 1]
            intensity: LUT intensity (0.0 = no effect, 1.0 = full effect)

        Returns:
            Color-graded image
        """
        if intensity == 0.0:
            return image.copy()

        # Ensure image is in [0, 1] range
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        # Apply 3D interpolation
        graded = self._trilinear_interpolation(image)

        # Blend with original based on intensity
        if intensity < 1.0:
            graded = image * (1.0 - intensity) + graded * intensity

        return graded

    def _trilinear_interpolation(self, image: np.ndarray) -> np.ndarray:
        """Apply trilinear interpolation for 3D LUT."""
        h, w, c = image.shape
        if c != 3:
            raise ValueError("Image must have 3 channels (RGB)")

        # Scale to LUT coordinates
        coords = image * (self.size - 1)

        # Get integer and fractional parts
        coords_int = np.floor(coords).astype(np.int32)
        coords_frac = coords - coords_int

        # Clamp to valid range
        coords_int = np.clip(coords_int, 0, self.size - 2)

        # Get the 8 corner values for trilinear interpolation
        result = np.zeros_like(image)

        for dr in [0, 1]:
            for dg in [0, 1]:
                for db in [0, 1]:
                    # Calculate weight for this corner
                    weight = (
                        (dr * coords_frac[:, :, 0] + (1 - dr) * (1 - coords_frac[:, :, 0])) *
                        (dg * coords_frac[:, :, 1] + (1 - dg) * (1 - coords_frac[:, :, 1])) *
                        (db * coords_frac[:, :, 2] + (1 - db) * (1 - coords_frac[:, :, 2]))
                    )

                    # Get LUT values at this corner
                    r_idx = coords_int[:, :, 0] + dr
                    g_idx = coords_int[:, :, 1] + dg
                    b_idx = coords_int[:, :, 2] + db

                    lut_values = self.data[r_idx, g_idx, b_idx]

                    # Add weighted contribution
                    result += lut_values * weight[:, :, np.newaxis]

        return result


class LUTLoader:
    """Loader for various LUT formats."""

    @staticmethod
    def load_cube(file_path: str) -> LUT3D:
        """
        Load a .cube LUT file.

        Args:
            file_path: Path to .cube file

        Returns:
            LUT3D object
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"LUT file not found: {file_path}")

        logger.info(f"Loading LUT from {file_path}")

        with open(path, 'r') as f:
            lines = f.readlines()

        # Parse header
        title = ""
        size = None
        domain_min = np.array([0.0, 0.0, 0.0])
        domain_max = np.array([1.0, 1.0, 1.0])

        data_start_idx = 0

        for i, line in enumerate(lines):
            line = line.strip()

            if line.startswith('TITLE'):
                title = line.split('"')[1] if '"' in line else line[5:].strip()
            elif line.startswith('LUT_3D_SIZE'):
                size = int(line.split()[-1])
            elif line.startswith('DOMAIN_MIN'):
                domain_min = np.array([float(x) for x in line.split()[1:4]])
            elif line.startswith('DOMAIN_MAX'):
                domain_max = np.array([float(x) for x in line.split()[1:4]])
            elif line and not line.startswith('#') and not line.startswith('LUT_'):
                # First data line
                data_start_idx = i
                break

        if size is None:
            raise ValueError("LUT_3D_SIZE not found in .cube file")

        # Parse data
        data_lines = lines[data_start_idx:]
        lut_data = []

        for line in data_lines:
            line = line.strip()
            if line and not line.startswith('#'):
                values = line.split()
                if len(values) >= 3:
                    try:
                        rgb = [float(values[0]), float(values[1]), float(values[2])]
                        lut_data.append(rgb)
                    except ValueError:
                        continue

        if len(lut_data) != size ** 3:
            raise ValueError(f"Expected {size**3} data points, got {len(lut_data)}")

        # Reshape data to 3D array
        lut_array = np.array(lut_data).reshape(size, size, size, 3)

        # Scale from domain to [0, 1] if needed
        if not np.allclose(domain_min, [0, 0, 0]) or not np.allclose(domain_max, [1, 1, 1]):
            logger.info(f"Scaling LUT from domain [{domain_min}, {domain_max}] to [0, 1]")
            lut_array = (lut_array - domain_min) / (domain_max - domain_min)

        logger.info(f"Loaded {size}x{size}x{size} LUT: {title}")
        return LUT3D(size, lut_array, title)

    @staticmethod
    def create_identity_lut(size: int = 33) -> LUT3D:
        """Create an identity LUT (no color change)."""
        coords = np.linspace(0, 1, size)
        r, g, b = np.meshgrid(coords, coords, coords, indexing='ij')
        data = np.stack([r, g, b], axis=-1)
        return LUT3D(size, data, "Identity")

    @staticmethod
    def list_available_luts(lut_dir: str = "luts") -> Dict[str, str]:
        """List available LUT files."""
        lut_path = Path(lut_dir)
        if not lut_path.exists():
            return {}

        luts = {}
        for file_path in lut_path.glob("*.cube"):
            luts[file_path.stem] = str(file_path)

        return luts


def load_lut(file_path: str) -> LUT3D:
    """
    Load a LUT from file.

    Args:
        file_path: Path to LUT file

    Returns:
        LUT3D object
    """
    path = Path(file_path)

    if path.suffix.lower() == '.cube':
        return LUTLoader.load_cube(file_path)
    else:
        raise ValueError(f"Unsupported LUT format: {path.suffix}")


def create_sample_lut(lut_path: str, lut_type: str = "warm") -> None:
    """Create a sample LUT file for testing."""
    size = 17
    coords = np.linspace(0, 1, size)
    r, g, b = np.meshgrid(coords, coords, coords, indexing='ij')

    if lut_type == "warm":
        # Warm color grading
        r_out = np.clip(r * 1.1 + 0.05, 0, 1)
        g_out = np.clip(g * 1.05, 0, 1)
        b_out = np.clip(b * 0.9, 0, 1)
        title = "Warm Tone"
    elif lut_type == "cool":
        # Cool color grading
        r_out = np.clip(r * 0.9, 0, 1)
        g_out = np.clip(g * 1.05, 0, 1)
        b_out = np.clip(b * 1.1 + 0.05, 0, 1)
        title = "Cool Tone"
    elif lut_type == "cinematic":
        # Cinematic look
        r_out = np.clip(r * 1.05 + 0.02, 0, 1)
        g_out = np.clip(g * 1.02, 0, 1)
        b_out = np.clip(b * 0.95 - 0.02, 0, 1)
        title = "Cinematic"
    else:
        # Identity
        r_out, g_out, b_out = r, g, b
        title = "Identity"

    # Write .cube file
    with open(lut_path, 'w') as f:
        f.write(f'TITLE "{title}"\n')
        f.write(f'LUT_3D_SIZE {size}\n')
        f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
        f.write('DOMAIN_MAX 1.0 1.0 1.0\n')
        f.write('\n')

        for i in range(size):
            for j in range(size):
                for k in range(size):
                    f.write(f'{r_out[i,j,k]:.6f} {g_out[i,j,k]:.6f} {b_out[i,j,k]:.6f}\n')

    logger.info(f"Created sample {lut_type} LUT: {lut_path}")