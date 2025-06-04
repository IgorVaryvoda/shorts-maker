# LUT Files Directory

Place your .cube LUT files in this directory for color grading.

## Supported Formats
- 3D LUTs in .cube format
- Sizes: 17x17x17, 33x33x33, 65x65x65

## Free LUT Sources
- RocketStock: https://www.rocketstock.com/free-after-effects-templates/35-free-luts-for-color-grading-videos/
- Ground Control: https://groundcontrol.film/free-luts/
- IWLTBAP: https://iwltbap.com/

## Commercial LUT Sources
- FilmConvert
- Color Grading Central
- LUT Robot

## Usage
```python
from src.core.color_grader import ColorGrader

grader = ColorGrader()
graded_video = grader.apply_lut(
    video_path="input.mp4",
    lut_path="luts/your_lut.cube",
    intensity=0.8
)
```
