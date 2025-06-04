# Shorts Creator - Automated Video Editor

An intelligent video editing system that automatically transforms long-form videos into engaging YouTube Shorts and TikTok content with advanced color grading and AI-powered scene detection.

## 🚀 Features

- **🎨 Advanced Color Grading**: Professional LUT (.cube) file support with 3D interpolation
- **🤖 Intelligent Scene Detection**: AI-powered segmentation using motion, audio, and visual analysis
- **📱 Platform Optimization**: Automatic formatting for YouTube Shorts (9:16) and TikTok
- **🎵 Smart Audio Processing**: Beat detection, noise reduction, and automatic music sync
- **✂️ Dynamic Cropping**: AI-powered subject tracking and reframing
- **📝 Auto Captions**: Speech-to-text with stylized subtitle rendering
- **⚡ GPU Acceleration**: CUDA support for fast processing

## 🛠️ Installation

### System Requirements
- Python 3.8+
- FFmpeg with GPU acceleration support
- CUDA toolkit (optional, for GPU acceleration)
- 8GB+ RAM recommended
- GPU with 4GB+ VRAM (optional)

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/shorts-creator.git
cd shorts-creator
```

2. **Install system dependencies (Ubuntu/Debian)**
```bash
sudo apt update
sudo apt install ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0
```

3. **Install system dependencies (Arch/Manjaro)**
```bash
sudo pacman -S ffmpeg opencv
```

4. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

5. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

6. **Install additional system packages**
```bash
# For OpenColorIO (LUT processing)
pip install OpenColorIO-Python

# For MediaPipe (if not installed automatically)
pip install mediapipe
```

## 🎯 Quick Start

### Basic Usage

```python
from src.core.video_processor import VideoProcessor
from src.core.color_grader import ColorGrader

# Initialize the processor
processor = VideoProcessor()
color_grader = ColorGrader()

# Process a video
input_video = "path/to/your/long_video.mp4"
output_dir = "output/"
lut_file = "luts/cinematic.cube"

# Generate shorts
shorts = processor.create_shorts(
    input_video=input_video,
    output_dir=output_dir,
    lut_file=lut_file,
    max_duration=60,  # seconds
    num_shorts=3
)

print(f"Generated {len(shorts)} shorts!")
```

### Command Line Interface

```bash
# Basic processing
python -m src.cli process --input video.mp4 --output ./shorts/ --lut luts/cinematic.cube

# Advanced options
python -m src.cli process \
    --input video.mp4 \
    --output ./shorts/ \
    --lut luts/vibrant.cube \
    --duration 45 \
    --count 5 \
    --quality high \
    --gpu
```

## 📁 Project Structure

```
shorts-creator/
├── src/
│   ├── core/                 # Core processing modules
│   │   ├── video_processor.py    # Main video processing
│   │   ├── scene_detector.py     # Scene detection algorithms
│   │   ├── color_grader.py       # LUT and color processing
│   │   └── audio_processor.py    # Audio analysis and processing
│   ├── algorithms/           # AI algorithms
│   │   ├── segmentation.py       # Video segmentation logic
│   │   ├── cropping.py           # Smart cropping algorithms
│   │   └── scoring.py            # Content scoring system
│   ├── utils/               # Utility functions
│   │   ├── lut_loader.py         # LUT file handling
│   │   ├── file_handler.py       # File I/O operations
│   │   └── config.py             # Configuration management
│   └── api/                 # Web API (optional)
│       ├── main.py               # FastAPI application
│       └── endpoints.py          # API endpoints
├── luts/                    # LUT files
│   ├── cinematic.cube
│   ├── vibrant.cube
│   └── vintage.cube
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
└── README.md
```

## 🎨 LUT (Color Grading) Support

The system supports industry-standard .cube LUT files for professional color grading:

### Supported LUT Formats
- **3D LUTs**: .cube format (most common)
- **Size**: 17x17x17, 33x33x33, 65x65x65 grids
- **Color Spaces**: Rec.709, sRGB, DCI-P3

### Using Custom LUTs
1. Place your .cube files in the `luts/` directory
2. Reference them in your processing pipeline:

```python
color_grader = ColorGrader()
graded_video = color_grader.apply_lut(
    video_path="input.mp4",
    lut_path="luts/your_custom_lut.cube",
    intensity=0.8  # 0.0 to 1.0
)
```

### Popular LUT Sources
- **Free**: RocketStock, Ground Control, IWLTBAP
- **Premium**: FilmConvert, Color Grading Central
- **Create Your Own**: DaVinci Resolve, Adobe Premiere Pro

## 🤖 Intelligent Segmentation Algorithm

The core algorithm analyzes videos using multiple factors:

### Scene Detection Methods
1. **Visual Analysis**
   - Histogram differences between frames
   - Motion vector analysis
   - Shot boundary detection
   - Object/face tracking continuity

2. **Audio Analysis**
   - Silence gap detection
   - Music beat synchronization
   - Speech pattern recognition
   - Audio energy levels

3. **Content Scoring**
   - Visual interest (contrast, motion, faces)
   - Audio engagement (music, speech clarity)
   - Action detection (sudden movements)
   - Emotional peak identification

### Algorithm Flow
```
Input Video → Frame Extraction → Scene Analysis → Content Scoring → Segment Selection → Output Shorts
```

## ⚙️ Configuration

Edit `config.yaml` to customize processing parameters:

```yaml
video:
  output_resolution: [1080, 1920]  # 9:16 aspect ratio
  fps: 30
  quality: "high"
  codec: "h264"

segmentation:
  min_duration: 15  # seconds
  max_duration: 60
  overlap_threshold: 0.3
  scene_threshold: 0.4

color_grading:
  default_lut: "luts/cinematic.cube"
  intensity: 0.8
  auto_exposure: true
  contrast_boost: 1.2

audio:
  normalize: true
  noise_reduction: true
  music_detection: true
  speech_enhancement: true
```

## 🚀 Performance Optimization

### GPU Acceleration
Enable GPU processing for 5-10x speed improvement:

```python
processor = VideoProcessor(use_gpu=True, gpu_device=0)
```

### Batch Processing
Process multiple videos efficiently:

```python
videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
processor.batch_process(videos, output_dir="batch_output/")
```

### Memory Management
For large videos, use streaming processing:

```python
processor = VideoProcessor(streaming_mode=True, chunk_size=1000)
```

## 📊 Quality Metrics

The system provides quality assessment:

- **Visual Quality**: PSNR, SSIM scores
- **Engagement Score**: Based on motion, faces, audio energy
- **Platform Compliance**: Aspect ratio, duration, file size validation
- **Processing Efficiency**: Time per minute of input video

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test category
pytest tests/test_color_grading.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

**FFmpeg not found**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

**CUDA out of memory**
```python
# Reduce batch size or disable GPU
processor = VideoProcessor(use_gpu=False)
```

**LUT file not loading**
- Ensure .cube file is properly formatted
- Check file permissions
- Verify color space compatibility

### Performance Tips

1. **Use SSD storage** for faster I/O
2. **Enable GPU acceleration** when available
3. **Adjust chunk_size** based on available RAM
4. **Use lower quality settings** for faster processing during development

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/shorts-creator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/shorts-creator/discussions)
- **Email**: support@shorts-creator.com

---

**Made with ❤️ for content creators**