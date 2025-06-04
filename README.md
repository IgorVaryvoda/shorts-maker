# Shorts Creator 🎬

Automated video editing system for YouTube Shorts and TikTok, specifically optimized for **FPV drone footage**. Transform your long-form aerial videos into engaging short-form content with AI-powered scene detection and professional color grading.

## ✨ Features

### 🎯 Core Requirements (Implemented)
- **✅ LUT Color Grading**: Professional .cube file support with 3D interpolation
- **✅ Intelligent Scene Detection**: AI-powered algorithm optimized for FPV drone footage

### 🚁 FPV Drone Optimizations
- **Motion-Based Scene Detection**: Analyzes optical flow and camera movement
- **Visual Interest Scoring**: Edge detection, contrast analysis, and texture complexity
- **Automatic Cropping**: Smart 9:16 aspect ratio conversion for vertical platforms
- **No Audio Processing**: Optimized for FPV footage (audio removed from dependencies)

### 🎨 Professional Color Grading
- **3D LUT Support**: Load and apply .cube LUT files with trilinear interpolation
- **Auto Corrections**: Automatic exposure, contrast, and saturation adjustments
- **Manual Controls**: Fine-tune highlights, shadows, and color temperature
- **Sample LUTs**: Built-in cinematic, warm, and cool color grades

### 🎬 Video Processing
- **Multiple Platforms**: YouTube Shorts, TikTok, Instagram Reels
- **Quality Presets**: Low, Medium, High, Ultra encoding options
- **GPU Acceleration**: NVIDIA CUDA support for faster processing
- **Batch Processing**: Generate multiple shorts from one video

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd shorts-creator

# Run the setup script (handles everything automatically)
chmod +x setup.sh
./setup.sh
```

### 2. Activate Environment

```bash
# For bash/zsh
source .venv/bin/activate

# For fish shell
source .venv/bin/activate.fish
```

### 3. System Check

```bash
python src/cli.py system-check
```

### 4. Process Your First Video

```bash
python src/cli.py process \
  --input your_fpv_video.mp4 \
  --output ./shorts \
  --lut luts/cinematic.cube \
  --duration 30 \
  --count 3 \
  --platform youtube
```

## 📖 Usage Guide

### Basic Commands

#### Process Video
```bash
# Basic processing
python src/cli.py process -i video.mp4 -o ./output

# With custom LUT and settings
python src/cli.py process \
  -i drone_footage.mp4 \
  -o ./shorts \
  --lut "luts/DJI Avata 2 D-Log M to Rec.709 V1._.cube" \
  --duration 45 \
  --count 5 \
  --quality ultra \
  --platform tiktok
```

#### Analyze Video (No Processing)
```bash
# Quick analysis
python src/cli.py analyze -i video.mp4

# Save analysis to file
python src/cli.py analyze -i video.mp4 -o analysis.json
```

#### LUT Management
```bash
# Validate LUT file
python src/cli.py validate-lut --lut luts/my_lut.cube

# Create sample LUT for testing
python src/cli.py create-lut --type cinematic --output luts/test.cube
```

#### Configuration
```bash
# Check current config
python src/cli.py config-check

# Check system requirements
python src/cli.py system-check
```

### Advanced Usage

#### Custom Configuration
Edit `config.yaml` to customize processing parameters:

```yaml
segmentation:
  scene_threshold: 0.4      # Scene change sensitivity
  motion_threshold: 0.3     # Motion detection sensitivity
  min_duration: 15          # Minimum clip length
  max_duration: 60          # Maximum clip length

color_grading:
  lut_intensity: 0.8        # LUT application strength
  auto_exposure: true       # Automatic exposure correction
  auto_contrast: true       # Automatic contrast enhancement

video:
  output_resolution: [1080, 1920]  # 9:16 aspect ratio
  fps: 30                   # Output frame rate
  quality: "high"           # Encoding quality
```

#### Platform-Specific Settings
```bash
# YouTube Shorts (up to 60s)
python src/cli.py process -i video.mp4 -o ./youtube --duration 60 --platform youtube

# TikTok (15-60s, optimized for 15s)
python src/cli.py process -i video.mp4 -o ./tiktok --duration 15 --platform tiktok

# Instagram Reels (up to 90s)
python src/cli.py process -i video.mp4 -o ./instagram --duration 30 --platform instagram
```

## 🎨 Color Grading

### Using LUT Files

The system supports professional .cube LUT files:

```bash
# Use your own LUT
python src/cli.py process -i video.mp4 -o ./output --lut path/to/your.cube

# Use built-in LUTs
python src/cli.py process -i video.mp4 -o ./output --lut luts/cinematic.cube
```

### Available Sample LUTs
- `cinematic.cube` - Film-like color grading
- `warm.cube` - Warm, golden hour tones
- `cool.cube` - Cool, blue-tinted look
- `DJI Avata 2 D-Log M to Rec.709 V1._.cube` - Professional DJI conversion

### Creating Custom LUTs
```bash
# Create test LUTs
python src/cli.py create-lut --type warm --output luts/my_warm.cube
python src/cli.py create-lut --type cool --output luts/my_cool.cube
python src/cli.py create-lut --type cinematic --output luts/my_cinematic.cube
```

## 🔧 Algorithm Details

### Scene Detection for FPV Footage

The system uses a multi-factor approach optimized for drone footage:

1. **Histogram Analysis**: Detects scene changes using HSV color histograms
2. **Optical Flow**: Analyzes camera movement and motion vectors
3. **Visual Interest Scoring**:
   - Edge density (terrain features, obstacles)
   - Contrast levels (dramatic lighting)
   - Color variance (diverse scenery)
   - Texture complexity (detail richness)

### Scoring Weights (FPV Optimized)
- Visual Interest: 60% (higher for scenic drone footage)
- Motion Activity: 40% (important for dynamic flight)
- Face Detection: 0% (not relevant for FPV)

### Color Grading Pipeline
1. **Auto Corrections**: Exposure, contrast, saturation
2. **LUT Application**: 3D color transformation with trilinear interpolation
3. **Manual Adjustments**: Highlights, shadows, fine-tuning

## 📊 Performance

### System Requirements
- **Python**: 3.9+
- **FFmpeg**: Latest version
- **GPU**: NVIDIA GPU recommended (CUDA support)
- **RAM**: 8GB+ recommended
- **Storage**: SSD recommended for faster processing

### Processing Speed
- **CPU-only**: ~2-5 minutes per 30s short
- **GPU-accelerated**: ~1-3 minutes per 30s short
- **Factors**: Input resolution, quality settings, LUT complexity

### Optimization Tips
```bash
# Faster processing (lower quality)
python src/cli.py process -i video.mp4 -o ./output --quality medium --no-gpu

# Maximum quality (slower)
python src/cli.py process -i video.mp4 -o ./output --quality ultra --gpu
```

## 🛠️ Development

### Project Structure
```
shorts-creator/
├── src/
│   ├── core/                 # Core processing modules
│   │   ├── scene_detector.py # FPV scene detection
│   │   ├── color_grader.py   # LUT color grading
│   │   └── video_processor.py # Main orchestrator
│   ├── utils/                # Utility modules
│   │   ├── config.py         # Configuration management
│   │   └── lut_loader.py     # LUT file handling
│   └── cli.py               # Command-line interface
├── luts/                    # LUT files directory
├── config.yaml             # Configuration file
└── pyproject.toml          # UV dependencies
```

### Adding New Features
1. **Scene Detection**: Modify `src/core/scene_detector.py`
2. **Color Grading**: Extend `src/core/color_grader.py`
3. **Video Processing**: Update `src/core/video_processor.py`
4. **CLI Commands**: Add to `src/cli.py`

### Testing
```bash
# Test with sample video
python src/cli.py analyze -i sample_video.mp4

# Validate LUT files
python src/cli.py validate-lut --lut luts/cinematic.cube

# System diagnostics
python src/cli.py system-check
```

## 🎯 FPV-Specific Tips

### Best Input Videos
- **Resolution**: 1080p or higher
- **Frame Rate**: 30fps or 60fps
- **Duration**: 2+ minutes for good scene variety
- **Content**: Varied flight patterns, different environments

### Optimal Settings for FPV
```bash
# Scenic flights (mountains, coastlines)
python src/cli.py process -i scenic_flight.mp4 -o ./output \
  --lut luts/cinematic.cube --duration 45 --quality high

# Action flights (racing, freestyle)
python src/cli.py process -i action_flight.mp4 -o ./output \
  --duration 15 --quality ultra --platform tiktok

# Sunrise/sunset flights
python src/cli.py process -i golden_hour.mp4 -o ./output \
  --lut luts/warm.cube --duration 30
```

### Scene Selection Tips
- The algorithm prioritizes high-motion, visually interesting segments
- Smooth camera movements score higher than shaky footage
- Varied terrain and lighting changes improve scene detection
- Avoid long segments of similar scenery

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

### Common Issues

**Import Errors**: Make sure you're in the virtual environment
```bash
source .venv/bin/activate.fish  # or .venv/bin/activate
```

**FFmpeg Not Found**: Install FFmpeg for your system
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# Arch/Manjaro
sudo pacman -S ffmpeg

# macOS
brew install ffmpeg
```

**GPU Not Detected**: Install CUDA drivers and PyTorch with CUDA support
```bash
uv add torch[cuda] torchvision[cuda]
```

**LUT Files Not Working**: Validate your LUT files
```bash
python src/cli.py validate-lut --lut your_file.cube
```

### Getting Help
- Check system requirements: `python src/cli.py system-check`
- Validate configuration: `python src/cli.py config-check`
- Use verbose mode: `python src/cli.py process -v -i video.mp4 -o output`

---

**Ready to transform your FPV footage into viral shorts!** 🚁✨