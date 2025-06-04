# Shorts Creator 🎬

Automated video editing system for YouTube Shorts and TikTok, specifically optimized for **FPV drone footage**. Transform your long-form aerial videos into engaging 10-20 second short-form content with AI-powered scene detection and professional color grading.

## ✨ Features

### 🎯 Core Requirements (Implemented)
- **✅ LUT Color Grading**: Professional .cube file support with DJI Avata 2 LUT included
- **✅ Intelligent Scene Detection**: Ultra-fast algorithm optimized for FPV drone footage
- **✅ Batch Processing**: Process entire directories with one command
- **✅ Dynamic Duration**: 10-20 second segments based on content quality
- **✅ Intelligent Background Music**: AI-powered music analysis with beat detection and energy scoring

### 🚁 FPV Drone Optimizations
- **Comprehensive Scene Detection**: Multiple detection strategies (motion spikes, visual interest peaks, contrast changes)
- **Uniform Video Coverage**: Analyzes entire video to avoid missing good content
- **Content Quality Filtering**: Automatically skips boring segments (landings, static scenes)
- **Dynamic Duration Extension**: High-quality segments automatically extended up to 25 seconds
- **GPU Acceleration**: NVIDIA hardware encoding for lightning-fast processing

### 🎨 Professional Color Grading
- **DJI Avata 2 LUT**: Professional D-Log M to Rec.709 conversion included
- **3D LUT Support**: Load and apply any .cube LUT files with trilinear interpolation
- **Auto Corrections**: Automatic exposure, contrast, and saturation adjustments
- **Manual Controls**: Fine-tune highlights, shadows, and color temperature

### 🎬 Video Processing
- **Smart Cropping**: Automatic 9:16 aspect ratio conversion for vertical platforms
- **Multiple Platforms**: YouTube Shorts, TikTok, Instagram Reels optimized
- **Quality Presets**: High-quality encoding with GPU acceleration
- **Organized Output**: Separate folders for each video processed

### 🎵 Intelligent Music System
- **AI Music Analysis**: 6-metric energy scoring (RMS energy, spectral centroid, beat strength, etc.)
- **Smart Segment Selection**: Automatically finds the most exciting parts of songs (chorus/drops)
- **Beat Detection**: Tempo analysis optimized for 120-180 BPM action music
- **Caching System**: Never re-analyzes the same song twice
- **Multiple Formats**: MP3, WAV, M4A, AAC, OGG, FLAC support
- **Auto-Mixing**: 30% volume with fade in/out effects

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

### 4. Process Your Videos (Super Simple!)

```bash
# 1. Put your FPV videos in the input/ folder
mkdir input
cp your_fpv_videos.mp4 input/

# 2. Add background music (optional)
mkdir music
cp your_music_files.mp3 music/

# 3. Process all videos with default settings (DJI Avata 2 LUT, 8 segments per video, intelligent music)
python src/cli.py process

# That's it! Your shorts will be in output/ folder with professional color grading and background music
```

## 📖 Usage Guide

### Simple Batch Processing (Recommended)

```bash
# Process all videos in input/ directory with default LUT
python src/cli.py process

# Process with custom settings
python src/cli.py process --max-segments 5 --output-dir my_shorts
```

### Single Video Processing

```bash
# Process single video
python src/cli.py process video.mp4

# With custom LUT
python src/cli.py process video.mp4 --lut luts/my_custom.cube
```

### Directory Processing

```bash
# Process specific directory
python src/cli.py process --input-dir /path/to/videos --output-dir /path/to/output
```

### Analysis (Preview Before Processing)

```bash
# Analyze video to see potential segments
python src/cli.py analyze video.mp4 --detailed

# Shows:
# - 9 scenes detected (10-20 second segments)
# - Quality scores for each segment
# - Visual interest and motion activity metrics
```

### Configuration Management

```bash
# Check current settings
python src/cli.py config-check

# Validate LUT files
python src/cli.py validate-lut luts/avata2.cube

# System requirements check
python src/cli.py system-check
```

### Music Management

```bash
# Check music library status
python src/cli.py music-check

# Analyze music for best segments (fast overview)
python src/cli.py music-analyze --fast

# Analyze top 10 tracks for energy scoring
python src/cli.py music-analyze --limit 10

# Analyze all tracks (takes time but builds complete cache)
python src/cli.py music-analyze --limit 0

# Process videos without music
python src/cli.py process --no-music
```

## 🎨 Color Grading

### Default LUT (DJI Avata 2)

The system comes with a professional DJI Avata 2 D-Log M to Rec.709 LUT:
- **File**: `luts/avata2.cube`
- **Purpose**: Converts flat D-Log footage to vibrant Rec.709 color space
- **Automatically applied** when you run `python src/cli.py process`

### Using Custom LUTs

```bash
# Use your own LUT
python src/cli.py process --lut path/to/your.cube

# Disable LUT (no color grading)
python src/cli.py process --use-default-lut=false
```

### Creating Test LUTs

```bash
# Create sample LUTs for testing
python src/cli.py create-lut --style cinematic --name my_cinematic
python src/cli.py create-lut --style warm --name golden_hour
python src/cli.py create-lut --style cool --name arctic_blue
```

## ⚙️ Configuration

### Key Settings (config.yaml)

```yaml
# Scene Detection - Optimized for 10-20s segments
segmentation:
  min_duration: 8           # Minimum 8 seconds
  max_duration: 25          # Maximum 25 seconds
  target_duration: 15       # Target 15 seconds
  scene_threshold: 0.05     # Very sensitive detection
  frame_sample_rate: 1.0    # Analyze 1 frame per second

# Dynamic Duration
  dynamic_duration: true
  quality_extension_threshold: 0.4  # Extend high-quality segments
  max_extension_duration: 15        # Up to 15 extra seconds

# Color Grading
color_grading:
  default_lut: "luts/avata2.cube"   # DJI Avata 2 LUT
  lut_intensity: 0.8                # 80% LUT strength

# Video Output
video:
  output_resolution: [1080, 1920]   # 9:16 vertical
  fps: 30                           # 30fps output
  quality: "high"                   # High quality encoding

# Background Music
audio:
  enable_music: true                # Enable background music
  music_directory: "music/"         # Music files location
  music_volume: 0.3                 # 30% volume
  fade_in_duration: 1.0            # 1 second fade in
  fade_out_duration: 1.0           # 1 second fade out
  random_selection: true           # Random track per video
```

## 🔧 Algorithm Details

### Ultra-Fast Scene Detection

The system uses multiple detection strategies for comprehensive coverage:

1. **Traditional Scene Changes**: Histogram-based detection with very low thresholds
2. **Motion Spike Detection**: Catches action sequences and dynamic flight maneuvers
3. **Visual Interest Peaks**: Identifies visually compelling content (scenery, obstacles)
4. **Contrast Changes**: Detects lighting transitions and environment changes
5. **Forced Intervals**: Ensures coverage every 25 seconds for long videos

### Content Quality Filtering

Automatically skips boring content:
- **Low Motion Segments**: Landings, hovering, static scenes
- **Poor Visual Interest**: Flat, monotonous footage
- **Low Contrast**: Overexposed or underexposed segments

### Dynamic Duration System

- **Base Duration**: 8-25 seconds based on content
- **Quality Extension**: High-scoring segments extended automatically
- **Smart Boundaries**: Respects natural scene transitions

### Performance Optimizations

- **Uniform Sampling**: Analyzes entire video (1 fps) to avoid missing content
- **GPU Processing**: NVIDIA hardware encoding for 10x faster output
- **Downsampled Analysis**: Processes at 1/4 resolution for speed
- **Batch Processing**: Handles multiple videos efficiently

### Intelligent Music Analysis

The AI music system analyzes tracks using 6 key metrics:

1. **RMS Energy**: Overall loudness and power (25% weight)
2. **Spectral Centroid**: Brightness and excitement (20% weight)
3. **Spectral Rolloff**: High-frequency energy distribution (15% weight)
4. **Zero Crossing Rate**: Percussive and rhythmic content (10% weight)
5. **Beat Strength**: Tempo and beat consistency (20% weight)
6. **Onset Strength**: Musical events and transitions (10% weight)

**Tempo Optimization**: Bonus scoring for 120-180 BPM (ideal for FPV action)
**Smart Caching**: Results cached with file modification tracking
**Segment Selection**: Finds chorus/drops instead of just using song beginnings

## 📊 Example Output

```bash
$ python src/cli.py process

🎬 Shorts Creator - Auto Batch Processing
📁 Input Directory: input/
==================================================
📂 Output: output
🎨 LUT: luts/avata2.cube
🔢 Max segments per video: 8
🎥 Videos to process: 1
==================================================

🎬 Processing Video 1/1: auto-test.mp4
============================================================
📊 Video Info: 3840x2160, 207.3s, 59.9fps, 12,425 frames
🎯 Target: 8 shorts, LUT: luts/avata2.cube

🎬 Stage 1: Scene Detection (47s)
✅ Found 9 scenes (10-20s segments)

🎯 Stage 2: Selecting Best Segments
✅ Selected 8 segments for processing

🎥 Stage 3: Processing 8 Segments
✅ Created: short_01_140s.mp4 (12.7s, 3.7MB)
✅ Created: short_02_67s.mp4 (11.8s, 3.4MB)
✅ Created: short_03_90s.mp4 (15.7s, 5.0MB)
... (5 more shorts)

🎉 Batch Processing Complete!
✅ Successfully processed 1/1 videos
📊 Created 8 total shorts (28.4MB)
📂 All files saved to: output/
🎨 Applied LUT: luts/avata2.cube
```

## 🎯 Platform Optimization

### YouTube Shorts
- **Duration**: 10-20 seconds (optimal for engagement)
- **Resolution**: 1080x1920 (9:16)
- **Quality**: High bitrate for crisp aerial footage

### TikTok
- **Duration**: 10-15 seconds (platform sweet spot)
- **Resolution**: 1080x1920 (9:16)
- **Quality**: Optimized for mobile viewing

### Instagram Reels
- **Duration**: 15-20 seconds (algorithm preference)
- **Resolution**: 1080x1920 (9:16)
- **Quality**: High quality for discovery

## 🚀 Performance

### Speed Benchmarks
- **Analysis**: 47 seconds for 207-second 4K video (4.4x real-time)
- **Processing**: GPU-accelerated, ~2-3 seconds per output segment
- **Total**: ~2-3 minutes for 8 high-quality shorts from 3.5-minute source

### System Requirements
- **Python**: 3.9+
- **GPU**: NVIDIA GPU recommended (10x faster processing)
- **RAM**: 8GB+ recommended for 4K footage
- **Storage**: ~50MB per minute of output video

## 🛠️ Troubleshooting

### Common Issues

**No videos found in input/ directory**
```bash
# Make sure videos are in input/ folder
ls input/
# Supported formats: .mp4, .mov, .avi, .mkv, .webm, .m4v, .flv
```

**LUT file not found**
```bash
# Check LUT exists
ls luts/avata2.cube
# Validate LUT file
python src/cli.py validate-lut luts/avata2.cube
```

**Slow processing (no GPU)**
```bash
# Check GPU availability
python src/cli.py system-check
# Look for "✅ CUDA available" message
```

**No scenes detected**
```bash
# Lower detection thresholds in config.yaml
segmentation:
  scene_threshold: 0.03     # Lower = more sensitive
  motion_threshold: 0.005   # Lower = more sensitive
```

**No music or audio issues**
```bash
# Check music library
python src/cli.py music-check

# Verify music files are supported formats
ls music/
# Supported: .mp3, .wav, .m4a, .aac, .ogg, .flac

# Test music analysis
python src/cli.py music-analyze --limit 5

# Disable music if having issues
python src/cli.py process --no-music
```

**Music analysis too slow**
```bash
# Use fast mode for library overview
python src/cli.py music-analyze --fast

# Analyze only top tracks
python src/cli.py music-analyze --limit 10

# Results are cached - second run is instant!
```

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`