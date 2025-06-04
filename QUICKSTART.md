# 🚀 Quick Start Guide

Get up and running with Shorts Creator in minutes!

## 📋 Prerequisites

- **Python 3.8+**
- **FFmpeg** (for video processing)
- **8GB+ RAM** (recommended)
- **GPU with CUDA** (optional, for acceleration)

## ⚡ One-Command Setup

```bash
# Clone and setup everything automatically
git clone https://github.com/yourusername/shorts-creator.git
cd shorts-creator
chmod +x setup.sh
./setup.sh
```

The setup script will:
- ✅ Install UV package manager
- ✅ Install system dependencies
- ✅ Create virtual environment
- ✅ Install Python packages
- ✅ Set up project structure

## 🎯 First Run

1. **Activate the environment:**
```bash
source .venv/bin/activate
```

2. **Check system status:**
```bash
shorts-creator system-check
```

3. **Add a LUT file** (for color grading):
```bash
# Download a free LUT or use your own
cp your-lut-file.cube luts/
```

4. **Process your first video:**
```bash
shorts-creator process \
    --input your-video.mp4 \
    --output ./output/ \
    --lut luts/your-lut.cube \
    --count 3
```

## 🎨 Core Features

### **Must-Have: LUT Color Grading**
- Professional .cube LUT file support
- 3D color interpolation
- Automatic exposure/contrast adjustment
- Multiple LUT presets

### **Must-Have: Intelligent Segmentation**
- AI-powered scene detection
- Motion and audio analysis
- Face/object tracking
- Content scoring algorithm

### **Additional Features**
- 9:16 aspect ratio optimization
- GPU acceleration
- Batch processing
- Multiple platform support

## 📁 Project Structure

```
shorts-creator/
├── src/                    # Core modules (to be implemented)
├── luts/                   # Your LUT files go here
├── config.yaml            # Configuration settings
├── pyproject.toml         # UV dependencies
└── setup.sh               # Automated setup
```

## 🔧 UV Commands

```bash
# Add new dependency
uv add package-name

# Install with optional features
uv pip install -e ".[cuda,api,dev]"

# Update dependencies
uv lock --upgrade

# Sync environment
uv sync
```

## 🎬 Algorithm Overview

### Scene Segmentation
1. **Frame Analysis** → Extract frames at 1fps
2. **Histogram Diff** → Detect scene boundaries
3. **Audio Analysis** → Find silence gaps and beats
4. **Content Scoring** → Rate visual interest and motion
5. **Segment Selection** → Choose best clips for shorts

### Color Grading Pipeline
1. **LUT Loading** → Parse .cube file
2. **Color Space** → Convert to working space
3. **3D Interpolation** → Apply color mapping
4. **Auto Adjust** → Exposure and contrast
5. **Output** → Convert to target format

## 🚧 Development Status

This is currently a **requirements and setup framework**. Core modules are under development:

- ✅ Project structure and dependencies
- ✅ Configuration system
- ✅ CLI interface (placeholder)
- 🚧 Video processing engine
- 🚧 Scene detection algorithms
- 🚧 LUT color grading system
- 🚧 Audio analysis
- 🚧 Smart cropping

## 🆘 Troubleshooting

**UV not found?**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

**FFmpeg missing?**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# Arch/Manjaro
sudo pacman -S ffmpeg
```

**Dependencies failing?**
```bash
# Try manual installation
uv pip install -e ".[dev]"
```

## 📚 Next Steps

1. **Read the full [README.md](README.md)** for detailed documentation
2. **Check [REQUIREMENTS.md](REQUIREMENTS.md)** for technical specifications
3. **Explore [config.yaml](config.yaml)** for customization options
4. **Join development** by implementing core modules

## 🎯 Example Workflow

```bash
# 1. Setup (one time)
./setup.sh

# 2. Activate environment
source .venv/bin/activate

# 3. Check system
shorts-creator system-check

# 4. Process videos
shorts-creator process -i long-video.mp4 -o shorts/ -n 5

# 5. Analyze results
shorts-creator analyze -i long-video.mp4
```

---

**Ready to create amazing shorts? Let's go! 🎬✨**