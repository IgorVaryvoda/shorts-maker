#!/bin/bash

# Shorts Creator Setup Script
# Automated installation for Linux systems using UV

set -e  # Exit on any error

echo "🎬 Shorts Creator Setup Script"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS
detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    elif type lsb_release >/dev/null 2>&1; then
        OS=$(lsb_release -si)
        VER=$(lsb_release -sr)
    else
        OS=$(uname -s)
        VER=$(uname -r)
    fi

    print_status "Detected OS: $OS $VER"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root"
        exit 1
    fi
}

# Install UV
install_uv() {
    print_status "Installing UV package manager..."

    if command -v uv &> /dev/null; then
        print_status "UV is already installed"
        UV_VERSION=$(uv --version)
        print_status "UV version: $UV_VERSION"
    else
        print_status "Downloading and installing UV..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add UV to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"

        # Verify installation
        if command -v uv &> /dev/null; then
            UV_VERSION=$(uv --version)
            print_success "UV installed successfully: $UV_VERSION"
        else
            print_error "UV installation failed"
            exit 1
        fi
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."

    case "$OS" in
        *"Ubuntu"*|*"Debian"*)
            sudo apt update
            sudo apt install -y \
                python3 python3-dev \
                ffmpeg \
                libsm6 libxext6 libxrender-dev libglib2.0-0 \
                libgl1-mesa-glx libfontconfig1 libice6 \
                build-essential cmake pkg-config \
                libjpeg-dev libtiff5-dev libpng-dev \
                libavcodec-dev libavformat-dev libswscale-dev \
                libgtk2.0-dev libcanberra-gtk-module \
                libxvidcore-dev libx264-dev libgtk-3-dev \
                libtbb2 libtbb-dev libdc1394-22-dev \
                libv4l-dev v4l-utils \
                libopenblas-dev libatlas-base-dev liblapack-dev gfortran \
                libhdf5-dev \
                curl
            ;;
        *"Arch"*|*"Manjaro"*)
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm \
                python python-pip \
                ffmpeg \
                opencv \
                gtk3 \
                base-devel cmake pkgconf \
                libjpeg-turbo libtiff libpng \
                hdf5 \
                openblas lapack \
                curl
            ;;
        *"Fedora"*|*"CentOS"*|*"Red Hat"*)
            sudo dnf update -y
            sudo dnf install -y \
                python3 python3-devel \
                ffmpeg ffmpeg-devel \
                opencv opencv-devel \
                gtk3-devel \
                gcc gcc-c++ cmake pkgconfig \
                libjpeg-turbo-devel libtiff-devel libpng-devel \
                hdf5-devel \
                openblas-devel lapack-devel \
                curl
            ;;
        *)
            print_warning "Unsupported OS. Please install dependencies manually:"
            print_warning "- Python 3.8+"
            print_warning "- FFmpeg"
            print_warning "- OpenCV development libraries"
            print_warning "- Build tools (gcc, cmake, pkg-config)"
            print_warning "- curl"
            ;;
    esac

    print_success "System dependencies installed"
}

# Check Python version
check_python() {
    print_status "Checking Python version..."

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_status "Python version: $PYTHON_VERSION"

        # Check if version is >= 3.9
        if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)'; then
            print_success "Python version is compatible"
        else
            print_error "Python 3.9+ is required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
}

# Create virtual environment with UV
create_venv() {
    print_status "Creating virtual environment with UV..."

    if [[ -d ".venv" ]]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf .venv
    fi

    # Create virtual environment with UV
    uv venv

    print_success "Virtual environment created with UV"
}

# Install Python dependencies with UV
install_python_deps() {
    print_status "Installing Python dependencies with UV..."

    # Install base dependencies
    print_status "Installing core dependencies..."
    uv pip install -e .

    # Install optional dependencies based on system capabilities
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected. Installing CUDA support..."
        uv pip install -e ".[cuda]"
    else
        print_status "No NVIDIA GPU detected. Skipping CUDA dependencies..."
    fi

    # Ask user if they want API dependencies
    read -p "Install web API dependencies? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Installing API dependencies..."
        uv pip install -e ".[api]"
    fi

    # Ask user if they want development dependencies
    read -p "Install development dependencies? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Installing development dependencies..."
        uv pip install -e ".[dev]"
    fi

    print_success "Python dependencies installed with UV"
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."

    mkdir -p src/{core,algorithms,utils,api}
    mkdir -p luts
    mkdir -p tests
    mkdir -p docs
    mkdir -p logs
    mkdir -p .cache

    # Create __init__.py files
    touch src/__init__.py
    touch src/core/__init__.py
    touch src/algorithms/__init__.py
    touch src/utils/__init__.py
    touch src/api/__init__.py

    print_success "Directory structure created"
}

# Download sample LUT files
download_sample_luts() {
    print_status "Setting up LUT directory..."

    cd luts

    # Create placeholder LUT files for testing
    cat > README.md << 'EOF'
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
EOF

    cd ..

    print_success "LUT directory prepared"
}

# Create UV lock file
create_lock_file() {
    print_status "Creating UV lock file..."

    # Generate uv.lock file
    uv lock

    print_success "UV lock file created"
}

# Run tests
run_tests() {
    print_status "Running basic tests..."

    # Activate virtual environment
    source .venv/bin/activate

    # Test Python imports
    python3 -c "
try:
    import cv2
    print('✓ OpenCV imported successfully')
except ImportError as e:
    print(f'✗ OpenCV import failed: {e}')

try:
    import numpy as np
    print('✓ NumPy imported successfully')
except ImportError as e:
    print(f'✗ NumPy import failed: {e}')

try:
    import torch
    print('✓ PyTorch imported successfully')
    if torch.cuda.is_available():
        print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        print('ℹ CUDA not available (CPU-only mode)')
except ImportError as e:
    print(f'✗ PyTorch import failed: {e}')

try:
    import librosa
    print('✓ Librosa imported successfully')
except ImportError as e:
    print(f'✗ Librosa import failed: {e}')
"

    # Test FFmpeg
    if command -v ffmpeg &> /dev/null; then
        print_success "FFmpeg is available"
        ffmpeg -version | head -1
    else
        print_error "FFmpeg is not available"
        exit 1
    fi

    # Test UV
    if command -v uv &> /dev/null; then
        print_success "UV is available"
        uv --version
    else
        print_error "UV is not available"
        exit 1
    fi

    print_success "Basic tests passed"
}

# Create development setup
setup_development() {
    print_status "Setting up development environment..."

    # Create pre-commit config if dev dependencies are installed
    if uv pip list | grep -q "pre-commit"; then
        cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.280
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.0
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML, types-requests]
EOF

        # Install pre-commit hooks
        source .venv/bin/activate
        pre-commit install
        print_success "Pre-commit hooks installed"
    fi

    print_success "Development environment setup complete"
}

# Main installation function
main() {
    echo
    print_status "Starting Shorts Creator installation with UV..."
    echo

    check_root
    detect_os
    check_python
    install_uv
    install_system_deps
    create_venv
    install_python_deps
    create_directories
    download_sample_luts
    create_lock_file
    setup_development
    run_tests

    echo
    print_success "🎉 Installation completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Activate the virtual environment: source .venv/bin/activate"
    echo "2. Add your LUT files to the luts/ directory"
    echo "3. Configure settings in config.yaml"
    echo "4. Run your first video processing job"
    echo
    echo "UV Commands:"
    echo "- Install new package: uv add <package>"
    echo "- Remove package: uv remove <package>"
    echo "- Update dependencies: uv lock --upgrade"
    echo "- Sync environment: uv sync"
    echo
    echo "For usage examples, see README.md"
    echo
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Shorts Creator Setup Script (UV Edition)"
        echo
        echo "Usage: $0 [options]"
        echo
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --deps-only    Install only system dependencies"
        echo "  --uv-only      Install only UV and Python dependencies"
        echo
        exit 0
        ;;
    --deps-only)
        detect_os
        install_system_deps
        ;;
    --uv-only)
        check_python
        install_uv
        create_venv
        install_python_deps
        ;;
    *)
        main
        ;;
esac