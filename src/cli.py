#!/usr/bin/env python3
"""
Shorts Creator CLI

Command-line interface for the automated video editing system.
"""

import click
import os
import sys
import logging
from pathlib import Path
from typing import Optional, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.core.video_processor import VideoProcessor
    from src.core.scene_detector import FPVSceneDetector
    from src.core.color_grader import ColorGrader
    from src.utils.config import Config
    from src.utils.lut_loader import LUT3D, LUTLoader, create_sample_lut
    from src.utils.progress import ProgressBar
except ImportError:
    # Fallback for development
    from core.video_processor import VideoProcessor
    from core.scene_detector import FPVSceneDetector
    from core.color_grader import ColorGrader
    from utils.config import Config
    from utils.lut_loader import LUT3D, LUTLoader, create_sample_lut
    from utils.progress import ProgressBar

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/shorts_creator.log'),
            logging.StreamHandler()
        ]
    )


@click.group()
@click.version_option(version="0.1.0")
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool):
    """
    Shorts Creator - Automated Video Editor for YouTube Shorts and TikTok

    Transform long-form FPV drone videos into engaging short-form content with AI-powered
    scene detection and professional color grading.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config or "config.yaml"
    ctx.obj["verbose"] = verbose

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        click.echo("🎬 Shorts Creator CLI v0.1.0")

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    setup_logging(verbose)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True), required=False)
@click.option('--input-dir', '-i', type=click.Path(exists=True), help='Input directory for batch processing')
@click.option('--output-dir', '-o', default='output', help='Output directory')
@click.option('--lut', '-l', help='Path to LUT file (.cube) - uses default if not specified')
@click.option('--max-segments', '-n', default=8, help='Maximum number of segments to create per video')
@click.option('--use-default-lut', is_flag=True, default=True, help='Use default LUT from config (enabled by default)')
@click.pass_context
def process(ctx, input_path, input_dir, output_dir, lut, max_segments, use_default_lut):
    """🎬 Process video(s) to create shorts with full progress tracking.

    Usage:
    - shorts-creator process video.mp4          # Process single video
    - shorts-creator process --input-dir ./     # Process all videos in directory
    - shorts-creator process                    # Process all videos in current directory with default LUT
    """

    try:
        # Load config to get default LUT
        config = Config(ctx.obj['config_path'])
        default_lut = config.get('color_grading.default_lut')

        # Determine LUT to use
        lut_to_use = None
        if lut:
            lut_to_use = lut
        elif use_default_lut and default_lut:
            lut_to_use = default_lut

        # Determine input files
        input_files = []

        if input_path:
            # Single file processing
            input_files = [input_path]
            print("🎬 Shorts Creator - Single Video Processing")
        elif input_dir:
            # Batch processing from specified directory
            input_files = _get_video_files(input_dir)
            print("🎬 Shorts Creator - Batch Processing")
            print(f"📁 Input Directory: {input_dir}")
        else:
            # Default: process all videos in current directory
            input_files = _get_video_files('.')
            print("🎬 Shorts Creator - Auto Batch Processing")
            print("📁 Input Directory: Current directory")

        if not input_files:
            click.echo("❌ No video files found to process", err=True)
            return

        print("=" * 50)
        print(f"📂 Output: {output_dir}")
        print(f"🎨 LUT: {lut_to_use or 'None'}")
        print(f"🔢 Max segments per video: {max_segments}")
        print(f"🎥 Videos to process: {len(input_files)}")
        print("=" * 50)

        # Validate LUT if provided
        if lut_to_use and not os.path.exists(lut_to_use):
            click.echo(f"❌ LUT file not found: {lut_to_use}", err=True)
            return

        # Initialize processor
        processor = VideoProcessor(ctx.obj['config_path'])

        # Process each video
        total_output_files = []
        successful_videos = 0

        for i, video_file in enumerate(input_files, 1):
            print(f"\n🎬 Processing Video {i}/{len(input_files)}: {os.path.basename(video_file)}")
            print("=" * 60)

            try:
                # Create video-specific output directory
                video_name = Path(video_file).stem
                video_output_dir = os.path.join(output_dir, video_name)

                # Process video with progress tracking
                output_files = processor.process_video(
                    input_path=video_file,
                    output_dir=video_output_dir,
                    lut_path=lut_to_use,
                    max_segments=max_segments
                )

                if output_files:
                    total_output_files.extend(output_files)
                    successful_videos += 1

                    print(f"\n✅ {os.path.basename(video_file)} - Created {len(output_files)} shorts")
                    for j, file_path in enumerate(output_files, 1):
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                        print(f"  {j}. {os.path.basename(file_path)} ({file_size:.1f}MB)")
                else:
                    print(f"\n⚠️  {os.path.basename(video_file)} - No suitable segments found")

            except Exception as e:
                print(f"\n❌ Failed to process {os.path.basename(video_file)}: {e}")
                if ctx.obj['verbose']:
                    import traceback
                    traceback.print_exc()
                continue

        # Display final results
        print("\n" + "=" * 60)
        print("🎉 Batch Processing Complete!")
        print("=" * 60)

        if total_output_files:
            total_size = sum(os.path.getsize(f) for f in total_output_files) / (1024 * 1024)
            print(f"✅ Successfully processed {successful_videos}/{len(input_files)} videos")
            print(f"📊 Created {len(total_output_files)} total shorts ({total_size:.1f}MB)")
            print(f"📂 All files saved to: {output_dir}")

            if lut_to_use:
                print(f"🎨 Applied LUT: {lut_to_use}")
        else:
            print("⚠️  No shorts were created from any videos")

    except Exception as e:
        click.echo(f"❌ Processing failed: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _get_video_files(directory: str) -> List[str]:
    """Get all video files from a directory."""
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.flv'}
    video_files = []

    directory_path = Path(directory)
    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(str(file_path))

    return sorted(video_files)


@cli.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.option('--detailed', '-d', is_flag=True, help='Show detailed scene information')
@click.pass_context
def analyze(ctx, input_video, detailed):
    """🔍 Analyze video content and show potential segments with progress tracking."""

    print("🔍 Video Analysis")
    print("=" * 40)
    print(f"📁 Input: {input_video}")
    print("=" * 40)

    try:
        # Initialize processor
        processor = VideoProcessor(ctx.obj['config_path'])

        # Analyze video with progress tracking
        analysis = processor.analyze_video(input_video)

        # Display results
        video_info = analysis['video_info']

        print("\n📊 Video Information")
        print("-" * 30)
        print(f"Duration: {video_info['duration']:.1f}s")
        print(f"Resolution: {video_info['resolution'][0]}x{video_info['resolution'][1]}")
        print(f"FPS: {video_info['fps']:.1f}")
        print(f"File size: {video_info['file_size_mb']:.1f}MB")
        print(f"Total frames: {video_info['frame_count']:,}")

        print(f"\n🎬 Scene Analysis")
        print("-" * 30)
        print(f"Total scenes detected: {analysis['total_scenes']}")
        print(f"Valid scenes (15-60s): {analysis['valid_scenes']}")

        if analysis['best_scenes']:
            print(f"\n⭐ Top {len(analysis['best_scenes'])} Scenes")
            print("-" * 30)

            for i, scene in enumerate(analysis['best_scenes'], 1):
                duration = scene['duration']
                score = scene['score']
                start_time = scene['start_time']

                print(f"{i}. {start_time:.1f}s - {start_time + duration:.1f}s "
                      f"({duration:.1f}s) Score: {score:.3f}")

                if detailed:
                    metrics = scene.get('metrics', {})
                    print(f"   Visual Interest: {metrics.get('visual_interest', 0):.3f}")
                    print(f"   Motion Activity: {metrics.get('motion_magnitude', 0):.3f}")
                    print(f"   Edge Density: {metrics.get('edge_density', 0):.3f}")
                    print(f"   Contrast: {metrics.get('contrast', 0):.3f}")
        else:
            print("\n⚠️  No suitable scenes found")

    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('lut_file', type=click.Path(exists=True))
@click.option('--test-image', help='Test LUT on a specific image')
@click.pass_context
def validate_lut(ctx, lut_file, test_image):
    """✅ Validate a LUT file and show its properties."""

    print("✅ LUT Validation")
    print("=" * 30)
    print(f"📁 LUT File: {lut_file}")

    # Show progress for LUT loading
    progress = ProgressBar(100, width=30, show_eta=False)

    try:
        progress.update(20, "Loading LUT file...")

        # Load and validate LUT
        lut = LUTLoader.load_cube(lut_file)

        progress.update(60, "Validating LUT structure...")

        print(f"\n📊 LUT Properties")
        print("-" * 25)
        print(f"Size: {lut.size}x{lut.size}x{lut.size}")
        print(f"Title: {lut.title or 'Untitled'}")
        print(f"Data points: {lut.data.size:,}")

        progress.update(80, "Checking data integrity...")

        # Validate data ranges
        min_val = lut.data.min()
        max_val = lut.data.max()

        print(f"\n🔍 Data Analysis")
        print("-" * 20)
        print(f"Value range: [{min_val:.3f}, {max_val:.3f}]")

        if min_val < 0 or max_val > 1:
            print("⚠️  Warning: Values outside [0,1] range detected")
        else:
            print("✅ All values within valid range")

        progress.update(100, "Validation complete")

        print(f"\n✅ LUT file is valid and ready to use!")

    except Exception as e:
        progress.update(100, "Validation failed")
        click.echo(f"\n❌ LUT validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def config_check(ctx):
    """⚙️ Check configuration and show current settings."""

    print("⚙️ Configuration Check")
    print("=" * 35)

    try:
        config = Config(ctx.obj['config_path'])

        print("📋 Current Settings")
        print("-" * 25)
        print(f"Output resolution: {config.get('video.output_resolution')}")
        print(f"Target FPS: {config.get('video.fps')}")
        print(f"Quality: {config.get('video.quality')}")
        print(f"Min duration: {config.get('segmentation.min_duration')}s")
        print(f"Max duration: {config.get('segmentation.max_duration')}s")
        print(f"Scene threshold: {config.get('segmentation.scene_threshold')}")
        print(f"Frame sample rate: {config.get('segmentation.frame_sample_rate')}")

        print(f"\n🎨 Color Grading")
        print("-" * 20)
        print(f"Default LUT: {config.get('color_grading.default_lut')}")
        print(f"LUT intensity: {config.get('color_grading.lut_intensity')}")
        print(f"Auto exposure: {config.get('color_grading.auto_exposure')}")
        print(f"Auto contrast: {config.get('color_grading.auto_contrast')}")

        print(f"\n⚡ Performance")
        print("-" * 15)
        print(f"Use GPU: {config.get('performance.use_gpu')}")
        print(f"Threads: {config.get('performance.num_threads')}")
        print(f"Chunk size: {config.get('performance.chunk_size')}")
        print(f"Enable cache: {config.get('performance.enable_cache')}")

        print(f"\n✅ Configuration is valid!")

    except Exception as e:
        click.echo(f"❌ Configuration check failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def system_check(ctx):
    """🔧 Check system dependencies and GPU availability."""

    print("🔧 System Check")
    print("=" * 25)

    # Check Python version
    print("🐍 Python Environment")
    print("-" * 25)

    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python version: {python_version}")

    if sys.version_info >= (3, 9):
        print("✅ Python version is compatible")
    else:
        print("❌ Python 3.9+ required")
        return

    # Check dependencies with progress
    print(f"\n📦 Dependencies")
    print("-" * 20)

    dependencies = [
        ('OpenCV', 'cv2'),
        ('NumPy', 'numpy'),
        ('PyTorch', 'torch'),
        ('Librosa', 'librosa'),
        ('PyYAML', 'yaml'),
        ('scikit-image', 'skimage')
    ]

    progress = ProgressBar(len(dependencies), width=30, show_eta=False)

    for i, (name, module) in enumerate(dependencies):
        try:
            __import__(module)
            print(f"✅ {name}")
            progress.update(i + 1, f"Checking {name}")
        except ImportError:
            print(f"❌ {name} - Not installed")
            progress.update(i + 1, f"Missing {name}")

    # Check FFmpeg
    print(f"\n🎥 Media Tools")
    print("-" * 15)

    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg: {version_line}")
        else:
            print("❌ FFmpeg - Not working properly")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ FFmpeg - Not found")

    # Check GPU
    print(f"\n🚀 GPU Support")
    print("-" * 15)

    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {gpu_name}")
            print(f"   GPU count: {gpu_count}")
        else:
            print("ℹ️  CUDA not available (CPU-only mode)")
    except ImportError:
        print("❌ PyTorch not available")

    print(f"\n🎯 System Status: Ready for video processing!")


@cli.command()
@click.option('--name', default='test_lut', help='LUT name')
@click.option('--size', default=33, help='LUT size (17, 33, or 65)')
@click.option('--style', type=click.Choice(['warm', 'cool', 'cinematic']),
              default='cinematic', help='LUT style')
@click.pass_context
def create_lut(ctx, name, size, style):
    """🎨 Create a sample LUT file for testing."""

    print("🎨 LUT Creation")
    print("=" * 20)
    print(f"Name: {name}")
    print(f"Size: {size}x{size}x{size}")
    print(f"Style: {style}")

    progress = ProgressBar(100, width=30)

    try:
        progress.update(20, "Initializing LUT...")

        progress.update(40, "Generating LUT data...")

        # Create output path
        output_path = f"luts/{name}_{style}.cube"
        os.makedirs('luts', exist_ok=True)

        progress.update(70, "Saving LUT file...")

        # Create sample LUT
        create_sample_lut(output_path, style)

        progress.update(100, "LUT created successfully")

        print(f"\n✅ LUT created: {output_path}")
        print(f"📊 Size: {size}x{size}x{size}")
        print(f"🎨 Style: {style}")

    except Exception as e:
        progress.update(100, "LUT creation failed")
        click.echo(f"\n❌ LUT creation failed: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()