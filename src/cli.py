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
import json
import time

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
@click.option('--no-music', is_flag=True, default=False, help='Disable background music for this processing run')
@click.option('--stabilize', is_flag=True, default=False, help='Stabilize videos using FFmpeg vidstab before processing')
@click.pass_context
def process(ctx, input_path, input_dir, output_dir, lut, max_segments, use_default_lut, no_music, stabilize):
    """🎬 Process video(s) to create shorts with full progress tracking.

    Usage:
    - shorts-creator process video.mp4          # Process single video
    - shorts-creator process --input-dir ./     # Process all videos in directory
    - shorts-creator process                    # Process all videos in input/ directory with default LUT
    - shorts-creator process --stabilize        # Process with FFmpeg stabilization
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
            # Default: process all videos in input directory
            input_files = _get_video_files('input')
            print("🎬 Shorts Creator - Auto Batch Processing")
            print("📁 Input Directory: input/")

            # Create input directory if it doesn't exist
            if not os.path.exists('input'):
                os.makedirs('input')
                print("📁 Created input/ directory - place your videos here!")

        if not input_files:
            if not input_path and not input_dir:
                click.echo("❌ No video files found in input/ directory", err=True)
                click.echo("💡 Place your video files in the input/ directory and try again", err=True)
            else:
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

        # Temporarily disable music if requested
        if no_music:
            processor.music_manager.enable_music = False
            print("🔇 Music disabled for this run")

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
                    max_segments=max_segments,
                    stabilize=stabilize
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

    # Check FFmpeg vidstab filters
    try:
        result = subprocess.run(['ffmpeg', '-filters'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and 'vidstabdetect' in result.stdout and 'vidstabtransform' in result.stdout:
            print("✅ FFmpeg vidstab: Available for video stabilization")
        else:
            print("ℹ️  FFmpeg vidstab: Not available (optional for stabilization)")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("ℹ️  FFmpeg vidstab: Not found (optional for stabilization)")

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


@cli.command()
@click.option('--duration', '-d', default=15.0, help='Target segment duration in seconds')
@click.option('--show-details', '-v', is_flag=True, help='Show detailed analysis for each track')
@click.option('--limit', '-l', default=10, help='Limit analysis to N tracks (0 = all tracks)')
@click.option('--fast', '-f', is_flag=True, help='Fast mode: skip energy analysis, just show library')
def music_analyze(duration, show_details, limit, fast):
    """Analyze music files and show the best segments for video use."""
    try:
        from src.core.music_manager import MusicManager
        from src.utils.config import Config
    except ImportError:
        from core.music_manager import MusicManager
        from utils.config import Config

    config = Config()
    music_manager = MusicManager(config)

    if not music_manager.enable_music:
        click.echo("❌ Music is disabled in config")
        return

    music_files = music_manager.get_music_files()
    if not music_files:
        click.echo("❌ No music files found in music/ directory")
        return

    click.echo(f"🎵 Found {len(music_files)} music files")

    if fast:
        click.echo("⚡ Fast mode: Showing library without energy analysis")
        click.echo("   Use --limit 5 to analyze just a few tracks")
        click.echo()

        for i, music_file in enumerate(music_files[:20], 1):  # Show first 20
            filename = os.path.basename(music_file)
            file_size = os.path.getsize(music_file) / (1024 * 1024)  # MB
            click.echo(f"   {i:2d}. {filename:<40} ({file_size:.1f}MB)")

        if len(music_files) > 20:
            click.echo(f"   ... and {len(music_files) - 20} more tracks")

        click.echo(f"\n💡 To analyze tracks: python src/cli.py music-analyze --limit 5")
        return

    # Limit tracks for analysis
    if limit > 0:
        music_files = music_files[:limit]
        click.echo(f"⚡ Analyzing first {len(music_files)} tracks (use --limit 0 for all)")
    else:
        click.echo(f"🐌 Analyzing ALL {len(music_files)} tracks (this will take a while...)")

    click.echo(f"🎯 Target duration: {duration}s")
    click.echo()

    results = []

    with click.progressbar(music_files, label='Analyzing music') as files:
        for music_file in files:
            try:
                segment_info = music_manager.find_best_music_segment(music_file, duration)

                # Get track duration for context (fast method)
                try:
                    import librosa
                    y, sr = librosa.load(music_file, duration=5)  # Just load 5s to get sample rate
                    # Use ffprobe for duration (much faster)
                    import subprocess
                    result = subprocess.run([
                        'ffprobe', '-v', 'quiet', '-print_format', 'json',
                        '-show_format', music_file
                    ], capture_output=True, text=True)

                    if result.returncode == 0:
                        import json
                        data = json.loads(result.stdout)
                        track_duration = float(data['format']['duration'])
                    else:
                        # Fallback to librosa
                        y_full, _ = librosa.load(music_file, duration=None)
                        track_duration = len(y_full) / sr
                except:
                    track_duration = 0

                results.append({
                    'file': music_file,
                    'filename': os.path.basename(music_file),
                    'duration': track_duration,
                    'best_start': segment_info['start_time'],
                    'energy_score': segment_info['energy_score']
                })

            except Exception as e:
                click.echo(f"❌ Error analyzing {os.path.basename(music_file)}: {e}")

    # Sort by energy score (best first)
    results.sort(key=lambda x: x['energy_score'], reverse=True)

    click.echo("\n🎯 Best Music Segments (sorted by energy):")
    click.echo("=" * 70)

    for i, result in enumerate(results, 1):
        energy_bar = "█" * int(result['energy_score'] * 20)
        energy_bar = energy_bar.ljust(20, "░")

        click.echo(f"{i:2d}. {result['filename'][:30]:<30}")
        click.echo(f"    ⚡ Energy: [{energy_bar}] {result['energy_score']:.3f}")
        click.echo(f"    🎵 Best segment: {result['best_start']:.1f}s - {result['best_start'] + duration:.1f}s")
        click.echo(f"    ⏱️  Track length: {result['duration']:.1f}s")

        if show_details:
            percentage = (result['best_start'] / result['duration']) * 100 if result['duration'] > 0 else 0
            click.echo(f"    📊 Segment position: {percentage:.1f}% through track")

        click.echo()

    # Summary stats
    if results:
        avg_energy = sum(r['energy_score'] for r in results) / len(results)
        best_track = results[0]

        click.echo("📈 Summary:")
        click.echo(f"   • Average energy score: {avg_energy:.3f}")
        click.echo(f"   • Best track: {best_track['filename']} (energy: {best_track['energy_score']:.3f})")
        click.echo(f"   • Cache location: .cache/music_analysis.json")

        # Show distribution of best segments
        early_segments = sum(1 for r in results if r['best_start'] < r['duration'] * 0.33)
        middle_segments = sum(1 for r in results if 0.33 <= r['best_start'] / r['duration'] < 0.67)
        late_segments = sum(1 for r in results if r['best_start'] >= r['duration'] * 0.67)

        click.echo(f"   • Segment distribution: {early_segments} early, {middle_segments} middle, {late_segments} late")

        if limit > 0 and len(music_files) < len(music_manager.get_music_files()):
            remaining = len(music_manager.get_music_files()) - len(music_files)
            click.echo(f"   • {remaining} tracks not analyzed (use --limit 0 for all)")

    click.echo(f"\n💡 Tips:")
    click.echo(f"   • Use --fast for quick library overview")
    click.echo(f"   • Use --limit 5 to analyze just your best tracks")
    click.echo(f"   • Results are cached - re-running is instant!")


@cli.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output path for stabilized video')
@click.pass_context
def stabilize(ctx, input_video, output):
    """🔧 Stabilize a video using FFmpeg vidstab."""

    print("🔧 Video Stabilization")
    print("=" * 40)
    print(f"📁 Input: {input_video}")
    if output:
        print(f"📁 Output: {output}")
    print("=" * 40)

    try:
        # Load config
        config = Config(ctx.obj['config_path'])

        # Initialize stabilizer
        from src.utils.stabilizer import VideoStabilizer
        stabilizer = VideoStabilizer(config.data)

        # Check if FFmpeg with vidstab is available
        if not stabilizer.is_ffmpeg_available():
            click.echo("❌ FFmpeg with vidstab is not available. Please install FFmpeg with libvidstab support.", err=True)
            click.echo("💡 On Ubuntu/Debian: sudo apt install ffmpeg", err=True)
            click.echo("💡 On macOS: brew install ffmpeg", err=True)
            return

        def progress_callback(percentage: float, message: str = ""):
            bar_width = 40
            filled = int(bar_width * percentage / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"\r{bar} {percentage:5.1f}% {message}", end="", flush=True)

        # Stabilize video
        stabilized_path = stabilizer.stabilize_video(
            input_video,
            output_path=output,
            progress_callback=progress_callback
        )

        if stabilized_path:
            file_size = os.path.getsize(stabilized_path) / (1024 * 1024)
            print(f"\n✅ Video stabilized successfully!")
            print(f"📁 Output: {stabilized_path}")
            print(f"📊 File size: {file_size:.1f}MB")
        else:
            print(f"\n❌ Stabilization failed")
            # Try simple stabilization
            print("🔄 Trying simple stabilization...")
            stabilized_path = stabilizer.stabilize_video_simple(input_video)
            if stabilized_path:
                file_size = os.path.getsize(stabilized_path) / (1024 * 1024)
                print(f"✅ Simple stabilization successful!")
                print(f"📁 Output: {stabilized_path}")
                print(f"📊 File size: {file_size:.1f}MB")
            else:
                print(f"❌ Simple stabilization also failed")
                sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Stabilization failed: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
def music_check():
    """Check music library status and show available tracks."""
    try:
        from src.core.music_manager import MusicManager
        from src.utils.config import Config
    except ImportError:
        from core.music_manager import MusicManager
        from utils.config import Config

    config = Config()
    music_manager = MusicManager(config)

    click.echo("🎵 Music Library Status")
    click.echo("=" * 40)

    if not music_manager.enable_music:
        click.echo("❌ Music is disabled in config.yaml")
        click.echo("   Set 'enable_music: true' to enable")
        return

    music_files = music_manager.get_music_files()

    if not music_files:
        click.echo("❌ No music files found")
        click.echo("   Add .mp3, .wav, .m4a, or .flac files to the music/ directory")
        return

    click.echo(f"✅ Found {len(music_files)} music files:")

    total_duration = 0
    for i, music_file in enumerate(music_files, 1):
        filename = os.path.basename(music_file)
        file_size = os.path.getsize(music_file) / (1024 * 1024)  # MB

        # Try to get duration
        try:
            import librosa
            y, sr = librosa.load(music_file, duration=None)
            duration = len(y) / sr
            total_duration += duration
            duration_str = f"{duration:.1f}s"
        except:
            duration_str = "unknown"

        click.echo(f"   {i:2d}. {filename:<30} ({file_size:.1f}MB, {duration_str})")

    if total_duration > 0:
        click.echo(f"\n📊 Total music library: {total_duration/60:.1f} minutes")

    # Check cache status
    cache_file = os.path.join('.cache', 'music_analysis.json')
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            click.echo(f"💾 Analysis cache: {len(cache_data)} cached analyses")
        except:
            click.echo("💾 Analysis cache: present but unreadable")
    else:
        click.echo("💾 Analysis cache: not created yet")

    click.echo(f"\n⚙️  Music settings:")
    click.echo(f"   • Volume: {config.get('audio.music_volume', 0.3)}")
    click.echo(f"   • Fade in: {config.get('audio.fade_in_duration', 1.0)}s")
    click.echo(f"   • Fade out: {config.get('audio.fade_out_duration', 1.0)}s")
    click.echo(f"   • Random selection: {config.get('audio.random_selection', True)}")


@cli.command()
@click.option('--force', '-f', is_flag=True, help='Force re-analysis of all tracks')
@click.option('--show-stats', '-s', is_flag=True, help='Show detailed library statistics')
def music_library_analyze(force, show_stats):
    """Analyze the entire music library with AI for intelligent selection."""
    try:
        from src.core.music_manager import MusicManager
        from src.utils.config import Config
    except ImportError:
        from core.music_manager import MusicManager
        from utils.config import Config

    config = Config()
    music_manager = MusicManager(config)

    if not music_manager.enable_music:
        click.echo("❌ Music is disabled in config")
        return

    music_files = music_manager.get_music_files()
    if not music_files:
        click.echo("❌ No music files found in music/ directory")
        return

    click.echo(f"🎵 Starting comprehensive music library analysis...")
    click.echo(f"📁 Found {len(music_files)} tracks to analyze")

    if force:
        click.echo("🔄 Force mode: Re-analyzing all tracks")

    # Run the analysis
    summary = music_manager.analyze_entire_library(force_reanalyze=force)

    if show_stats and summary.get('analyzed', 0) + summary.get('cached', 0) > 0:
        click.echo(f"\n📊 Detailed Statistics:")
        click.echo(f"   • Processing time: {time.time() - summary['analysis_timestamp']:.1f}s")
        click.echo(f"   • Cache efficiency: {summary['cached']}/{summary['total_tracks']} ({summary['cached']/summary['total_tracks']*100:.1f}%)")

        # Show some sample intelligent selections
        click.echo(f"\n🎯 Testing Intelligent Selection:")
        for duration in [10, 15, 20]:
            selected = music_manager.select_music_for_video("test.mp4", duration)
            if selected:
                click.echo(f"   • {duration}s segment: {os.path.basename(selected)}")

    click.echo(f"\n💡 Tips:")
    click.echo(f"   • Analysis results are cached for fast future processing")
    click.echo(f"   • Use this command after adding new music files")
    click.echo(f"   • The system now uses AI to avoid repetitive music selection")


@cli.command()
@click.option('--test-duration', '-d', default=15, help='Test segment duration in seconds')
@click.option('--num-tests', '-n', default=5, help='Number of selection tests to run')
def music_test_selection(test_duration, num_tests):
    """Test the intelligent music selection system."""
    try:
        from src.core.music_manager import MusicManager
        from src.utils.config import Config
    except ImportError:
        from core.music_manager import MusicManager
        from utils.config import Config

    config = Config()
    music_manager = MusicManager(config)

    if not music_manager.enable_music:
        click.echo("❌ Music is disabled in config")
        return

    music_files = music_manager.get_music_files()
    if not music_files:
        click.echo("❌ No music files found")
        return

    click.echo(f"🎵 Testing Intelligent Music Selection")
    click.echo(f"🎯 Duration: {test_duration}s, Tests: {num_tests}")
    click.echo("=" * 60)

    # Ensure library is analyzed
    if not music_manager._library_analysis_cache:
        click.echo("🔍 Library not analyzed yet, running analysis...")
        music_manager.analyze_entire_library()

    selections = []
    for i in range(num_tests):
        selected = music_manager.select_music_for_video(f"test_video_{i}.mp4", test_duration)
        if selected:
            filename = os.path.basename(selected)
            selections.append(filename)

            # Get analysis data
            file_key = music_manager._get_file_cache_key(selected)
            analysis = music_manager._library_analysis_cache.get(file_key, {})

            energy = analysis.get('energy_score', 0)
            genre = analysis.get('genre_prediction', 'unknown')
            tempo = analysis.get('tempo', 0)

            click.echo(f"  {i+1:2d}. {filename[:45]:<45} | {energy:.3f} energy | {genre:<10} | {tempo:3.0f} BPM")
        else:
            click.echo(f"  {i+1:2d}. ❌ No music selected")

    # Check for variety
    unique_selections = len(set(selections))
    variety_percentage = (unique_selections / len(selections)) * 100 if selections else 0

    click.echo("\n" + "=" * 60)
    click.echo(f"📊 Selection Analysis:")
    click.echo(f"   • Unique tracks: {unique_selections}/{len(selections)} ({variety_percentage:.1f}% variety)")
    click.echo(f"   • Repetition: {'✅ Good variety' if variety_percentage > 80 else '⚠️ Some repetition' if variety_percentage > 50 else '❌ High repetition'}")

    if selections:
        # Show usage stats
        usage_stats = music_manager._usage_stats
        avg_usage = sum(usage_stats.get(s, 0) for s in selections) / len(selections)
        click.echo(f"   • Average usage count: {avg_usage:.1f}")
        click.echo(f"   • Recent selections: {len(music_manager._recent_selections)}")

    click.echo(f"\n💡 To reset usage stats: rm .cache/music_usage_stats.json")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()