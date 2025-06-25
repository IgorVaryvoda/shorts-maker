#!/usr/bin/env python3
"""
Test script for apply-lut functionality
"""

import os
import subprocess
import time
from pathlib import Path


def test_lut_application():
    """Test LUT application on a single video."""

    # Find a test video
    input_dir = "input"
    test_files = [
        f for f in os.listdir(input_dir) if f.endswith(".mp4") and "_colorized" not in f
    ]

    if not test_files:
        print("❌ No test video files found")
        return False

    test_video = os.path.join(input_dir, test_files[0])
    lut_file = "luts/avata2.cube"
    output_video = os.path.join(
        input_dir, f"{Path(test_files[0]).stem}_test_colorized.mp4"
    )

    print("🎬 Testing LUT application...")
    print(f"📹 Input: {test_files[0]}")
    print(f"🎨 LUT: {os.path.basename(lut_file)}")
    print(f"📁 Output: {os.path.basename(output_video)}")

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-i",
        test_video,
        "-vf",
        f"lut3d={lut_file}",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "slow",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        "-y",
        output_video,
    ]

    print("🎞️  Processing with FFmpeg...")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        output_size = os.path.getsize(output_video) / (1024 * 1024)
        print("✅ LUT application successful!")
        print(f"⏱️  Processing time: {elapsed:.1f}s")
        print(f"📊 Output size: {output_size:.1f}MB")
        return True
    else:
        print(f"❌ FFmpeg failed: {result.stderr}")
        return False


if __name__ == "__main__":
    success = test_lut_application()
    if success:
        print("\n🎉 LUT application test passed!")
    else:
        print("\n❌ LUT application test failed!")
