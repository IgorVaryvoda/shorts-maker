"""
Music Manager for FPV Shorts Creator

Handles background music selection, audio processing, and integration.
"""

import os
import random
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import subprocess
import tempfile
import json

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("Librosa not available - advanced audio features disabled")

try:
    from ..utils.config import Config
except ImportError:
    from utils.config import Config

logger = logging.getLogger(__name__)


class MusicManager:
    """Manages background music for FPV shorts."""

    def __init__(self, config: Config):
        """Initialize music manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Music settings
        self.enable_music = config.get('audio.enable_music', True)
        self.music_directory = config.get('audio.music_directory', 'music/')
        self.music_volume = config.get('audio.music_volume', 0.3)
        self.fade_in_duration = config.get('audio.fade_in_duration', 1.0)
        self.fade_out_duration = config.get('audio.fade_out_duration', 1.0)
        self.random_selection = config.get('audio.random_selection', True)
        self.preferred_genres = config.get('audio.preferred_genres', [])

        # Audio processing
        self.normalize_audio = config.get('audio.normalize_audio', True)
        self.original_audio_volume = config.get('audio.original_audio_volume', 0.0)

        # Beat detection
        self.beat_detection = config.get('audio.beat_detection', True)
        self.sync_cuts_to_beats = config.get('audio.sync_cuts_to_beats', False)

        # Supported audio formats
        self.supported_formats = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}

        # Create music directory if it doesn't exist
        os.makedirs(self.music_directory, exist_ok=True)

        # Cache for music files
        self._music_cache = None

        # Cache for music analysis results
        self._analysis_cache = {}
        self._cache_file = os.path.join('.cache', 'music_analysis.json')
        self._load_analysis_cache()

        logger.info(f"Initialized Music Manager - Music: {self.enable_music}")

    def _load_analysis_cache(self):
        """Load cached music analysis results."""
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, 'r') as f:
                    self._analysis_cache = json.load(f)
                self.logger.debug(f"Loaded {len(self._analysis_cache)} cached music analyses")
        except Exception as e:
            self.logger.warning(f"Could not load music analysis cache: {e}")
            self._analysis_cache = {}

    def _save_analysis_cache(self):
        """Save music analysis results to cache."""
        try:
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            with open(self._cache_file, 'w') as f:
                json.dump(self._analysis_cache, f, indent=2)
            self.logger.debug(f"Saved {len(self._analysis_cache)} music analyses to cache")
        except Exception as e:
            self.logger.warning(f"Could not save music analysis cache: {e}")

    def _get_cache_key(self, music_path: str, duration: float) -> str:
        """Generate cache key for music analysis."""
        # Include file modification time to detect changes
        try:
            mtime = os.path.getmtime(music_path)
            return f"{os.path.basename(music_path)}_{duration}s_{mtime}"
        except:
            return f"{os.path.basename(music_path)}_{duration}s"

    def get_music_files(self) -> List[str]:
        """Get all available music files."""
        if self._music_cache is not None:
            return self._music_cache

        music_files = []
        music_path = Path(self.music_directory)

        if not music_path.exists():
            self.logger.warning(f"Music directory not found: {self.music_directory}")
            return []

        for file_path in music_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                music_files.append(str(file_path))

        self._music_cache = sorted(music_files)
        self.logger.info(f"Found {len(music_files)} music files")
        return self._music_cache

    def select_music_for_video(self, video_path: str, segment_duration: float) -> Optional[str]:
        """Select appropriate music for a video segment."""
        if not self.enable_music:
            return None

        music_files = self.get_music_files()
        if not music_files:
            self.logger.warning("No music files found - creating music directory")
            self._create_music_readme()
            return None

        if self.random_selection:
            selected = random.choice(music_files)
        else:
            # Could implement more sophisticated selection based on:
            # - Video content analysis
            # - Genre preferences
            # - Duration matching
            selected = music_files[0]

        self.logger.info(f"Selected music: {os.path.basename(selected)}")
        return selected

    def find_best_music_segment(self, music_path: str, target_duration: float) -> Dict[str, float]:
        """Find the most energetic/exciting segment of a music track using AI analysis."""
        # Check cache first
        cache_key = self._get_cache_key(music_path, target_duration)
        if cache_key in self._analysis_cache:
            cached_result = self._analysis_cache[cache_key]
            self.logger.debug(f"Using cached analysis for {os.path.basename(music_path)}")
            return cached_result

        if not LIBROSA_AVAILABLE:
            self.logger.warning("Librosa not available - using beginning of track")
            result = {'start_time': 0.0, 'energy_score': 0.5}
            self._analysis_cache[cache_key] = result
            self._save_analysis_cache()
            return result

        try:
            # Load audio file with faster settings
            self.logger.debug(f"Analyzing music: {os.path.basename(music_path)}")

            # Load at lower sample rate for faster processing
            y, sr = librosa.load(music_path, sr=22050, duration=None)  # 22kHz instead of 44kHz

            # Calculate track duration
            track_duration = len(y) / sr

            if track_duration <= target_duration:
                # Track is shorter than needed, use entire track
                result = {'start_time': 0.0, 'energy_score': 1.0}
                self._analysis_cache[cache_key] = result
                self._save_analysis_cache()
                return result

            # Faster analysis with larger windows and less overlap
            window_size = 5.0  # Larger windows (was 2.0)
            hop_size = 2.0     # Less overlap (was 0.5)

            # Calculate number of analysis windows
            num_windows = int((track_duration - target_duration) / hop_size) + 1

            if num_windows <= 1:
                result = {'start_time': 0.0, 'energy_score': 1.0}
                self._analysis_cache[cache_key] = result
                self._save_analysis_cache()
                return result

            # Limit analysis windows for very long tracks
            max_windows = 20  # Don't analyze more than 20 segments
            if num_windows > max_windows:
                # Sample evenly across the track
                step = num_windows // max_windows
                window_indices = list(range(0, num_windows, step))[:max_windows]
            else:
                window_indices = list(range(num_windows))

            self.logger.debug(f"Analyzing {len(window_indices)} segments of {target_duration}s each")

            best_segment = {'start_time': 0.0, 'energy_score': 0.0}

            for i in window_indices:
                start_time = i * hop_size
                end_time = start_time + target_duration

                if end_time > track_duration:
                    break

                # Extract segment for analysis
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment = y[start_sample:end_sample]

                # Calculate energy score for this segment
                energy_score = self._calculate_segment_energy(segment, sr)

                if energy_score > best_segment['energy_score']:
                    best_segment = {
                        'start_time': start_time,
                        'energy_score': energy_score
                    }

            self.logger.info(f"Best segment: {best_segment['start_time']:.1f}s (energy: {best_segment['energy_score']:.3f})")

            # Cache the result
            self._analysis_cache[cache_key] = best_segment
            self._save_analysis_cache()

            return best_segment

        except Exception as e:
            self.logger.error(f"Music analysis failed: {e}")
            # Fallback to middle of track (often better than beginning)
            try:
                # Quick duration check using ffprobe
                import subprocess
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_format', music_path
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    import json
                    data = json.loads(result.stdout)
                    track_duration = float(data['format']['duration'])
                    fallback_start = max(0, (track_duration - target_duration) / 3)
                else:
                    fallback_start = 0.0
            except:
                fallback_start = 0.0

            result = {'start_time': fallback_start, 'energy_score': 0.5}
            self._analysis_cache[cache_key] = result
            self._save_analysis_cache()
            return result

    def _calculate_segment_energy(self, segment: np.ndarray, sr: int) -> float:
        """Calculate energy score for a music segment using multiple metrics."""
        try:
            # 1. RMS Energy (overall loudness/power)
            rms_energy = np.mean(librosa.feature.rms(y=segment)[0])

            # 2. Spectral Centroid (brightness/excitement)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)[0])
            spectral_centroid_norm = min(spectral_centroid / 4000.0, 1.0)  # Normalize to 0-1

            # 3. Spectral Rolloff (energy distribution)
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr)[0])
            spectral_rolloff_norm = min(spectral_rolloff / 8000.0, 1.0)  # Normalize to 0-1

            # 4. Zero Crossing Rate (percussive content)
            zcr = np.mean(librosa.feature.zero_crossing_rate(segment)[0])
            zcr_norm = min(zcr * 10, 1.0)  # Normalize to 0-1

            # 5. Tempo and Beat Strength
            tempo, beats = librosa.beat.beat_track(y=segment, sr=sr)
            beat_strength = len(beats) / (len(segment) / sr)  # Beats per second
            beat_strength_norm = min(beat_strength / 3.0, 1.0)  # Normalize (3 BPS = max)

            # 6. Onset Strength (musical events/transitions)
            onset_strength = np.mean(librosa.onset.onset_strength(y=segment, sr=sr))

            # Weighted combination of all metrics
            # Higher weights for metrics that indicate "exciting" music
            energy_score = (
                rms_energy * 0.25 +           # Overall energy
                spectral_centroid_norm * 0.20 + # Brightness
                spectral_rolloff_norm * 0.15 +  # High frequency content
                zcr_norm * 0.10 +              # Percussive elements
                beat_strength_norm * 0.20 +    # Strong beat
                onset_strength * 0.10          # Musical events
            )

            # Bonus for segments with strong tempo (good for FPV)
            if 120 <= tempo <= 180:  # Sweet spot for action music
                energy_score *= 1.1

            return min(energy_score, 1.0)  # Cap at 1.0

        except Exception as e:
            self.logger.error(f"Energy calculation failed: {e}")
            return 0.5  # Neutral score on error

    def add_music_to_video(
        self,
        video_path: str,
        output_path: str,
        music_path: Optional[str] = None,
        segment_duration: Optional[float] = None
    ) -> str:
        """Add background music to a video."""
        if not self.enable_music or not music_path:
            # Just copy the video if no music
            if video_path != output_path:
                subprocess.run([
                    'ffmpeg', '-i', video_path, '-c', 'copy', '-y', output_path
                ], check=True, capture_output=True)
            return output_path

        # Get video duration if not provided
        if segment_duration is None:
            segment_duration = self._get_video_duration(video_path)

        # Create temporary audio file with proper duration and fades
        temp_audio = self._prepare_audio_track(music_path, segment_duration)

        try:
            # Check if input video has audio
            has_video_audio = self._has_audio_stream(video_path)

            if has_video_audio and self.original_audio_volume > 0:
                # Mix video audio with music
                cmd = [
                    'ffmpeg',
                    '-i', video_path,           # Input video
                    '-i', temp_audio,           # Input music
                    '-c:v', 'copy',             # Copy video stream
                    '-filter_complex', f'[0:a]volume={self.original_audio_volume}[va];[1:a]volume=1.0[ma];[va][ma]amix=inputs=2:duration=shortest[out]',
                    '-map', '0:v:0',            # Map video from first input
                    '-map', '[out]',            # Map mixed audio
                    '-c:a', 'aac',              # Encode audio as AAC
                    '-b:a', '128k',             # Audio bitrate
                    '-shortest',                # End when shortest stream ends
                    '-y',                       # Overwrite output
                    output_path
                ]
            else:
                # Video has no audio or original audio is disabled - just add music
                cmd = [
                    'ffmpeg',
                    '-i', video_path,           # Input video
                    '-i', temp_audio,           # Input music
                    '-c:v', 'copy',             # Copy video stream
                    '-c:a', 'aac',              # Encode audio as AAC
                    '-b:a', '128k',             # Audio bitrate
                    '-map', '0:v:0',            # Map video from first input
                    '-map', '1:a:0',            # Map music audio from second input
                    '-shortest',                # End when shortest stream ends
                    '-y',                       # Overwrite output
                    output_path
                ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                # Fallback: copy original video
                subprocess.run(['ffmpeg', '-i', video_path, '-c', 'copy', '-y', output_path],
                             check=True, capture_output=True)
            else:
                self.logger.info(f"Added music to {os.path.basename(output_path)}")

        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio):
                os.remove(temp_audio)

        return output_path

    def _prepare_audio_track(self, music_path: str, duration: float) -> str:
        """Prepare audio track with proper duration and fades, using the best segment."""
        temp_fd, temp_audio = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)

        # Find the best segment of the music
        best_segment = self.find_best_music_segment(music_path, duration)
        start_time = best_segment['start_time']
        energy_score = best_segment['energy_score']

        self.logger.debug(f"Using segment at {start_time:.1f}s (energy: {energy_score:.3f})")

        # FFmpeg command to process audio with intelligent segment selection
        cmd = [
            'ffmpeg',
            '-i', music_path,
            '-ss', str(start_time),        # Start at the best segment
            '-t', str(duration),           # Trim to segment duration
            '-af', self._build_audio_filter(duration),
            '-ar', '44100',                # Sample rate
            '-ac', '2',                    # Stereo
            '-y',
            temp_audio
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.error(f"Audio processing error: {result.stderr}")
            # Fallback: try without segment selection
            cmd_fallback = [
                'ffmpeg',
                '-i', music_path,
                '-t', str(duration),
                '-af', self._build_audio_filter(duration),
                '-ar', '44100',
                '-ac', '2',
                '-y',
                temp_audio
            ]
            result = subprocess.run(cmd_fallback, capture_output=True, text=True)
            if result.returncode != 0:
                # Create silent audio as final fallback
                self._create_silent_audio(temp_audio, duration)

        return temp_audio

    def _build_audio_filter(self, duration: float) -> str:
        """Build FFmpeg audio filter string."""
        filters = []

        # Volume adjustment
        if self.music_volume != 1.0:
            filters.append(f"volume={self.music_volume}")

        # Fade in
        if self.fade_in_duration > 0:
            filters.append(f"afade=t=in:st=0:d={self.fade_in_duration}")

        # Fade out
        if self.fade_out_duration > 0:
            fade_start = max(0, duration - self.fade_out_duration)
            filters.append(f"afade=t=out:st={fade_start}:d={self.fade_out_duration}")

        # Normalization
        if self.normalize_audio:
            filters.append("loudnorm")

        return ','.join(filters) if filters else "anull"

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using FFprobe."""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            video_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            data = json.loads(result.stdout)
            return float(data['format']['duration'])
        except Exception as e:
            self.logger.error(f"Could not get video duration: {e}")
            return 15.0  # Default fallback

    def _has_audio_stream(self, video_path: str) -> bool:
        """Check if video file has an audio stream."""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'csv=p=0',
            video_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return bool(result.stdout.strip())
        except Exception:
            return False

    def _create_silent_audio(self, output_path: str, duration: float):
        """Create silent audio as fallback."""
        cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100',
            '-t', str(duration),
            '-y',
            output_path
        ]
        subprocess.run(cmd, capture_output=True)

    def _create_music_readme(self):
        """Create README in music directory with instructions."""
        readme_path = os.path.join(self.music_directory, 'README.md')

        content = """# Music Directory 🎵

Place your background music files here for FPV shorts.

## Supported Formats
- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- AAC (.aac)
- OGG (.ogg)
- FLAC (.flac)

## Recommended Music
- **Electronic/EDM**: High energy for action sequences
- **Cinematic**: Epic orchestral for scenic flights
- **Upbeat**: Pop/rock for general FPV content
- **Ambient**: Chill tracks for smooth flights

## Free Music Sources
- **YouTube Audio Library**: https://studio.youtube.com/channel/UC.../music
- **Pixabay Music**: https://pixabay.com/music/
- **Freesound**: https://freesound.org/
- **Zapsplat**: https://zapsplat.com/ (free with account)
- **Epidemic Sound**: https://epidemicsound.com/ (paid)

## Usage
The system will automatically:
1. Randomly select music for each video
2. Trim to match segment duration
3. Add fade in/out effects
4. Mix at 30% volume (configurable)
5. Normalize audio levels

## Tips
- Use copyright-free music for social media
- 10-20 second tracks work best for shorts
- Higher energy music works well for FPV footage
- Avoid tracks with vocals (can be distracting)
"""

        with open(readme_path, 'w') as f:
            f.write(content)

        self.logger.info(f"Created music directory guide: {readme_path}")

    def analyze_music_beats(self, music_path: str) -> Optional[List[float]]:
        """Analyze music for beat detection (requires librosa)."""
        if not LIBROSA_AVAILABLE or not self.beat_detection:
            return None

        try:
            # Load audio
            y, sr = librosa.load(music_path)

            # Beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beats, sr=sr)

            self.logger.info(f"Detected {len(beat_times)} beats, tempo: {tempo:.1f} BPM")
            return beat_times.tolist()

        except Exception as e:
            self.logger.error(f"Beat detection failed: {e}")
            return None

    def get_music_info(self) -> Dict[str, Any]:
        """Get information about available music."""
        music_files = self.get_music_files()

        info = {
            'enabled': self.enable_music,
            'music_directory': self.music_directory,
            'total_tracks': len(music_files),
            'supported_formats': list(self.supported_formats),
            'settings': {
                'volume': self.music_volume,
                'fade_in': self.fade_in_duration,
                'fade_out': self.fade_out_duration,
                'random_selection': self.random_selection
            }
        }

        if music_files:
            info['sample_tracks'] = [os.path.basename(f) for f in music_files[:5]]

        return info