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
import hashlib
from collections import defaultdict
import time

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
    """Manages background music for FPV shorts with intelligent selection and caching."""

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

        # Enhanced caching system
        os.makedirs('.cache', exist_ok=True)
        self._analysis_cache = {}
        self._library_analysis_cache = {}
        self._usage_stats = defaultdict(int)
        self._recent_selections = []

        # Cache files
        self._cache_file = os.path.join('.cache', 'music_analysis.json')
        self._library_cache_file = os.path.join('.cache', 'music_library_analysis.json')
        self._usage_stats_file = os.path.join('.cache', 'music_usage_stats.json')

        self._load_all_caches()

        logger.info(f"Initialized Enhanced Music Manager - Music: {self.enable_music}")

    def _load_all_caches(self):
        """Load all cached data."""
        self._load_analysis_cache()
        self._load_library_analysis()
        self._load_usage_stats()

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

    def _load_library_analysis(self):
        """Load cached library analysis results."""
        try:
            if os.path.exists(self._library_cache_file):
                with open(self._library_cache_file, 'r') as f:
                    self._library_analysis_cache = json.load(f)
                self.logger.debug(f"Loaded library analysis for {len(self._library_analysis_cache)} tracks")
        except Exception as e:
            self.logger.warning(f"Could not load library analysis cache: {e}")
            self._library_analysis_cache = {}

    def _load_usage_stats(self):
        """Load music usage statistics."""
        try:
            if os.path.exists(self._usage_stats_file):
                with open(self._usage_stats_file, 'r') as f:
                    data = json.load(f)
                    self._usage_stats = defaultdict(int, data.get('usage_stats', {}))
                    self._recent_selections = data.get('recent_selections', [])
                self.logger.debug(f"Loaded usage stats for {len(self._usage_stats)} tracks")
        except Exception as e:
            self.logger.warning(f"Could not load usage stats: {e}")
            self._usage_stats = defaultdict(int)
            self._recent_selections = []

    def _save_analysis_cache(self):
        """Save music analysis results to cache."""
        try:
            with open(self._cache_file, 'w') as f:
                json.dump(self._analysis_cache, f, indent=2)
            self.logger.debug(f"Saved {len(self._analysis_cache)} music analyses to cache")
        except Exception as e:
            self.logger.warning(f"Could not save music analysis cache: {e}")

    def _save_library_analysis(self):
        """Save library analysis results to cache."""
        try:
            with open(self._library_cache_file, 'w') as f:
                json.dump(self._library_analysis_cache, f, indent=2)
            self.logger.debug(f"Saved library analysis for {len(self._library_analysis_cache)} tracks")
        except Exception as e:
            self.logger.warning(f"Could not save library analysis cache: {e}")

    def _save_usage_stats(self):
        """Save music usage statistics."""
        try:
            data = {
                'usage_stats': dict(self._usage_stats),
                'recent_selections': self._recent_selections,
                'last_updated': time.time()
            }
            with open(self._usage_stats_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.debug(f"Saved usage stats for {len(self._usage_stats)} tracks")
        except Exception as e:
            self.logger.warning(f"Could not save usage stats: {e}")

    def analyze_entire_library(self, force_reanalyze: bool = False) -> Dict[str, Any]:
        """Analyze the entire music library and cache results."""
        music_files = self.get_music_files()
        if not music_files:
            self.logger.warning("No music files found for library analysis")
            return {}

        print(f"🎵 Analyzing Music Library ({len(music_files)} tracks)")
        print("=" * 60)

        analyzed_count = 0
        skipped_count = 0
        failed_count = 0

        for i, music_file in enumerate(music_files, 1):
            filename = os.path.basename(music_file)
            file_key = self._get_file_cache_key(music_file)

            # Check if already analyzed (unless forcing reanalysis)
            if not force_reanalyze and file_key in self._library_analysis_cache:
                skipped_count += 1
                print(f"   {i:3d}/{len(music_files)} ⚡ {filename[:50]:<50} [CACHED]")
                continue

            try:
                print(f"   {i:3d}/{len(music_files)} 🔍 {filename[:50]:<50} [ANALYZING...]", end="", flush=True)

                # Analyze the track
                analysis = self._analyze_track_comprehensive(music_file)

                # Cache the results
                self._library_analysis_cache[file_key] = analysis
                analyzed_count += 1

                print(f"\r   {i:3d}/{len(music_files)} ✅ {filename[:50]:<50} [COMPLETE]")

            except Exception as e:
                failed_count += 1
                print(f"\r   {i:3d}/{len(music_files)} ❌ {filename[:50]:<50} [FAILED: {str(e)[:20]}]")

        # Save results
        self._save_library_analysis()

        # Generate summary
        summary = {
            'total_tracks': len(music_files),
            'analyzed': analyzed_count,
            'cached': skipped_count,
            'failed': failed_count,
            'analysis_timestamp': time.time()
        }

        print("\n" + "=" * 60)
        print("📊 Library Analysis Complete!")
        print(f"   • {analyzed_count} tracks analyzed")
        print(f"   • {skipped_count} tracks from cache")
        print(f"   • {failed_count} tracks failed")
        print(f"   • Total library: {len(music_files)} tracks")

        if analyzed_count > 0 or skipped_count > 0:
            # Show some interesting stats
            self._show_library_stats()

        return summary

    def _analyze_track_comprehensive(self, music_path: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of a music track."""
        analysis = {
            'file_path': music_path,
            'filename': os.path.basename(music_path),
            'file_size': os.path.getsize(music_path),
            'analysis_timestamp': time.time()
        }

        if not LIBROSA_AVAILABLE:
            # Basic analysis without librosa
            analysis.update({
                'duration': self._get_track_duration_ffprobe(music_path),
                'energy_score': 0.5,
                'tempo': 120,
                'genre_prediction': 'unknown',
                'mood': 'neutral',
                'analysis_method': 'basic'
            })
            return analysis

        try:
            # Load audio at lower sample rate for faster processing
            y, sr = librosa.load(music_path, sr=22050, duration=None)

            # Basic properties
            duration = len(y) / sr
            analysis['duration'] = duration
            analysis['sample_rate'] = sr

            # Energy and spectral features
            rms_energy = np.mean(librosa.feature.rms(y=y)[0])
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
            zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])

            # Tempo and beat information
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_strength = len(beats) / duration if duration > 0 else 0

            # Advanced features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)

            # Calculate overall energy score
            energy_score = self._calculate_comprehensive_energy_score(
                rms_energy, spectral_centroid, spectral_rolloff, zcr, beat_strength, tempo
            )

            # Genre and mood prediction (simple heuristics)
            genre_prediction = self._predict_genre(tempo, energy_score, spectral_centroid)
            mood = self._predict_mood(energy_score, tempo, mfcc_mean)

            analysis.update({
                'energy_score': float(energy_score),
                'rms_energy': float(rms_energy),
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff),
                'zero_crossing_rate': float(zcr),
                'tempo': float(tempo),
                'beat_strength': float(beat_strength),
                'genre_prediction': genre_prediction,
                'mood': mood,
                'mfcc_features': [float(x) for x in mfcc_mean],
                'analysis_method': 'comprehensive'
            })

        except Exception as e:
            self.logger.warning(f"Comprehensive analysis failed for {os.path.basename(music_path)}: {e}")
            # Fallback to basic analysis
            analysis.update({
                'duration': self._get_track_duration_ffprobe(music_path),
                'energy_score': 0.5,
                'tempo': 120,
                'genre_prediction': 'unknown',
                'mood': 'neutral',
                'analysis_method': 'fallback',
                'error': str(e)
            })

        return analysis

    def _calculate_comprehensive_energy_score(self, rms_energy: float, spectral_centroid: float,
                                            spectral_rolloff: float, zcr: float,
                                            beat_strength: float, tempo: float) -> float:
        """Calculate comprehensive energy score using multiple metrics."""
        # Normalize components
        rms_norm = min(rms_energy * 5, 1.0)  # RMS is usually very small
        centroid_norm = min(spectral_centroid / 4000.0, 1.0)
        rolloff_norm = min(spectral_rolloff / 8000.0, 1.0)
        zcr_norm = min(zcr * 10, 1.0)
        beat_norm = min(beat_strength / 3.0, 1.0)

        # Tempo bonus (good for action videos)
        tempo_bonus = 1.0
        if 120 <= tempo <= 180:
            tempo_bonus = 1.2
        elif 80 <= tempo <= 120:
            tempo_bonus = 1.0
        else:
            tempo_bonus = 0.8

        # Weighted combination
        energy_score = (
            rms_norm * 0.30 +         # Overall loudness
            centroid_norm * 0.25 +    # Brightness
            rolloff_norm * 0.15 +     # High-frequency energy
            zcr_norm * 0.10 +         # Percussive content
            beat_norm * 0.20          # Beat strength
        ) * tempo_bonus

        return min(energy_score, 1.0)

    def _predict_genre(self, tempo: float, energy: float, centroid: float) -> str:
        """Simple genre prediction based on audio features."""
        if tempo > 140 and energy > 0.7:
            if centroid > 3000:
                return "electronic"
            else:
                return "rock"
        elif tempo < 90 and energy < 0.4:
            return "ambient"
        elif 90 <= tempo <= 120 and energy > 0.5:
            return "pop"
        elif tempo > 120 and energy > 0.6:
            return "upbeat"
        else:
            return "cinematic"

    def _predict_mood(self, energy: float, tempo: float, mfcc_features) -> str:
        """Simple mood prediction."""
        if energy > 0.7 and tempo > 120:
            return "energetic"
        elif energy < 0.3:
            return "calm"
        elif energy > 0.5 and tempo > 100:
            return "uplifting"
        else:
            return "neutral"

    def _show_library_stats(self):
        """Show interesting statistics about the music library."""
        if not self._library_analysis_cache:
            return

        analyses = list(self._library_analysis_cache.values())

        # Energy distribution
        energies = [a.get('energy_score', 0.5) for a in analyses]
        avg_energy = sum(energies) / len(energies)
        high_energy_count = sum(1 for e in energies if e > 0.7)

        # Tempo distribution
        tempos = [a.get('tempo', 120) for a in analyses]
        avg_tempo = sum(tempos) / len(tempos)

        # Duration stats
        durations = [a.get('duration', 0) for a in analyses]
        total_duration = sum(durations)
        avg_duration = total_duration / len(durations)

        # Genre distribution
        genres = [a.get('genre_prediction', 'unknown') for a in analyses]
        genre_counts = {}
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

        print(f"\n🎵 Library Statistics:")
        print(f"   • Average energy: {avg_energy:.3f}")
        print(f"   • High energy tracks: {high_energy_count}/{len(analyses)} ({high_energy_count/len(analyses)*100:.1f}%)")
        print(f"   • Average tempo: {avg_tempo:.1f} BPM")
        print(f"   • Total duration: {total_duration/60:.1f} minutes")
        print(f"   • Average track length: {avg_duration:.1f}s")

        # Show top genres
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"   • Top genres: {', '.join([f'{g} ({c})' for g, c in sorted_genres[:3]])}")

    def select_music_for_video(self, video_path: str, segment_duration: float) -> Optional[str]:
        """Select appropriate music using intelligent selection with variety."""
        if not self.enable_music:
            return None

        music_files = self.get_music_files()
        if not music_files:
            self.logger.warning("No music files found - creating music directory")
            self._create_music_readme()
            return None

        # Ensure library is analyzed
        if not self._library_analysis_cache:
            print("🔍 Music library not analyzed yet. Running quick analysis...")
            self.analyze_entire_library()

        # Use intelligent selection if library is analyzed
        if self._library_analysis_cache:
            selected = self._intelligent_music_selection(music_files, segment_duration)
        else:
            # Fallback to random selection with recent avoidance
            selected = self._random_selection_with_variety(music_files)

        if selected:
            # Update usage statistics
            self._track_music_usage(selected)
            self.logger.info(f"Selected music: {os.path.basename(selected)}")

        return selected

    def _intelligent_music_selection(self, music_files: List[str], segment_duration: float) -> Optional[str]:
        """Intelligent music selection based on analysis and usage patterns."""
        candidates = []

        for music_file in music_files:
            file_key = self._get_file_cache_key(music_file)
            analysis = self._library_analysis_cache.get(file_key, {})

            if not analysis:
                continue

            # Calculate selection score
            score = self._calculate_selection_score(music_file, analysis, segment_duration)
            candidates.append((music_file, score, analysis))

        if not candidates:
            return self._random_selection_with_variety(music_files)

        # Sort by score and select from top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Select from top 20% to maintain some variety
        top_count = max(1, len(candidates) // 5)
        top_candidates = candidates[:top_count]

        # Use weighted random selection from top candidates
        weights = [c[1] for c in top_candidates]
        selected_idx = self._weighted_random_choice(weights)

        return top_candidates[selected_idx][0]

    def _calculate_selection_score(self, music_file: str, analysis: Dict[str, Any],
                                 segment_duration: float) -> float:
        """Calculate selection score for a music track."""
        score = 0.0

        # Base energy score (favor high energy for FPV)
        energy = analysis.get('energy_score', 0.5)
        score += energy * 40  # 0-40 points

        # Duration matching (prefer tracks longer than segment)
        track_duration = analysis.get('duration', 0)
        if track_duration >= segment_duration:
            duration_score = min(20, track_duration - segment_duration)  # 0-20 points
        else:
            duration_score = max(0, 10 - (segment_duration - track_duration))  # Penalty for short tracks
        score += duration_score

        # Genre preference
        genre = analysis.get('genre_prediction', 'unknown')
        if genre in self.preferred_genres:
            score += 15  # 15 point bonus
        elif genre in ['electronic', 'upbeat', 'rock']:  # Good for FPV
            score += 10
        elif genre == 'cinematic':
            score += 5

        # Tempo preference (good tempo for action videos)
        tempo = analysis.get('tempo', 120)
        if 120 <= tempo <= 180:
            score += 10
        elif 90 <= tempo <= 120:
            score += 5

        # Usage frequency penalty (avoid overused tracks)
        usage_count = self._usage_stats.get(os.path.basename(music_file), 0)
        usage_penalty = min(15, usage_count * 3)  # Up to 15 point penalty
        score -= usage_penalty

        # Recent usage penalty (strong penalty for recently used)
        if os.path.basename(music_file) in self._recent_selections[-10:]:  # Last 10 selections
            recent_penalty = 20 - (self._recent_selections[::-1].index(os.path.basename(music_file)) * 2)
            score -= max(10, recent_penalty)

        return max(0, score)  # Ensure non-negative score

    def _weighted_random_choice(self, weights: List[float]) -> int:
        """Select index using weighted random choice."""
        if not weights:
            return 0

        total = sum(weights)
        if total <= 0:
            return random.randint(0, len(weights) - 1)

        r = random.uniform(0, total)
        cumulative = 0

        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return i

        return len(weights) - 1

    def _random_selection_with_variety(self, music_files: List[str]) -> Optional[str]:
        """Random selection that avoids recent selections."""
        if not music_files:
            return None

        # Filter out recently used tracks
        available = []
        for music_file in music_files:
            filename = os.path.basename(music_file)
            if filename not in self._recent_selections[-5:]:  # Avoid last 5 selections
                available.append(music_file)

        # If all tracks were recently used, use all tracks
        if not available:
            available = music_files

        return random.choice(available)

    def _track_music_usage(self, music_file: str):
        """Track music usage for intelligent selection."""
        filename = os.path.basename(music_file)

        # Update usage count
        self._usage_stats[filename] += 1

        # Update recent selections (keep last 20)
        self._recent_selections.append(filename)
        if len(self._recent_selections) > 20:
            self._recent_selections = self._recent_selections[-20:]

        # Save updated stats
        self._save_usage_stats()

    def _get_file_cache_key(self, music_path: str) -> str:
        """Generate cache key for a music file."""
        try:
            # Use file path and modification time for cache key
            mtime = os.path.getmtime(music_path)
            path_hash = hashlib.md5(music_path.encode()).hexdigest()[:8]
            return f"{path_hash}_{int(mtime)}"
        except:
            return hashlib.md5(music_path.encode()).hexdigest()[:12]

    def _get_track_duration_ffprobe(self, music_path: str) -> float:
        """Get track duration using ffprobe."""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', music_path
            ], capture_output=True, text=True)

            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                return float(data['format']['duration'])
        except:
            pass
        return 0.0

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

        # Volume adjustment - always apply for consistency
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

    def find_best_music_segment(self, music_path: str, target_duration: float) -> Dict[str, float]:
        """Find the most energetic/exciting segment of a music track using AI analysis."""
        # Check cache first
        cache_key = self._get_cache_key(music_path, target_duration)
        if cache_key in self._analysis_cache:
            cached_result = self._analysis_cache[cache_key]
            self.logger.debug(f"Using cached analysis for {os.path.basename(music_path)}")
            return cached_result

        # Use library analysis if available
        file_key = self._get_file_cache_key(music_path)
        if file_key in self._library_analysis_cache:
            analysis = self._library_analysis_cache[file_key]
            duration = analysis.get('duration', target_duration)

            if duration <= target_duration:
                result = {'start_time': 0.0, 'energy_score': analysis.get('energy_score', 0.5)}
            else:
                # Use intelligent segment selection based on energy
                energy_score = analysis.get('energy_score', 0.5)
                # Start from 1/3 through the track for better segments
                start_time = max(0, (duration - target_duration) / 3)
                result = {'start_time': start_time, 'energy_score': energy_score}

            self._analysis_cache[cache_key] = result
            self._save_analysis_cache()
            return result

        # Fallback to original analysis method
        return self._find_best_segment_original(music_path, target_duration)

    def _get_cache_key(self, music_path: str, duration: float) -> str:
        """Generate cache key for music analysis."""
        # Include file modification time to detect changes
        try:
            mtime = os.path.getmtime(music_path)
            return f"{os.path.basename(music_path)}_{duration}s_{mtime}"
        except:
            return f"{os.path.basename(music_path)}_{duration}s"

    def _find_best_segment_original(self, music_path: str, target_duration: float) -> Dict[str, float]:
        """Find the most energetic/exciting segment of a music track using original analysis."""
        if not LIBROSA_AVAILABLE:
            self.logger.warning("Librosa not available - using beginning of track")
            result = {'start_time': 0.0, 'energy_score': 0.5}
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
                return result

            # Faster analysis with larger windows and less overlap
            window_size = 5.0  # Larger windows (was 2.0)
            hop_size = 2.0     # Less overlap (was 0.5)

            # Calculate number of analysis windows
            num_windows = int((track_duration - target_duration) / hop_size) + 1

            if num_windows <= 1:
                result = {'start_time': 0.0, 'energy_score': 1.0}
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

            return best_segment

        except Exception as e:
            self.logger.error(f"Music analysis failed: {e}")
            # Fallback to middle of track (often better than beginning)
            try:
                track_duration = self._get_track_duration_ffprobe(music_path)
                fallback_start = max(0, (track_duration - target_duration) / 3)
            except:
                fallback_start = 0.0

            result = {'start_time': fallback_start, 'energy_score': 0.5}
            return result

    def _calculate_segment_energy(self, segment, sr: int) -> float:
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