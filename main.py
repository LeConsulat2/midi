import sys
import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import pretty_midi

# Optional dependencies with graceful fallbacks
try:
    from scipy.signal import find_peaks, butter, filtfilt
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not installed. Using basic algorithms.")
    SCIPY_AVAILABLE = False

try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
except ImportError:
    print("Warning: moviepy not available. Video files not supported.")
    MOVIEPY_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("piano_transcription")


class AdvancedPianoTranscriber:
    """Advanced piano transcription with balanced detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sr = 22050
        
    def transcribe_audio(self, input_audio: str) -> str:
        """Main transcription method with hybrid approach"""
        logger.info(f"Transcribing: {input_audio}")
        
        if not os.path.isfile(input_audio):
            raise FileNotFoundError(f"Input file not found: {input_audio}")
            
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle video files
        audio_file = self._prepare_audio_file(input_audio, output_dir)
        
        # Load and preprocess audio
        y, sr = librosa.load(audio_file, sr=self.sr)
        y = self._preprocess_audio(y)
        
        logger.info(f"Audio loaded: {len(y)/sr:.1f} seconds")
        
        # Create MIDI
        pm = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0, name="Piano")
        
        # Use hybrid approach: CQT analysis + smart filtering
        notes = self._hybrid_transcribe(y, sr)
        piano.notes.extend(notes)
        
        # Smart cleanup (less aggressive than before)
        piano.notes = self._smart_cleanup(piano.notes)
        
        pm.instruments.append(piano)
        
        # Save
        base_name = os.path.splitext(os.path.basename(input_audio))[0]
        midi_path = os.path.join(output_dir, f"{base_name}_advanced.mid")
        pm.write(midi_path)
        
        logger.info(f"Created MIDI with {len(piano.notes)} notes")
        return midi_path
    
    def _prepare_audio_file(self, input_file: str, output_dir: str) -> str:
        """Extract audio from video if needed"""
        file_ext = os.path.splitext(input_file)[1].lower()
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
        
        if file_ext in video_extensions:
            if not MOVIEPY_AVAILABLE:
                raise RuntimeError("Video file detected but moviepy not installed")
                
            logger.info("Extracting audio from video...")
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            temp_audio = os.path.join(output_dir, f"{base_name}_extracted.wav")
            
            try:
                video = mp.VideoFileClip(input_file)
                if video.audio is None:
                    raise RuntimeError("No audio track in video")
                video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
                video.close()
                return temp_audio
            except Exception as e:
                raise RuntimeError(f"Failed to extract audio: {e}")
        
        return input_file
    
    def _preprocess_audio(self, y: np.ndarray) -> np.ndarray:
        """Clean up audio but preserve musical content"""
        # Remove DC offset
        y = y - np.mean(y)
        
        # Normalize
        y = librosa.util.normalize(y)
        
        # Very gentle high-pass filter (just remove extreme low frequencies)
        if SCIPY_AVAILABLE:
            b, a = butter(1, 40.0 / (self.sr / 2), btype='high')  # Gentler filter
            y = filtfilt(b, a, y)
        
        return y
    
    def _hybrid_transcribe(self, y: np.ndarray, sr: int) -> List[pretty_midi.Note]:
        """Hybrid approach: comprehensive detection + smart filtering"""
        notes = []
        
        hop_length = 512
        
        # 1. Use CQT (Constant-Q Transform) for better musical note detection
        logger.info("Computing CQT analysis...")
        
        # CQT parameters for piano range
        fmin = librosa.note_to_hz('A0')  # Lowest piano note
        n_bins = 88  # 88 piano keys
        bins_per_octave = 12
        
        try:
            # Try modern librosa first
            cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, 
                                   fmin=fmin, n_bins=n_bins, 
                                   bins_per_octave=bins_per_octave))
        except TypeError:
            # Fallback for older versions
            cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, 
                                   fmin=fmin, n_bins=n_bins))
        
        # Normalize CQT
        cqt = librosa.util.normalize(cqt, axis=1)
        
        # Time axis
        times = librosa.frames_to_time(np.arange(cqt.shape[1]), 
                                     sr=sr, hop_length=hop_length)
        
        logger.info(f"CQT shape: {cqt.shape}, analyzing {len(times):.1f}s")
        
        # 2. For each piano key, do smarter detection
        detection_threshold = 0.15  # Lower threshold to catch more notes
        min_note_duration = 0.05    # Minimum 50ms
        
        for note_idx in range(n_bins):
            midi_note = 21 + note_idx  # A0 = MIDI 21
            
            # Get magnitude profile for this note
            magnitude = cqt[note_idx, :]
            
            # Smooth slightly to reduce noise
            if SCIPY_AVAILABLE:
                magnitude = ndimage.gaussian_filter1d(magnitude, sigma=1.0)
            else:
                # Simple moving average
                kernel = np.ones(5) / 5
                magnitude = np.convolve(magnitude, kernel, mode='same')
            
            # Adaptive threshold based on this note's characteristics
            note_mean = np.mean(magnitude)
            note_std = np.std(magnitude)
            
            # Use percentile-based threshold (more robust)
            threshold = max(detection_threshold, 
                          np.percentile(magnitude, 75))  # 75th percentile
            threshold = min(threshold, 0.4)  # Cap at reasonable level
            
            # Find note regions
            note_regions = self._find_note_regions(magnitude, times, threshold, min_note_duration)
            
            # Create notes for each region
            for start_time, end_time, strength in note_regions:
                # Calculate velocity based on strength and note position
                velocity = self._calculate_velocity(strength, midi_note)
                
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=midi_note,
                    start=start_time,
                    end=end_time
                )
                notes.append(note)
        
        logger.info(f"Initial detection: {len(notes)} notes")
        return notes
    
    def _find_note_regions(self, magnitude: np.ndarray, times: np.ndarray, 
                          threshold: float, min_duration: float) -> List[Tuple[float, float, float]]:
        """Find continuous regions where note is active"""
        regions = []
        
        # Find where magnitude exceeds threshold
        above_threshold = magnitude > threshold
        
        # Find continuous regions
        in_note = False
        start_idx = 0
        
        for i, is_above in enumerate(above_threshold):
            if is_above and not in_note:
                # Start of note
                start_idx = i
                in_note = True
            elif not is_above and in_note:
                # End of note
                end_idx = i
                duration = times[end_idx] - times[start_idx]
                
                if duration >= min_duration:
                    # Calculate average strength in this region
                    strength = np.mean(magnitude[start_idx:end_idx])
                    regions.append((times[start_idx], times[end_idx], strength))
                
                in_note = False
        
        # Handle case where note continues to end
        if in_note:
            end_idx = len(magnitude) - 1
            duration = times[end_idx] - times[start_idx]
            if duration >= min_duration:
                strength = np.mean(magnitude[start_idx:end_idx])
                regions.append((times[start_idx], times[end_idx], strength))
        
        return regions
    
    def _calculate_velocity(self, strength: float, midi_note: int) -> int:
        """Calculate MIDI velocity based on strength and note characteristics"""
        # Base velocity from strength
        base_velocity = int(strength * 100)
        
        # Adjust for piano characteristics
        # Higher notes tend to be played softer, lower notes harder
        if midi_note < 40:  # Low notes
            velocity = base_velocity + 10
        elif midi_note > 80:  # High notes
            velocity = base_velocity - 5
        else:
            velocity = base_velocity
        
        # Add some musical variation
        velocity += np.random.randint(-8, 8)
        
        # Clamp to valid MIDI range
        return max(20, min(127, velocity))
    
    def _smart_cleanup(self, notes: List[pretty_midi.Note]) -> List[pretty_midi.Note]:
        """Intelligent cleanup that preserves musical content"""
        if not notes:
            return notes
        
        # Sort by start time
        notes.sort(key=lambda x: x.start)
        
        logger.info(f"Cleaning up {len(notes)} notes...")
        
        # 1. Remove extremely short notes (likely artifacts)
        notes = [n for n in notes if (n.end - n.start) >= 0.04]  # Min 40ms
        
        # 2. Group notes by pitch and clean up obvious duplicates
        pitch_groups = {}
        for note in notes:
            if note.pitch not in pitch_groups:
                pitch_groups[note.pitch] = []
            pitch_groups[note.pitch].append(note)
        
        cleaned_notes = []
        
        for pitch, pitch_notes in pitch_groups.items():
            # Sort by start time
            pitch_notes.sort(key=lambda x: x.start)
            
            # Remove notes that are too close together (likely duplicates)
            filtered_pitch_notes = []
            
            for note in pitch_notes:
                # Check if this note overlaps significantly with recent notes
                too_close = False
                
                for existing in filtered_pitch_notes[-2:]:  # Check last 2 notes only
                    time_gap = note.start - existing.start
                    
                    if time_gap < 0.1:  # Within 100ms
                        # Keep the stronger note
                        if note.velocity > existing.velocity:
                            filtered_pitch_notes.remove(existing)
                        else:
                            too_close = True
                        break
                
                if not too_close:
                    filtered_pitch_notes.append(note)
            
            cleaned_notes.extend(filtered_pitch_notes)
        
        # 3. Final sort by start time
        cleaned_notes.sort(key=lambda x: x.start)
        
        # 4. Adjust overlapping notes (less aggressive)
        for i in range(len(cleaned_notes) - 1):
            current = cleaned_notes[i]
            next_note = cleaned_notes[i + 1]
            
            # If current note overlaps with next note significantly
            overlap = current.end - next_note.start
            if overlap > 0.2 and current.pitch != next_note.pitch:
                # Reduce overlap but don't eliminate it completely
                current.end = next_note.start + 0.05  # 50ms overlap allowed
        
        logger.info(f"Cleanup result: {len(notes)} -> {len(cleaned_notes)} notes")
        return cleaned_notes
    
    def process_midi(self, midi_path: str) -> str:
        """Apply post-processing to MIDI"""
        logger.info("Post-processing MIDI...")
        
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            logger.error(f"Failed to load MIDI: {e}")
            return midi_path
        
        # Apply quantization if requested (gentler)
        if self.config.get("quantize", False):  # Default off
            pm = self._gentle_quantize(pm)
        
        # Separate hands if requested
        if self.config.get("separate_hands", True):
            pm = self._separate_hands(pm)
        
        # Adjust velocities for more natural sound
        pm = self._adjust_velocities(pm)
        
        # Save processed version
        output_path = midi_path.replace(".mid", "_processed.mid")
        try:
            pm.write(output_path)
            logger.info(f"Processed MIDI saved: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save processed MIDI: {e}")
            return midi_path
    
    def _gentle_quantize(self, pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Gentle quantization that preserves musical expression"""
        division = 16  # 16th notes
        
        # Get tempo
        tempo_changes = pm.get_tempo_changes()
        tempo = tempo_changes[1][0] if len(tempo_changes[1]) > 0 else 120.0
        
        beat_length = 60.0 / tempo
        grid_size = beat_length / (division / 4)
        
        for instrument in pm.instruments:
            if instrument.is_drum:
                continue
            
            for note in instrument.notes:
                # Only quantize if note is close to grid (preserve intentional timing)
                grid_position = note.start / grid_size
                nearest_grid = round(grid_position)
                distance = abs(grid_position - nearest_grid)
                
                if distance < 0.3:  # Only quantize if within 30% of grid
                    note.start = nearest_grid * grid_size
        
        return pm
    
    def _separate_hands(self, pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Separate into left and right hand parts"""
        split_point = self.config.get("hand_split_point", 60)  # Middle C
        
        new_pm = pretty_midi.PrettyMIDI()
        right_hand = pretty_midi.Instrument(program=0, name="Right Hand")
        left_hand = pretty_midi.Instrument(program=0, name="Left Hand")
        
        # Collect all notes
        all_notes = []
        for instrument in pm.instruments:
            if not instrument.is_drum:
                all_notes.extend(instrument.notes)
        
        # Split by pitch with some overlap for natural playing
        for note in all_notes:
            if note.pitch >= split_point:
                right_hand.notes.append(note)
            else:
                left_hand.notes.append(note)
        
        # Add non-empty instruments
        if right_hand.notes:
            new_pm.instruments.append(right_hand)
        if left_hand.notes:
            new_pm.instruments.append(left_hand)
        
        return new_pm
    
    def _adjust_velocities(self, pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Adjust velocities for more natural sound"""
        for instrument in pm.instruments:
            if instrument.is_drum:
                continue
            
            if not instrument.notes:
                continue
            
            # Get velocity statistics
            velocities = [note.velocity for note in instrument.notes]
            mean_vel = np.mean(velocities)
            std_vel = np.std(velocities)
            
            # If velocities are too uniform, add some variation
            if std_vel < 10:
                for note in instrument.notes:
                    # Add musical variation
                    variation = np.random.randint(-12, 12)
                    note.velocity = max(30, min(120, note.velocity + variation))
        
        return pm


def main():
    # Configuration
    input_dir = r"C:\Users\Jonathan\Documents\midi"
    output_dir = input_dir
    
    # Supported formats
    audio_formats = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]
    video_formats = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"]
    valid_formats = audio_formats + video_formats
    
    # Find files
    files = [f for f in os.listdir(input_dir) 
             if os.path.splitext(f)[1].lower() in valid_formats]
    
    if not files:
        print(f"No audio/video files found in {input_dir}")
        return 1
    
    # Configuration
    config = {
        "output_dir": output_dir,
        "quantize": False,  # Keep natural timing
        "separate_hands": True,
        "hand_split_point": 60,  # Middle C
    }
    
    # Process each file
    success_count = 0
    for file in files:
        input_file = os.path.join(input_dir, file)
        try:
            print(f"\n🎵 Processing: {file}")
            
            transcriber = AdvancedPianoTranscriber(config)
            
            # Transcribe
            midi_path = transcriber.transcribe_audio(input_file)
            
            # Post-process
            processed_path = transcriber.process_midi(midi_path)
            
            print(f"✅ Success: {os.path.basename(processed_path)}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Failed to process {file}: {e}")
            logger.error(f"Error processing {file}", exc_info=True)
    
    print(f"\n🎹 Completed: {success_count}/{len(files)} files processed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())