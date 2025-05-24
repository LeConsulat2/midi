import sys
import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Check for scipy
try:
    from scipy.signal import find_peaks
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not installed. Install with: pip install scipy")
    SCIPY_AVAILABLE = False

# Check for required libraries and provide helpful error messages
try:
    import pretty_midi
except ImportError:
    print("Error: pretty_midi not installed. Install with: pip install pretty_midi")
    sys.exit(1)

try:
    import librosa
except ImportError:
    print("Error: librosa not installed. Install with: pip install librosa")
    sys.exit(1)

try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
except ImportError:
    print("Warning: moviepy not installed. MP4/video files not supported.")
    print("Install with: pip install moviepy")
    MOVIEPY_AVAILABLE = False

# Try to import tensorflow for magenta
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ImportError:
    print("Warning: TensorFlow not installed. Some transcription methods may not work.")

# Try to import magenta
try:
    from magenta.models.onsets_frames_transcription import infer_util
    from magenta.models.onsets_frames_transcription import constants
    MAGENTA_AVAILABLE = True
except ImportError:
    print("Warning: Magenta not installed. Falling back to enhanced transcription.")
    print("For best results, install with: pip install magenta")
    MAGENTA_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("piano_transcription")


class PianoTranscriber:
    """Enhanced Piano transcription pipeline with improved algorithms"""

    def __init__(self, config: Dict):
        self.config = config

    def transcribe_audio(self, input_audio: str) -> str:
        """
        Transcribes piano audio to MIDI using available methods
        Returns path to the created MIDI file
        """
        logger.info(f"Transcribing piano audio: {input_audio}")
        
        if not os.path.isfile(input_audio):
            raise FileNotFoundError(f"Input audio file not found: {input_audio}")
        
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        # Handle video files by extracting audio first
        audio_file = self._prepare_audio_file(input_audio, output_dir)

        midi_path = None
        
        # Method 1: Try Basic Pitch (Spotify's open-source model)
        if self.config.get("use_basic_pitch", True):
            try:
                midi_path = self._transcribe_with_basic_pitch(audio_file, output_dir)
            except Exception as e:
                logger.warning(f"Basic Pitch transcription failed: {e}")
        
        # Method 2: Try Magenta if available
        if not midi_path and MAGENTA_AVAILABLE and self.config.get("checkpoint_dir"):
            try:
                midi_path = self._transcribe_with_magenta(audio_file, output_dir)
            except Exception as e:
                logger.warning(f"Magenta transcription failed: {e}")
        
        # Method 3: Enhanced multi-pitch transcription
        if not midi_path:
            if SCIPY_AVAILABLE:
                logger.info("Using enhanced multi-pitch transcription method")
                midi_path = self._transcribe_enhanced(audio_file, output_dir)
            else:
                logger.info("Using basic transcription method (scipy not available)")
                midi_path = self._transcribe_basic_improved(audio_file, output_dir)

        if not midi_path or not os.path.isfile(midi_path):
            raise FileNotFoundError(f"Transcription failed - no MIDI file created")

        logger.info(f"Raw transcription saved to: {midi_path}")
        return midi_path

    def _prepare_audio_file(self, input_file: str, output_dir: str) -> str:
        """Prepare audio file for transcription"""
        file_ext = os.path.splitext(input_file)[1].lower()
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
        
        if file_ext in video_extensions:
            if not MOVIEPY_AVAILABLE:
                raise RuntimeError(f"Video file detected but moviepy not installed")
            
            logger.info(f"Extracting audio from video file: {input_file}")
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            temp_audio = os.path.join(output_dir, f"{base_name}_extracted.wav")
            
            try:
                video = mp.VideoFileClip(input_file)
                audio = video.audio
                if audio is None:
                    raise RuntimeError("No audio track found in video file")
                
                audio.write_audiofile(temp_audio, verbose=False, logger=None)
                audio.close()
                video.close()
                
                logger.info(f"Audio extracted to: {temp_audio}")
                return temp_audio
                
            except Exception as e:
                raise RuntimeError(f"Failed to extract audio from video: {e}")
        
        return input_file

    def _transcribe_with_basic_pitch(self, input_audio: str, output_dir: str) -> str:
        """Try to use Basic Pitch (best free option)"""
        try:
            # Check if basic-pitch is available
            result = subprocess.run(["basic-pitch", "--help"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                raise RuntimeError("basic-pitch command not found")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            raise RuntimeError("Basic Pitch not installed. Install with: pip install basic-pitch")

        logger.info("Using Basic Pitch for transcription")
        
        # Run Basic Pitch
        cmd = [
            "basic-pitch", 
            output_dir, 
            input_audio,
            "--onset-threshold", "0.5",
            "--frame-threshold", "0.3", 
            "--minimum-note-length", "58",  # ~0.13 seconds at 22050 Hz
            "--minimum-frequency", str(librosa.note_to_hz('A0')),  # Piano range
            "--maximum-frequency", str(librosa.note_to_hz('C8'))
        ]
        
        try:
            subprocess.run(cmd, check=True, timeout=600)  # 10 minute timeout
        except subprocess.TimeoutExpired:
            raise RuntimeError("Basic Pitch transcription timed out")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Basic Pitch failed with error: {e}")
        
        # Find the created MIDI file
        base_name = os.path.splitext(os.path.basename(input_audio))[0]
        possible_paths = [
            os.path.join(output_dir, f"{base_name}_basic_pitch.mid"),
            os.path.join(output_dir, f"{base_name}.mid")
        ]
        
        for path in possible_paths:
            if os.path.isfile(path):
                return path
                
        raise FileNotFoundError("Basic Pitch didn't create expected MIDI file")

    def _transcribe_with_magenta(self, input_audio: str, output_dir: str) -> str:
        """Transcribe using Magenta's onsets-and-frames model"""
        checkpoint = self.config["checkpoint_dir"]
        
        # Load audio at correct sample rate
        audio, sr = librosa.load(input_audio, sr=constants.SAMPLE_RATE)
        
        # Run inference
        sequence = infer_util.infer(audio, checkpoint)
        
        # Save MIDI
        base = os.path.splitext(os.path.basename(input_audio))[0]
        midi_path = os.path.join(output_dir, f"{base}_magenta.mid")
        
        pretty_midi.sequence_proto_to_pretty_midi(sequence).write(midi_path)
        return midi_path

    def _transcribe_enhanced(self, input_audio: str, output_dir: str) -> str:
        """Enhanced multi-pitch transcription using spectral analysis"""
        logger.info("Using enhanced spectral transcription method")
        
        # Load audio with higher sample rate for better frequency resolution
        y, sr = librosa.load(input_audio, sr=44100)
        
        # Enhance audio quality
        y = self._preprocess_audio(y, sr)
        
        # Create MIDI object
        pm = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        
        # Get spectral representation
        hop_length = 512
        n_fft = 4096  # Higher for better frequency resolution
        
        # Compute CQT (Constant-Q Transform) - better for musical notes
        # Use compatible parameters for older librosa versions
        try:
            cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, 
                                    fmin=librosa.note_to_hz('A0'), 
                                    fmax=librosa.note_to_hz('C8'),
                                    n_bins=88))  # 88 piano keys
        except TypeError:
            # Fallback for older librosa versions
            cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, 
                                    fmin=librosa.note_to_hz('A0'), 
                                    n_bins=88))  # 88 piano keys
        
        # Normalize
        cqt = librosa.util.normalize(cqt, axis=0)
        
        # Time axis
        times = librosa.frames_to_time(np.arange(cqt.shape[1]), 
                                     sr=sr, hop_length=hop_length)
        
        # For each piano key, detect note events
        min_note_frames = int(0.1 * sr / hop_length)  # Minimum 100ms note
        
        for note_idx in range(88):  # 88 piano keys
            midi_note = 21 + note_idx  # A0 = MIDI 21
            
            # Get the magnitude profile for this note
            magnitude = cqt[note_idx, :]
            
            # Smooth the signal
            magnitude = ndimage.gaussian_filter1d(magnitude, sigma=2)
            
            # Find note onsets and offsets
            notes = self._detect_note_events(magnitude, times, midi_note, min_note_frames)
            piano.notes.extend(notes)
        
        # Sort notes by start time
        piano.notes.sort(key=lambda x: x.start)
        
        # Remove very short notes and overlaps
        piano.notes = self._clean_notes(piano.notes)
        
        pm.instruments.append(piano)
        
        # Save MIDI
        base = os.path.splitext(os.path.basename(input_audio))[0]
        midi_path = os.path.join(output_dir, f"{base}_enhanced.mid")
        pm.write(midi_path)
        
        return midi_path

    def _preprocess_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Preprocess audio for better transcription"""
        # Remove DC offset
        y = y - np.mean(y)
        
        # Normalize
        y = librosa.util.normalize(y)
        
        # Apply subtle high-pass filter to reduce low-frequency noise
        y = librosa.effects.preemphasis(y, coef=0.97)
        
        return y

    def _detect_note_events(self, magnitude: np.ndarray, times: np.ndarray, 
                          midi_note: int, min_note_frames: int) -> List[pretty_midi.Note]:
        """Detect individual note events from magnitude profile"""
        notes = []
        
        # Adaptive threshold
        threshold = np.mean(magnitude) + 2 * np.std(magnitude)
        threshold = max(threshold, 0.1)  # Minimum threshold
        
        # Find peaks (note onsets)
        if SCIPY_AVAILABLE:
            peaks, properties = find_peaks(magnitude, 
                                         height=threshold,
                                         distance=min_note_frames,
                                         prominence=threshold * 0.5)
        else:
            # Simple peak detection fallback
            peaks = []
            for i in range(min_note_frames, len(magnitude) - min_note_frames):
                if (magnitude[i] > threshold and 
                    magnitude[i] > magnitude[i-1] and 
                    magnitude[i] > magnitude[i+1]):
                    # Check if it's far enough from previous peaks
                    if not peaks or (i - peaks[-1]) >= min_note_frames:
                        peaks.append(i)
            peaks = np.array(peaks)
        
        if len(peaks) == 0:
            return notes
        
        # For each peak, find the corresponding note duration
        for peak_idx in peaks:
            start_time = times[peak_idx]
            
            # Find note end by looking for when magnitude drops significantly
            end_idx = peak_idx
            peak_magnitude = magnitude[peak_idx]
            
            # Look forward to find where note ends
            for i in range(peak_idx + 1, min(len(magnitude), peak_idx + int(5 * 44100 / 512))):
                if magnitude[i] < peak_magnitude * 0.3:  # 30% of peak
                    end_idx = i
                    break
            else:
                # Default duration if no clear end found
                end_idx = min(len(magnitude) - 1, peak_idx + int(1.0 * 44100 / 512))
            
            end_time = times[end_idx]
            duration = end_time - start_time
            
            # Only add notes with reasonable duration
            if duration >= 0.05 and duration <= 20.0:  # 50ms to 20 seconds
                velocity = int(min(127, max(1, peak_magnitude * 127)))
                
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=midi_note,
                    start=start_time,
                    end=end_time
                )
                notes.append(note)
        
        return notes

    def _clean_notes(self, notes: List[pretty_midi.Note]) -> List[pretty_midi.Note]:
        """Clean up the note list by removing very short notes and fixing overlaps"""
        if not notes:
            return notes
        
        # Remove notes shorter than 50ms
        notes = [note for note in notes if (note.end - note.start) >= 0.05]
        
        # Sort by start time
        notes.sort(key=lambda x: x.start)
        
        # Remove excessive overlaps for same pitch
        cleaned_notes = []
        for note in notes:
            # Check if this note overlaps significantly with recent notes of same pitch
            overlapping = False
            for existing in cleaned_notes[-10:]:  # Check last 10 notes
                if (existing.pitch == note.pitch and 
                    existing.end > note.start and 
                    (existing.end - note.start) > 0.1):  # 100ms overlap
                    overlapping = True
                    break
            
            if not overlapping:
                cleaned_notes.append(note)
        
        return cleaned_notes

    def _transcribe_basic_improved(self, input_audio: str, output_dir: str) -> str:
        """Improved basic transcription without scipy dependency"""
        logger.info("Using improved basic transcription method")
        
        # Load audio with higher sample rate
        y, sr = librosa.load(input_audio, sr=44100)
        
        # Enhance audio quality
        y = self._preprocess_audio(y, sr)
        
        # Create MIDI object
        pm = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        
        # Parameters
        hop_length = 512
        n_fft = 4096
        
        # Get STFT for analysis
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(D)
        
        # Get time and frequency axes
        times = librosa.frames_to_time(np.arange(magnitude.shape[1]), 
                                     sr=sr, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Piano frequency range (A0 to C8)
        piano_notes = []
        for midi_note in range(21, 109):  # A0 to C8
            freq = librosa.note_to_hz(librosa.midi_to_note(midi_note))
            piano_notes.append((midi_note, freq))
        
        # For each piano note, find the closest frequency bin
        for midi_note, target_freq in piano_notes:
            freq_bin = np.argmin(np.abs(freqs - target_freq))
            
            if freq_bin >= len(freqs):
                continue
                
            # Get magnitude profile for this frequency
            note_magnitude = magnitude[freq_bin, :]
            
            # Simple peak detection without scipy
            notes = self._detect_notes_simple(note_magnitude, times, midi_note)
            piano.notes.extend(notes)
        
        # Sort and clean notes
        piano.notes.sort(key=lambda x: x.start)
        piano.notes = self._clean_notes_simple(piano.notes)
        
        pm.instruments.append(piano)
        
        # Save MIDI
        base = os.path.splitext(os.path.basename(input_audio))[0]
        midi_path = os.path.join(output_dir, f"{base}_basic_improved.mid")
        pm.write(midi_path)
        
        return midi_path

    def _detect_notes_simple(self, magnitude: np.ndarray, times: np.ndarray, 
                           midi_note: int) -> List[pretty_midi.Note]:
        """Simple note detection without scipy"""
        notes = []
        
        # Smooth the signal manually
        window_size = 5
        smoothed = np.convolve(magnitude, np.ones(window_size)/window_size, mode='same')
        
        # Adaptive threshold
        mean_mag = np.mean(smoothed)
        std_mag = np.std(smoothed)
        threshold = mean_mag + 1.5 * std_mag
        threshold = max(threshold, 0.05)
        
        # Simple peak detection
        above_threshold = smoothed > threshold
        
        # Find onset and offset points
        i = 0
        while i < len(above_threshold) - 1:
            if above_threshold[i] and not (i > 0 and above_threshold[i-1]):
                # Found onset
                onset_idx = i
                
                # Find offset
                offset_idx = onset_idx
                for j in range(onset_idx + 1, len(above_threshold)):
                    if not above_threshold[j]:
                        offset_idx = j
                        break
                else:
                    offset_idx = len(above_threshold) - 1
                
                # Create note if long enough
                duration = times[offset_idx] - times[onset_idx]
                if duration >= 0.1 and duration <= 10.0:  # 100ms to 10 seconds
                    peak_magnitude = np.max(smoothed[onset_idx:offset_idx+1])
                    velocity = int(min(127, max(30, peak_magnitude * 300)))
                    
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=midi_note,
                        start=times[onset_idx],
                        end=times[offset_idx]
                    )
                    notes.append(note)
                
                i = offset_idx
            else:
                i += 1
        
        return notes

    def _clean_notes_simple(self, notes: List[pretty_midi.Note]) -> List[pretty_midi.Note]:
        """Simple note cleaning without scipy"""
        if not notes:
            return notes
        
        # Remove very short notes
        notes = [note for note in notes if (note.end - note.start) >= 0.08]
        
        # Sort by start time
        notes.sort(key=lambda x: x.start)
        
        # Remove overlapping notes of same pitch
        cleaned_notes = []
        for note in notes:
            # Check for overlap with recent notes of same pitch
            should_add = True
            for existing in cleaned_notes[-5:]:  # Check last 5 notes
                if (existing.pitch == note.pitch and 
                    existing.end > note.start + 0.05):  # 50ms tolerance
                    should_add = False
                    break
            
            if should_add:
                cleaned_notes.append(note)
        
        return cleaned_notes

    def process_midi(self, midi_path: str) -> str:
        """Process MIDI with quantization, hand separation and other enhancements"""
        logger.info("Processing MIDI with advanced features")

        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            logger.error(f"Failed to load MIDI file {midi_path}: {e}")
            return midi_path

        # Apply quantization if requested
        if self.config.get("quantize", True):
            pm = self._quantize_midi(pm)

        # Apply hand separation
        if self.config.get("separate_hands", True):
            pm = self._separate_hands(pm)

        # Apply piano-specific processing
        pm = self._process_piano_specifics(pm)

        # Save the processed MIDI
        output_path = midi_path.replace(".mid", "_processed.mid")
        try:
            pm.write(output_path)
            logger.info(f"Processed MIDI saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save processed MIDI: {e}")
            return midi_path

    def _quantize_midi(self, pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Quantize note timings to a musical grid"""
        division = self.config.get("quantize_division", 16)
        
        tempo_changes = pm.get_tempo_changes()
        tempo = tempo_changes[1][0] if len(tempo_changes[1]) > 0 else 120.0
        
        beat_length = 60.0 / tempo
        grid_size = beat_length / (division / 4)

        logger.info(f"Quantizing to {division} divisions per bar at {tempo} BPM")

        for instrument in pm.instruments:
            if instrument.is_drum:
                continue
                
            for note in instrument.notes:
                grid_position = round(note.start / grid_size)
                quantized_start = grid_position * grid_size
                
                original_duration = note.end - note.start
                duration_grid_units = max(1, round(original_duration / grid_size))
                
                note.start = quantized_start
                note.end = quantized_start + (duration_grid_units * grid_size)

        return pm

    def _separate_hands(self, pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Separate notes into left and right hand piano parts"""
        logger.info("Separating notes into left and right hand parts")

        new_pm = pretty_midi.PrettyMIDI()
        
        tempo_changes = pm.get_tempo_changes()
        if len(tempo_changes[1]) > 0:
            new_pm.tempo = tempo_changes[1][0]

        right_hand = pretty_midi.Instrument(program=0, name="Right Hand")
        left_hand = pretty_midi.Instrument(program=0, name="Left Hand")

        all_notes = []
        for instrument in pm.instruments:
            if not instrument.is_drum:
                all_notes.extend(instrument.notes)

        split_point = self.config.get("hand_split_point", 60)

        for note in all_notes:
            if note.pitch >= split_point:
                right_hand.notes.append(note)
            else:
                left_hand.notes.append(note)

        if right_hand.notes:
            new_pm.instruments.append(right_hand)
        if left_hand.notes:
            new_pm.instruments.append(left_hand)

        new_pm.time_signature_changes = pm.time_signature_changes.copy()
        return new_pm

    def _process_piano_specifics(self, pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Apply piano-specific processing"""
        piano_type = self.config.get("piano_type", "grand")

        program_map = {
            "grand": 0,
            "bright": 1,
            "digital": 4
        }
        program_num = program_map.get(piano_type, 0)

        for instrument in pm.instruments:
            if not instrument.is_drum:
                instrument.program = program_num

                if self.config.get("normalize_velocity", True):
                    self._normalize_velocity(instrument)

        return pm

    def _normalize_velocity(self, instrument: pretty_midi.Instrument) -> None:
        """Normalize note velocities"""
        if not instrument.notes:
            return

        velocities = [note.velocity for note in instrument.notes]
        if not velocities:
            return
            
        min_vel = min(velocities)
        max_vel = max(velocities)

        if min_vel >= 30 and max_vel <= 120:
            return

        if max_vel == min_vel:
            for note in instrument.notes:
                note.velocity = 75
        else:
            for note in instrument.notes:
                normalized = ((note.velocity - min_vel) / (max_vel - min_vel)) * 70 + 40
                note.velocity = int(max(1, min(127, normalized)))


def main():
    fixed_dir = r"C:\Users\Jonathan\Documents\midi"
    output_dir = fixed_dir

    valid_exts = [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"]

    files = [f for f in os.listdir(fixed_dir) if os.path.splitext(f)[1].lower() in valid_exts]

    if not files:
        print("No audio/video files found in", fixed_dir)
        return 1

    checkpoint_dir = None

    config = {
        "checkpoint_dir": checkpoint_dir,
        "output_dir": output_dir,
        "min_pitch": 21,
        "max_pitch": 108,
        "quantize": True,
        "quantize_division": 16,
        "separate_hands": True,
        "hand_split_point": 60,
        "piano_type": "grand",
        "normalize_velocity": True,
        "use_cli": True,
        "use_basic_pitch": True  # New option
    }

    for file in files:
        input_file = os.path.join(fixed_dir, file)
        try:
            print(f"Starting transcription of: {input_file}")
            transcriber = PianoTranscriber(config)
            midi_path = transcriber.transcribe_audio(input_file)
            print("Processing MIDI...")
            processed_midi = transcriber.process_midi(midi_path)
            print(f"✅ {file} → {os.path.basename(processed_midi)} 변환 완료!")
        except Exception as e:
            print(f"❌ {file} 변환 실패: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())