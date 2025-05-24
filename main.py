import sys
import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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
    # Suppress tensorflow warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ImportError:
    print("Warning: TensorFlow not installed. Some transcription methods may not work.")
    print("Install with: pip install tensorflow")

# Try to import magenta
try:
    from magenta.models.onsets_frames_transcription import infer_util
    from magenta.models.onsets_frames_transcription import constants
    MAGENTA_AVAILABLE = True
except ImportError:
    print("Warning: Magenta not installed. Falling back to basic transcription.")
    print("For full functionality, install with: pip install magenta")
    MAGENTA_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("piano_transcription")


class PianoTranscriber:
    """Piano transcription pipeline specialized for piano recordings"""

    def __init__(self, config: Dict):
        self.config = config

    def transcribe_audio(self, input_audio: str) -> str:
        """
        Transcribes piano audio to MIDI using available methods
        Supports MP3, WAV, MP4, and other audio/video formats
        Returns path to the created MIDI file
        """
        logger.info(f"Transcribing piano audio: {input_audio}")
        
        # Check if input file exists
        if not os.path.isfile(input_audio):
            raise FileNotFoundError(f"Input audio file not found: {input_audio}")
        
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        # Handle video files (MP4, etc.) by extracting audio first
        audio_file = self._prepare_audio_file(input_audio, output_dir)

        # Try different transcription methods
        midi_path = None
        
        # Method 1: Try Magenta if available
        if MAGENTA_AVAILABLE and self.config.get("checkpoint_dir"):
            try:
                midi_path = self._transcribe_with_magenta(audio_file, output_dir)
            except Exception as e:
                logger.warning(f"Magenta transcription failed: {e}")
        
        # Method 2: Try command line tool if available
        if not midi_path and self.config.get("use_cli", True):
            try:
                midi_path = self._transcribe_with_cli(audio_file, output_dir)
            except Exception as e:
                logger.warning(f"CLI transcription failed: {e}")
        
        # Method 3: Fallback to basic onset detection
        if not midi_path:
            logger.info("Using fallback basic transcription method")
            midi_path = self._transcribe_basic(audio_file, output_dir)

        if not midi_path or not os.path.isfile(midi_path):
            raise FileNotFoundError(f"Transcription failed - no MIDI file created")

        logger.info(f"Raw transcription saved to: {midi_path}")
        return midi_path

    def _prepare_audio_file(self, input_file: str, output_dir: str) -> str:
        """
        Prepare audio file for transcription
        Extracts audio from video files (MP4, etc.) if needed
        Returns path to audio file ready for processing
        """
        file_ext = os.path.splitext(input_file)[1].lower()
        
        # Video file extensions that need audio extraction
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
        
        if file_ext in video_extensions:
            if not MOVIEPY_AVAILABLE:
                raise RuntimeError(
                    f"Video file detected ({file_ext}) but moviepy not installed. "
                    "Install with: pip install moviepy"
                )
            
            logger.info(f"Extracting audio from video file: {input_file}")
            
            # Create temporary audio file
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            temp_audio = os.path.join(output_dir, f"{base_name}_extracted.wav")
            
            try:
                # Extract audio using moviepy
                video = mp.VideoFileClip(input_file)
                audio = video.audio
                
                if audio is None:
                    raise RuntimeError("No audio track found in video file")
                
                # Write audio to temporary file
                audio.write_audiofile(temp_audio, verbose=False, logger=None)
                
                # Clean up
                audio.close()
                video.close()
                
                logger.info(f"Audio extracted to: {temp_audio}")
                return temp_audio
                
            except Exception as e:
                raise RuntimeError(f"Failed to extract audio from video: {e}")
        
        else:
            # Regular audio file, return as-is
            return input_file

    def _transcribe_with_magenta(self, input_audio: str, output_dir: str) -> str:
        """Transcribe using Magenta's onsets-and-frames model"""
        checkpoint = self.config["checkpoint_dir"]
        
        # Load audio
        audio, sr = librosa.load(input_audio, sr=constants.SAMPLE_RATE)
        
        # Run inference
        sequence = infer_util.infer(audio, checkpoint)
        
        # Save MIDI
        base = os.path.splitext(os.path.basename(input_audio))[0]
        midi_path = os.path.join(output_dir, f"{base}.mid")
        
        pretty_midi.sequence_proto_to_pretty_midi(sequence).write(midi_path)
        return midi_path

    def _transcribe_with_cli(self, input_audio: str, output_dir: str) -> str:
        """Try to use command line transcription tools"""
        # Check for various CLI tools
        cli_commands = [
            "onsets_frames_transcription_transcribe",
            "piano-transcription",
            "basic_pitch"  # Spotify's Basic Pitch
        ]
        
        for cmd_name in cli_commands:
            if self._check_command_exists(cmd_name):
                return self._run_cli_transcription(cmd_name, input_audio, output_dir)
        
        raise RuntimeError("No CLI transcription tools found")

    def _check_command_exists(self, command: str) -> bool:
        """Check if a command exists in the system PATH"""
        try:
            subprocess.run([command, "--help"], 
                         capture_output=True, check=False, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _run_cli_transcription(self, cmd_name: str, input_audio: str, output_dir: str) -> str:
        """Run CLI transcription command"""
        base = os.path.splitext(os.path.basename(input_audio))[0]
        midi_path = os.path.join(output_dir, f"{base}.mid")
        
        if cmd_name == "basic_pitch":
            # Spotify's Basic Pitch syntax
            cmd = ["basic-pitch", output_dir, input_audio]
        else:
            # Generic Magenta-style syntax
            cmd = [cmd_name, "--model_dir", self.config.get("checkpoint_dir", ""), 
                   input_audio, "--output_dir", output_dir]
            
            if self.config.get("min_pitch"):
                cmd.extend(["--min_pitch", str(self.config["min_pitch"])])
            if self.config.get("max_pitch"):
                cmd.extend(["--max_pitch", str(self.config["max_pitch"])])

        subprocess.run(cmd, check=True, timeout=300)  # 5 minute timeout
        
        # Look for created MIDI file
        possible_paths = [
            midi_path,
            os.path.join(output_dir, f"{base}_basic_pitch.mid"),
            os.path.join(output_dir, f"{base}_transcription.mid")
        ]
        
        for path in possible_paths:
            if os.path.isfile(path):
                return path
                
        raise FileNotFoundError("CLI transcription didn't create expected MIDI file")

    def _transcribe_basic(self, input_audio: str, output_dir: str) -> str:
        """Basic transcription using librosa onset detection"""
        logger.info("Using basic onset detection for transcription")
        
        # Load audio
        y, sr = librosa.load(input_audio, sr=22050)
        
        # Detect onsets
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Estimate pitches using chroma features
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Create a simple MIDI file
        pm = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        
        # Convert onsets to notes (very basic approach)
        for i, onset_time in enumerate(onset_times):
            if i < len(onset_times) - 1:
                duration = onset_times[i + 1] - onset_time
            else:
                duration = 0.5  # Default duration for last note
            
            # Find dominant pitch at this time
            frame_idx = librosa.time_to_frames(onset_time, sr=sr)
            if frame_idx < chromagram.shape[1]:
                pitch_class = np.argmax(chromagram[:, frame_idx])
                # Map to MIDI note (C4 = 60)
                midi_note = 60 + pitch_class
                
                # Create note
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=midi_note,
                    start=onset_time,
                    end=onset_time + min(duration, 2.0)  # Cap duration
                )
                piano.notes.append(note)
        
        pm.instruments.append(piano)
        
        # Save MIDI
        base = os.path.splitext(os.path.basename(input_audio))[0]
        midi_path = os.path.join(output_dir, f"{base}_basic.mid")
        pm.write(midi_path)
        
        return midi_path

    def process_midi(self, midi_path: str) -> str:
        """Process MIDI with quantization, hand separation and other enhancements"""
        logger.info("Processing MIDI with advanced features")

        try:
            # Load the MIDI file
            pm = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            logger.error(f"Failed to load MIDI file {midi_path}: {e}")
            return midi_path  # Return original path if processing fails

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
            return midi_path  # Return original if save fails

    def _quantize_midi(self, pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Quantize note timings to a musical grid"""
        division = self.config.get("quantize_division", 16)
        
        # Get tempo (default to 120 BPM if none found)
        tempo_changes = pm.get_tempo_changes()
        tempo = tempo_changes[1][0] if len(tempo_changes[1]) > 0 else 120.0
        
        beat_length = 60.0 / tempo
        grid_size = beat_length / (division / 4)

        logger.info(f"Quantizing to {division} divisions per bar at {tempo} BPM")

        for instrument in pm.instruments:
            if instrument.is_drum:
                continue
                
            for note in instrument.notes:
                # Quantize start time
                grid_position = round(note.start / grid_size)
                quantized_start = grid_position * grid_size
                
                # Preserve duration but quantize it too
                original_duration = note.end - note.start
                duration_grid_units = max(1, round(original_duration / grid_size))
                
                note.start = quantized_start
                note.end = quantized_start + (duration_grid_units * grid_size)

        return pm

    def _separate_hands(self, pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Separate notes into left and right hand piano parts (simplified)"""
        logger.info("Separating notes into left and right hand parts")

        # Create new MIDI object
        new_pm = pretty_midi.PrettyMIDI()
        
        # Get tempo
        tempo_changes = pm.get_tempo_changes()
        if len(tempo_changes[1]) > 0:
            new_pm.tempo_changes.append(
                pretty_midi.TempoChange(tempo_changes[1][0], 0)
            )

        # Create instruments for left and right hands
        right_hand = pretty_midi.Instrument(program=0, name="Right Hand")
        left_hand = pretty_midi.Instrument(program=0, name="Left Hand")

        # Collect all notes
        all_notes = []
        for instrument in pm.instruments:
            if not instrument.is_drum:
                all_notes.extend(instrument.notes)

        # Simple separation based on pitch
        split_point = self.config.get("hand_split_point", 60)  # Middle C

        for note in all_notes:
            if note.pitch >= split_point:
                right_hand.notes.append(note)
            else:
                left_hand.notes.append(note)

        # Add instruments to new MIDI
        if right_hand.notes:
            new_pm.instruments.append(right_hand)
        if left_hand.notes:
            new_pm.instruments.append(left_hand)

        # Copy time signatures
        for ts in pm.time_signature_changes:
            new_pm.time_signature_changes.append(ts)

        return new_pm

    def _process_piano_specifics(self, pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Apply piano-specific processing"""
        piano_type = self.config.get("piano_type", "grand")

        # Set program number based on piano type
        program_map = {
            "grand": 0,    # Acoustic Grand Piano
            "bright": 1,   # Bright Acoustic Piano
            "digital": 4   # Electric Piano
        }
        program_num = program_map.get(piano_type, 0)

        for instrument in pm.instruments:
            if not instrument.is_drum:
                instrument.program = program_num

                # Normalize velocities if requested
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

        # Skip if already in good range
        if min_vel >= 30 and max_vel <= 120:
            return

        # Normalize to range 40-110
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

    # 변환할 확장자 목록 (필요시 추가 가능)
    valid_exts = [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"]

    # 폴더 안의 모든 오디오/비디오 파일을 찾음
    files = [f for f in os.listdir(fixed_dir) if os.path.splitext(f)[1].lower() in valid_exts]

    if not files:
        print("No audio/video files found in", fixed_dir)
        return 1

    # 필요시 체크포인트 경로 지정 (없으면 None)
    checkpoint_dir = None  # 예: r"C:\\Users\\Jonathan\\Documents\\midi\\onsets_frames_checkpoint"

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
        "use_cli": True
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