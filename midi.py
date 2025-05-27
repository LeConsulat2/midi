import librosa
import numpy as np
import scipy.signal
from scipy import ndimage
from sklearn.decomposition import NMF
import pretty_midi
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

class AdvancedPianoToMIDI:
    def __init__(self, sample_rate=22050, hop_length=256):  # Smaller hop for better temporal resolution
        self.sr = sample_rate
        self.hop_length = hop_length
        self.frame_time = hop_length / sample_rate
        
        # Piano-specific parameters - MUCH more conservative
        self.piano_range = (21, 108)  # A0 to C8
        self.min_note_duration = 0.1  # 100ms minimum (was too short at 50ms)
        self.max_polyphony = 6  # Reduced from 10 - most piano music has 4-6 simultaneous notes max
        
        # CQT parameters - MORE CONSERVATIVE for stability
        self.bins_per_octave = 12  # Standard semitone resolution (was 36 - too high)
        self.n_bins = 88  # Exact piano keys
        self.fmin = librosa.midi_to_hz(21)  # A0
        
        # Critical thresholds
        self.onset_threshold = 0.3  # Higher threshold to avoid false positives
        self.pitch_threshold_percentile = 85  # More selective
        
    def load_and_preprocess(self, audio_path):
        """Load and preprocess audio with CONSERVATIVE piano-specific optimizations"""
        print(f"Loading audio: {audio_path}")
        
        # Load audio with error handling
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        except Exception as e:
            raise Exception(f"Failed to load audio: {e}")
            
        if len(y) == 0:
            raise Exception("Empty audio file")
            
        print(f"Audio loaded: {len(y)/self.sr:.2f} seconds, {self.sr} Hz")
        
        # Remove DC offset
        y = y - np.mean(y)
        
        # Gentle pre-emphasis for piano clarity
        y = scipy.signal.lfilter([1, -0.95], [1], y)
        
        # Remove very low frequencies (below piano range)
        nyquist = self.sr / 2
        low_cutoff = 20 / nyquist  # 20 Hz
        high_cutoff = 8000 / nyquist  # 8 kHz (piano harmonics)
        
        b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        y = scipy.signal.filtfilt(b, a, y)
        
        # GENTLE normalization - preserve dynamics
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val * 0.8  # Leave headroom
        
        return y
    
    def extract_piano_optimized_cqt(self, y):
        """Extract CQT specifically optimized for piano transcription"""
        print("Computing Constant-Q Transform...")
        
        # Use standard CQT with piano-specific parameters
        cqt_complex = librosa.cqt(
            y, 
            sr=self.sr, 
            hop_length=self.hop_length,
            fmin=self.fmin, 
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            filter_scale=1.0,  # Standard filtering
            norm=1,
            sparsity=0.01
        )
        
        # Get magnitude
        cqt_mag = np.abs(cqt_complex)
        
        # Apply logarithmic compression to handle piano's wide dynamic range
        cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)
        
        # Normalize to 0-1 range
        cqt_normalized = (cqt_db - np.min(cqt_db)) / (np.max(cqt_db) - np.min(cqt_db))
        
        return cqt_normalized, cqt_complex
    
    def smart_onset_detection(self, y):
        """Robust onset detection for piano"""
        print("Detecting note onsets...")
        
        # Multiple onset detection methods
        onset_envelope = librosa.onset.onset_strength(
            y=y, 
            sr=self.sr, 
            hop_length=self.hop_length,
            aggregate=np.median,
            fmax=4000,
            center=True
        )
        
        # Peak picking with conservative parameters
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=self.sr,
            hop_length=self.hop_length,
            pre_max=3,    # Smaller window
            post_max=3,
            pre_avg=3,
            post_avg=3,
            delta=self.onset_threshold,  # Higher threshold
            wait=10       # Minimum time between onsets (frames)
        )
        
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=self.hop_length)
        print(f"Found {len(onset_times)} potential onsets")
        
        return onset_frames, onset_times, onset_envelope
    
    def piano_pitch_detection(self, cqt_normalized):
        """Piano-specific pitch detection with better accuracy"""
        print("Detecting pitches...")
        
        detected_notes = []
        n_frames = cqt_normalized.shape[1]
        
        # Calculate adaptive threshold for each MIDI note
        note_thresholds = np.percentile(cqt_normalized, self.pitch_threshold_percentile, axis=1)
        
        for frame_idx in range(n_frames):
            frame_time = frame_idx * self.frame_time
            spectrum = cqt_normalized[:, frame_idx]
            
            # Find active notes in this frame
            active_notes = []
            
            for midi_idx in range(self.n_bins):
                midi_note = 21 + midi_idx  # A0 = 21
                
                if spectrum[midi_idx] > note_thresholds[midi_idx]:
                    # Additional validation: check local peak
                    window_start = max(0, midi_idx - 1)
                    window_end = min(self.n_bins, midi_idx + 2)
                    local_window = spectrum[window_start:window_end]
                    
                    # Must be local maximum
                    if spectrum[midi_idx] == np.max(local_window):
                        confidence = spectrum[midi_idx]
                        active_notes.append({
                            'midi_note': midi_note,
                            'time': frame_time,
                            'confidence': confidence,
                            'frame': frame_idx
                        })
            
            # Keep only top notes by confidence
            active_notes.sort(key=lambda x: x['confidence'], reverse=True)
            detected_notes.extend(active_notes[:self.max_polyphony])
        
        print(f"Detected {len(detected_notes)} pitch events")
        return detected_notes
    
    def intelligent_note_tracking(self, detected_notes, onset_times):
        """Convert pitch detections to musical notes with smart tracking"""
        print("Tracking and consolidating notes...")
        
        if not detected_notes:
            return []
        
        # Group by MIDI note number
        note_groups = defaultdict(list)
        for note in detected_notes:
            note_groups[note['midi_note']].append(note)
        
        final_notes = []
        
        for midi_note, detections in note_groups.items():
            if len(detections) < 5:  # Need sufficient evidence
                continue
                
            # Sort by time
            detections.sort(key=lambda x: x['time'])
            
            # Group into note segments using onset information
            segments = self.segment_note_detections(detections, onset_times)
            
            # Convert segments to final notes
            for segment in segments:
                if len(segment) < 3:  # Too short to be a real note
                    continue
                    
                start_time = segment[0]['time']
                end_time = segment[-1]['time'] + self.frame_time
                duration = end_time - start_time
                
                if duration >= self.min_note_duration:
                    # Calculate velocity from confidence
                    avg_confidence = np.mean([d['confidence'] for d in segment])
                    # Scale velocity more conservatively
                    velocity = int(np.clip(30 + avg_confidence * 80, 30, 110))
                    
                    final_notes.append({
                        'midi_note': midi_note,
                        'start_time': start_time,
                        'end_time': end_time,
                        'velocity': velocity
                    })
        
        # Sort by start time
        final_notes.sort(key=lambda x: x['start_time'])
        
        print(f"Final notes: {len(final_notes)}")
        return final_notes
    
    def segment_note_detections(self, detections, onset_times):
        """Segment continuous detections into individual notes"""
        if not detections:
            return []
        
        segments = []
        current_segment = [detections[0]]
        
        # Maximum gap between detections in same note
        max_gap = self.frame_time * 8  # Allow some discontinuity
        
        for i in range(1, len(detections)):
            current_time = detections[i]['time']
            prev_time = detections[i-1]['time']
            time_gap = current_time - prev_time
            
            # Check if there's an onset between prev and current
            onset_between = any(prev_time < onset < current_time for onset in onset_times)
            
            if time_gap <= max_gap and not onset_between:
                # Continue current segment
                current_segment.append(detections[i])
            else:
                # Start new segment
                if len(current_segment) >= 3:
                    segments.append(current_segment)
                current_segment = [detections[i]]
        
        # Don't forget last segment
        if len(current_segment) >= 3:
            segments.append(current_segment)
        
        return segments
    
    def create_midi(self, notes, output_path):
        """Create MIDI file from detected notes"""
        if not notes:
            raise Exception("No notes detected - cannot create MIDI file")
        
        print(f"Creating MIDI file with {len(notes)} notes...")
        
        # Create MIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        
        # Create piano instrument
        piano = pretty_midi.Instrument(program=0, is_drum=False, name='Piano')
        
        for note_data in notes:
            try:
                note = pretty_midi.Note(
                    velocity=int(note_data['velocity']),
                    pitch=int(note_data['midi_note']),
                    start=float(note_data['start_time']),
                    end=float(note_data['end_time'])
                )
                piano.notes.append(note)
            except Exception as e:
                print(f"Warning: Skipping invalid note {note_data}: {e}")
                continue
        
        if not piano.notes:
            raise Exception("No valid notes created")
        
        midi.instruments.append(piano)
        
        # Write MIDI file
        try:
            midi.write(output_path)
            print(f"MIDI file saved: {output_path}")
        except Exception as e:
            raise Exception(f"Failed to write MIDI file: {e}")
        
        return midi
    
    def convert(self, input_path, output_path, debug=False):
        """Main conversion pipeline - IMPROVED"""
        try:
            # Step 1: Load and preprocess
            y = self.load_and_preprocess(input_path)
            
            # Step 2: Extract spectral features
            cqt_normalized, cqt_complex = self.extract_piano_optimized_cqt(y)
            
            # Step 3: Onset detection
            onset_frames, onset_times, onset_envelope = self.smart_onset_detection(y)
            
            # Step 4: Pitch detection
            detected_notes = self.piano_pitch_detection(cqt_normalized)
            
            # Step 5: Note tracking and consolidation
            final_notes = self.intelligent_note_tracking(detected_notes, onset_times)
            
            if not final_notes:
                print("⚠️  No notes detected in audio")
                return None, []
            
            # Step 6: Create MIDI
            midi = self.create_midi(final_notes, output_path)
            
            if debug:
                self.plot_analysis(y, cqt_normalized, onset_envelope, onset_frames, final_notes)
            
            return midi, final_notes
            
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return None, []
    
    def plot_analysis(self, y, cqt, onset_strength, onset_frames, notes):
        """Debug visualization - IMPROVED"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Waveform
        times = np.linspace(0, len(y)/self.sr, len(y))
        axes[0].plot(times, y)
        axes[0].set_title('Audio Waveform')
        axes[0].set_ylabel('Amplitude')
        
        # CQT spectrogram
        times_frames = librosa.frames_to_time(np.arange(cqt.shape[1]), sr=self.sr, hop_length=self.hop_length)
        midi_notes = np.arange(21, 109)  # A0 to C8
        
        im = axes[1].imshow(cqt, aspect='auto', origin='lower', 
                           extent=[times_frames[0], times_frames[-1], midi_notes[0], midi_notes[-1]],
                           cmap='magma')
        axes[1].set_title('Piano Roll Spectrogram (CQT)')
        axes[1].set_ylabel('MIDI Note')
        
        # Onset detection
        onset_times_plot = librosa.frames_to_time(np.arange(len(onset_strength)), 
                                                 sr=self.sr, hop_length=self.hop_length)
        axes[2].plot(onset_times_plot, onset_strength, label='Onset Strength', alpha=0.7)
        onset_time_points = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=self.hop_length)
        axes[2].vlines(onset_time_points, 0, np.max(onset_strength), 
                      color='red', alpha=0.8, label='Detected Onsets')
        axes[2].set_title('Onset Detection')
        axes[2].set_ylabel('Onset Strength')
        axes[2].legend()
        
        # Final notes (Piano roll)
        for note in notes:
            duration = note['end_time'] - note['start_time']
            alpha = min(1.0, note['velocity'] / 127.0)
            axes[3].barh(note['midi_note'], duration, left=note['start_time'], 
                        height=0.8, alpha=alpha, color='green', edgecolor='darkgreen')
        
        axes[3].set_title(f'Detected Notes ({len(notes)} notes)')
        axes[3].set_xlabel('Time (seconds)')
        axes[3].set_ylabel('MIDI Note')
        axes[3].set_ylim(20, 110)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Usage example
if __name__ == "__main__":
    # Try to import moviepy for video support
    try:
        import moviepy.editor as mp
        MOVIEPY_AVAILABLE = True
    except ImportError:
        MOVIEPY_AVAILABLE = False
        print("Warning: moviepy not installed. Video files will be skipped.")

    input_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = input_dir
    
    audio_formats = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]
    video_formats = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"]
    valid_formats = audio_formats + video_formats

    files = [f for f in os.listdir(input_dir)
             if os.path.splitext(f)[1].lower() in valid_formats]

    if not files:
        print(f"No audio/video files found in {input_dir}")
        sys.exit(1)

    converter = AdvancedPianoToMIDI()
    success_count = 0
    for file in files:
        input_path = os.path.join(input_dir, file)
        base_name, ext = os.path.splitext(file)
        output_file = os.path.join(output_dir, f"{base_name}_converted.mid")
        temp_audio = None
        try:
            print(f"\n🎵 Processing: {file}")
            # If video, extract audio
            if ext.lower() in video_formats:
                if not MOVIEPY_AVAILABLE:
                    print(f"❌ Skipping {file}: moviepy not installed.")
                    continue
                print("Extracting audio from video...")
                temp_audio = os.path.join(output_dir, f"{base_name}_temp.wav")
                video = mp.VideoFileClip(input_path)
                if video.audio is None:
                    print(f"❌ No audio track in video: {file}")
                    continue
                video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
                video.close()
                input_to_process = temp_audio
            else:
                input_to_process = input_path
            
            midi, notes = converter.convert(input_to_process, output_file, debug=False)
            if midi and notes:
                print(f"✅ Success: {os.path.basename(output_file)} ({len(notes)} notes)")
                success_count += 1
            else:
                print(f"❌ Failed: No notes detected in {file}")
            
            # Clean up temp audio
            if temp_audio and os.path.exists(temp_audio):
                os.remove(temp_audio)
        except Exception as e:
            print(f"❌ Failed to process {file}: {e}")
        
    print(f"\n🎹 Completed: {success_count}/{len(files)} files processed successfully")