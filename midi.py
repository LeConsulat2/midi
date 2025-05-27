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
    def __init__(self, sample_rate=22050, hop_length=512):
        self.sr = sample_rate
        self.hop_length = hop_length
        self.frame_time = hop_length / sample_rate
        
        # Piano-specific parameters
        self.piano_range = (21, 108)  # A0 to C8
        self.min_note_duration = 0.05  # 50ms minimum note
        self.max_polyphony = 10  # Maximum simultaneous notes
        
        # CQT parameters optimized for piano
        self.bins_per_octave = 36  # Higher resolution for better pitch accuracy
        self.n_bins = 7 * self.bins_per_octave  # 7 octaves
        self.fmin = librosa.midi_to_hz(21)  # A0
        
    def load_and_preprocess(self, audio_path):
        """Load and preprocess audio with piano-specific optimizations"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Remove DC offset
        y = y - np.mean(y)
        
        # Apply gentle high-pass filter to remove rumble
        b, a = scipy.signal.butter(4, 80, btype='high', fs=self.sr)
        y = scipy.signal.filtfilt(b, a, y)
        
        # Dynamic range compression (gentle)
        y = np.sign(y) * np.power(np.abs(y), 0.8)
        
        # Normalize
        y = librosa.util.normalize(y)
        
        return y
    
    def extract_spectral_features(self, y):
        """Extract multiple spectral representations"""
        # High-resolution CQT for pitch detection
        cqt = np.abs(librosa.cqt(y, sr=self.sr, hop_length=self.hop_length,
                                 fmin=self.fmin, n_bins=self.n_bins,
                                 bins_per_octave=self.bins_per_octave,
                                 filter_scale=0.8))  # Sharper filters
        
        # Chroma for harmonic analysis
        chroma = librosa.feature.chroma_cqt(C=cqt, bins_per_octave=self.bins_per_octave)
        
        # Spectral centroid and rolloff for note quality assessment
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sr, hop_length=self.hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr, hop_length=self.hop_length)[0]
        
        return cqt, chroma, spectral_centroids, spectral_rolloff
    
    def advanced_onset_detection(self, y, cqt):
        """Multi-method onset detection with piano-specific tuning"""
        # Method 1: Spectral flux with adaptive threshold
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_length,
                                                 aggregate=np.median, fmax=4000)
        
        # Method 2: Complex domain flux from CQT
        cqt_complex = librosa.cqt(y, sr=self.sr, hop_length=self.hop_length,
                                  fmin=self.fmin, n_bins=self.n_bins,
                                  bins_per_octave=self.bins_per_octave)
        complex_flux = np.sum(np.abs(np.diff(cqt_complex, axis=1)), axis=0)
        complex_flux = np.concatenate([[0], complex_flux])
        
        # Method 3: High-frequency content for percussive onsets
        hfc = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_length,
                                          feature=librosa.feature.spectral_centroid)
        
        # Combine methods with weights
        combined_onset = 0.4 * onset_env + 0.4 * complex_flux + 0.2 * hfc
        
        # Adaptive peak picking
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=combined_onset,
            sr=self.sr,
            hop_length=self.hop_length,
            pre_max=5,
            post_max=5,
            pre_avg=5,
            post_avg=5,
            delta=0.1,
            wait=5
        )
        
        return onset_frames, combined_onset
    
    def harmonic_suppression(self, cqt):
        """Suppress harmonics to isolate fundamentals"""
        # Create harmonic suppression matrix
        suppressed_cqt = cqt.copy()
        
        for frame in range(cqt.shape[1]):
            spectrum = cqt[:, frame]
            
            # Find peaks
            peaks, properties = scipy.signal.find_peaks(
                spectrum, 
                height=np.max(spectrum) * 0.1,
                distance=3
            )
            
            if len(peaks) > 0:
                # For each peak, suppress its harmonics
                for peak in peaks:
                    # Calculate harmonic positions
                    fundamental_freq = self.fmin * (2 ** (peak / self.bins_per_octave))
                    
                    for harmonic in [2, 3, 4, 5, 6]:
                        harmonic_freq = fundamental_freq * harmonic
                        if harmonic_freq < self.sr / 2:
                            harmonic_bin = int(self.bins_per_octave * np.log2(harmonic_freq / self.fmin))
                            if harmonic_bin < len(spectrum):
                                # Reduce harmonic energy
                                suppression_factor = 0.3 / harmonic  # Stronger suppression for higher harmonics
                                suppressed_cqt[harmonic_bin, frame] *= suppression_factor
        
        return suppressed_cqt
    
    def polyphonic_nmf_separation(self, cqt, n_components=88):
        """Use NMF to separate overlapping notes"""
        # Transpose for NMF (features x samples)
        cqt_nmf = cqt.T
        
        # Apply NMF
        model = NMF(n_components=n_components, init='nndsvd', max_iter=200, 
                   alpha_W=0.1, alpha_H=0.1, random_state=42)
        W = model.fit_transform(cqt_nmf)  # Time x Components
        H = model.components_  # Components x Frequency
        
        # Reconstruct individual components
        separated_spectrograms = []
        for i in range(n_components):
            component_spectrogram = np.outer(W[:, i], H[i, :]).T
            separated_spectrograms.append(component_spectrogram)
        
        return separated_spectrograms, W, H
    
    def advanced_pitch_tracking(self, cqt, onset_frames):
        """Track pitches with temporal continuity and polyphony handling"""
        # Apply harmonic suppression
        clean_cqt = self.harmonic_suppression(cqt)
        
        # Apply median filtering to reduce noise
        clean_cqt = ndimage.median_filter(clean_cqt, size=(3, 1))
        
        # Adaptive threshold per frequency bin
        thresholds = np.percentile(clean_cqt, 75, axis=1, keepdims=True)
        
        detected_notes = []
        
        for frame_idx in range(clean_cqt.shape[1]):
            frame_spectrum = clean_cqt[:, frame_idx]
            frame_time = frame_idx * self.frame_time
            
            # Find peaks above adaptive threshold
            peaks, properties = scipy.signal.find_peaks(
                frame_spectrum,
                height=thresholds.flatten(),
                distance=2,  # Minimum separation between peaks
                prominence=0.1
            )
            
            # Convert peaks to MIDI notes
            frame_notes = []
            for peak in peaks:
                midi_note = 21 + (peak * 12 / self.bins_per_octave)
                
                # Round to nearest semitone
                midi_note = int(np.round(midi_note))
                
                # Only keep notes in piano range
                if self.piano_range[0] <= midi_note <= self.piano_range[1]:
                    confidence = frame_spectrum[peak]
                    frame_notes.append({
                        'midi_note': midi_note,
                        'time': frame_time,
                        'confidence': confidence,
                        'frame': frame_idx
                    })
            
            # Sort by confidence and keep top N
            frame_notes.sort(key=lambda x: x['confidence'], reverse=True)
            detected_notes.extend(frame_notes[:self.max_polyphony])
        
        return detected_notes
    
    def note_consolidation_and_tracking(self, detected_notes, onset_frames):
        """Convert detected pitches into proper note events with durations"""
        # Group detections by MIDI note
        note_groups = defaultdict(list)
        for note in detected_notes:
            note_groups[note['midi_note']].append(note)
        
        final_notes = []
        
        for midi_note, detections in note_groups.items():
            if not detections:
                continue
                
            # Sort by time
            detections.sort(key=lambda x: x['time'])
            
            # Group into continuous segments
            segments = []
            current_segment = [detections[0]]
            
            for i in range(1, len(detections)):
                time_gap = detections[i]['time'] - detections[i-1]['time']
                
                # If gap is small, continue segment
                if time_gap <= self.frame_time * 3:  # Allow small gaps
                    current_segment.append(detections[i])
                else:
                    # Start new segment
                    if len(current_segment) >= 3:  # Minimum segment length
                        segments.append(current_segment)
                    current_segment = [detections[i]]
            
            # Don't forget the last segment
            if len(current_segment) >= 3:
                segments.append(current_segment)
            
            # Convert segments to notes
            for segment in segments:
                start_time = segment[0]['time']
                end_time = segment[-1]['time'] + self.frame_time
                duration = end_time - start_time
                
                # Filter out very short notes (likely noise)
                if duration >= self.min_note_duration:
                    # Calculate average confidence as velocity
                    avg_confidence = np.mean([d['confidence'] for d in segment])
                    velocity = int(np.clip(avg_confidence * 127, 20, 127))
                    
                    final_notes.append({
                        'midi_note': midi_note,
                        'start_time': start_time,
                        'end_time': end_time,
                        'velocity': velocity
                    })
        
        return final_notes
    
    def create_midi(self, notes, output_path):
        """Create MIDI file from detected notes"""
        # Create a PrettyMIDI object
        midi = pretty_midi.PrettyMIDI()
        
        # Create piano instrument
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        
        for note_data in notes:
            note = pretty_midi.Note(
                velocity=note_data['velocity'],
                pitch=note_data['midi_note'],
                start=note_data['start_time'],
                end=note_data['end_time']
            )
            piano.notes.append(note)
        
        midi.instruments.append(piano)
        
        # Save MIDI file
        midi.write(output_path)
        
        return midi
    
    def convert(self, input_path, output_path, debug=False):
        """Main conversion pipeline"""
        print("Loading and preprocessing audio...")
        y = self.load_and_preprocess(input_path)
        
        print("Extracting spectral features...")
        cqt, chroma, centroids, rolloff = self.extract_spectral_features(y)
        
        print("Detecting onsets...")
        onset_frames, onset_strength = self.advanced_onset_detection(y, cqt)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=self.hop_length)
        
        print(f"Found {len(onset_frames)} onsets")
        
        print("Tracking pitches...")
        detected_notes = self.advanced_pitch_tracking(cqt, onset_frames)
        
        print(f"Detected {len(detected_notes)} note candidates")
        
        print("Consolidating notes...")
        final_notes = self.note_consolidation_and_tracking(detected_notes, onset_frames)
        
        print(f"Final note count: {len(final_notes)}")
        
        print("Creating MIDI file...")
        midi = self.create_midi(final_notes, output_path)
        
        if debug:
            self.plot_analysis(y, cqt, onset_strength, onset_frames, final_notes)
        
        return midi, final_notes
    
    def plot_analysis(self, y, cqt, onset_strength, onset_frames, notes):
        """Debug visualization"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Waveform
        times = librosa.frames_to_time(np.arange(len(y)), sr=self.sr)
        axes[0].plot(times, y)
        axes[0].set_title('Waveform')
        axes[0].set_ylabel('Amplitude')
        
        # CQT
        librosa.display.specshow(librosa.amplitude_to_db(cqt, ref=np.max),
                                sr=self.sr, hop_length=self.hop_length,
                                fmin=self.fmin, bins_per_octave=self.bins_per_octave,
                                x_axis='time', y_axis='cqt_note', ax=axes[1])
        axes[1].set_title('Constant-Q Transform')
        
        # Onset detection
        onset_times = librosa.frames_to_time(np.arange(len(onset_strength)), 
                                           sr=self.sr, hop_length=self.hop_length)
        axes[2].plot(onset_times, onset_strength, label='Onset Strength')
        onset_time_points = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=self.hop_length)
        axes[2].vlines(onset_time_points, 0, max(onset_strength), color='r', alpha=0.8, label='Onsets')
        axes[2].set_title('Onset Detection')
        axes[2].legend()
        
        # Piano roll
        for note in notes:
            axes[3].barh(note['midi_note'], note['end_time'] - note['start_time'],
                        left=note['start_time'], height=0.8,
                        alpha=0.7, color='blue')
        axes[3].set_title('Detected Notes (Piano Roll)')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('MIDI Note')
        axes[3].set_ylim(20, 110)
        
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
            print(f"✅ Success: {os.path.basename(output_file)} ({len(notes)} notes)")
            success_count += 1
            
            # Clean up temp audio
            if temp_audio and os.path.exists(temp_audio):
                os.remove(temp_audio)
        except Exception as e:
            print(f"❌ Failed to process {file}: {e}")
        
    print(f"\n🎹 Completed: {success_count}/{len(files)} files processed successfully")