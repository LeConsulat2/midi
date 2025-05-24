import sys
import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi

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
        Transcribes piano audio to MIDI using Onsets-and-Frames
        Returns path to the created MIDI file
        """
        logger.info(f"Transcribing piano audio: {input_audio}")
        checkpoint = self.config["checkpoint_dir"]
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        # Call the transcription binary
        cmd = [
            "onsets_frames_transcription_transcribe",
            "--model_dir",
            checkpoint,
            input_audio,
            "--output_dir",
            output_dir,
        ]

        if self.config.get("min_pitch"):
            cmd.extend(["--min_pitch", str(self.config["min_pitch"])])
        if self.config.get("max_pitch"):
            cmd.extend(["--max_pitch", str(self.config["max_pitch"])])

        subprocess.run(cmd, check=True)

        # The CLI writes a .mid with same base name as the input
        base = os.path.splitext(os.path.basename(input_audio))[0]
        midi_path = os.path.join(output_dir, f"{base}.mid")

        if not os.path.isfile(midi_path):
            raise FileNotFoundError(f"Expected MIDI file not created at {midi_path}")

        logger.info(f"Raw transcription saved to: {midi_path}")
        return midi_path

    def process_midi(self, midi_path: str) -> str:
        """Process MIDI with quantization, hand separation and other enhancements"""
        logger.info("Processing MIDI with advanced features")

        # Load the MIDI file
        pm = pretty_midi.PrettyMIDI(midi_path)

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
        pm.write(output_path)
        logger.info(f"Processed MIDI saved to: {output_path}")

        return output_path

    def _quantize_midi(self, pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Quantize note timings to a musical grid"""
        # Extract quantization parameters
        division = self.config.get(
            "quantize_division", 16
        )  # e.g., 16 = sixteenth notes
        swing = self.config.get("swing", 0.0)  # 0.0 = straight, 0.33 = medium swing

        tempo = (
            pm.get_tempo_changes()[1][0]
            if pm.get_tempo_changes()[1].size > 0
            else 120.0
        )
        beat_length = 60.0 / tempo
        grid_size = beat_length / (division / 4)  # convert to quarter note divisions

        logger.info(
            f"Quantizing to {division} divisions per bar with swing={swing:.2f}"
        )

        for instrument in pm.instruments:
            for note in instrument.notes:
                # Calculate which grid position this note is closest to
                grid_position = round(note.start / grid_size)
                quantized_start = grid_position * grid_size

                # Apply swing if needed (on every other division)
                if swing > 0 and grid_position % 2 == 1:
                    quantized_start += grid_size * swing

                # Calculate note duration in grid units, maintaining original duration
                duration_grid_units = round((note.end - note.start) / grid_size)
                if duration_grid_units < 1:
                    duration_grid_units = 1

                # Set the quantized times
                note.end = quantized_start + (duration_grid_units * grid_size)
                note.start = quantized_start

        return pm

    def _separate_hands(self, pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Separate notes into left and right hand piano parts"""
        logger.info("Separating notes into left and right hand parts")

        # Create new PrettyMIDI object
        new_pm = pretty_midi.PrettyMIDI(
            initial_tempo=(
                pm.get_tempo_changes()[1][0]
                if len(pm.get_tempo_changes()[1]) > 0
                else 120
            )
        )

        # Create instruments for left and right hands
        right_hand = pretty_midi.Instrument(
            program=self.config.get("piano_program", 0), name="Right Hand"
        )
        left_hand = pretty_midi.Instrument(
            program=self.config.get("piano_program", 0), name="Left Hand"
        )

        # Get all notes from existing instruments
        all_notes = []
        for instrument in pm.instruments:
            if not instrument.is_drum:  # Skip drum tracks if any
                all_notes.extend(instrument.notes)

        # Sort by start time, then pitch (highest first)
        all_notes.sort(key=lambda x: (x.start, -x.pitch))

        # Define the split point (middle C = 60 by default)
        split_point = self.config.get("hand_split_point", 60)

        # Set up parameters for the hand separation algorithm
        hand_max_polyphony = self.config.get("max_polyphony_per_hand", 5)
        overlap_threshold = self.config.get(
            "hand_overlap_threshold", 0.6
        )  # 0.6 = 60% overlap required

        # Group notes by time slices
        time_slices = {}
        for note in all_notes:
            start_time = round(note.start, 3)  # Round to nearest millisecond
            if start_time not in time_slices:
                time_slices[start_time] = []
            time_slices[start_time].append(note)

        # Process each time slice
        for time, notes in sorted(time_slices.items()):
            # Simple split: notes above split_point go to right hand
            lower_notes = [n for n in notes if n.pitch < split_point]
            upper_notes = [n for n in notes if n.pitch >= split_point]

            # Handle edge cases (notes near the split point)
            if (
                len(lower_notes) > hand_max_polyphony
                and len(upper_notes) < hand_max_polyphony
            ):
                # Move some notes from left to right
                overlap_candidates = [
                    n for n in lower_notes if n.pitch >= split_point - 12
                ]
                overlap_candidates.sort(
                    key=lambda x: -x.pitch
                )  # Sort by pitch, highest first

                # Move highest notes from left to right until polyphony is balanced
                while (
                    len(lower_notes) > hand_max_polyphony
                    and len(upper_notes) < hand_max_polyphony
                    and overlap_candidates
                ):
                    note_to_move = overlap_candidates.pop(0)
                    lower_notes.remove(note_to_move)
                    upper_notes.append(note_to_move)

            # Same for moving from right to left if needed
            elif (
                len(upper_notes) > hand_max_polyphony
                and len(lower_notes) < hand_max_polyphony
            ):
                overlap_candidates = [
                    n for n in upper_notes if n.pitch <= split_point + 12
                ]
                overlap_candidates.sort(
                    key=lambda x: x.pitch
                )  # Sort by pitch, lowest first

                while (
                    len(upper_notes) > hand_max_polyphony
                    and len(lower_notes) < hand_max_polyphony
                    and overlap_candidates
                ):
                    note_to_move = overlap_candidates.pop(0)
                    upper_notes.remove(note_to_move)
                    lower_notes.append(note_to_move)

            # Add the separated notes to their respective hands
            left_hand.notes.extend(lower_notes)
            right_hand.notes.extend(upper_notes)

        # Add instruments to the new MIDI
        new_pm.instruments.append(right_hand)
        new_pm.instruments.append(left_hand)

        # Copy tempo changes and time signature changes
        for ts in pm.time_signature_changes:
            new_pm.time_signature_changes.append(ts)

        for tc_time, tc_tempo in zip(*pm.get_tempo_changes()):
            new_pm.tempo_changes.append(pretty_midi.TempoChange(tc_tempo, tc_time))

        return new_pm

    def _process_piano_specifics(
        self, pm: pretty_midi.PrettyMIDI
    ) -> pretty_midi.PrettyMIDI:
        """Apply piano-specific processing (pedal, dynamics, etc.)"""
        # Get piano type
        piano_type = self.config.get("piano_type", "grand")

        # Set appropriate program numbers based on piano type
        program_num = 0  # Default: Acoustic Grand Piano
        if piano_type == "digital":
            program_num = 4  # Electric Piano
        elif piano_type == "bright":
            program_num = 1  # Bright Acoustic Piano

        for instrument in pm.instruments:
            # Set the program number
            instrument.program = program_num

            # Process dynamics if requested
            if self.config.get("normalize_velocity", True):
                self._normalize_velocity(instrument)

            # Process pedal if requested
            if self.config.get("clean_pedal", True):
                self._clean_pedal(instrument)

        return pm

    def _normalize_velocity(self, instrument: pretty_midi.Instrument) -> None:
        """Normalize note velocities while preserving dynamics"""
        if not instrument.notes:
            return

        # Calculate velocity statistics
        velocities = [note.velocity for note in instrument.notes]
        min_vel = min(velocities)
        max_vel = max(velocities)

        # Skip if already in good range
        if min_vel >= 30 and max_vel <= 120 and (max_vel - min_vel) >= 40:
            return

        # Target velocity range (ensures good dynamic range)
        target_min = 40
        target_max = 110

        # Prevent division by zero
        if max_vel == min_vel:
            # All notes have the same velocity
            for note in instrument.notes:
                note.velocity = 75
        else:
            # Scale velocities to the target range
            for note in instrument.notes:
                note.velocity = int(
                    ((note.velocity - min_vel) / (max_vel - min_vel))
                    * (target_max - target_min)
                    + target_min
                )

    def _clean_pedal(self, instrument: pretty_midi.Instrument) -> None:
        """Clean up pedal data for more realistic piano sound"""
        # Process only pedal control changes (CC 64 = sustain pedal)
        pedal_ccs = [cc for cc in instrument.control_changes if cc.number == 64]

        if not pedal_ccs:
            return

        # Sort by time
        pedal_ccs.sort(key=lambda x: x.time)

        # Remove redundant pedal events
        new_pedal_ccs = []
        last_value = -1

        for cc in pedal_ccs:
            # Skip if it's the same value as the last one
            if cc.value == last_value:
                continue

            # Store this as a good control change and remember its value
            new_pedal_ccs.append(cc)
            last_value = cc.value

        # Replace with cleaned pedal data
        instrument.control_changes = [
            cc for cc in instrument.control_changes if cc.number != 64
        ]
        instrument.control_changes.extend(new_pedal_ccs)


def main():
    """Main function to run the transcription pipeline"""
    parser = argparse.ArgumentParser(
        description="Advanced Piano Audio to MIDI Transcription"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input audio file (WAV, MP3, etc.)"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        required=True,
        help="Directory containing the Onsets-and-Frames pretrained model",
    )
    parser.add_argument(
        "--output", "-o", default="output", help="Directory for MIDI output files"
    )

    # Transcription parameters
    parser.add_argument(
        "--min-pitch",
        type=int,
        default=21,
        help="Minimum MIDI pitch to detect (default: 21, piano lowest A)",
    )
    parser.add_argument(
        "--max-pitch",
        type=int,
        default=108,
        help="Maximum MIDI pitch to detect (default: 108, piano highest C)",
    )

    # MIDI processing parameters
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=True,
        help="Quantize notes to musical grid",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_false",
        dest="quantize",
        help="Disable quantization",
    )
    parser.add_argument(
        "--quantize-division",
        type=int,
        default=16,
        help="Quantization grid divisions per bar (4=quarter, 8=eighth, 16=sixteenth)",
    )
    parser.add_argument(
        "--swing",
        type=float,
        default=0.0,
        help="Swing amount (0.0=straight, 0.33=medium swing)",
    )

    # Hand separation parameters
    parser.add_argument(
        "--separate-hands",
        action="store_true",
        default=True,
        help="Separate notes into left and right hand parts",
    )
    parser.add_argument(
        "--no-separate-hands",
        action="store_false",
        dest="separate_hands",
        help="Don't separate hands",
    )
    parser.add_argument(
        "--hand-split-point",
        type=int,
        default=60,
        help="MIDI note number for hand split point (default: 60, middle C)",
    )
    parser.add_argument(
        "--max-polyphony",
        type=int,
        default=5,
        help="Maximum polyphony per hand (default: 5)",
    )

    # Piano-specific parameters
    parser.add_argument(
        "--piano-type",
        choices=["grand", "digital", "bright"],
        default="grand",
        help="Type of piano sound (affects MIDI program selection)",
    )
    parser.add_argument(
        "--normalize-velocity",
        action="store_true",
        default=True,
        help="Normalize note velocities while preserving dynamics",
    )
    parser.add_argument(
        "--clean-pedal",
        action="store_true",
        default=True,
        help="Clean and optimize pedal data",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Prepare configuration
    config = {
        "checkpoint_dir": args.checkpoint,
        "output_dir": args.output,
        "min_pitch": args.min_pitch,
        "max_pitch": args.max_pitch,
        "quantize": args.quantize,
        "quantize_division": args.quantize_division,
        "swing": args.swing,
        "separate_hands": args.separate_hands,
        "hand_split_point": args.hand_split_point,
        "max_polyphony_per_hand": args.max_polyphony,
        "piano_type": args.piano_type,
        "normalize_velocity": args.normalize_velocity,
        "clean_pedal": args.clean_pedal,
    }

    # Run the transcription pipeline
    try:
        transcriber = PianoTranscriber(config)

        # Step 1: Transcribe audio to MIDI
        midi_path = transcriber.transcribe_audio(args.input)

        # Step 2: Process MIDI with advanced features
        processed_midi = transcriber.process_midi(midi_path)

        print(f"\nTranscription completed successfully!")
        print(f"Processed MIDI saved to: {processed_midi}")

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
