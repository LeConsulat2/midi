#!/usr/bin/env python3
"""
piano_transcription.py - Specialized Piano Transcription Tool

Features:

1. Transcribes piano audio (WAV, MP3, etc.) to MIDI using Onsets-and-Frames
2. Intelligent left/right hand separation based on pitch and overlaps
3. Advanced MIDI post-processing:
   - Configurable quantization
   - Velocity normalization and dynamics preservation
   - Optional pedal detection and cleaning
4. Output options for different piano types (acoustic grand, digital piano)
   """
