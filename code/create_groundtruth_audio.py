import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

# Paths to your stems
base_dir = Path("..")
stems = {
    "vocal": "Nude_vocal.wav",
    "drum": "Nude_drum.wav",
    "bas": "Nude_bass.wav",
    "guitar": "Nude_guitar.wav",
    "synth": "Nude_synth.wav"
}

# Load all stems
sr = 44100
audio = {}
for name, path in stems.items():
    y, _ = librosa.load(base_dir / "audio" / path, sr=sr, mono=False)  # keep stereo
    audio[name] = y

# Mix 
instrumental = sum(audio[name] for name in stems if name != "vocal")
all_mix = sum(audio[name] for name in stems)

# Normalize to avoid clipping
max_amp = np.max(np.abs(instrumental))
if max_amp > 1:
    instrumental /= max_amp

# Save outputs
sf.write(base_dir / "audio" / "Nude_mix.wav", all_mix.T, sr)
sf.write(base_dir / "audio" / "Nude_instrumental.wav", instrumental.T, sr)