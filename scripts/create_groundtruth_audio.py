import librosa
import numpy as np
import soundfile as sf
from pathlib import Path


script_dir = Path(__file__).parent

# Paths to your stems
base_dir = script_dir.parent
stems = {
    "vocal": "Nude_vocal.wav",
    "drum": "Nude_drum.wav",
    "bass": "Nude_bass.wav",
    "guitar": "Nude_guitar.wav",
    "synth": "Nude_synth.wav"
}


def main():
    # Load all stems
    sr = 44100
    audio = {}
    for name, path in stems.items():
        y, _ = librosa.load(base_dir / "audio" / "stems" / path, sr=sr, mono=False)  # keep stereo
        audio[name] = y

    # Mix
    instrumental = sum(audio[name] for name in stems if name != "vocal")
    all_mix = sum(audio[name] for name in stems)

    # Normalize to avoid clipping
    # Find max amplitude across both
    max_amp = np.max([np.abs(instrumental).max(), np.abs(all_mix).max()])

    # Apply same scaling to both
    if max_amp > 1:
        instrumental /= max_amp
        all_mix /= max_amp

    # Save outputs
    sf.write(base_dir / "audio" / "originals" / "Nude_mix.wav", all_mix.T, sr)
    sf.write(base_dir / "audio" / "originals" / "Nude_instrumental.wav", instrumental.T, sr)


if __name__ == "__main__":
    main()