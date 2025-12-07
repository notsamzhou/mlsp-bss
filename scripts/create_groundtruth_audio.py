import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import yaml

script_dir = Path(__file__).parent.parent
audio_dir = script_dir / "audio" / "stems"
sr = 44100

def load_metadata(yaml_path):
    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)

    stems_info = []

    for stem_id, stem_data in meta["stems"].items():
        stem_file = stem_data["filename"]
        instrument = stem_data.get("instrument", "unknown")

        stems_info.append({
            "id": stem_id,
            "instrument": instrument,
            "filename": stem_file
        })

    return stems_info

def main():
    for audio_path in audio_dir.glob("[!.]*"):
        audio_fullname = audio_path.name
        print(f"Processing file: {audio_fullname}")
        if audio_fullname == "Radiohead_Nude":
            yaml_path = audio_path / f"{audio_fullname}_STEMINFO.yaml"
            with open(yaml_path, "r") as f:
                stems_info = yaml.safe_load(f)
        else:
            yaml_path = audio_path / f"{audio_fullname}_METADATA.yaml"
            stems_info = load_metadata(yaml_path)
        
        audio = {}
        for stem in stems_info:
            instr_name = stem["instrument"].replace(" ", "_")  # safe dictionary key
            file_path = audio_path / f"{audio_fullname}_STEMS" / stem["filename"]
            y, _ = librosa.load(file_path, sr=sr, mono=False)
            audio[instr_name] = y
            
        vocal_keys = [
            k for k in audio
            if "vocal" in k.lower() or "singer" in k.lower()
        ]
            
        # Mix
        vocal = sum(audio[k] for k in audio if k in vocal_keys)
        instrumental = sum(audio[k] for k in audio if k not in vocal_keys)
        all_mix = sum(audio[k] for k in audio)

        # Normalize to avoid clipping
        # Find max amplitude across both
        max_amp = np.max([np.abs(instrumental).max(), np.abs(all_mix).max(), np.abs(vocal).max()])

        # Apply same scaling to both
        if max_amp > 1:
            vocal /= max_amp
            instrumental /= max_amp
            all_mix /= max_amp

        # Save outputs
        sf.write(audio_dir.parent / "originals" / f"{audio_fullname}_mix.wav", all_mix.T, sr)
        sf.write(audio_dir.parent / "originals" / f"{audio_fullname}_vocal.wav", vocal.T, sr)
        sf.write(audio_dir.parent / "originals" / f"{audio_fullname}_instrumental.wav", instrumental.T, sr)

if __name__ == "__main__":
    main()